"""LLM-as-judge groundedness evaluation on a 30-question sample (§6.5, §7.3).

Protocol:
  1. Load Quiz-mode outputs for a stratified sample of 30 questions.
  2. For each: call judge LLM (opposite provider from answerer) with
     judge_prompts.GROUNDEDNESS_JUDGE_SYSTEM.
  3. Parse JSON scores.
  4. Write results/qualitative_sample.md with all entries.

Usage:
    python eval/run_groundedness_judge.py [--sample-size 30]
    python eval/run_groundedness_judge.py --validate-only   # first 10 for kappa check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
RESULTS_DIR = _REPO_ROOT / "results"
RAG_CSV_PATH = RESULTS_DIR / "eval_rag.csv"
# Sample from the MCQ pool (not the held-out test set) so QuizLoader can find the items
TEST_PATH = _REPO_ROOT / "data" / "afrimedqa_mcq_pool.jsonl"

from eval.judge_prompts import GROUNDEDNESS_JUDGE_SYSTEM
from tutor.llm_client import build_judge_client, build_llm_client
from tutor.orchestrator import handle_quiz_submit, _render_chunks
from tutor.quiz import QuizLoader
from tutor.retriever import build_retriever
from tutor.schemas import MCQOption, MCQItem, JudgeScore


def load_test_items() -> dict[str, MCQItem]:
    items: dict[str, MCQItem] = {}
    with open(TEST_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            options = raw.get("options", [])
            if isinstance(options, dict):
                options = [MCQOption(key=k, text=v) for k, v in options.items()]
            else:
                options = [MCQOption(**o) for o in options]
            raw["options"] = options
            item = MCQItem(**raw)
            items[item.question_id] = item
    return items


def _parse_judge_json(text: str) -> dict | None:
    """Extract JSON from judge response (it may have surrounding text)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _stratified_sample(items: dict[str, MCQItem], n: int = 30) -> list[MCQItem]:
    import random
    from collections import defaultdict

    by_specialty: dict[str, list[MCQItem]] = defaultdict(list)
    for item in items.values():
        by_specialty[item.specialty].append(item)

    rng = random.Random(99)
    chosen: list[MCQItem] = []
    specialties = sorted(by_specialty.keys())
    per_specialty = max(1, n // len(specialties))
    extra = n - per_specialty * len(specialties)

    for i, spec in enumerate(specialties):
        take = per_specialty + (1 if i < extra else 0)
        pool = by_specialty[spec]
        chosen.extend(rng.sample(pool, min(take, len(pool))))

    return chosen[:n]


def score_item(
    item: MCQItem,
    quiz_loader: QuizLoader,
    llm_client,
    retriever,
    judge_client,
) -> JudgeScore | None:
    # Re-run quiz submit to get explanation + chunks
    # We use the gold answer as the student's answer (ensures non-trivial explanation)
    try:
        resp = handle_quiz_submit(
            question_id=item.question_id,
            student_choice=item.gold_answer,
            llm_client=llm_client,
            retriever=retriever,
            quiz_loader=quiz_loader,
        )
    except Exception as e:
        log.error("Quiz submit failed for %s: %s", item.question_id, e)
        return None

    chunks_text = _render_chunks(resp.retrieved_chunks)
    judge_prompt = GROUNDEDNESS_JUDGE_SYSTEM.format(
        question=item.question,
        generated_explanation=resp.explanation,
        retrieved_chunks=chunks_text,
        gold_rationale=item.gold_rationale,
    )
    try:
        judge_result = judge_client.complete(
            system=judge_prompt, user="Please evaluate.", max_tokens=512
        )
    except Exception as e:
        log.error("Judge call failed for %s: %s", item.question_id, e)
        return None

    parsed = _parse_judge_json(judge_result.text)
    if not parsed:
        log.warning("Could not parse judge JSON for %s: %s", item.question_id, judge_result.text[:200])
        return None

    return JudgeScore(
        question_id=item.question_id,
        groundedness=int(parsed.get("groundedness", 0)),
        groundedness_justification=parsed.get("groundedness_justification", ""),
        citation_accuracy=int(parsed.get("citation_accuracy", 0)),
        citation_justification=parsed.get("citation_justification", ""),
        consistency=int(parsed.get("consistency", 0)),
        consistency_justification=parsed.get("consistency_justification", ""),
        generated_explanation=resp.explanation,
        retrieved_chunks_text=chunks_text,
        gold_rationale=item.gold_rationale,
    )


def write_qualitative_md(scores: list[JudgeScore], path: Path) -> None:
    lines = ["# AfriMed Tutor — Qualitative Groundedness Sample\n"]
    for i, s in enumerate(scores, 1):
        lines.append(f"## Entry {i} — {s.question_id}\n")
        lines.append(f"**Groundedness:** {s.groundedness}/2 — {s.groundedness_justification}\n")
        lines.append(f"**Citation Accuracy:** {s.citation_accuracy}/2 — {s.citation_justification}\n")
        lines.append(f"**Consistency with Gold:** {s.consistency}/2 — {s.consistency_justification}\n")
        lines.append(f"\n### Generated Explanation\n{s.generated_explanation}\n")
        lines.append(f"\n### Gold Rationale\n{s.gold_rationale}\n")
        lines.append(f"\n### Retrieved Chunks\n{s.retrieved_chunks_text}\n")
        lines.append("---\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s (%d entries).", path, len(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--validate-only", action="store_true",
                        help="Run on first 10 items only for manual kappa validation")
    args = parser.parse_args()

    n = 10 if args.validate_only else args.sample_size
    test_items = load_test_items()
    sample = _stratified_sample(test_items, n=n)
    log.info("Sample size: %d items.", len(sample))

    llm_client = build_llm_client()
    judge_client = build_judge_client()
    retriever = build_retriever()
    quiz_loader = QuizLoader()

    scores: list[JudgeScore] = []
    for i, item in enumerate(sample):
        log.info("[judge] %d/%d  %s", i + 1, len(sample), item.question_id)
        score = score_item(item, quiz_loader, llm_client, retriever, judge_client)
        if score:
            scores.append(score)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_name = "qualitative_validation.md" if args.validate_only else "qualitative_sample.md"
    write_qualitative_md(scores, RESULTS_DIR / out_name)

    if scores:
        avg_g = sum(s.groundedness for s in scores) / len(scores)
        avg_c = sum(s.citation_accuracy for s in scores) / len(scores)
        avg_k = sum(s.consistency for s in scores) / len(scores)
        log.info(
            "Mean scores — groundedness=%.2f  citation=%.2f  consistency=%.2f",
            avg_g, avg_c, avg_k,
        )


if __name__ == "__main__":
    main()
