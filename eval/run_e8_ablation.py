from __future__ import annotations

"""E8 — Prompt Ablation: Citation and Refusal Instructions (§6.5).

Measures the effect of removing individual prompt constraints on output
quality as scored by the LLM judge.

Usage (Colab):
    !python eval/run_e8_ablation.py --sample-size 5 --variant baseline
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

RESULTS_DIR = _REPO_ROOT / "results" / "e8_ablation"
DATA_PATH = _REPO_ROOT / "data" / "afrimedqa_mcq_pool.jsonl"

from eval.judge_prompts import GROUNDEDNESS_JUDGE_SYSTEM
from tutor.llm_client import build_judge_client, build_llm_client
from tutor.orchestrator import _render_chunks
from tutor.prompts_ablation import ABLATION_VARIANTS, VARIANT_LABELS
from tutor.quiz import QuizLoader
from tutor.retriever import build_retriever
from tutor.schemas import MCQItem, MCQOption


def load_items() -> dict[str, MCQItem]:
    items: dict[str, MCQItem] = {}
    with open(DATA_PATH) as f:
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


def stratified_sample(items: dict[str, MCQItem], n: int = 30) -> list[MCQItem]:
    import random
    from collections import defaultdict
    by_specialty: dict[str, list[MCQItem]] = defaultdict(list)
    for item in items.values():
        by_specialty[item.specialty].append(item)
    rng = random.Random(42)
    chosen: list[MCQItem] = []
    specialties = sorted(by_specialty.keys())
    per_specialty = max(1, n // len(specialties))
    extra = n - per_specialty * len(specialties)
    for i, spec in enumerate(specialties):
        take = per_specialty + (1 if i < extra else 0)
        pool = by_specialty[spec]
        chosen.extend(rng.sample(pool, min(take, len(pool))))
    return chosen[:n]


def parse_judge_json(text: str) -> dict | None:
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


def generate_explanation(item, prompt_template, llm_client, retriever, quiz_loader):
    try:
        chunks = retriever.search(item.question)
        chunks_text = _render_chunks(chunks)
        options_text = "\n".join(f"{opt.key}. {opt.text}" for opt in item.options)
        system_prompt = prompt_template.format(
            question=item.question,
            options=options_text,
            student_choice=item.gold_answer,
            gold_answer=item.gold_answer,
            gold_rationale=item.gold_rationale or "(not provided)",
            retrieved_chunks=chunks_text,
        )
        result = llm_client.complete(system=system_prompt, user="Please provide your explanation.", max_tokens=1024, temperature=0.0)
        return result.text, chunks_text
    except Exception as e:
        log.error("Explanation failed for %s: %s", item.question_id, e)
        return None


def judge_explanation(item, explanation, chunks_text, judge_client):
    judge_prompt = GROUNDEDNESS_JUDGE_SYSTEM.format(
        question=item.question,
        generated_explanation=explanation,
        retrieved_chunks=chunks_text,
        gold_rationale=item.gold_rationale or "(not provided)",
    )
    try:
        result = judge_client.complete(system=judge_prompt, user="Please evaluate.", max_tokens=512, temperature=0.0)
        return parse_judge_json(result.text)
    except Exception as e:
        log.error("Judge failed for %s: %s", item.question_id, e)
        return None


def mean(values):
    return sum(values) / len(values) if values else 0.0


def run_variant(variant_key, sample, llm_client, judge_client, retriever, quiz_loader):
    prompt_template = ABLATION_VARIANTS[variant_key]
    label = VARIANT_LABELS[variant_key]
    log.info("=" * 55)
    log.info("Running variant: %s", label)
    log.info("=" * 55)
    results = []
    for i, item in enumerate(sample):
        log.info("[%s] %d/%d  %s", variant_key, i + 1, len(sample), item.question_id)
        output = generate_explanation(item, prompt_template, llm_client, retriever, quiz_loader)
        if not output:
            continue
        explanation, chunks_text = output
        scores = judge_explanation(item, explanation, chunks_text, judge_client)
        if not scores:
            continue
        results.append({
            "question_id": item.question_id,
            "specialty": item.specialty,
            "variant": variant_key,
            "groundedness": int(scores.get("groundedness", 0)),
            "groundedness_justification": scores.get("groundedness_justification", ""),
            "citation_accuracy": int(scores.get("citation_accuracy", 0)),
            "citation_justification": scores.get("citation_justification", ""),
            "consistency": int(scores.get("consistency", 0)),
            "consistency_justification": scores.get("consistency_justification", ""),
            "explanation": explanation,
        })
    return results


def generate_report(all_results):
    lines = [
        "# E8 — Prompt Ablation Results",
        "## AfriMed Tutor | ICS4554 NLP | Ashesi University\n",
        "### Mean Scores Per Variant\n",
        "| Variant | Groundedness | Citation Accuracy | Consistency |",
        "|---------|-------------|-------------------|-------------|",
    ]
    summary = {}
    for key in ["baseline", "no_citation", "no_refusal", "no_outside_ban"]:
        if key not in all_results or not all_results[key]:
            continue
        results = all_results[key]
        g = mean([r["groundedness"] for r in results])
        c = mean([r["citation_accuracy"] for r in results])
        k = mean([r["consistency"] for r in results])
        summary[key] = {"groundedness": g, "citation_accuracy": c, "consistency": k}
        lines.append(f"| {VARIANT_LABELS[key]} | {g:.2f} | {c:.2f} | {k:.2f} |")
    if "baseline" in summary:
        lines += ["\n### Delta vs Baseline\n",
                  "| Variant | ΔGroundedness | ΔCitation | ΔConsistency |",
                  "|---------|--------------|-----------|--------------|"]
        base = summary["baseline"]
        for key in ["no_citation", "no_refusal", "no_outside_ban"]:
            if key not in summary:
                continue
            s = summary[key]
            lines.append(f"| {VARIANT_LABELS[key]} | {s['groundedness']-base['groundedness']:+.2f} | {s['citation_accuracy']-base['citation_accuracy']:+.2f} | {s['consistency']-base['consistency']:+.2f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(ABLATION_VARIANTS.keys()) + ["all"], default="all")
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()

    log.info("Loading MCQ items...")
    items = load_items()
    sample = stratified_sample(items, n=args.sample_size)
    log.info("Sample: %d items", len(sample))

    llm_client = build_llm_client()
    judge_client = build_llm_client()  # use paxsenix for both answerer and judge
    retriever = build_retriever()
    quiz_loader = QuizLoader()

    variants_to_run = list(ABLATION_VARIANTS.keys()) if args.variant == "all" else [args.variant]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}

    for variant_key in variants_to_run:
        result_path = RESULTS_DIR / f"{variant_key}.json"
        if result_path.exists():
            log.info("Loading cached results for: %s", variant_key)
            all_results[variant_key] = json.loads(result_path.read_text())
            continue
        results = run_variant(variant_key, sample, llm_client, judge_client, retriever, quiz_loader)
        all_results[variant_key] = results
        result_path.write_text(json.dumps(results, indent=2))
        log.info("Saved %d results to %s", len(results), result_path)
        if results:
            log.info("  %s — groundedness=%.2f  citation=%.2f  consistency=%.2f",
                     VARIANT_LABELS[variant_key],
                     mean([r["groundedness"] for r in results]),
                     mean([r["citation_accuracy"] for r in results]),
                     mean([r["consistency"] for r in results]))

    if len(all_results) == len(ABLATION_VARIANTS):
        report = generate_report(all_results)
        report_path = RESULTS_DIR / "e8_report.md"
        report_path.write_text(report)
        log.info("Report saved to %s", report_path)
        print("\n" + "=" * 60)
        print(report)

if __name__ == "__main__":
    main()
