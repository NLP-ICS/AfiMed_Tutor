"""Quantitative MCQ evaluation — baseline and RAG conditions (§7.1).

Produces:
  results/eval_baseline.csv   (LLM, no retrieval)
  results/eval_rag.csv        (LLM + RAG pipeline)

Usage:
    python eval/run_mcq_eval.py
    python eval/run_mcq_eval.py --condition baseline
    python eval/run_mcq_eval.py --condition rag
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "results"
TEST_PATH = _REPO_ROOT / "data" / "afrimedqa_mcq_test.jsonl"

import sys
sys.path.insert(0, str(_REPO_ROOT))

from tutor.llm_client import build_llm_client
from tutor.retriever import build_retriever
from tutor.quiz import format_options
from tutor.schemas import MCQOption, MCQItem


_BASELINE_PROMPT = """\
You are a medical knowledge assessment system. Answer the following \
multiple-choice question by selecting the single best answer. Reply with \
ONLY the option letter (A, B, C, or D). Do not explain.

QUESTION: {question}
OPTIONS:
{options}

Answer:\
"""

_RAG_PROMPT = """\
You are a medical knowledge assessment system. Using ONLY the provided \
guideline excerpts and your clinical reasoning, select the single best answer. \
Reply with ONLY the option letter (A, B, C, or D). Do not explain.

GUIDELINE EXCERPTS:
{retrieved_chunks}

QUESTION: {question}
OPTIONS:
{options}

Answer:\
"""


def _render_chunks(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Excerpt {i}] {c.source_doc} — {c.section_title}\n{c.text}")
    return "\n\n---\n\n".join(parts) if parts else "(No relevant excerpts found.)"


def _extract_answer(text: str) -> str:
    text = text.strip()
    for char in text:
        if char.upper() in "ABCDE":
            return char.upper()
    return text[:1].upper()


def load_test_items() -> list[MCQItem]:
    items: list[MCQItem] = []
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
            items.append(MCQItem(**raw))
    return items


def run_condition(
    condition: str,
    items: list[MCQItem],
    llm_client,
    retriever=None,
) -> list[dict]:
    rows: list[dict] = []
    for i, item in enumerate(items):
        log.info("[%s] %d/%d  %s", condition, i + 1, len(items), item.question_id)
        options_str = format_options(item.options)
        t0 = time.perf_counter()

        if condition == "baseline":
            prompt = _BASELINE_PROMPT.format(
                question=item.question, options=options_str
            )
            result = llm_client.complete(system=prompt, user="", max_tokens=10)
        else:
            chunks = retriever.search(item.question)
            prompt = _RAG_PROMPT.format(
                retrieved_chunks=_render_chunks(chunks),
                question=item.question,
                options=options_str,
            )
            result = llm_client.complete(system=prompt, user="", max_tokens=10)

        latency_ms = (time.perf_counter() - t0) * 1000
        predicted = _extract_answer(result.text)
        correct = predicted == item.gold_answer.strip().upper()

        rows.append(
            {
                "question_id": item.question_id,
                "specialty": item.specialty,
                "gold_answer": item.gold_answer,
                "predicted_answer": predicted,
                "correct": int(correct),
                "latency_ms": round(latency_ms, 1),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    accuracy = sum(r["correct"] for r in rows) / len(rows)
    log.info("Saved %s  (accuracy=%.3f)", path.name, accuracy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["baseline", "rag", "both"],
        default="both",
    )
    args = parser.parse_args()

    items = load_test_items()
    log.info("Loaded %d test items.", len(items))

    llm_client = build_llm_client()
    retriever = build_retriever() if args.condition in ("rag", "both") else None

    if args.condition in ("baseline", "both"):
        rows = run_condition("baseline", items, llm_client)
        write_csv(rows, RESULTS_DIR / "eval_baseline.csv")

    if args.condition in ("rag", "both"):
        rows = run_condition("rag", items, llm_client, retriever=retriever)
        write_csv(rows, RESULTS_DIR / "eval_rag.csv")


if __name__ == "__main__":
    main()
