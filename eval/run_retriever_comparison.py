"""Dense vs sparse retriever comparison on the 100-question test set (§7.2).

Produces:
  results/eval_retriever_comparison.csv

Columns: question_id, specialty, gold_answer, predicted_dense,
         predicted_sparse, correct_dense, correct_sparse, latency_ms_dense,
         latency_ms_sparse

Usage:
    python eval/run_retriever_comparison.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
RESULTS_DIR = _REPO_ROOT / "results"
TEST_PATH = _REPO_ROOT / "data" / "afrimedqa_mcq_test.jsonl"

from tutor.llm_client import build_llm_client
from tutor.retriever import DenseRetriever, SparseRetriever
from tutor.quiz import format_options
from tutor.schemas import MCQOption, MCQItem

_RAG_PROMPT = """\
You are a medical assessment system. Using the provided guideline excerpts, \
select the single best answer. Reply with ONLY the option letter.

GUIDELINE EXCERPTS:
{retrieved_chunks}

QUESTION: {question}
OPTIONS:
{options}

Answer:\
"""


def _render_chunks(chunks) -> str:
    parts = [
        f"[Excerpt {i}] {c.source_doc} — {c.section_title}\n{c.text}"
        for i, c in enumerate(chunks, 1)
    ]
    return "\n\n---\n\n".join(parts) if parts else "(No excerpts found.)"


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


def main() -> None:
    items = load_test_items()
    log.info("Loaded %d test items.", len(items))

    llm_client = build_llm_client()
    dense = DenseRetriever()
    sparse = SparseRetriever()

    rows: list[dict] = []
    for i, item in enumerate(items):
        log.info("%d/%d  %s", i + 1, len(items), item.question_id)
        options_str = format_options(item.options)
        gold = item.gold_answer.strip().upper()

        # Dense
        t0 = time.perf_counter()
        d_chunks = dense.search(item.question)
        d_prompt = _RAG_PROMPT.format(
            retrieved_chunks=_render_chunks(d_chunks),
            question=item.question,
            options=options_str,
        )
        d_result = llm_client.complete(system=d_prompt, user="", max_tokens=10)
        d_latency = (time.perf_counter() - t0) * 1000
        d_pred = _extract_answer(d_result.text)

        # Sparse
        t0 = time.perf_counter()
        s_chunks = sparse.search(item.question)
        s_prompt = _RAG_PROMPT.format(
            retrieved_chunks=_render_chunks(s_chunks),
            question=item.question,
            options=options_str,
        )
        s_result = llm_client.complete(system=s_prompt, user="", max_tokens=10)
        s_latency = (time.perf_counter() - t0) * 1000
        s_pred = _extract_answer(s_result.text)

        rows.append(
            {
                "question_id": item.question_id,
                "specialty": item.specialty,
                "gold_answer": gold,
                "predicted_dense": d_pred,
                "predicted_sparse": s_pred,
                "correct_dense": int(d_pred == gold),
                "correct_sparse": int(s_pred == gold),
                "latency_ms_dense": round(d_latency, 1),
                "latency_ms_sparse": round(s_latency, 1),
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "eval_retriever_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    acc_d = sum(r["correct_dense"] for r in rows) / n
    acc_s = sum(r["correct_sparse"] for r in rows) / n
    log.info("Saved %s", out_path)
    log.info("Dense accuracy=%.3f  Sparse accuracy=%.3f", acc_d, acc_s)


if __name__ == "__main__":
    main()
