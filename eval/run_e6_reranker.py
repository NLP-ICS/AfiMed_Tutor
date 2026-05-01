"""E6 — Cross-Encoder Re-Ranking (Person 3).

Compares two retriever conditions on the 100-question MCQ test set:
  - dense    : FAISS bi-encoder, top-5 (existing baseline, accuracy ~0.700)
  - reranked : dense retrieves 20 candidates → cross-encoder re-scores → top-5

Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~85 MB download, runs on CPU, no API key required.
  - Jointly encodes (query, passage) for a more accurate relevance score
    than the bi-encoder's independent embeddings.

Produces:
  results/eval_e6_reranker.csv          — per-question detail (both conditions)
  results/eval_e6_reranker_summary.csv  — accuracy + latency trade-off table

Usage:
    python eval/run_e6_reranker.py
    python eval/run_e6_reranker.py --backends dense          # dense only
    python eval/run_e6_reranker.py --backends reranked       # reranked only
    python eval/run_e6_reranker.py --n-candidates 30         # override candidate pool
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=True)
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = _REPO_ROOT / "results"
TEST_PATH = _REPO_ROOT / "data" / "afrimedqa_mcq_test.jsonl"

from tutor.llm_client import build_llm_client
from tutor.retriever import DenseRetriever, RerankedRetriever
from tutor.quiz import format_options
from tutor.schemas import MCQOption, MCQItem

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

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
    parts = [
        f"[Excerpt {i}] {c.source_doc} — {c.section_title}\n{c.text}"
        for i, c in enumerate(chunks, 1)
    ]
    return "\n\n---\n\n".join(parts) if parts else "(No relevant excerpts found.)"


def _extract_answer(text: str) -> str:
    text = text.strip()
    for char in text:
        if char.upper() in "ABCDE":
            return char.upper()
    return text[:1].upper() if text else "?"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-backend evaluation
# ---------------------------------------------------------------------------

def run_backend(
    backend_name: str,
    retriever,
    items: list[MCQItem],
    llm_client,
    delay: float = 0.5,
) -> list[dict]:
    rows: list[dict] = []
    for i, item in enumerate(items):
        log.info("[%s] %d/%d  %s", backend_name, i + 1, len(items), item.question_id)

        # Time retrieval separately so we can report the latency cost of re-ranking
        t0 = time.perf_counter()
        chunks = retriever.search(item.question)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        options_str = format_options(item.options)
        prompt = _RAG_PROMPT.format(
            retrieved_chunks=_render_chunks(chunks),
            question=item.question,
            options=options_str,
        )

        t1 = time.perf_counter()
        result = llm_client.complete(system=prompt, user="", max_tokens=10)
        llm_ms = (time.perf_counter() - t1) * 1000

        predicted = _extract_answer(result.text)
        gold = item.gold_answer.strip().upper()

        rows.append(
            {
                "backend": backend_name,
                "question_id": item.question_id,
                "specialty": item.specialty,
                "gold_answer": gold,
                "predicted_answer": predicted,
                "correct": int(predicted == gold),
                "n_chunks_returned": len(chunks),
                "mean_chunk_score": (
                    round(sum(c.score for c in chunks) / len(chunks), 4)
                    if chunks else 0.0
                ),
                "retrieval_ms": round(retrieval_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "end_to_end_ms": round(retrieval_ms + llm_ms, 1),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        )
        if delay > 0:
            time.sleep(delay)
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(all_rows: list[dict]) -> list[dict]:
    """Accuracy and mean latency per (backend, specialty) + overall."""
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        groups[(row["backend"], row["specialty"])].append(row)
        groups[(row["backend"], "OVERALL")].append(row)

    summary = []
    for (backend, specialty), rows in sorted(groups.items()):
        summary.append(
            {
                "backend": backend,
                "specialty": specialty,
                "n": len(rows),
                "accuracy": round(sum(int(r["correct"]) for r in rows) / len(rows), 4),
                "mean_retrieval_ms": round(
                    sum(float(r["retrieval_ms"]) for r in rows) / len(rows), 1
                ),
                "mean_end_to_end_ms": round(
                    sum(float(r["end_to_end_ms"]) for r in rows) / len(rows), 1
                ),
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E6: Cross-encoder re-ranking evaluation")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["dense", "reranked"],
        default=["dense", "reranked"],
        help="Which backends to evaluate (default: both)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=20,
        help="Number of dense candidates to pass to the cross-encoder (default: 20)",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace cross-encoder model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N questions per backend (default: all 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls to avoid rate limits (default: 0.5)",
    )
    args = parser.parse_args()

    items = load_test_items()
    log.info("Loaded %d test items.", len(items))

    eval_items = items[: args.limit] if args.limit else items
    if args.limit:
        log.info("Limiting to %d questions per backend.", args.limit)

    llm_client = build_llm_client()

    all_rows: list[dict] = []

    for name in args.backends:
        log.info("\n=== Evaluating backend: %s ===", name)
        if name == "dense":
            retriever = DenseRetriever()
        else:  # reranked
            log.info(
                "Cross-encoder model: %s  (n_candidates=%d)",
                args.cross_encoder_model,
                args.n_candidates,
            )
            retriever = RerankedRetriever(
                n_candidates=args.n_candidates,
                model_name=args.cross_encoder_model,
            )

        rows = run_backend(name, retriever, eval_items, llm_client, delay=args.delay)
        all_rows.extend(rows)

    # Merge with any previously saved results (so backends can be run separately)
    RESULTS_DIR.mkdir(exist_ok=True)
    detail_path = RESULTS_DIR / "eval_e6_reranker.csv"
    existing_rows: list[dict] = []
    if detail_path.exists():
        with open(detail_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r["backend"] not in args.backends]
        if existing_rows:
            log.info("Keeping %d existing rows from previous runs.", len(existing_rows))

    merged_rows = existing_rows + all_rows
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)
    log.info("Saved per-question results → %s", detail_path)

    # Save summary over all accumulated results
    summary = compute_summary(merged_rows)
    summary_path = RESULTS_DIR / "eval_e6_reranker_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    log.info("Saved summary → %s", summary_path)

    # Print overall accuracy + latency for each backend
    log.info("\n--- E6 Overall Results ---")
    for row in summary:
        if row["specialty"] == "OVERALL":
            log.info(
                "  %-12s  accuracy=%.4f  mean_retrieval_ms=%.1f  mean_e2e_ms=%.1f",
                row["backend"],
                row["accuracy"],
                row["mean_retrieval_ms"],
                row["mean_end_to_end_ms"],
            )


if __name__ == "__main__":
    main()
