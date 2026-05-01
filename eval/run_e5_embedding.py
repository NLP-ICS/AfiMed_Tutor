"""E5 — Local Embedding Model Comparison (Person 3).

Compares three freely available local embedding models on:
  - MCQ accuracy on the 100-question test set
  - Corpus embedding time
  - Model size (parameter count / disk footprint)
  - Retrieval score distributions

Models:
  1. all-MiniLM-L6-v2       (current default, 384-dim, ~22M params, fast)
  2. all-mpnet-base-v2       (768-dim, ~110M params, higher quality general-purpose)
  3. medicalai/ClinicalBERT  (768-dim, ~110M params, clinical domain-specific)

All three run locally — no API key required.
Models are downloaded automatically from HuggingFace on first run (~200–400 MB each).

Produces:
  results/eval_e5_embedding.csv          — per-question accuracy for each model
  results/eval_e5_embedding_summary.csv  — accuracy / timing / size comparison table
  corpus/faiss_<model_slug>.index        — per-model FAISS indexes (reused on re-run)

Usage:
    python eval/run_e5_embedding.py
    python eval/run_e5_embedding.py --models all-MiniLM-L6-v2 all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=True)
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = _REPO_ROOT / "results"
CORPUS_DIR = _REPO_ROOT / "corpus"
TEST_PATH = _REPO_ROOT / "data" / "afrimedqa_mcq_test.jsonl"
CHUNKS_META_PATH = CORPUS_DIR / "chunks_meta.jsonl"

# ---------------------------------------------------------------------------
# Models to compare (E5 spec)
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "medicalai/ClinicalBERT",
]

# Filesystem-safe slug for each model name
def _slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Prompt (same as run_mcq_eval.py RAG prompt for fair comparison)
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

def load_chunks_meta() -> list[dict]:
    meta: list[dict] = []
    with open(CHUNKS_META_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return meta


def load_test_items():
    from tutor.schemas import MCQOption, MCQItem
    items = []
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
# Per-model FAISS index (build or load from cache)
# ---------------------------------------------------------------------------

def build_or_load_index(model_name: str, chunks_meta: list[dict]) -> tuple:
    """Return (faiss_index, embedding_time_s, model_size_mb).

    If a cached index for this model already exists, load it (embedding_time=0).
    Otherwise embed all chunks and build a fresh IndexFlatIP.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    index_path = CORPUS_DIR / f"faiss_{_slug(model_name)}.index"

    model = SentenceTransformer(model_name)

    # Model size: sum of parameter bytes → MB
    try:
        import torch
        n_params = sum(p.numel() for p in model[0].auto_model.parameters())
        # rough: float32 = 4 bytes per param
        model_size_mb = round(n_params * 4 / 1024 / 1024, 1)
    except Exception:
        model_size_mb = -1

    if index_path.exists():
        log.info("[%s] Loading cached FAISS index from %s", model_name, index_path.name)
        index = faiss.read_index(str(index_path))
        embedding_time_s = 0.0
        return index, embedding_time_s, model_size_mb

    log.info("[%s] Building FAISS index — embedding %d chunks…", model_name, len(chunks_meta))
    texts = [m["text"] for m in chunks_meta]

    t0 = time.perf_counter()
    batch_size = 64
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)
        log.info(
            "  [%s] embedded %d/%d chunks",
            model_name,
            min(i + batch_size, len(texts)),
            len(texts),
        )
    embedding_time_s = time.perf_counter() - t0

    vecs_np = np.vstack(all_vecs).astype("float32")
    # Normalise for cosine similarity via inner product
    norms = np.linalg.norm(vecs_np, axis=1, keepdims=True)
    vecs_np = vecs_np / np.where(norms > 0, norms, 1)

    dim = vecs_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs_np)
    faiss.write_index(index, str(index_path))
    log.info(
        "[%s] Index saved (%d vectors, dim=%d) in %.1fs",
        model_name,
        index.ntotal,
        dim,
        embedding_time_s,
    )
    return index, embedding_time_s, model_size_mb


# ---------------------------------------------------------------------------
# Per-model retrieval (inline, bypasses DenseRetriever to swap the index)
# ---------------------------------------------------------------------------

def retrieve_with_model(
    query: str,
    model,
    index,
    chunks_meta: list[dict],
    k: int = 5,
    threshold: float = 0.30,
):
    from tutor.schemas import Chunk

    vec = model.encode(query, normalize_embeddings=True).astype("float32").reshape(1, -1)
    scores, indices = index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or float(score) < threshold:
            continue
        m = chunks_meta[idx]
        results.append(
            Chunk(
                chunk_id=m["chunk_id"],
                text=m["text"],
                source_doc=m["source_doc"],
                section_title=m["section_title"],
                condition=m.get("condition"),
                page_number=m.get("page_number"),
                score=float(score),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    index,
    embedding_model,
    chunks_meta: list[dict],
    items,
    llm_client,
    delay: float = 0.5,
) -> list[dict]:
    from tutor.quiz import format_options

    rows: list[dict] = []
    for i, item in enumerate(items):
        log.info("[%s] %d/%d  %s", model_name, i + 1, len(items), item.question_id)

        t0 = time.perf_counter()
        chunks = retrieve_with_model(item.question, embedding_model, index, chunks_meta)
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
                "model": model_name,
                "question_id": item.question_id,
                "specialty": item.specialty,
                "gold_answer": gold,
                "predicted_answer": predicted,
                "correct": int(predicted == gold),
                "n_chunks_returned": len(chunks),
                "mean_retrieval_score": (
                    round(sum(c.score for c in chunks) / len(chunks), 4)
                    if chunks else 0.0
                ),
                "retrieval_ms": round(retrieval_ms, 1),
                "llm_ms": round(llm_ms, 1),
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

def compute_summary(
    all_rows: list[dict],
    timing_info: dict[str, dict],
) -> list[dict]:
    """One summary row per model with accuracy + corpus-level metadata."""
    from collections import defaultdict

    model_corrects: dict[str, list[int]] = defaultdict(list)
    model_scores: dict[str, list[float]] = defaultdict(list)

    for row in all_rows:
        model_corrects[row["model"]].append(row["correct"])
        if row["mean_retrieval_score"] > 0:
            model_scores[row["model"]].append(row["mean_retrieval_score"])

    summary = []
    for model_name in timing_info:
        corrects = model_corrects.get(model_name, [])
        scores = model_scores.get(model_name, [])
        info = timing_info[model_name]
        summary.append(
            {
                "model": model_name,
                "n_questions": len(corrects),
                "accuracy": round(sum(corrects) / len(corrects), 4) if corrects else 0.0,
                "mean_retrieval_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "embedding_dim": info.get("embedding_dim", -1),
                "embedding_time_s": info.get("embedding_time_s", 0.0),
                "model_size_mb": info.get("model_size_mb", -1),
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E5: Local embedding model comparison")
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help=f"Models to compare (default: all three). Options: {ALL_MODELS}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N questions per model (default: all 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls to avoid rate limits (default: 0.5)",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from tutor.llm_client import build_llm_client

    chunks_meta = load_chunks_meta()
    log.info("Loaded %d chunk metadata entries.", len(chunks_meta))

    items = load_test_items()
    log.info("Loaded %d test items.", len(items))

    eval_items = items[: args.limit] if args.limit else items
    if args.limit:
        log.info("Limiting to %d questions per model.", args.limit)

    llm_client = build_llm_client()

    all_rows: list[dict] = []
    timing_info: dict[str, dict] = {}

    for model_name in args.models:
        log.info("\n=== Model: %s ===", model_name)

        index, embed_time, model_size_mb = build_or_load_index(model_name, chunks_meta)
        embedding_model = SentenceTransformer(model_name)
        embedding_dim = embedding_model.get_sentence_embedding_dimension()

        timing_info[model_name] = {
            "embedding_time_s": round(embed_time, 1),
            "model_size_mb": model_size_mb,
            "embedding_dim": embedding_dim,
        }

        rows = run_model(
            model_name, index, embedding_model, chunks_meta, eval_items, llm_client,
            delay=args.delay,
        )
        all_rows.extend(rows)

    # Save per-question detail
    RESULTS_DIR.mkdir(exist_ok=True)
    detail_path = RESULTS_DIR / "eval_e5_embedding.csv"
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    log.info("Saved per-question results → %s", detail_path)

    # Save summary
    summary = compute_summary(all_rows, timing_info)
    summary_path = RESULTS_DIR / "eval_e5_embedding_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    log.info("Saved summary → %s", summary_path)

    log.info("\n--- E5 Model Comparison ---")
    for row in summary:
        log.info(
            "  %-35s  accuracy=%.4f  dim=%d  embed_time=%.1fs  size=%.1fMB",
            row["model"],
            row["accuracy"],
            row["embedding_dim"],
            row["embedding_time_s"],
            row["model_size_mb"],
        )


if __name__ == "__main__":
    main()
