#!/usr/bin/env python3
"""Dump top-k retrieved chunks for side-by-side qualitative comparison (E1).

Example:
    set CORPUS_ARTIFACT_DIR=artifacts\\corpus\\e1_structure
    python experiments/dump_coherence_examples.py --compare artifacts/corpus/e1_naive --out results/person2/coherence_pairs.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env", override=True)

from tutor.retriever import DenseRetriever  # noqa: E402
from tutor.schemas import MCQOption, MCQItem  # noqa: E402


def load_test(path: Path, n: int) -> list[MCQItem]:
    items: list[MCQItem] = []
    with open(path, encoding="utf-8") as f:
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
            if len(items) >= n:
                break
    return items


def _render_topk(retriever: DenseRetriever, query: str, k: int) -> str:
    chunks = retriever.search(query, k=k)
    lines = []
    for i, c in enumerate(chunks, 1):
        preview = c.text[:500].replace("\n", " ")
        if len(c.text) > 500:
            preview += " …"
        lines.append(
            f"{i}. score={c.score:.4f} | {c.source_doc} | {c.section_title}\n   {preview}\n"
        )
    return "\n".join(lines) if lines else "(no chunks above threshold)\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-path",
        type=Path,
        default=REPO_ROOT / "data" / "afrimedqa_mcq_test.jsonl",
        help="MCQ JSONL source for queries",
    )
    parser.add_argument("--n-queries", type=int, default=10, metavar="N", help="first N queries from test")
    parser.add_argument("--top-k", type=int, default=5, metavar="K")
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="?",
        default=None,
        help="Second corpus dir (chunks_meta.jsonl + faiss.index). Primary uses CORPUS_ARTIFACT_DIR.",
    )
    parser.add_argument("--out", type=Path, required=True, metavar="FILE.md")
    args = parser.parse_args()

    primary_root = os.environ.get("CORPUS_ARTIFACT_DIR")
    if not primary_root:
        raise SystemExit("Set CORPUS_ARTIFACT_DIR to the primary corpus directory.")

    idx_a = Path(primary_root) / "faiss.index"
    meta_a = Path(primary_root) / "chunks_meta.jsonl"
    ret_a = DenseRetriever(index_path=idx_a, chunks_meta_path=meta_a)

    ret_b = None
    if args.compare:
        br = Path(args.compare).expanduser().resolve()
        ret_b = DenseRetriever(index_path=br / "faiss.index", chunks_meta_path=br / "chunks_meta.jsonl")

    items = load_test(args.test_path, args.n_queries)
    lines = [
        f"# Retrieval coherence examples (top-{args.top_k})\n\n",
        f"Primary CORPUS_ARTIFACT_DIR: `{primary_root}`\n\n",
    ]
    if args.compare:
        lines.append(f"Compared to: `{args.compare}`\n\n")

    for item in items:
        lines.append(f"## {item.question_id}\n\n")
        lines.append(f"_Specialty:_ {item.specialty}\n\n")
        lines.append(f"**Q:** {item.question}\n\n")
        lines.append("### Primary\n")
        lines.append(_render_topk(ret_a, item.question, args.top_k))
        if ret_b is not None:
            lines.append("\n### Alternate\n")
            lines.append(_render_topk(ret_b, item.question, args.top_k))
        lines.append("\n---\n\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
