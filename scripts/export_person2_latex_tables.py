#!/usr/bin/env python3
"""Emit LaTeX tabular rows from results/person2/summary_tables.json (internal reproducibility).

Usage (from repo root):
    python scripts/export_person2_latex_tables.py

Prints to stdout; redirect to a snippet file if desired. Not required for the manuscript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "results" / "person2" / "summary_tables.json"


def main() -> None:
    if not SUMMARY.exists():
        print(f"Missing {SUMMARY}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    by_cond = data["by_condition"]
    chunk_stats = data.get("chunk_file_stats_by_condition", {})

    print("% --- E2 ---")
    for cid in ["e2_t100", "e2_t200", "e2_t400", "e2_t800"]:
        r = by_cond[cid]
        lo, hi = r["accuracy_ci95_bootstrap"]
        max_tok = cid.split("_t")[1]
        print(
            f"{max_tok} & {r['accuracy']:.2f} & {lo:.2f} & {hi:.2f} \\\\"
        )

    print("\n% --- E1 ---")
    mc = data.get("e1", {}).get("mcnemar_primary_vs_naive", {})
    for k, v in mc.items():
        print(f"% mcnemar.{k}={v}")

    print("\n% --- E3 ---")
    order = ["e3_k1", "e3_k3", "e3_k5", "e3_k10", "e3_k15"]
    for cid in order:
        k = cid.replace("e3_k", "")
        r = by_cond[cid]
        lo, hi = r["accuracy_ci95_bootstrap"]
        print(
            f"{k} & {r['accuracy']:.2f} & "
            f"{r['input_tokens_mean']:.2f} & {r['latency_mean_ms']:.1f} "
            f"& [{lo:.2f},\\,{hi:.2f}] \\\\"
        )

    print("\n% --- chunk stats ---")
    for cid in ["e2_t100", "e2_t200", "e2_t400", "e2_t800", "e1_structure", "e1_naive"]:
        st = chunk_stats.get(cid) or {}
        if not st:
            continue
        print(
            f"% {cid}: n={st.get('num_chunks')} "
            f"mean_words={st.get('mean_words')} median={st.get('median_words')}"
        )


if __name__ == "__main__":
    main()
