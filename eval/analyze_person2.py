"""Aggregate Person 2 experiment CSVs into summary tables and simple figures.

Reads:
    results/person2/<condition>/eval_rag.csv
    results/person2/e2_summary.json (optional — for labeling best E2 corpus)
Writes:
    results/person2/summary_tables.json
    results/person2/figures/*.png (if matplotlib installed)

Usage:
    python eval/analyze_person2.py

McNemar (exact binomial): applied to E1 structural vs naive when both CSVs exist.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSON2 = REPO_ROOT / "results" / "person2"
FIG_DIR = PERSON2 / "figures"

try:
    from scipy.stats import binomtest
except ImportError:
    binomtest = None  # type: ignore[misc, assignment]


def _read_eval(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    correct = [int(r["correct"]) for r in rows]
    latency = [float(r["latency_ms"]) for r in rows]
    inputs = [int(r["input_tokens"]) for r in rows if str(r.get("input_tokens", "")).isdigit()]
    top1_scores: list[float] = []
    for r in rows:
        s = r.get("retrieval_top1_score")
        if s is None or str(s).strip() == "":
            continue
        try:
            top1_scores.append(float(s))
        except ValueError:
            pass
    by_spec: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_spec[str(r["specialty"])].append(int(r["correct"]))

    return {
        "n": len(rows),
        "accuracy": sum(correct) / len(rows),
        "latency_mean_ms": statistics.mean(latency),
        "latency_std_ms": statistics.pstdev(latency) if len(latency) > 1 else 0.0,
        "input_tokens_mean": statistics.mean(inputs) if inputs else None,
        "retrieval_top1_mean": statistics.mean(top1_scores) if top1_scores else None,
        "retrieval_top1_std": statistics.pstdev(top1_scores) if len(top1_scores) > 1 else 0.0,
        "accuracy_by_specialty": {
            spec: sum(vals) / len(vals) for spec, vals in by_spec.items()
        },
        "accuracy_ci95_bootstrap": _bootstrap_accuracy(correct, seed=42),
    }


def _bootstrap_accuracy(flags: list[int], *, seed: int, n_boot: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.array(flags, dtype=np.float64)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(float(x[idx].mean()))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _mcnemar_exact(r1: list[dict], r2: list[dict]) -> dict | None:
    if binomtest is None:
        return {"error": "scipy.stats.binomtest not available"}
    idx1 = {r["question_id"]: int(r["correct"]) for r in r1}
    idx2 = {r["question_id"]: int(r["correct"]) for r in r2}
    common = sorted(set(idx1) & set(idx2))
    b = sum(1 for q in common if idx1[q] == 0 and idx2[q] == 1)
    c = sum(1 for q in common if idx1[q] == 1 and idx2[q] == 0)
    n = b + c
    if n == 0:
        p_val = 1.0
    else:
        p_val = float(binomtest(min(b, c), n, p=0.5, alternative="two-sided").pvalue)
    return {"discordant_wrong_right": b, "discordant_right_wrong": c, "n_discordant": n, "p_value": p_val}


def _chunk_meta_stats(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    word_counts = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text", "")
            word_counts.append(len(t.split()))
    if not word_counts:
        return {}
    return {
        "num_chunks": len(word_counts),
        "mean_words": statistics.mean(word_counts),
        "median_words": statistics.median(word_counts),
    }


def _discover_conditions() -> list[str]:
    if not PERSON2.is_dir():
        return []
    return sorted({p.parent.name for p in PERSON2.glob("*/eval_rag.csv")})


def _parse_e3_k(cid: str) -> int | None:
    m = re.match(r"^e3_k(\d+)$", cid)
    return int(m.group(1)) if m else None


def main() -> None:
    PERSON2.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    condition_ids = _discover_conditions()
    summaries: dict[str, dict] = {}
    corpus_stats: dict[str, dict] = {}

    for cid in condition_ids:
        rows = _read_eval(PERSON2 / cid / "eval_rag.csv")
        summaries[cid] = _summarize(rows)
        meta_path = REPO_ROOT / "artifacts" / "corpus" / cid / "chunks_meta.jsonl"
        corpus_stats[cid] = _chunk_meta_stats(meta_path)

    e1_comparison = {}
    rows_struct = _read_eval(PERSON2 / "e1_structure" / "eval_rag.csv")
    rows_naive = _read_eval(PERSON2 / "e1_naive" / "eval_rag.csv")
    if rows_struct and rows_naive:
        e1_comparison["mcnemar_primary_vs_naive"] = _mcnemar_exact(rows_struct, rows_naive)

    out_payload = {"by_condition": summaries, "chunk_file_stats_by_condition": corpus_stats, "e1": e1_comparison}

    summary_path = PERSON2 / "summary_tables.json"
    summary_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures.")
        return

    figures_written = 0
    e2_points = [(corpus_stats[c].get("mean_words", math.nan), summaries[c]["accuracy"], c)
                 for c in condition_ids if c.startswith("e2_t")]
    e2_points = [(mw, acc, c) for mw, acc, c in e2_points if summaries[c]]

    if e2_points:
        e2_points.sort(key=lambda t: t[0])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([p[0] for p in e2_points], [p[1] for p in e2_points], "o-")
        ax.set_xlabel("Mean chunk length (words)")
        ax.set_ylabel("MCQ accuracy")
        ax.set_title("E2: accuracy vs mean chunk size")
        fig.savefig(FIG_DIR / "e2_accuracy_vs_chunksize.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures_written += 1

    e3_points = []
    for cid in condition_ids:
        k = _parse_e3_k(cid)
        if k is None:
            continue
        if summaries[cid]:
            e3_points.append((k, summaries[cid]["accuracy"], summaries[cid].get("input_tokens_mean")))
    if e3_points:
        e3_points.sort(key=lambda t: t[0])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([p[0] for p in e3_points], [p[1] for p in e3_points], "s-")
        ax.set_xlabel("Top-k retrieval")
        ax.set_ylabel("MCQ accuracy")
        ax.set_title("E3: accuracy vs top-k")
        fig.savefig(FIG_DIR / "e3_accuracy_vs_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures_written += 1

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        toks = [p[2] or 0 for p in e3_points]
        ax2.plot([p[0] for p in e3_points], toks, "d-")
        ax2.set_xlabel("Top-k retrieval")
        ax2.set_ylabel("Mean input tokens")
        ax2.set_title("E3: input tokens vs top-k")
        fig2.savefig(FIG_DIR / "e3_input_tokens_vs_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        figures_written += 1

    if figures_written:
        print(
            f"Wrote {figures_written} figure(s) under "
            f"{FIG_DIR.relative_to(REPO_ROOT)}"
        )
    else:
        print("No E2/E3 data found — skipping figure generation.")


if __name__ == "__main__":
    main()
