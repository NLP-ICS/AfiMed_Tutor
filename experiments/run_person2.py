#!/usr/bin/env python3
"""Drive ICS4554 Person 2 experiments: E2 chunk-size sweep, E1 structure vs naive, E3 top-k.

Examples:
    python experiments/run_person2.py --phases manifest
    python experiments/run_person2.py --phases e2 --limit 5
    python experiments/run_person2.py --phases e2 e1 e3
    python experiments/run_person2.py --dry-run --phases all

Requires: corpus PDFs cached, embeddings/LLM keys in .env, and data/afrimedqa_mcq_test.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.constants import (  # noqa: E402
    E1_NAIVE,
    E1_STRUCT,
    E2_CONDITIONS,
    E2_MAX_CHUNK_TOKENS,
    E3_K_VALUES,
    E3_PREFIX,
    corpus_artifact_dir,
    person2_manifest_path,
    person2_result_csv,
)

SUMMARY_JSON = REPO_ROOT / "results" / "person2" / "e2_summary.json"


def _git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def write_manifest(limit: int | None) -> None:
    manifest = person2_manifest_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "notes": (
            "Control variables held fixed across phases: dataset split (100-item test set), "
            "LLM_PROVIDER / OPENAI_MODEL / ANTHROPIC_MODEL per .env, EMBEDDING_PROVIDER, dense retriever."
        ),
        "smoke_limit": limit,
        "artifacts_root": str(REPO_ROOT / "artifacts" / "corpus"),
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {manifest.relative_to(REPO_ROOT)}")


def run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    display = " ".join(cmd)
    if dry_run:
        print(f"DRY RUN: {display}")
        return
    print(display)
    merged = os.environ.copy()
    merged.update(env)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=merged, check=True)


def accuracy_and_latency(csv_path: Path) -> tuple[float | None, float | None, int]:
    if not csv_path.exists():
        return None, None, 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, 0
    acc = sum(int(r["correct"]) for r in rows) / len(rows)
    lat = sum(float(r["latency_ms"]) for r in rows) / len(rows)
    return acc, lat, len(rows)


def corpus_build_cmd(
    out_dir: Path,
    *,
    naive: bool,
    max_chunk_tokens: int | None,
    naive_window: int,
    naive_overlap: int,
) -> list[str]:
    cmd = [sys.executable, "corpus/build_corpus.py", "--out-dir", str(out_dir)]
    if naive:
        cmd.extend(
            ["--naive", "--naive-window-chars", str(naive_window), "--naive-overlap-chars", str(naive_overlap)]
        )
    else:
        assert max_chunk_tokens is not None
        cmd.extend(["--max-chunk-tokens", str(max_chunk_tokens)])
    return cmd


def eval_cmd(result_csv: Path, limit: int | None) -> list[str]:
    cmd = [
        sys.executable,
        "eval/run_mcq_eval.py",
        "--condition",
        "rag",
        "--rag-output",
        str(result_csv),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd


def run_e2(limit: int | None, dry_run: bool) -> None:
    rows_summary: list[dict] = []
    for cid in E2_CONDITIONS:
        artifact = corpus_artifact_dir(cid)
        tok = E2_MAX_CHUNK_TOKENS[cid]
        run_cmd(corpus_build_cmd(artifact, naive=False, max_chunk_tokens=tok, naive_window=800, naive_overlap=200), {}, dry_run)
        out_csv = person2_result_csv(cid)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(eval_cmd(out_csv, limit), {"CORPUS_ARTIFACT_DIR": str(artifact)}, dry_run)

        acc, lat, n = accuracy_and_latency(out_csv)
        rows_summary.append(
            {
                "condition_id": cid,
                "max_chunk_tokens": tok,
                "accuracy": acc,
                "mean_latency_ms": lat,
                "n_evaluated": n,
                "artifact_dir": str(artifact),
            }
        )

    usable = [r for r in rows_summary if r["accuracy"] is not None]
    if usable:
        best = max(
            usable,
            key=lambda r: (float(r["accuracy"]), -float(r["mean_latency_ms"] or 0.0)),
        )
        SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
        payload = {"conditions": rows_summary, "best_condition_id": best["condition_id"]}
        if not dry_run:
            SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote {SUMMARY_JSON.relative_to(REPO_ROOT)} (best: {best['condition_id']})")


def run_e1(limit: int | None, dry_run: bool) -> None:
    # Structure-aware baseline aligned with 800-token ceiling (matches e2_t800).
    sdir = corpus_artifact_dir(E1_STRUCT)
    run_cmd(corpus_build_cmd(sdir, naive=False, max_chunk_tokens=800, naive_window=800, naive_overlap=200), {}, dry_run)
    s_csv = person2_result_csv(E1_STRUCT)
    s_csv.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(eval_cmd(s_csv, limit), {"CORPUS_ARTIFACT_DIR": str(sdir)}, dry_run)

    ndir = corpus_artifact_dir(E1_NAIVE)
    run_cmd(corpus_build_cmd(ndir, naive=True, max_chunk_tokens=None, naive_window=800, naive_overlap=200), {}, dry_run)
    n_csv = person2_result_csv(E1_NAIVE)
    n_csv.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(eval_cmd(n_csv, limit), {"CORPUS_ARTIFACT_DIR": str(ndir)}, dry_run)


def _best_e2_artifact() -> Path:
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_JSON}. Run `--phases e2` first "
            "(or place e2_summary.json with best_condition_id)."
        )
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    bid = data.get("best_condition_id")
    if not bid:
        raise ValueError(f"best_condition_id missing in {SUMMARY_JSON}")
    return corpus_artifact_dir(str(bid))


def run_e3(limit: int | None, dry_run: bool, artifact_override: Path | None) -> None:
    if artifact_override is not None:
        artifact = artifact_override.expanduser().resolve()
    elif dry_run:
        # Summary is only written after a real E2 run; placeholder path for printing commands.
        artifact = corpus_artifact_dir(E2_CONDITIONS[-1])
    else:
        artifact = _best_e2_artifact()
    for k in E3_K_VALUES:
        cid = f"{E3_PREFIX}_k{k}"
        out_csv = person2_result_csv(cid)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            eval_cmd(out_csv, limit),
            {"CORPUS_ARTIFACT_DIR": str(artifact), "RETRIEVER_TOP_K": str(k)},
            dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["manifest", "e2", "e1", "e3", "all"],
        default=["all"],
        help="experiment phases (use 'manifest' alone to emit metadata only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="first N MCQ items only for every eval (smoke test)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
    parser.add_argument(
        "--e3-corpus-dir",
        type=Path,
        default=None,
        help="override CORPUS_ARTIFACT_DIR for E3 (skip reading e2_summary.json)",
    )
    args = parser.parse_args()

    phases = set(args.phases)
    if "all" in phases:
        phases = {"manifest", "e2", "e1", "e3"}

    if "manifest" in phases:
        write_manifest(args.limit)

    try:
        if "e2" in phases:
            run_e2(args.limit, args.dry_run)
        if "e1" in phases:
            run_e1(args.limit, args.dry_run)
        if "e3" in phases:
            run_e3(args.limit, args.dry_run, args.e3_corpus_dir)
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)


if __name__ == "__main__":
    main()
