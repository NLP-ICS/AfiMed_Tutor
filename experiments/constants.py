"""Person 2 experiment condition IDs and path helpers."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def corpus_artifact_dir(condition_id: str) -> Path:
    return _REPO_ROOT / "artifacts" / "corpus" / condition_id


def person2_result_csv(condition_id: str) -> Path:
    return _REPO_ROOT / "results" / "person2" / condition_id / "eval_rag.csv"


def person2_manifest_path() -> Path:
    return _REPO_ROOT / "results" / "person2" / "experiment_manifest.json"


# E2 chunk-size ablation (structure-aware chunker).
E2_CONDITIONS = ["e2_t100", "e2_t200", "e2_t400", "e2_t800"]
E2_MAX_CHUNK_TOKENS = {"e2_t100": 100, "e2_t200": 200, "e2_t400": 400, "e2_t800": 800}

# E1 structure vs naive dense baseline (matching E2 nominal max token ceiling).
E1_STRUCT = "e1_structure"
E1_NAIVE = "e1_naive"

# E3 top-k sweep (uses best E2 corpus by accuracy from summary file).
E3_K_VALUES = [1, 3, 5, 10, 15]
E3_PREFIX = "e3"
