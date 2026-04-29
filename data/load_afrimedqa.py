"""Download AfriMed-QA from Hugging Face and produce three stratified JSONL splits (§8).

Outputs:
  data/afrimedqa_mcq_pool.jsonl   — all MCQs minus held-out test set
  data/afrimedqa_mcq_test.jsonl   — exactly 100 stratified held-out MCQs
  data/afrimedqa_saq.jsonl        — short-answer items for Explain mode
  data/question_embeddings.npy    — pre-computed question embeddings (pool)

Usage:
    python data/load_afrimedqa.py
    python data/load_afrimedqa.py --skip-embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"

MCQ_POOL_PATH = DATA_DIR / "afrimedqa_mcq_pool.jsonl"
MCQ_TEST_PATH = DATA_DIR / "afrimedqa_mcq_test.jsonl"
SAQ_PATH = DATA_DIR / "afrimedqa_saq.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "question_embeddings.npy"

# The five specialties for stratified sampling (§7.1)
TARGET_SPECIALTIES = [
    "Surgery",
    "Obstetrics & Gynecology",
    "Pediatrics",
    "Infectious Disease",
    "Internal Medicine",
]
TEST_PER_SPECIALTY = 20  # 5 × 20 = 100


# ---------------------------------------------------------------------------
# Normalisation helpers — tuned to actual intronhealth/afrimedqa_v2 schema
# Fields: sample_id, question_type, question, answer_options (JSON str),
#         correct_answer (option1..option5), answer_rationale, specialty
# ---------------------------------------------------------------------------

# Map dataset's underscore specialty names → our canonical five + others
_SPECIALTY_ALIASES: dict[str, str] = {
    # The five target specialties (§7.1)
    "general_surgery": "Surgery",
    "orthopedic_surgery": "Surgery",
    "surgery": "Surgery",
    "obstetrics_and_gynecology": "Obstetrics & Gynecology",
    "obstetrics & gynecology": "Obstetrics & Gynecology",
    "obstetrics & gynaecology": "Obstetrics & Gynecology",
    "ob/gyn": "Obstetrics & Gynecology",
    "pediatrics": "Pediatrics",
    "paediatrics": "Pediatrics",
    "infectious_disease": "Infectious Disease",
    "infectious disease": "Infectious Disease",
    "infectious diseases": "Infectious Disease",
    "internal_medicine": "Internal Medicine",
    "internal medicine": "Internal Medicine",
    "medicine": "Internal Medicine",
}

# Keys used in answer_options dict, in order
_OPTION_KEYS = ["option1", "option2", "option3", "option4", "option5"]
_LETTER_KEYS = ["A", "B", "C", "D", "E"]


def _normalise_specialty(raw: str | None) -> str:
    if not raw:
        return "General"
    return _SPECIALTY_ALIASES.get(raw.lower().strip(),
           _SPECIALTY_ALIASES.get(raw.strip(), raw.strip()))


def _parse_answer_options(raw) -> list[dict]:
    """Parse answer_options JSON string → list[{key: "A", text: "..."}]."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(raw, dict):
        result = []
        for option_key, letter in zip(_OPTION_KEYS, _LETTER_KEYS):
            if option_key in raw and raw[option_key]:
                result.append({"key": letter, "text": str(raw[option_key])})
        return result
    return []


def _correct_answer_to_letter(correct_answer: str | None, options: list[dict]) -> str:
    """Convert 'option4' → 'D' (letter matching position in options list)."""
    if not correct_answer:
        return ""
    ca = correct_answer.strip().lower()
    # Already a letter
    if ca.upper() in _LETTER_KEYS:
        return ca.upper()
    # option1, option2, ...
    for i, ok in enumerate(_OPTION_KEYS):
        if ca == ok and i < len(options):
            return options[i]["key"]
    return ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_from_hf() -> tuple[list[dict], list[dict]]:
    """Return (mcq_items, saq_items) from HuggingFace."""
    from datasets import load_dataset

    log.info("Loading intronhealth/afrimedqa_v2 from Hugging Face...")
    # trust_remote_code removed — deprecated in newer datasets versions
    ds = load_dataset("intronhealth/afrimedqa_v2")

    mcq_items: list[dict] = []
    saq_items: list[dict] = []

    for split_name in ds:
        for i, row in enumerate(ds[split_name]):
            qtype = str(row.get("question_type", "")).lower()
            if qtype == "mcq":
                item = _normalise_mcq(row, f"{split_name}_{i}")
                if item:
                    mcq_items.append(item)
            elif qtype == "saq":
                item = _normalise_saq(row, f"{split_name}_{i}")
                if item:
                    saq_items.append(item)
            # skip consumer_queries and other types

    log.info("Loaded %d MCQs and %d SAQs.", len(mcq_items), len(saq_items))
    return mcq_items, saq_items


def _normalise_mcq(row: dict, fallback_id: str) -> dict | None:
    question = (row.get("question_clean") or row.get("question") or "").strip()
    if not question:
        return None

    options = _parse_answer_options(row.get("answer_options"))
    if not options:
        return None

    gold = _correct_answer_to_letter(str(row.get("correct_answer") or ""), options)
    if not gold:
        return None

    rationale = str(row.get("answer_rationale") or "").strip()
    specialty = _normalise_specialty(row.get("specialty"))
    qid = str(row.get("sample_id") or fallback_id)

    return {
        "question_id": qid,
        "question": question,
        "options": options,
        "gold_answer": gold,
        "gold_rationale": rationale,
        "specialty": specialty,
        "source": str(row.get("country") or ""),
    }


def _normalise_saq(row: dict, fallback_id: str) -> dict | None:
    scenario = (row.get("question_clean") or row.get("question") or "").strip()
    if not scenario:
        return None
    expert = str(row.get("answer_rationale") or "").strip()
    specialty = _normalise_specialty(row.get("specialty"))
    return {
        "case_id": str(row.get("sample_id") or fallback_id),
        "scenario": scenario,
        "expert_answer": expert,
        "specialty": specialty,
    }


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_test_split(
    mcq_items: list[dict],
    target_specialties: list[str] = TARGET_SPECIALTIES,
    n_per_specialty: int = TEST_PER_SPECIALTY,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Return (pool, test) with exactly n_per_specialty items per target specialty in test."""
    rng = random.Random(seed)
    by_specialty: dict[str, list[dict]] = defaultdict(list)
    for item in mcq_items:
        by_specialty[item["specialty"]].append(item)

    test_ids: set[str] = set()
    test: list[dict] = []
    for spec in target_specialties:
        candidates = by_specialty.get(spec, [])
        n = min(n_per_specialty, len(candidates))
        if n < n_per_specialty:
            log.warning(
                "Only %d items for specialty %r (wanted %d).", n, spec, n_per_specialty
            )
        chosen = rng.sample(candidates, n)
        test.extend(chosen)
        test_ids.update(item["question_id"] for item in chosen)

    pool = [item for item in mcq_items if item["question_id"] not in test_ids]
    log.info("Test set: %d items. Pool: %d items.", len(test), len(pool))
    return pool, test


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def compute_question_embeddings(pool: list[dict]) -> np.ndarray:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    questions = [item["question"] for item in pool]
    batch_size = 100
    all_embs: list[list[float]] = []

    if provider == "local":
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        log.info("Loading local embedding model: %s", model_name)
        model = SentenceTransformer(model_name)
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embs.extend(v.tolist() for v in vecs)
            log.info("  Embedded %d/%d questions", min(i + batch_size, len(questions)), len(questions))
        return np.array(all_embs, dtype="float32")

    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
            all_embs.extend(e.embedding for e in resp.data)
        elif provider == "voyage":
            import voyageai
            vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
            result = vo.embed(batch, model="voyage-3-lite", input_type="document")
            all_embs.extend(result.embeddings)
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")
        log.info("  Embedded questions %d/%d", min(i + batch_size, len(questions)), len(questions))

    vecs = np.array(all_embs, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load and split AfriMed-QA dataset")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    mcq_items, saq_items = load_from_hf()

    pool, test = stratified_test_split(mcq_items)

    MCQ_POOL_PATH.write_text(
        "\n".join(json.dumps(item) for item in pool), encoding="utf-8"
    )
    MCQ_TEST_PATH.write_text(
        "\n".join(json.dumps(item) for item in test), encoding="utf-8"
    )
    SAQ_PATH.write_text(
        "\n".join(json.dumps(item) for item in saq_items), encoding="utf-8"
    )
    log.info(
        "Wrote %d pool, %d test, %d SAQ items.",
        len(pool), len(test), len(saq_items),
    )

    if not args.skip_embeddings:
        if not pool:
            log.warning("Pool is empty — skipping embedding computation.")
        else:
            log.info("Computing question embeddings for pool (%d questions)...", len(pool))
            embs = compute_question_embeddings(pool)
            np.save(EMBEDDINGS_PATH, embs)
            log.info("Saved embeddings to %s (shape %s).", EMBEDDINGS_PATH, embs.shape)


if __name__ == "__main__":
    main()
