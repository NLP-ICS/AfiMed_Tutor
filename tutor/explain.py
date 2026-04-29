"""SAQ clinical scenario loader for Explain mode (§2, §6.3)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from tutor.schemas import SAQItem

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAQ_PATH = _REPO_ROOT / "data" / "afrimedqa_saq.jsonl"


class SAQLoader:
    """Loads short-answer clinical scenarios from the AfriMed-QA SAQ split."""

    def __init__(self, saq_path: Path = _SAQ_PATH) -> None:
        self._cases: list[SAQItem] = []
        with open(saq_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._cases.append(SAQItem(**json.loads(line)))

    def sample(
        self,
        specialty: str | None = None,
        exclude_ids: set[str] | None = None,
        rng: random.Random | None = None,
    ) -> SAQItem:
        rng = rng or random.Random()
        exclude_ids = exclude_ids or set()
        candidates = [
            c for c in self._cases
            if c.case_id not in exclude_ids
            and (specialty is None or c.specialty == specialty)
        ]
        if not candidates:
            raise ValueError("No SAQ candidates available.")
        return rng.choice(candidates)

    def get_by_id(self, case_id: str) -> SAQItem:
        for case in self._cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"case_id {case_id!r} not found.")

    def __len__(self) -> int:
        return len(self._cases)
