"""MCQ sampling and quiz flow (§2, §6.3)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from tutor.schemas import MCQItem, MCQOption

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MCQ_POOL_PATH = _REPO_ROOT / "data" / "afrimedqa_mcq_pool.jsonl"


class QuizLoader:
    """Loads MCQ items from the pool and samples by specialty."""

    def __init__(self, pool_path: Path = _MCQ_POOL_PATH) -> None:
        self._pool: list[MCQItem] = []
        with open(pool_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw = json.loads(line)
                    # Normalise options: expect list of {"key": ..., "text": ...}
                    # or a dict {"A": ..., "B": ...}
                    options = raw.get("options", [])
                    if isinstance(options, dict):
                        options = [MCQOption(key=k, text=v) for k, v in options.items()]
                    else:
                        options = [MCQOption(**o) for o in options]
                    raw["options"] = options
                    self._pool.append(MCQItem(**raw))

    def sample(
        self,
        specialty: str | None = None,
        exclude_ids: set[str] | None = None,
        rng: random.Random | None = None,
    ) -> MCQItem:
        rng = rng or random.Random()
        exclude_ids = exclude_ids or set()
        candidates = [
            item for item in self._pool
            if item.question_id not in exclude_ids
            and (specialty is None or item.specialty == specialty)
        ]
        if not candidates:
            raise ValueError(
                f"No MCQ candidates available for specialty={specialty!r} "
                f"after excluding {len(exclude_ids)} ids."
            )
        return rng.choice(candidates)

    def get_by_id(self, question_id: str) -> MCQItem:
        for item in self._pool:
            if item.question_id == question_id:
                return item
        raise KeyError(f"question_id {question_id!r} not found in pool.")

    @property
    def specialties(self) -> list[str]:
        return sorted({item.specialty for item in self._pool})

    def __len__(self) -> int:
        return len(self._pool)


def format_options(options: list[MCQOption]) -> str:
    """Render option list for prompt insertion."""
    return "\n".join(f"{o.key}. {o.text}" for o in options)
