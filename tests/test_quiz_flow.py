"""Tests for tutor/quiz.py — MCQ sampling and option formatting."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.quiz import QuizLoader, format_options
from tutor.schemas import MCQItem, MCQOption


def _make_pool(n: int = 20) -> list[dict]:
    specialties = ["Surgery", "Pediatrics", "Internal Medicine", "Infectious Disease"]
    items = []
    for i in range(n):
        items.append(
            {
                "question_id": f"q{i:04d}",
                "question": f"What is treatment for condition {i}?",
                "options": [
                    {"key": "A", "text": "Option A"},
                    {"key": "B", "text": "Option B"},
                    {"key": "C", "text": "Option C"},
                    {"key": "D", "text": "Option D"},
                ],
                "gold_answer": "B",
                "gold_rationale": f"Because of reason {i}.",
                "specialty": specialties[i % len(specialties)],
                "source": "test",
            }
        )
    return items


def _build_loader(pool: list[dict]) -> QuizLoader:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pool.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in pool))
        return QuizLoader(pool_path=path)


def test_loader_loads_all_items():
    pool = _make_pool(20)
    loader = _build_loader(pool)
    assert len(loader) == 20


def test_sample_returns_item():
    loader = _build_loader(_make_pool(10))
    item = loader.sample()
    assert isinstance(item, MCQItem)
    assert item.options


def test_sample_by_specialty():
    loader = _build_loader(_make_pool(20))
    item = loader.sample(specialty="Surgery")
    assert item.specialty == "Surgery"


def test_sample_excludes_ids():
    pool = _make_pool(5)
    loader = _build_loader(pool)
    all_ids = {item["question_id"] for item in pool}
    # Exclude all but one
    keep = list(all_ids)[:1]
    exclude = all_ids - set(keep)
    item = loader.sample(exclude_ids=exclude)
    assert item.question_id not in exclude


def test_get_by_id():
    pool = _make_pool(5)
    loader = _build_loader(pool)
    item = loader.get_by_id("q0002")
    assert item.question_id == "q0002"


def test_get_by_id_missing():
    loader = _build_loader(_make_pool(3))
    with pytest.raises(KeyError):
        loader.get_by_id("nonexistent_id")


def test_specialties_property():
    loader = _build_loader(_make_pool(20))
    specs = loader.specialties
    assert isinstance(specs, list)
    assert len(specs) > 0
    assert specs == sorted(specs)


def test_format_options():
    options = [
        MCQOption(key="A", text="First option"),
        MCQOption(key="B", text="Second option"),
    ]
    rendered = format_options(options)
    assert "A." in rendered
    assert "B." in rendered
    assert "First option" in rendered


def test_loader_accepts_dict_options():
    item = {
        "question_id": "q_dict",
        "question": "Test?",
        "options": {"A": "Alpha", "B": "Beta", "C": "Gamma"},
        "gold_answer": "A",
        "gold_rationale": "Because alpha.",
        "specialty": "Surgery",
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pool.jsonl"
        path.write_text(json.dumps(item))
        loader = QuizLoader(pool_path=path)
    assert len(loader) == 1
    loaded = loader.get_by_id("q_dict")
    assert len(loaded.options) == 3
