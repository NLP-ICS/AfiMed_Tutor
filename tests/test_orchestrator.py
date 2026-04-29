"""Tests for tutor/orchestrator.py using mock LLM client and retriever."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.schemas import Chunk, CompletionResult, MCQItem, MCQOption
from tutor.orchestrator import _render_chunks, handle_ask, handle_quiz_submit


def _mock_chunk(i: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"doc__{i:05d}",
        text=f"Guideline text for condition {i}.",
        source_doc="test_doc",
        section_title=f"Section {i}",
        score=0.8,
    )


def _mock_llm(text: str = "Test answer.") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = CompletionResult(
        text=text,
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
        model_name="test-model",
    )
    return client


def _mock_retriever(chunks=None) -> MagicMock:
    ret = MagicMock()
    ret.search.return_value = chunks or [_mock_chunk(0), _mock_chunk(1)]
    return ret


def _mock_quiz_loader(item: MCQItem) -> MagicMock:
    loader = MagicMock()
    loader.get_by_id.return_value = item
    return loader


def _sample_mcq_item() -> MCQItem:
    return MCQItem(
        question_id="q001",
        question="What is the first-line treatment for uncomplicated malaria?",
        options=[
            MCQOption(key="A", text="Chloroquine"),
            MCQOption(key="B", text="Artemisinin combination therapy"),
            MCQOption(key="C", text="Quinine"),
            MCQOption(key="D", text="Doxycycline"),
        ],
        gold_answer="B",
        gold_rationale="ACT is recommended as first-line for uncomplicated malaria.",
        specialty="Infectious Disease",
    )


# ── _render_chunks ────────────────────────────────────────────────────────────

def test_render_chunks_empty():
    rendered = _render_chunks([])
    assert "No relevant" in rendered


def test_render_chunks_includes_source():
    chunks = [_mock_chunk(0)]
    rendered = _render_chunks(chunks)
    assert "test_doc" in rendered
    assert "Section 0" in rendered
    assert "Guideline text" in rendered


# ── handle_ask ────────────────────────────────────────────────────────────────

def test_handle_ask_returns_ask_response():
    from tutor.schemas import AskResponse
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "tutor.orchestrator.build_related_retriever", side_effect=Exception("no data")
    ):
        resp = handle_ask(
            question="What is the treatment for malaria?",
            llm_client=_mock_llm("Malaria is treated with ACT."),
            retriever=_mock_retriever(),
        )
    assert isinstance(resp, AskResponse)
    assert resp.answer == "Malaria is treated with ACT."
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50


def test_handle_ask_calls_retriever():
    retriever = _mock_retriever()
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "tutor.orchestrator.build_related_retriever", side_effect=Exception("no data")
    ):
        handle_ask("test question", _mock_llm(), retriever)
    retriever.search.assert_called_once()


# ── handle_quiz_submit ────────────────────────────────────────────────────────

def test_handle_quiz_submit_correct():
    from tutor.schemas import QuizResponse
    item = _sample_mcq_item()
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "tutor.orchestrator.build_related_retriever", side_effect=Exception("no data")
    ):
        resp = handle_quiz_submit(
            question_id="q001",
            student_choice="B",
            llm_client=_mock_llm("Part 1 — Verdict: Correct!"),
            retriever=_mock_retriever(),
            quiz_loader=_mock_quiz_loader(item),
        )
    assert isinstance(resp, QuizResponse)
    assert resp.is_correct is True


def test_handle_quiz_submit_incorrect():
    from tutor.schemas import QuizResponse
    item = _sample_mcq_item()
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "tutor.orchestrator.build_related_retriever", side_effect=Exception("no data")
    ):
        resp = handle_quiz_submit(
            question_id="q001",
            student_choice="A",
            llm_client=_mock_llm("Part 1 — Verdict: Incorrect."),
            retriever=_mock_retriever(),
            quiz_loader=_mock_quiz_loader(item),
        )
    assert resp.is_correct is False


def test_handle_quiz_submit_case_insensitive():
    item = _sample_mcq_item()
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "tutor.orchestrator.build_related_retriever", side_effect=Exception("no data")
    ):
        resp = handle_quiz_submit(
            question_id="q001",
            student_choice="b",  # lowercase
            llm_client=_mock_llm(),
            retriever=_mock_retriever(),
            quiz_loader=_mock_quiz_loader(item),
        )
    assert resp.is_correct is True
