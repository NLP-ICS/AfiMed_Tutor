"""Tests embedding API input budget splitting in corpus/build_corpus.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tiktoken = pytest.importorskip("tiktoken")

from corpus.build_corpus import (  # noqa: E402
    _split_text_at_token_budget,
    enforce_embedding_input_budget,
)


@pytest.fixture()
def Encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_split_text_at_budget_single_piece_under_limit(Encoding):
    t = "short text"
    assert _split_text_at_token_budget(t, Encoding, max_tokens=100) == ["short text"]


def test_split_text_at_budget_multiple_pieces(Encoding):
    tiny_budget = 8
    long = "apple " * 200
    pieces = _split_text_at_token_budget(long, Encoding, tiny_budget)
    assert len(pieces) >= 3
    for p in pieces:
        assert len(Encoding.encode(p)) <= tiny_budget


def test_enforce_embedding_budget_splits_with_openai_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MAX_INPUT_TOKENS", "120")
    long_text = "banana " * 800
    inp = [
        {
            "chunk_id": "doc__00042",
            "text": long_text,
            "source_doc": "doc",
            "section_title": "S",
            "condition": None,
            "page_number": None,
        }
    ]
    out = enforce_embedding_input_budget(inp)
    assert len(out) >= 2
    assert all("__emb" in o["chunk_id"] for o in out)
    assert {o["source_doc"] for o in out} == {"doc"}


def test_enforce_budget_noop_when_local(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    inp = [{"chunk_id": "a", "text": "hello " * 10000}]
    assert enforce_embedding_input_budget(inp) is inp


def test_spaces_free_giant_under_heuristic_but_over_encodes(Encoding):
    """Mimics PDF glue: whitespace-free span can underestimate word-split heuristics."""
    tiny_budget = 64
    giant = ("Q" * 500) + ("\n\n" + "final bit")
    pieces = _split_text_at_token_budget(giant, Encoding, tiny_budget)
    assert len(pieces) >= 2

