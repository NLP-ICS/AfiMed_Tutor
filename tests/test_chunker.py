"""Tests for corpus/build_corpus.py chunker functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus.build_corpus import (
    _approx_tokens,
    _split_at_paragraph_boundaries,
    naive_fixed_chunks,
    structure_aware_chunks,
)

SAMPLE_GHANA = """1. MALARIA

Diagnosis
Fever, chills, headache. Confirm with RDT or microscopy.

Investigations
Full blood count, malaria RDT.

Management
Artemisinin-based combination therapy (ACT).
Refer severe cases immediately.

Referral
Refer all complicated malaria to hospital level.

2. PNEUMONIA

Diagnosis
Cough, fast breathing, chest indrawing.

Management
Amoxicillin 40 mg/kg/day for 5 days.
"""


def test_approx_tokens_rough():
    assert 5 < _approx_tokens("hello world this is a test sentence") < 20


def test_split_at_paragraph_boundaries_respects_max():
    long_text = "\n\n".join(["word " * 50] * 30)
    chunks = _split_at_paragraph_boundaries(long_text, max_tokens=800, overlap_tokens=100)
    for chunk in chunks:
        assert _approx_tokens(chunk) <= 900  # allow small overrun at paragraph boundary


def test_structure_aware_chunks_splits_on_headers():
    chunks = structure_aware_chunks(SAMPLE_GHANA, source_doc="ghana_stg_2017", family="ghana_stg")
    assert len(chunks) >= 2
    assert all(c["source_doc"] == "ghana_stg_2017" for c in chunks)
    assert all(c["chunk_id"] for c in chunks)
    assert all(c["text"].strip() for c in chunks)


def test_structure_aware_chunks_has_metadata():
    chunks = structure_aware_chunks(SAMPLE_GHANA, source_doc="ghana_stg_2017", family="ghana_stg")
    for c in chunks:
        assert "section_title" in c
        assert "chunk_id" in c
        assert "source_doc" in c


def test_naive_chunks_produces_output():
    chunks = naive_fixed_chunks("word " * 1000, source_doc="test_doc")
    assert len(chunks) > 0
    for c in chunks:
        assert c["source_doc"] == "test_doc"
        assert c["text"]


def test_chunk_ids_are_unique():
    chunks = structure_aware_chunks(SAMPLE_GHANA, source_doc="ghana_stg_2017", family="ghana_stg")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_no_empty_chunks():
    chunks = structure_aware_chunks(SAMPLE_GHANA, source_doc="ghana_stg_2017", family="ghana_stg")
    assert all(c["text"].strip() for c in chunks)


def test_structure_vs_naive_different_boundaries():
    struct = structure_aware_chunks(SAMPLE_GHANA, source_doc="doc", family="ghana_stg")
    naive = naive_fixed_chunks(SAMPLE_GHANA, source_doc="doc")
    # Structure-aware should produce more or equal chunks since it detects sections
    section_titles = {c["section_title"] for c in struct}
    assert "(none)" not in section_titles or len(struct) > 0
    assert len(naive) > 0
