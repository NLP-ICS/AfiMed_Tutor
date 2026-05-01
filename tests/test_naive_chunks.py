import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus.build_corpus import naive_fixed_chunks_chars


@pytest.mark.parametrize("window,overlap", [(800, 200), (100, 20)])
def test_naive_chunks_cover_text_and_slide(window: int, overlap: int):
    text = "x" * 2500
    chunks = naive_fixed_chunks_chars(
        text,
        source_doc="doc_a",
        window_chars=window,
        overlap_chars=overlap,
    )
    assert chunks
    assert all(c["source_doc"] == "doc_a" for c in chunks)
    assert all(len(c["text"]) <= window for c in chunks)
    overlap_ch = min(max(0, overlap), window - 1)
    stride = window - overlap_ch
    assert stride > 0
    assert len(chunks) >= max(1, (len(text) - window) // stride + 1)


def test_naive_sections_metadata_placeholder():
    text = "One two three.\n" * 100
    out = naive_fixed_chunks_chars(text, source_doc="s", window_chars=50, overlap_chars=10)
    for row in out:
        assert row["section_title"] == "(none)"
        assert row["condition"] is None
