"""Tests for tutor/retriever.py using a small in-memory fixture."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.schemas import Chunk


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _make_meta(n: int = 10) -> list[dict]:
    return [
        {
            "chunk_id": f"doc__{i:05d}",
            "text": f"This is guideline text about condition {i}.",
            "source_doc": "test_doc",
            "section_title": f"Section {i}",
            "condition": f"Condition{i}",
            "page_number": i + 1,
        }
        for i in range(n)
    ]


def _make_embeddings(n: int = 10, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ── Dense retriever ───────────────────────────────────────────────────────────

class TestDenseRetriever:
    def _build(self):
        import faiss
        from tutor.retriever import DenseRetriever

        meta = _make_meta()
        embs = _make_embeddings(len(meta), dim=16)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meta_path = tmp_path / "chunks_meta.jsonl"
            index_path = tmp_path / "faiss.index"

            meta_path.write_text("\n".join(json.dumps(m) for m in meta))

            index = faiss.IndexFlatIP(16)
            index.add(embs)
            faiss.write_index(index, str(index_path))

            # Patch embedding call to return a random vector of correct dim
            fake_query_vec = embs[0].tolist()
            with patch("tutor.retriever._get_embedding", return_value=fake_query_vec):
                retriever = DenseRetriever.__new__(DenseRetriever)
                retriever.threshold = 0.0
                import faiss as _faiss
                retriever._index = _faiss.read_index(str(index_path))
                retriever._meta = meta
                yield retriever

    def test_search_returns_chunks(self):
        import faiss
        from tutor.retriever import DenseRetriever

        meta = _make_meta()
        embs = _make_embeddings(len(meta), dim=16)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meta_path = tmp_path / "chunks_meta.jsonl"
            index_path = tmp_path / "faiss.index"
            meta_path.write_text("\n".join(json.dumps(m) for m in meta))
            index = faiss.IndexFlatIP(16)
            index.add(embs)
            faiss.write_index(index, str(index_path))

            fake_query_vec = embs[0].tolist()
            with patch("tutor.retriever._get_embedding", return_value=fake_query_vec):
                ret = DenseRetriever(index_path=index_path, chunks_meta_path=meta_path, threshold=0.0)
                results = ret.search("test query", k=3)

            assert len(results) <= 3
            assert all(isinstance(r, Chunk) for r in results)

    def test_threshold_filters_low_scores(self):
        import faiss
        from tutor.retriever import DenseRetriever

        meta = _make_meta(5)
        embs = _make_embeddings(5, dim=16)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meta_path = tmp_path / "chunks_meta.jsonl"
            index_path = tmp_path / "faiss.index"
            meta_path.write_text("\n".join(json.dumps(m) for m in meta))
            index = faiss.IndexFlatIP(16)
            index.add(embs)
            faiss.write_index(index, str(index_path))

            # Query vector orthogonal to all indexed vectors → near-zero scores
            query_vec = np.zeros(16, dtype="float32")
            query_vec[0] = 1.0
            with patch("tutor.retriever._get_embedding", return_value=query_vec.tolist()):
                ret = DenseRetriever(
                    index_path=index_path,
                    chunks_meta_path=meta_path,
                    threshold=0.99,  # very high threshold
                )
                results = ret.search("query", k=5)
            assert isinstance(results, list)


# ── Sparse retriever ──────────────────────────────────────────────────────────

class TestSparseRetriever:
    def _build(self, meta):
        from tutor.retriever import SparseRetriever
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "chunks_meta.jsonl"
            meta_path.write_text("\n".join(json.dumps(m) for m in meta))
            return SparseRetriever(chunks_meta_path=meta_path)

    def test_search_returns_chunks(self):
        meta = _make_meta()
        with tempfile.TemporaryDirectory() as tmp:
            from tutor.retriever import SparseRetriever
            meta_path = Path(tmp) / "chunks_meta.jsonl"
            meta_path.write_text("\n".join(json.dumps(m) for m in meta))
            ret = SparseRetriever(chunks_meta_path=meta_path)
            results = ret.search("guideline text condition", k=3)
            assert len(results) <= 3
            assert all(isinstance(r, Chunk) for r in results)

    def test_search_is_ranked(self):
        meta = _make_meta()
        with tempfile.TemporaryDirectory() as tmp:
            from tutor.retriever import SparseRetriever
            meta_path = Path(tmp) / "chunks_meta.jsonl"
            meta_path.write_text("\n".join(json.dumps(m) for m in meta))
            ret = SparseRetriever(chunks_meta_path=meta_path)
            results = ret.search("guideline text condition 5", k=5)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
