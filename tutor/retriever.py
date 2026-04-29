"""FAISS dense retriever and BM25 sparse retriever (§6.2, §10).

Both implement the Retriever protocol so they are interchangeable at runtime.
Active backend is chosen by RETRIEVER_BACKEND env var (default: dense).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from tutor.schemas import Chunk

# Paths to index artifacts (relative to repo root, resolved at load time)
_REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = _REPO_ROOT / "corpus" / "chunks.jsonl"
CHUNKS_META_PATH = _REPO_ROOT / "corpus" / "chunks_meta.jsonl"
FAISS_INDEX_PATH = _REPO_ROOT / "corpus" / "faiss.index"


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_local_model = None

def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _local_model = SentenceTransformer(model_name)
    return _local_model


def _get_embedding(text: str) -> list[float]:
    """Embed a single text using the configured provider."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    if provider == "openai":
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    if provider == "voyage":
        import voyageai
        vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        result = vo.embed([text], model="voyage-3-lite", input_type="query")
        return result.embeddings[0]
    if provider == "local":
        model = _get_local_model()
        return model.encode(text, normalize_embeddings=True).tolist()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """FAISS IndexFlatIP over normalized embeddings (cosine similarity via IP)."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        chunks_meta_path: Path = CHUNKS_META_PATH,
        threshold: float | None = None,
    ) -> None:
        import faiss

        self.threshold = threshold if threshold is not None else float(
            os.getenv("RETRIEVER_THRESHOLD", "0.30")
        )
        self._index = faiss.read_index(str(index_path))
        self._meta: list[dict] = []
        with open(chunks_meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._meta.append(json.loads(line))

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        if k is None:
            k = int(os.getenv("RETRIEVER_TOP_K", "5"))
        vec = _normalize(np.array(_get_embedding(query), dtype="float32")).reshape(1, -1)
        scores, indices = self._index.search(vec, k)

        results: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < self.threshold:
                continue
            m = self._meta[idx]
            results.append(
                Chunk(
                    chunk_id=m["chunk_id"],
                    text=m["text"],
                    source_doc=m["source_doc"],
                    section_title=m["section_title"],
                    condition=m.get("condition"),
                    page_number=m.get("page_number"),
                    score=float(score),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Sparse retriever (BM25)
# ---------------------------------------------------------------------------

class SparseRetriever:
    """BM25 over chunk texts using rank_bm25."""

    def __init__(self, chunks_meta_path: Path = CHUNKS_META_PATH) -> None:
        from rank_bm25 import BM25Okapi

        self._meta: list[dict] = []
        with open(chunks_meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._meta.append(json.loads(line))

        tokenized = [m["text"].lower().split() for m in self._meta]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        if k is None:
            k = int(os.getenv("RETRIEVER_TOP_K", "5"))
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[Chunk] = []
        for idx in top_indices:
            m = self._meta[int(idx)]
            results.append(
                Chunk(
                    chunk_id=m["chunk_id"],
                    text=m["text"],
                    source_doc=m["source_doc"],
                    section_title=m["section_title"],
                    condition=m.get("condition"),
                    page_number=m.get("page_number"),
                    score=float(scores[idx]),
                )
            )
        return results


def build_retriever(backend: str | None = None) -> Retriever:
    """Factory: return the correct retriever from RETRIEVER_BACKEND env var."""
    backend = backend or os.getenv("RETRIEVER_BACKEND", "dense")
    if backend == "dense":
        return DenseRetriever()
    if backend == "sparse":
        return SparseRetriever()
    raise ValueError(f"Unknown RETRIEVER_BACKEND: {backend!r}. Choose 'dense' or 'sparse'.")
