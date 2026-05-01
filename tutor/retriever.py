"""FAISS dense retriever, BM25 sparse retriever, hybrid RRF retriever,
and cross-encoder re-ranked retriever (§6.2, §10).

All four implement the Retriever protocol so they are interchangeable.
Active backend is chosen by RETRIEVER_BACKEND env var (default: dense).

Person 3 additions (E4, E6):
  - HybridRetriever   : Reciprocal Rank Fusion over dense + sparse (E4)
  - RerankedRetriever : Cross-encoder re-ranking on top of dense (E6)
  Select via RETRIEVER_BACKEND=hybrid  or  RETRIEVER_BACKEND=reranked.
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


# ---------------------------------------------------------------------------
# Hybrid retriever — Reciprocal Rank Fusion (E4, Person 3)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combine dense and sparse rankings with Reciprocal Rank Fusion.

    RRF score = Σ 1 / (RRF_K + rank)  across retrievers
    (Cormack, Clarke & Buettcher, SIGIR 2009).

    Protocol (E4):
      - Fetch k_candidates=10 results from each retriever.
      - Accumulate RRF scores per chunk_id.
      - Return the top-k chunks by fused score.

    Select via: RETRIEVER_BACKEND=hybrid
    """

    RRF_K: int = 60  # standard constant from the original paper

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        chunks_meta_path: Path = CHUNKS_META_PATH,
        k_candidates: int = 10,
    ) -> None:
        self._dense = DenseRetriever(
            index_path=index_path, chunks_meta_path=chunks_meta_path
        )
        self._sparse = SparseRetriever(chunks_meta_path=chunks_meta_path)
        self._k_candidates = k_candidates

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        if k is None:
            k = int(os.getenv("RETRIEVER_TOP_K", "5"))

        dense_results = self._dense.search(query, k=self._k_candidates)
        sparse_results = self._sparse.search(query, k=self._k_candidates)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, chunk in enumerate(dense_results):
            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0.0)
                + 1.0 / (self.RRF_K + rank + 1)
            )
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_results):
            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0.0)
                + 1.0 / (self.RRF_K + rank + 1)
            )
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        sorted_ids = sorted(
            rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True
        )[:k]

        return [
            chunk_map[cid].model_copy(update={"score": rrf_scores[cid]})
            for cid in sorted_ids
        ]


# ---------------------------------------------------------------------------
# Cross-encoder re-ranked retriever (E6, Person 3)
# ---------------------------------------------------------------------------

class RerankedRetriever:
    """Dense retrieval followed by cross-encoder re-ranking.

    Step 1: Fetch n_candidates=20 from DenseRetriever.
    Step 2: Re-score all candidates with a cross-encoder that jointly encodes
            the (query, passage) pair — more accurate than bi-encoder scoring.
    Step 3: Return top-k by re-ranked score.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~85 MB, local, no API key).

    Select via: RETRIEVER_BACKEND=reranked
    """

    _DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        chunks_meta_path: Path = CHUNKS_META_PATH,
        n_candidates: int = 20,
        model_name: str | None = None,
    ) -> None:
        self._dense = DenseRetriever(
            index_path=index_path, chunks_meta_path=chunks_meta_path
        )
        self._n_candidates = n_candidates
        self._model_name = model_name or os.getenv(
            "CROSS_ENCODER_MODEL", self._DEFAULT_MODEL
        )
        self._cross_encoder = None  # lazy-loaded on first search()

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            self._cross_encoder = CrossEncoder(self._model_name)
        return self._cross_encoder

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        if k is None:
            k = int(os.getenv("RETRIEVER_TOP_K", "5"))

        candidates = self._dense.search(query, k=self._n_candidates)
        if not candidates:
            return []

        cross_encoder = self._get_cross_encoder()
        pairs = [(query, chunk.text) for chunk in candidates]
        scores = cross_encoder.predict(pairs)

        # Sort by cross-encoder score descending, take top-k
        ranked = sorted(
            zip(scores.tolist(), candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            chunk.model_copy(update={"score": float(score)})
            for score, chunk in ranked[:k]
        ]


def build_retriever(backend: str | None = None) -> Retriever:
    """Factory: return the correct retriever from RETRIEVER_BACKEND env var.

    Supported backends: dense | sparse | hybrid | reranked
    """
    backend = backend or os.getenv("RETRIEVER_BACKEND", "dense")
    if backend == "dense":
        return DenseRetriever()
    if backend == "sparse":
        return SparseRetriever()
    if backend == "hybrid":
        return HybridRetriever()
    if backend == "reranked":
        return RerankedRetriever()
    raise ValueError(
        f"Unknown RETRIEVER_BACKEND: {backend!r}. "
        "Choose 'dense', 'sparse', 'hybrid', or 'reranked'."
    )
