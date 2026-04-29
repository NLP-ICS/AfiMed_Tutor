"""Related-question retrieval (§6.4).

Two methods:
  Method A (semantic)  — cosine similarity over pre-computed question embeddings.
  Method B (keyword)   — specialty filter + BM25 over question text.

Default is Method A; selectable via RELATED_Q_METHOD env var.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

from tutor.schemas import MCQItem, RelatedQuestion

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MCQ_POOL_PATH = _REPO_ROOT / "data" / "afrimedqa_mcq_pool.jsonl"
_EMBEDDINGS_PATH = _REPO_ROOT / "data" / "question_embeddings.npy"


def _load_pool() -> list[MCQItem]:
    items: list[MCQItem] = []
    with open(_MCQ_POOL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(MCQItem(**json.loads(line)))
    return items


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


_local_model = None

def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _local_model = SentenceTransformer(model_name)
    return _local_model


def _embed_query(text: str) -> np.ndarray:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    if provider == "openai":
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(resp.data[0].embedding, dtype="float32")
    if provider == "voyage":
        import voyageai
        vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        result = vo.embed([text], model="voyage-3-lite", input_type="query")
        return np.array(result.embeddings[0], dtype="float32")
    if provider == "local":
        model = _get_local_model()
        return model.encode(text, normalize_embeddings=True).astype("float32")
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


class SemanticRelatedRetriever:
    """Method A: cosine similarity over pre-computed question embeddings."""

    def __init__(self, exclude_id: str | None = None) -> None:
        self._pool = _load_pool()
        self._embeddings = np.load(_EMBEDDINGS_PATH)  # shape (N, D), normalized
        self._exclude_id = exclude_id

    def get_related(self, query: str, k: int = 3) -> list[RelatedQuestion]:
        query_vec = _normalize(_embed_query(query)).reshape(1, -1)
        scores = (self._embeddings @ query_vec.T).flatten()

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[RelatedQuestion] = []
        for idx, score in indexed:
            item = self._pool[idx]
            if item.question_id == self._exclude_id:
                continue
            results.append(
                RelatedQuestion(
                    question_id=item.question_id,
                    question=item.question,
                    specialty=item.specialty,
                    retrieval_score=float(score),
                )
            )
            if len(results) >= k:
                break
        return results


class KeywordRelatedRetriever:
    """Method B: specialty filter + BM25 over question text."""

    def __init__(self, exclude_id: str | None = None) -> None:
        from rank_bm25 import BM25Okapi

        self._pool = _load_pool()
        self._exclude_id = exclude_id
        # Build a BM25 index per specialty for efficiency
        self._by_specialty: dict[str, list[int]] = {}
        for i, item in enumerate(self._pool):
            self._by_specialty.setdefault(item.specialty, []).append(i)

        tokenized = [item.question.lower().split() for item in self._pool]
        self._bm25 = BM25Okapi(tokenized)

    def get_related(
        self, query: str, specialty: str | None = None, k: int = 3
    ) -> list[RelatedQuestion]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Restrict to same specialty if provided
        if specialty:
            mask = np.full(len(self._pool), -np.inf)
            for idx in self._by_specialty.get(specialty, []):
                mask[idx] = scores[idx]
            scores = mask

        top_indices = np.argsort(scores)[::-1]
        results: list[RelatedQuestion] = []
        for idx in top_indices:
            item = self._pool[int(idx)]
            if item.question_id == self._exclude_id:
                continue
            if scores[idx] == -np.inf:
                continue
            results.append(
                RelatedQuestion(
                    question_id=item.question_id,
                    question=item.question,
                    specialty=item.specialty,
                    retrieval_score=float(scores[idx]),
                )
            )
            if len(results) >= k:
                break
        return results


def build_related_retriever(
    method: str | None = None,
    exclude_id: str | None = None,
) -> SemanticRelatedRetriever | KeywordRelatedRetriever:
    method = method or os.getenv("RELATED_Q_METHOD", "semantic")
    if method == "semantic":
        return SemanticRelatedRetriever(exclude_id=exclude_id)
    if method == "keyword":
        return KeywordRelatedRetriever(exclude_id=exclude_id)
    raise ValueError(f"Unknown RELATED_Q_METHOD: {method!r}. Choose 'semantic' or 'keyword'.")
