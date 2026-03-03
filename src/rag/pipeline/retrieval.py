"""Hybrid retrieval: dense + BM25 sparse + RRF fusion + MMR reranking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.rag.core.logging import get_logger
from src.rag.domain.search import SearchResult
from src.rag.infrastructure.vector_store.base import VectorStore

logger = get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class BM25Scorer:
    """In-memory BM25 scorer over a corpus of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[dict] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def index(self, docs: list[dict]) -> None:
        self._corpus = docs
        tokenized = [d["content"].lower().split() for d in docs]
        N = len(tokenized)
        self._avgdl = sum(len(t) for t in tokenized) / N if N else 1.0
        df: dict[str, int] = {}
        for tokens in tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_terms = query.lower().split()
        scores: list[tuple[str, float]] = []
        for doc in self._corpus:
            tokens = doc["content"].lower().split()
            dl = len(tokens)
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            doc_score = 0.0
            for term in query_terms:
                if term not in self._idf:
                    continue
                tf = tf_map.get(term, 0)
                idf = self._idf[term]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                doc_score += idf * (num / den)
            scores.append((doc["id"], doc_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _to_search_result(r: Any) -> SearchResult:
    return SearchResult(
        chunk_id=r.id,
        document_id=r.payload.get("document_id", ""),
        content=r.payload.get("content", ""),
        score=r.score,
        chunk_index=r.payload.get("chunk_index", 0),
        metadata={k: v for k, v in r.payload.items() if k not in {"content", "document_id"}},
        source=r.payload.get("source"),
    )


class HybridRetriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store

    async def dense_search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        raw = await self.vs.search(
            collection_name, query_vector, top_k=top_k, filters=filters
        )
        return [_to_search_result(r) for r in raw]

    async def hybrid_fusion(
        self,
        collection_name: str,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        fetch_k = top_k * 4

        dense_raw = await self.vs.search(collection_name, query_vector, top_k=fetch_k)
        dense_ids = [r.id for r in dense_raw]
        dense_map = {r.id: r for r in dense_raw}

        bm25 = BM25Scorer()
        corpus = [{"id": r.id, "content": r.payload.get("content", "")} for r in dense_raw]
        bm25.index(corpus)
        sparse_scores = bm25.score(query, top_k=fetch_k)
        sparse_ids = [sid for sid, _ in sparse_scores]

        rrf_scores = _reciprocal_rank_fusion([dense_ids, sparse_ids], k=rrf_k)
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for chunk_id, fused_score in ranked:
            if chunk_id in dense_map:
                raw = dense_map[chunk_id]
                results.append(SearchResult(
                    chunk_id=raw.id,
                    document_id=raw.payload.get("document_id", ""),
                    content=raw.payload.get("content", ""),
                    score=round(fused_score, 4),
                    chunk_index=raw.payload.get("chunk_index", 0),
                    metadata={k: v for k, v in raw.payload.items() if k not in {"content", "document_id"}},
                    source=raw.payload.get("source"),
                ))
        return results

    async def mmr_rerank(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[SearchResult]:
        """Maximal Marginal Relevance: balance relevance and diversity."""
        candidates_raw = await self.vs.search(collection_name, query_vector, top_k=fetch_k)
        if not candidates_raw:
            return []

        candidates = [_to_search_result(r) for r in candidates_raw]
        embeddings = {r.chunk_id: candidates_raw[i].payload.get("embedding") for i, r in enumerate(candidates)}

        selected: list[SearchResult] = []
        remaining = list(candidates)

        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda r: r.score)
            else:
                def mmr_score(r: SearchResult) -> float:
                    relevance = r.score
                    if embeddings.get(r.chunk_id) and any(embeddings.get(s.chunk_id) for s in selected):
                        sims = [
                            _cosine_similarity(embeddings[r.chunk_id], embeddings[s.chunk_id])
                            for s in selected
                            if embeddings.get(s.chunk_id)
                        ]
                        redundancy = max(sims) if sims else 0.0
                    else:
                        redundancy = 0.0
                    return lambda_mult * relevance - (1 - lambda_mult) * redundancy

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return selected
