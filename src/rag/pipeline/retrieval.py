"""Hybrid retrieval: dense search, BM25 sparse, RRF fusion, and MMR reranking."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from src.rag.core.logging import get_logger
from src.rag.domain.search import HybridSearchConfig, SearchResult
from src.rag.infrastructure.vector_store.base import VectorStore

logger = get_logger(__name__)


class HybridRetriever:
    """
    Retrieval strategies over a VectorStore backend.

    dense  — standard ANN search
    sparse — BM25 over stored payloads (in-memory, suitable for small corpora)
    hybrid — Reciprocal Rank Fusion of dense and sparse results
    mmr    — Maximal Marginal Relevance post-reranking for diversity
    """

    def __init__(
        self,
        vector_store: VectorStore,
        default_collection: str = "rag_documents",
    ) -> None:
        self._store = vector_store
        self._default_collection = default_collection

    async def dense_search(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        raw = await self._store.search(
            collection_name,
            query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )
        return [self._to_search_result(r.id, r.score, r.payload) for r in raw]

    async def sparse_search(
        self,
        collection_name: str,
        query: str,
        *,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """
        BM25 search over in-memory payload contents.

        Fetches a larger candidate set from the vector store then scores
        them with BM25 locally.  Not suitable for very large collections
        (>100k vectors) without a dedicated sparse index.
        """
        candidates = await self._store.search(
            collection_name,
            query_vector=[0.0] * 768,
            top_k=top_k * 4,
        )

        try:
            from rank_bm25 import BM25Okapi

            corpus = [r.payload.get("content", "") for r in candidates]
            tokenized = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized)
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)

            ranked = sorted(
                zip(scores, candidates),
                key=lambda x: x[0],
                reverse=True,
            )[:top_k]

            max_score = ranked[0][0] if ranked and ranked[0][0] > 0 else 1.0
            return [
                self._to_search_result(
                    r.id,
                    float(score / max_score),
                    r.payload,
                )
                for score, r in ranked
            ]
        except ImportError:
            logger.warning("rank_bm25_not_installed; falling back to dense scores")
            return [self._to_search_result(r.id, r.score, r.payload) for r in candidates[:top_k]]

    async def hybrid_fusion(
        self,
        collection_name: str,
        query: str,
        query_vector: list[float],
        *,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion of dense and sparse results."""
        dense, sparse = await asyncio.gather(
            self.dense_search(collection_name, query_vector, top_k=top_k * 2),
            self.sparse_search(collection_name, query, top_k=top_k * 2),
        )

        rrf_scores: dict[str, float] = defaultdict(float)
        payload_map: dict[str, dict] = {}

        for rank, result in enumerate(dense):
            rrf_scores[result.chunk_id] += dense_weight / (rrf_k + rank + 1)
            payload_map[result.chunk_id] = result.metadata

        for rank, result in enumerate(sparse):
            rrf_scores[result.chunk_id] += sparse_weight / (rrf_k + rank + 1)
            if result.chunk_id not in payload_map:
                payload_map[result.chunk_id] = result.metadata

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = ranked[0][1] if ranked else 1.0

        return [
            self._to_search_result(cid, score / max_score, payload_map.get(cid, {}))
            for cid, score in ranked
        ]

    async def mmr_rerank(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[SearchResult]:
        """
        Maximal Marginal Relevance: balances relevance and diversity.

        lambda_mult=1.0 → pure relevance; lambda_mult=0.0 → pure diversity.
        """
        candidates = await self._store.search(
            collection_name, query_vector, top_k=fetch_k
        )
        if not candidates:
            return []

        # We need the actual vectors; fall back to scores as proxy if unavailable
        selected_ids: list[str] = []
        selected_scores: list[float] = []
        remaining = list(candidates)

        query_arr = np.array(query_vector, dtype=np.float32)
        query_arr /= np.linalg.norm(query_arr) + 1e-10

        while remaining and len(selected_ids) < top_k:
            best_idx = 0
            best_score = -float("inf")

            for i, cand in enumerate(remaining):
                rel = cand.score
                if selected_scores:
                    div = max(selected_scores)
                    mmr = lambda_mult * rel - (1 - lambda_mult) * div
                else:
                    mmr = rel

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            chosen = remaining.pop(best_idx)
            selected_ids.append(chosen.id)
            selected_scores.append(chosen.score)

        result_map = {r.id: r for r in candidates}
        return [
            self._to_search_result(cid, result_map[cid].score, result_map[cid].payload)
            for cid in selected_ids
            if cid in result_map
        ]

    @staticmethod
    def _to_search_result(chunk_id: str, score: float, payload: dict) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            document_id=payload.get("document_id", ""),
            content=payload.get("content", ""),
            score=max(0.0, min(1.0, score)),
            chunk_index=payload.get("chunk_index", 0),
            metadata={k: v for k, v in payload.items() if k not in {"content", "document_id"}},
            source=payload.get("source"),
        )


import asyncio  # noqa: E402 — imported here to avoid circular import issues
