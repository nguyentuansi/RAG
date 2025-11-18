"""Search API endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.rag.api.dependencies import CacheDep, CurrentUserDep, EmbeddingProviderDep, SettingsDep, VectorStoreDep
from src.rag.core.logging import get_logger
from src.rag.domain.search import HybridSearchConfig, RetrievalStrategy, SearchQuery, SearchResponse, SearchResult

router = APIRouter()
logger = get_logger(__name__)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0
    collection_name: str | None = None
    filters: dict | None = None


@router.post("", response_model=SearchResponse, summary="Dense vector similarity search")
async def search(
    body: SearchRequest,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    cache: CacheDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> SearchResponse:
    t0 = time.monotonic()
    collection = body.collection_name or settings.collection_name

    cache_key = cache.make_key("search", body.query, str(body.top_k), collection)
    cached = await cache.get(cache_key)
    if cached:
        logger.debug("search_cache_hit", query=body.query[:60])
        return SearchResponse(**cached)

    query_vector = await embedding_provider.embed_text(body.query)
    results = await vector_store.search(
        collection,
        query_vector,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        filters=body.filters,
    )

    search_results = [
        SearchResult(
            chunk_id=r.id,
            document_id=r.payload.get("document_id", ""),
            content=r.payload.get("content", ""),
            score=r.score,
            chunk_index=r.payload.get("chunk_index", 0),
            metadata={k: v for k, v in r.payload.items() if k not in {"content", "document_id"}},
            source=r.payload.get("source"),
        )
        for r in results
    ]

    response = SearchResponse(
        query=body.query,
        results=search_results,
        total_found=len(search_results),
        retrieval_strategy=RetrievalStrategy.DENSE,
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        collection_name=collection,
    )

    await cache.set(cache_key, response.model_dump(), ttl=300)
    logger.info("search_completed", query=body.query[:60], results=len(search_results))
    return response


@router.post("/hybrid", response_model=SearchResponse, summary="Hybrid dense+sparse search")
async def hybrid_search(
    body: SearchRequest,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    cache: CacheDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
    dense_weight: float = Query(default=0.7, ge=0.0, le=1.0),
) -> SearchResponse:
    from src.rag.pipeline.retrieval import HybridRetriever

    t0 = time.monotonic()
    collection = body.collection_name or settings.collection_name

    query_vector = await embedding_provider.embed_text(body.query)
    retriever = HybridRetriever(vector_store=vector_store)

    results = await retriever.hybrid_fusion(
        collection_name=collection,
        query=body.query,
        query_vector=query_vector,
        top_k=body.top_k,
        dense_weight=dense_weight,
        sparse_weight=1.0 - dense_weight,
    )

    return SearchResponse(
        query=body.query,
        results=results,
        total_found=len(results),
        retrieval_strategy=RetrievalStrategy.HYBRID,
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        collection_name=collection,
    )


@router.post("/mmr", response_model=SearchResponse, summary="MMR diversified search")
async def mmr_search(
    body: SearchRequest,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
    lambda_mult: float = Query(default=0.5, ge=0.0, le=1.0, description="MMR diversity factor"),
) -> SearchResponse:
    from src.rag.pipeline.retrieval import HybridRetriever

    t0 = time.monotonic()
    collection = body.collection_name or settings.collection_name

    query_vector = await embedding_provider.embed_text(body.query)
    retriever = HybridRetriever(vector_store=vector_store)

    results = await retriever.mmr_rerank(
        collection_name=collection,
        query_vector=query_vector,
        top_k=body.top_k,
        fetch_k=body.top_k * 4,
        lambda_mult=lambda_mult,
    )

    return SearchResponse(
        query=body.query,
        results=results,
        total_found=len(results),
        retrieval_strategy=RetrievalStrategy.MMR,
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        collection_name=collection,
    )


@router.get("/collections/{name}/stats", summary="Get collection statistics")
async def collection_stats(
    name: str,
    vector_store: VectorStoreDep,
    current_user: CurrentUserDep,
) -> dict:
    info = await vector_store.get_collection_info(name)
    return {
        "collection": info.name,
        "vector_count": info.vector_count,
        "indexed_vectors": info.indexed_vector_count,
        "distance_metric": info.distance_metric,
        "status": info.status,
    }
