"""Search API endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter

from src.rag.api.dependencies import (
    CacheDep,
    CurrentUserDep,
    EmbeddingProviderDep,
    SettingsDep,
    VectorStoreDep,
)
from src.rag.core.logging import get_logger
from src.rag.domain.search import (
    RetrievalStrategy,
    SearchQuery,
    SearchResponse,
    SearchResult,
)
from src.rag.infrastructure.cache.redis import RedisCache

router = APIRouter()
logger = get_logger(__name__)


@router.post("/search", response_model=SearchResponse)
async def search(
    query: SearchQuery,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    cache: CacheDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> SearchResponse:
    """Dense vector similarity search."""
    t0 = time.monotonic()
    collection = query.collection_name or settings.collection_name
    cache_key = RedisCache.make_key("search", query.query, collection, str(query.top_k))

    cached = await cache.get(cache_key)
    if cached:
        logger.debug("search_cache_hit", query=query.query[:60])
        return SearchResponse(**cached)

    query_vector = await embedding_provider.embed_text(query.query)
    raw_results = await vector_store.search(
        collection_name=collection,
        query_vector=query_vector,
        top_k=query.top_k,
        score_threshold=query.score_threshold,
    )

    results = [
        SearchResult(
            chunk_id=r.id,
            document_id=r.payload.get("document_id", ""),
            content=r.payload.get("content", ""),
            score=r.score,
            chunk_index=r.payload.get("chunk_index", 0),
            metadata=r.payload,
            source=r.payload.get("source"),
        )
        for r in raw_results
    ]

    response = SearchResponse(
        query=query.query,
        results=results,
        total_found=len(results),
        retrieval_strategy=RetrievalStrategy.DENSE,
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        collection_name=collection,
    )

    await cache.set(cache_key, response.model_dump(), ttl=300)

    logger.info(
        "search_completed",
        query=query.query[:60],
        results=len(results),
        latency_ms=response.latency_ms,
        user=current_user.get("sub"),
    )
    return response


@router.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(
    query: SearchQuery,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> SearchResponse:
    """Hybrid dense+sparse search with Reciprocal Rank Fusion."""
    t0 = time.monotonic()
    from src.rag.pipeline.retrieval import HybridRetriever

    collection = query.collection_name or settings.collection_name
    retriever = HybridRetriever(vector_store, embedding_provider)

    results = await retriever.hybrid_search(
        query=query.query,
        collection_name=collection,
        top_k=query.top_k,
        hybrid_config=query.hybrid_config,
    )

    return SearchResponse(
        query=query.query,
        results=results,
        total_found=len(results),
        retrieval_strategy=RetrievalStrategy.HYBRID,
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        collection_name=collection,
    )


@router.post("/search/mmr", response_model=SearchResponse)
async def mmr_search(
    query: SearchQuery,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> SearchResponse:
    """Maximal Marginal Relevance search for diverse result sets."""
    t0 = time.monotonic()
    from src.rag.pipeline.retrieval import HybridRetriever

    collection = query.collection_name or settings.collection_name
    retriever = HybridRetriever(vector_store, embedding_provider)

    results = await retriever.mmr_search(
        query=query.query,
        collection_name=collection,
        top_k=query.top_k,
    )

    return SearchResponse(
        query=query.query,
        results=results,
        total_found=len(results),
        retrieval_strategy=RetrievalStrategy.MMR,
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        collection_name=collection,
    )


@router.get("/collections/{collection_name}/stats")
async def collection_stats(
    collection_name: str,
    vector_store: VectorStoreDep,
    current_user: CurrentUserDep,
) -> dict:
    """Return statistics for a collection."""
    info = await vector_store.get_collection_info(collection_name)
    return {
        "name": info.name,
        "vector_count": info.vector_count,
        "indexed_vector_count": info.indexed_vector_count,
        "vector_size": info.vector_size,
        "distance_metric": info.distance_metric,
        "status": info.status,
    }
