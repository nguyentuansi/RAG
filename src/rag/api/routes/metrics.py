"""Prometheus metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Response

router = APIRouter()


def _get_registry():
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess

        registry = CollectorRegistry()
        try:
            multiprocess.MultiProcessCollector(registry)
        except Exception:
            from prometheus_client import REGISTRY
            registry = REGISTRY
        return registry, CONTENT_TYPE_LATEST, generate_latest
    except ImportError:
        return None, None, None


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose Prometheus metrics. Scrape this with your Prometheus server."""
    registry, content_type, generate_latest = _get_registry()
    if registry is None:
        return Response(content="# prometheus_client not installed\n", media_type="text/plain")
    return Response(content=generate_latest(registry), media_type=content_type)


class RAGMetrics:
    """
    Central Prometheus metrics registry for the RAG platform.

    Instantiate once at startup and store on app.state.metrics.
    """

    def __init__(self) -> None:
        try:
            from prometheus_client import Counter, Gauge, Histogram

            self.requests_total = Counter(
                "rag_requests_total",
                "Total HTTP requests",
                ["method", "endpoint", "status_code"],
            )
            self.request_duration_seconds = Histogram(
                "rag_request_duration_seconds",
                "HTTP request duration in seconds",
                ["method", "endpoint"],
                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            )
            self.documents_processed_total = Counter(
                "rag_documents_processed_total",
                "Total documents ingested",
                ["format", "status"],
            )
            self.search_latency_seconds = Histogram(
                "rag_search_latency_seconds",
                "Search operation latency",
                ["strategy"],
                buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            )
            self.embedding_duration_seconds = Histogram(
                "rag_embedding_duration_seconds",
                "Embedding generation latency",
                ["model", "batch_size"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            )
            self.vector_store_size = Gauge(
                "rag_vector_store_size",
                "Number of vectors in the store",
                ["collection"],
            )
            self.cache_hits_total = Counter(
                "rag_cache_hits_total",
                "Cache hits",
                ["operation"],
            )
            self.cache_misses_total = Counter(
                "rag_cache_misses_total",
                "Cache misses",
                ["operation"],
            )
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available
