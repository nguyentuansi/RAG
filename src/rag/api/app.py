"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.rag.core.config import get_settings
from src.rag.core.exceptions import RAGException
from src.rag.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.is_production)
    logger.info("rag_api_starting", version=app.version, environment=settings.environment)

    from src.rag.infrastructure.vector_store.qdrant import QdrantVectorStore
    from src.rag.infrastructure.embeddings.sentence_transformer import SentenceTransformerProvider
    from src.rag.infrastructure.cache.redis import RedisCache

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )
    embedding_provider = SentenceTransformerProvider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    cache = RedisCache(
        url=settings.redis_url,
        default_ttl=settings.cache_ttl_seconds,
        max_connections=settings.cache_max_connections,
    )

    try:
        await cache.connect()
        await embedding_provider.warm_up()
        await vector_store.create_collection(
            name=settings.collection_name,
            vector_size=embedding_provider.dimension,
        )
    except Exception as exc:
        logger.error("startup_failed", error=str(exc))
        raise

    app.state.vector_store = vector_store
    app.state.embedding_provider = embedding_provider
    app.state.cache = cache

    logger.info("rag_api_ready")
    yield

    logger.info("rag_api_shutting_down")
    await cache.close()
    await embedding_provider.close()
    await vector_store.close()
    logger.info("rag_api_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Platform API",
        description="Production-grade Retrieval-Augmented Generation platform",
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RAGException)
    async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
        logger.warning("request_error", error_code=exc.error_code, path=request.url.path)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_error", error=str(exc), path=request.url.path, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
        )

    from src.rag.api.routes import health, documents, search, metrics

    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(documents.router, prefix="/documents", tags=["Documents"])
    app.include_router(search.router, prefix="/search", tags=["Search"])
    app.include_router(metrics.router, tags=["Metrics"])

    return app


app = create_app()
