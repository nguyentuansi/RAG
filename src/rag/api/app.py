"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.rag.core.config import Settings, get_settings
from src.rag.core.exceptions import AuthenticationError, AuthorizationError, RAGException, RateLimitError
from src.rag.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory — returns a configured FastAPI instance."""
    cfg = settings or get_settings()

    configure_logging(log_level=cfg.log_level, json_logs=cfg.is_production)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("startup", environment=cfg.environment, version="0.1.0")
        # Initialise singletons stored on app.state
        from src.rag.infrastructure.vector_store.qdrant import QdrantVectorStore
        from src.rag.infrastructure.embeddings.sentence_transformer import SentenceTransformerProvider
        from src.rag.infrastructure.cache.redis import RedisCache

        vector_store = QdrantVectorStore(
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
            api_key=cfg.qdrant_api_key,
            prefer_grpc=cfg.qdrant_prefer_grpc,
        )
        embedding_provider = SentenceTransformerProvider(
            model_name=cfg.embedding_model,
            device=cfg.embedding_device,
            batch_size=cfg.embedding_batch_size,
        )
        cache = RedisCache(
            url=cfg.redis_url,
            default_ttl=cfg.cache_ttl_seconds,
        )

        app.state.vector_store = vector_store
        app.state.embedding_provider = embedding_provider
        app.state.cache = cache
        app.state.settings = cfg

        try:
            await cache.connect()
        except Exception as exc:
            logger.warning("redis_connect_failed", error=str(exc))

        await embedding_provider.warm_up()

        yield

        logger.info("shutdown")
        await vector_store.close()
        await embedding_provider.close()
        await cache.close()

    app = FastAPI(
        title="RAG Platform API",
        description="Production-grade Retrieval-Augmented Generation platform",
        version="0.1.0",
        docs_url="/docs" if not cfg.is_production else None,
        redoc_url="/redoc" if not cfg.is_production else None,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(RAGException)
    async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(AuthenticationError)
    async def auth_exception_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers={"Retry-After": "60"},
        )

    # Routers
    from src.rag.api.routes.health import router as health_router
    from src.rag.api.routes.documents import router as documents_router
    from src.rag.api.routes.search import router as search_router

    app.include_router(health_router, tags=["health"])
    app.include_router(documents_router, prefix="/api/v1", tags=["documents"])
    app.include_router(search_router, prefix="/api/v1", tags=["search"])

    return app
