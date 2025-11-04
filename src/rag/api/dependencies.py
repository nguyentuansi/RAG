"""FastAPI dependency injection functions."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.rag.core.config import Settings
from src.rag.core.exceptions import AuthenticationError
from src.rag.infrastructure.cache.redis import RedisCache
from src.rag.infrastructure.embeddings.base import EmbeddingProvider
from src.rag.infrastructure.vector_store.base import VectorStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider


def get_cache(request: Request) -> RedisCache:
    return request.app.state.cache


async def get_current_user(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """Extract and validate the authenticated user from the request."""
    from src.rag.security.auth import verify_token

    token: str | None = None

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]

    if not token:
        api_key = request.headers.get(settings.api_key_header)
        if api_key:
            from src.rag.security.auth import verify_api_key
            return await verify_api_key(api_key)

    if not token:
        raise AuthenticationError("Missing authentication credentials")

    return await verify_token(token, settings.jwt_secret_key, settings.jwt_algorithm)


# Type aliases for DI
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbeddingProviderDep = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
CacheDep = Annotated[RedisCache, Depends(get_cache)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
CurrentUserDep = Annotated[dict, Depends(get_current_user)]
