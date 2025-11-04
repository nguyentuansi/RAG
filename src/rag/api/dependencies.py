"""FastAPI dependency injection functions."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from src.rag.core.config import Settings, get_settings
from src.rag.infrastructure.cache.redis import RedisCache
from src.rag.infrastructure.embeddings.base import EmbeddingProvider
from src.rag.infrastructure.vector_store.base import VectorStore


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider


def get_cache(request: Request) -> RedisCache:
    return request.app.state.cache


async def get_current_user(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    """Authenticate via JWT bearer token or API key."""
    from src.rag.security.auth import decode_access_token, verify_api_key
    from src.rag.core.exceptions import AuthenticationError

    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
        try:
            return decode_access_token(token, settings.jwt_secret_key, settings.jwt_algorithm)
        except AuthenticationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    if x_api_key:
        user = await verify_api_key(x_api_key, cache=request.app.state.cache)
        if user:
            return user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbeddingProviderDep = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
CacheDep = Annotated[RedisCache, Depends(get_cache)]
CurrentUserDep = Annotated[dict, Depends(get_current_user)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
