"""Redis-backed async cache with TTL and cache-aside pattern."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, TypeVar

import redis.asyncio as aioredis

from src.rag.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RedisCache:
    """Async Redis cache with JSON serialization, TTL management, and cache-aside helper."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        key_prefix: str = "rag",
        max_connections: int = 20,
    ) -> None:
        self._url = url
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix
        self._max_connections = max_connections
        self._pool: aioredis.ConnectionPool | None = None
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._pool = aioredis.ConnectionPool.from_url(
            self._url,
            max_connections=self._max_connections,
            decode_responses=True,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        await self._client.ping()
        logger.info("redis_connected", url=self._url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()
        logger.info("redis_disconnected")

    async def get(self, key: str) -> Any | None:
        client = self._require_client()
        raw = await client.get(self._prefixed(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        client = self._require_client()
        serialized = json.dumps(value, default=str)
        await client.setex(
            self._prefixed(key),
            ttl if ttl is not None else self._default_ttl,
            serialized,
        )

    async def delete(self, key: str) -> bool:
        client = self._require_client()
        deleted = await client.delete(self._prefixed(key))
        return bool(deleted)

    async def exists(self, key: str) -> bool:
        client = self._require_client()
        return bool(await client.exists(self._prefixed(key)))

    async def delete_pattern(self, pattern: str) -> int:
        client = self._require_client()
        keys = await client.keys(self._prefixed(pattern))
        if not keys:
            return 0
        return await client.delete(*keys)

    async def increment(self, key: str, amount: int = 1) -> int:
        client = self._require_client()
        return await client.incrby(self._prefixed(key), amount)

    async def expire(self, key: str, ttl: int) -> bool:
        client = self._require_client()
        return await client.expire(self._prefixed(key), ttl)

    @staticmethod
    def make_key(*parts: str) -> str:
        """Build a cache key and hash it to a fixed length."""
        raw = ":".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def cache_aside(
        self,
        key_fn: Callable[..., str],
        ttl: int | None = None,
    ) -> Callable:
        """Decorator that wraps an async function with cache-aside logic."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                cache_key = key_fn(*args, **kwargs)
                cached = await self.get(cache_key)
                if cached is not None:
                    logger.debug("cache_hit", key=cache_key)
                    return cached
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                logger.debug("cache_miss_stored", key=cache_key)
                return result
            return wrapper
        return decorator

    def _prefixed(self, key: str) -> str:
        return f"{self._key_prefix}:{key}"

    def _require_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisCache not connected. Call connect() first.")
        return self._client
