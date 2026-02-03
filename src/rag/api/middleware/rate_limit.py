"""Sliding-window rate limiter middleware backed by Redis."""

from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.rag.core.logging import get_logger

logger = get_logger(__name__)

EXEMPT_PATHS = {"/health", "/health/ready", "/health/live", "/metrics"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-user sliding window rate limiter.

    Uses a Redis sorted set where each member is a timestamp.
    Members older than the window are purged on each request.
    """

    def __init__(self, app, requests_per_minute: int = 60, burst: int = 10) -> None:
        super().__init__(app)
        self.rpm = requests_per_minute
        self.burst = burst
        self.window_seconds = 60

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        try:
            cache = request.app.state.cache
            user_key = self._identify(request)
            redis_key = f"ratelimit:{user_key}"
            now = time.time()
            window_start = now - self.window_seconds

            client = cache._require_client()
            pipe = client.pipeline()
            pipe.zremrangebyscore(cache._prefixed(redis_key), 0, window_start)
            pipe.zcard(cache._prefixed(redis_key))
            pipe.zadd(cache._prefixed(redis_key), {str(now): now})
            pipe.expire(cache._prefixed(redis_key), self.window_seconds + 1)
            _, count, *_ = await pipe.execute()

            remaining = max(0, self.rpm - count)
            reset_at = int(now) + self.window_seconds

            if count > self.rpm + self.burst:
                logger.warning("rate_limit_exceeded", user_key=user_key, count=count)
                return Response(
                    content='{"error":"RATE_LIMIT_EXCEEDED","message":"Too many requests"}',
                    status_code=429,
                    media_type="application/json",
                    headers={
                        "X-RateLimit-Limit": str(self.rpm),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_at),
                        "Retry-After": str(self.window_seconds),
                    },
                )

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.rpm)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_at)
            return response

        except Exception:
            # Never block a request due to rate limiter failure
            return await call_next(request)

    def _identify(self, request: Request) -> str:
        if auth := request.headers.get("authorization"):
            return f"token:{hash(auth) & 0xFFFFFF}"
        if api_key := request.headers.get("x-api-key"):
            return f"key:{hash(api_key) & 0xFFFFFF}"
        forwarded = request.headers.get("x-forwarded-for")
        ip = forwarded.split(",")[0].strip() if forwarded else request.client.host if request.client else "unknown"
        return f"ip:{ip}"
