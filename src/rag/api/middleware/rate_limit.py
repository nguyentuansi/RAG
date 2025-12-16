"""Redis-backed sliding-window rate limiter middleware."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from src.rag.core.logging import get_logger

logger = get_logger(__name__)

_SKIP_PATHS = frozenset({"/health", "/health/live", "/health/ready"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter backed by Redis.

    Limits are applied per (IP, user_id) pair when a user is authenticated,
    otherwise per IP address.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests_per_minute: int = 60,
        burst: int = 10,
    ) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._burst = burst
        self._window = 60  # seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        try:
            cache = request.app.state.cache
        except AttributeError:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        user_id = getattr(request.state, "user_id", None)
        key = f"rate:{user_id or client_ip}"

        now = int(time.time())
        window_start = now - self._window

        try:
            pipe_key = f"rl:{key}:{now // self._window}"
            count = await cache.increment(pipe_key)
            if count == 1:
                await cache.expire(pipe_key, self._window * 2)

            limit = self._rpm + self._burst
            if count > limit:
                logger.warning(
                    "rate_limit_exceeded",
                    client=client_ip,
                    user=user_id,
                    count=count,
                    limit=limit,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": f"Too many requests. Limit: {self._rpm}/min",
                        "retry_after": self._window,
                    },
                    headers={
                        "Retry-After": str(self._window),
                        "X-RateLimit-Limit": str(self._rpm),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self._rpm)
            response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
            return response

        except Exception as exc:
            logger.warning("rate_limit_check_failed", error=str(exc))
            return await call_next(request)
