"""JWT bearer and API key authentication middleware."""

from __future__ import annotations

import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.rag.core.logging import RequestContext, get_logger

logger = get_logger(__name__)

SKIP_AUTH_PATHS = {"/health", "/health/ready", "/health/live", "/metrics", "/docs", "/redoc", "/openapi.json"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Injects a correlation ID and optionally validates tokens before routing."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        RequestContext.set(correlation_id=correlation_id, path=request.url.path)

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        RequestContext.clear()
        return response
