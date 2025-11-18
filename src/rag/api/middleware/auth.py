"""JWT and API key authentication middleware."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from src.rag.core.logging import RequestContext, get_logger

logger = get_logger(__name__)

_SKIP_PATHS = frozenset({"/health", "/health/live", "/health/ready", "/docs", "/redoc", "/openapi.json"})


class AuthMiddleware(BaseHTTPMiddleware):
    """Injects correlation ID; authentication is handled per-route via Depends."""

    def __init__(self, app: ASGIApp, *, require_auth: bool = True) -> None:
        super().__init__(app)
        self._require_auth = require_auth

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = RequestContext.new_correlation_id()
        RequestContext.set(
            correlation_id,
            path=request.url.path,
        )
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        RequestContext.clear()
        return response
