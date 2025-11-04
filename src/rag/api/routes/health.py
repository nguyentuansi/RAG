"""Health check endpoints."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


async def _check_vector_store(request: Request) -> dict[str, Any]:
    try:
        vs = request.app.state.vector_store
        from src.rag.core.config import get_settings
        settings = get_settings()
        count = await vs.count(settings.collection_name)
        return {"status": "ok", "vector_count": count}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


async def _check_cache(request: Request) -> dict[str, Any]:
    try:
        cache = request.app.state.cache
        probe_key = "health:probe"
        await cache.set(probe_key, "1", ttl=10)
        val = await cache.get(probe_key)
        return {"status": "ok" if val == "1" else "degraded"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


async def _check_embedding(request: Request) -> dict[str, Any]:
    try:
        ep = request.app.state.embedding_provider
        ready = await ep.is_ready()
        return {"status": "ok" if ready else "loading", "model": ep.model_name}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@router.get("", summary="Full health check with component details")
async def health(request: Request) -> JSONResponse:
    start = time.monotonic()

    components = {
        "vector_store": await _check_vector_store(request),
        "cache": await _check_cache(request),
        "embedding": await _check_embedding(request),
    }

    all_ok = all(c["status"] == "ok" for c in components.values())
    any_error = any(c["status"] == "error" for c in components.values())

    overall = "healthy" if all_ok else ("degraded" if not any_error else "unhealthy")
    status_code = 200 if overall in {"healthy", "degraded"} else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall,
            "latency_ms": round((time.monotonic() - start) * 1000, 2),
            "components": components,
        },
    )


@router.get("/ready", summary="Readiness probe (k8s)")
async def readiness(request: Request) -> JSONResponse:
    """Returns 200 once all dependencies are ready to serve traffic."""
    ep = request.app.state.embedding_provider
    if not await ep.is_ready():
        return JSONResponse(status_code=503, content={"ready": False, "reason": "model_loading"})
    return JSONResponse(status_code=200, content={"ready": True})


@router.get("/live", summary="Liveness probe (k8s)")
async def liveness() -> JSONResponse:
    """Always returns 200 while the process is alive."""
    return JSONResponse(status_code=200, content={"alive": True})
