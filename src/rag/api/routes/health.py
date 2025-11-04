"""Health check endpoints."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class ComponentStatus(BaseModel):
    name: str
    status: str
    latency_ms: float | None = None
    details: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    components: list[ComponentStatus]


_start_time = time.monotonic()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Full health check with component status."""
    components: list[ComponentStatus] = []

    # Vector store check
    try:
        t0 = time.monotonic()
        vector_store = request.app.state.vector_store
        await vector_store.collection_exists("_health_probe")
        components.append(ComponentStatus(
            name="vector_store",
            status="ok",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        ))
    except Exception as exc:
        components.append(ComponentStatus(
            name="vector_store",
            status="degraded",
            details={"error": str(exc)},
        ))

    # Redis check
    try:
        t0 = time.monotonic()
        cache = request.app.state.cache
        await cache.get("_health_probe")
        components.append(ComponentStatus(
            name="cache",
            status="ok",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        ))
    except Exception as exc:
        components.append(ComponentStatus(
            name="cache",
            status="degraded",
            details={"error": str(exc)},
        ))

    # Embedding model check
    try:
        embedding_provider = request.app.state.embedding_provider
        ready = await embedding_provider.is_ready()
        components.append(ComponentStatus(
            name="embedding_model",
            status="ok" if ready else "loading",
            details={"model": embedding_provider.model_name},
        ))
    except Exception as exc:
        components.append(ComponentStatus(
            name="embedding_model",
            status="degraded",
            details={"error": str(exc)},
        ))

    overall = "ok" if all(c.status == "ok" for c in components) else "degraded"

    return HealthResponse(
        status=overall,
        version="0.1.0",
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        components=components,
    )


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe — just confirms process is up."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(request: Request) -> dict[str, str]:
    """Kubernetes readiness probe — confirms app can serve traffic."""
    try:
        embedding_provider = request.app.state.embedding_provider
        ready = await embedding_provider.is_ready()
        if not ready:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="Embedding model still loading")
        return {"status": "ready"}
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Not ready")
