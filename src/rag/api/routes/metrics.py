"""Prometheus metrics placeholder (populated in Phase 10)."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return PlainTextResponse("# prometheus_client not installed\n")
