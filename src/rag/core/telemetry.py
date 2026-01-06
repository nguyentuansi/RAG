"""OpenTelemetry tracing setup."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from src.rag.core.logging import get_logger

logger = get_logger(__name__)


def configure_telemetry(
    service_name: str = "rag-api",
    otlp_endpoint: str | None = None,
    *,
    enabled: bool = True,
) -> None:
    """Configure OpenTelemetry TracerProvider with optional OTLP export."""
    if not enabled:
        logger.info("telemetry_disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)

        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("otlp_exporter_configured", endpoint=otlp_endpoint)
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        logger.info("telemetry_configured", service=service_name)

    except ImportError:
        logger.warning(
            "opentelemetry_not_installed",
            hint="pip install opentelemetry-sdk opentelemetry-exporter-otlp",
        )


def instrument_fastapi(app: Any) -> None:
    """Auto-instrument a FastAPI app for HTTP tracing."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("fastapi_instrumented")
    except ImportError:
        logger.warning("fastapi_instrumentor_not_available")


@contextmanager
def trace_span(name: str, **attributes: Any) -> Generator[Any, None, None]:
    """Context manager that creates a named span with optional attributes."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            yield span
    except ImportError:
        yield None
