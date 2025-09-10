"""Structured JSON logging with correlation ID support."""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_request_path: ContextVar[str] = ContextVar("request_path", default="")
_user_id: ContextVar[str] = ContextVar("user_id", default="")


class RequestContext:
    """Thread-local request context for log enrichment."""

    @staticmethod
    def set(correlation_id: str, path: str = "", user_id: str = "") -> None:
        _correlation_id.set(correlation_id)
        _request_path.set(path)
        _user_id.set(user_id)

    @staticmethod
    def get_correlation_id() -> str:
        return _correlation_id.get() or str(uuid.uuid4())

    @staticmethod
    def new_correlation_id() -> str:
        cid = str(uuid.uuid4())
        _correlation_id.set(cid)
        return cid

    @staticmethod
    def clear() -> None:
        _correlation_id.set("")
        _request_path.set("")
        _user_id.set("")


def _add_correlation_id(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    event_dict["correlation_id"] = _correlation_id.get() or "-"
    if path := _request_path.get():
        event_dict["path"] = path
    if uid := _user_id.get():
        event_dict["user_id"] = uid
    return event_dict


def _add_service_info(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    event_dict["service"] = "rag-api"
    return event_dict


def configure_logging(log_level: str = "INFO", *, json_logs: bool = True) -> None:
    """Configure structlog for the application."""
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_correlation_id,
        _add_service_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger."""
    return structlog.get_logger(name)
