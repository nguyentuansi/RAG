"""Custom exception hierarchy for the RAG platform."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any


class RAGException(Exception):
    """Base exception for all RAG platform errors."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code: str = "RAG_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# --- Document errors ---


class DocumentNotFoundError(RAGException):
    status_code = HTTPStatus.NOT_FOUND
    error_code = "DOCUMENT_NOT_FOUND"


class DocumentAlreadyExistsError(RAGException):
    status_code = HTTPStatus.CONFLICT
    error_code = "DOCUMENT_ALREADY_EXISTS"


class UnsupportedDocumentFormatError(RAGException):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    error_code = "UNSUPPORTED_FORMAT"


class DocumentTooLargeError(RAGException):
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    error_code = "DOCUMENT_TOO_LARGE"


# --- Pipeline errors ---


class ChunkingError(RAGException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "CHUNKING_ERROR"


class EmbeddingError(RAGException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "EMBEDDING_ERROR"


class ModelNotLoadedError(EmbeddingError):
    error_code = "MODEL_NOT_LOADED"


# --- Search errors ---


class SearchError(RAGException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "SEARCH_ERROR"


class CollectionNotFoundError(SearchError):
    status_code = HTTPStatus.NOT_FOUND
    error_code = "COLLECTION_NOT_FOUND"


# --- Storage errors ---


class VectorStoreError(RAGException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "VECTOR_STORE_ERROR"


class VectorStoreConnectionError(VectorStoreError):
    error_code = "VECTOR_STORE_CONNECTION_ERROR"


# --- Auth errors ---


class AuthenticationError(RAGException):
    status_code = HTTPStatus.UNAUTHORIZED
    error_code = "AUTHENTICATION_FAILED"


class TokenExpiredError(AuthenticationError):
    error_code = "TOKEN_EXPIRED"


class InvalidTokenError(AuthenticationError):
    error_code = "INVALID_TOKEN"


class AuthorizationError(RAGException):
    status_code = HTTPStatus.FORBIDDEN
    error_code = "AUTHORIZATION_FAILED"


# --- Rate limiting ---


class RateLimitError(RAGException):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    error_code = "RATE_LIMIT_EXCEEDED"


# --- Validation ---


class ValidationError(RAGException):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    error_code = "VALIDATION_ERROR"


# --- Security ---


class PromptInjectionError(RAGException):
    status_code = HTTPStatus.BAD_REQUEST
    error_code = "PROMPT_INJECTION_DETECTED"
