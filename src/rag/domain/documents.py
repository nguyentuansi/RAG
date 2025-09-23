"""Document domain models."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"
    HTML = "html"
    JSON = "json"
    CSV = "csv"

    @classmethod
    def from_extension(cls, ext: str) -> DocumentFormat:
        mapping = {
            ".pdf": cls.PDF,
            ".docx": cls.DOCX,
            ".doc": cls.DOCX,
            ".txt": cls.TXT,
            ".md": cls.MARKDOWN,
            ".markdown": cls.MARKDOWN,
            ".html": cls.HTML,
            ".htm": cls.HTML,
            ".json": cls.JSON,
            ".csv": cls.CSV,
        }
        normalized = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        if normalized not in mapping:
            raise ValueError(f"Unsupported file extension: {ext}")
        return mapping[normalized]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str = Field(description="Origin path or URL of the document")
    author: str | None = None
    title: str | None = None
    created_at: datetime | None = None
    language: str = Field(default="en")
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1)
    format: DocumentFormat
    metadata: DocumentMetadata
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()

    @computed_field  # type: ignore[misc]
    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def mark_processing(self) -> None:
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.now(timezone.utc)

    def mark_indexed(self, chunk_count: int) -> None:
        self.status = DocumentStatus.INDEXED
        self.chunk_count = chunk_count
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str) -> None:
        self.status = DocumentStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.now(timezone.utc)


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_char_range(self) -> DocumentChunk:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class ProcessedDocument(BaseModel):
    """Output of the full ingestion pipeline for one document."""

    document: Document
    chunks: list[DocumentChunk]
    processing_time_ms: float
    embedding_model: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
