"""Chunk domain models and chunking configuration."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    MARKDOWN_HEADER = "markdown_header"
    SENTENCE = "sentence"


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=100, ge=0)
    min_chunk_size: int = Field(default=64, ge=1)
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    preserve_sentences: bool = True
    include_metadata: bool = True


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    section: str | None = None
    heading: str | None = None
    page_number: int | None = None
    paragraph_index: int | None = None
    is_table: bool = False
    is_code: bool = False
    language: str | None = None


class TextChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: UUID
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    token_estimate: int = Field(default=0)

    @property
    def char_length(self) -> int:
        return self.end_char - self.start_char


class EmbeddedChunk(BaseModel):
    """A TextChunk that has been through the embedding model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk: TextChunk
    embedding: list[float]
    embedding_model: str
    embedding_dimension: int

    @property
    def embedding_array(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)

    def to_vector_payload(self) -> dict[str, Any]:
        """Payload format for vector store upsert."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": str(self.chunk.document_id),
            "content": self.chunk.content,
            "chunk_index": self.chunk.chunk_index,
            "start_char": self.chunk.start_char,
            "end_char": self.chunk.end_char,
            "token_estimate": self.chunk.token_estimate,
            **self.chunk.chunk_metadata.model_dump(exclude_none=True),
        }
