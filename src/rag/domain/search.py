"""Search request and response domain models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    MMR = "mmr"


class SearchFilter(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ids: list[UUID] | None = None
    tags: list[str] | None = None
    language: str | None = None
    source_pattern: str | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=2048)
    collection_name: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    filters: SearchFilter | None = None
    include_content: bool = True
    include_metadata: bool = True
    hybrid_config: HybridSearchConfig | None = None


class HybridSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, description="RRF constant for score fusion")

    @property
    def sparse_weight_computed(self) -> float:
        return 1.0 - self.dense_weight


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    highlights: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
    retrieval_strategy: RetrievalStrategy
    latency_ms: float
    collection_name: str

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    @property
    def top_result(self) -> SearchResult | None:
        return self.results[0] if self.results else None
