"""Abstract vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionInfo:
    name: str
    vector_size: int
    distance_metric: str
    vector_count: int
    indexed_vector_count: int
    status: str


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        vector_size: int,
        *,
        distance: str = "cosine",
        on_disk: bool = False,
    ) -> None:
        """Create a new vector collection. No-op if already exists."""

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its vectors."""

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Return True if the collection exists."""

    @abstractmethod
    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Return metadata about a collection."""

    @abstractmethod
    async def upsert_vectors(
        self,
        collection_name: str,
        records: list[VectorRecord],
        *,
        batch_size: int = 100,
    ) -> int:
        """Insert or update vectors. Returns count of upserted records."""

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Perform nearest-neighbour search."""

    @abstractmethod
    async def delete_vectors(
        self,
        collection_name: str,
        ids: list[str],
    ) -> int:
        """Delete vectors by ID. Returns count deleted."""

    @abstractmethod
    async def get_vector(
        self,
        collection_name: str,
        vector_id: str,
    ) -> VectorRecord | None:
        """Retrieve a single vector by ID."""

    @abstractmethod
    async def count(self, collection_name: str) -> int:
        """Return number of vectors in the collection."""

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources / connections."""
