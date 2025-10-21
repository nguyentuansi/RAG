"""Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for text embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced embedding vectors."""

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Maximum tokens the model can process per input."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text. Returns normalised float vector."""

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed a list of texts in optimised batches."""

    @abstractmethod
    async def is_ready(self) -> bool:
        """Return True if the model is loaded and ready to serve requests."""

    @abstractmethod
    async def warm_up(self) -> None:
        """Pre-load the model into memory / GPU. Called at startup."""

    @abstractmethod
    async def close(self) -> None:
        """Release model resources."""
