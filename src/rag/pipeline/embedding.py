"""Async embedding pipeline with batching, retries, and progress callbacks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from src.rag.core.exceptions import EmbeddingError
from src.rag.core.logging import get_logger
from src.rag.domain.chunks import EmbeddedChunk, TextChunk
from src.rag.infrastructure.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


class AsyncEmbeddingPipeline:
    """
    Batch-processes TextChunks into EmbeddedChunks.

    Handles retries with exponential back-off and reports progress
    via an optional callback.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._provider = provider
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_delay = retry_base_delay

    async def embed_chunks(
        self,
        chunks: list[TextChunk],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        if not await self._provider.is_ready():
            await self._provider.warm_up()

        embedded: list[EmbeddedChunk] = []
        total = len(chunks)

        for batch_start in range(0, total, self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            texts = [c.content for c in batch]

            vectors = await self._embed_with_retry(texts)

            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"Embedding provider returned {len(vectors)} vectors for {len(batch)} inputs"
                )

            for chunk, vector in zip(batch, vectors):
                embedded.append(
                    EmbeddedChunk(
                        chunk=chunk,
                        embedding=vector,
                        embedding_model=self._provider.model_name,
                        embedding_dimension=len(vector),
                    )
                )

            if on_progress:
                on_progress(min(batch_start + self._batch_size, total), total)

        logger.info(
            "embedding_pipeline_complete",
            total_chunks=total,
            model=self._provider.model_name,
            dimension=self._provider.dimension,
        )
        return embedded

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._provider.embed_batch(texts)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "embedding_retry",
                        attempt=attempt + 1,
                        delay_s=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

        raise EmbeddingError(
            f"Embedding failed after {self._max_retries} attempts",
            cause=last_exc,
        )
