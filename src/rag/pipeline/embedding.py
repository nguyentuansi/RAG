"""Async embedding pipeline with batching and retry logic."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from src.rag.core.exceptions import EmbeddingError
from src.rag.core.logging import get_logger
from src.rag.domain.chunks import EmbeddedChunk, TextChunk
from src.rag.infrastructure.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


class AsyncEmbeddingPipeline:
    """Embeds TextChunks in concurrent batches with exponential-backoff retry."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.provider = provider
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.progress_callback = progress_callback

    async def embed_chunks(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        embeddings = await self._embed_with_retry(texts)

        # Validate dimension consistency
        if embeddings and len(embeddings[0]) != self.provider.dimension:
            raise EmbeddingError(
                f"Dimension mismatch: expected {self.provider.dimension}, got {len(embeddings[0])}"
            )

        result = [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                embedding_model=self.provider.model_name,
                embedding_dimension=len(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        logger.info("chunks_embedded", count=len(result), model=self.provider.model_name)
        return result

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_embeddings: list[list[float]] = []

        for batch_idx, batch in enumerate(batches):
            embeddings = await self._retry_batch(batch, batch_idx)
            all_embeddings.extend(embeddings)
            if self.progress_callback:
                self.progress_callback(min((batch_idx + 1) * self.batch_size, len(texts)), len(texts))

        return all_embeddings

    async def _retry_batch(self, batch: list[str], batch_idx: int) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self.provider.embed_batch(batch)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "embedding_batch_retry",
                        batch=batch_idx,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
        raise EmbeddingError(
            f"Embedding batch {batch_idx} failed after {self.max_retries} attempts",
            cause=last_error,
        )
