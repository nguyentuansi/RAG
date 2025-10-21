"""SentenceTransformer embedding provider."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.rag.core.exceptions import EmbeddingError, ModelNotLoadedError
from src.rag.core.logging import get_logger

from .base import EmbeddingProvider

logger = get_logger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Async wrapper around sentence-transformers.

    Model loading is deferred until first use (or explicit warm_up call)
    to avoid blocking startup.  All CPU-bound encoding is pushed to a
    private thread pool so the event loop stays free.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        max_workers: int = 2,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize_embeddings
        self._model: Any = None
        self._dimension: int | None = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="st-embed")
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise ModelNotLoadedError(
                "Model not loaded yet; call warm_up() first or embed a text."
            )
        return self._dimension

    @property
    def max_sequence_length(self) -> int:
        if self._model is None:
            return 512
        return getattr(self._model, "max_seq_length", 512)

    async def warm_up(self) -> None:
        await self._ensure_loaded()
        logger.info("embedding_model_warmed_up", model=self._model_name, device=self._device)

    async def is_ready(self) -> bool:
        return self._model is not None

    async def embed_text(self, text: str) -> list[float]:
        model = await self._ensure_loaded()
        loop = asyncio.get_running_loop()
        try:
            vector: np.ndarray = await loop.run_in_executor(
                self._executor,
                lambda: model.encode(
                    [text],
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )[0],
            )
            return vector.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text", cause=exc) from exc

    async def embed_batch(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> list[list[float]]:
        if not texts:
            return []

        model = await self._ensure_loaded()
        loop = asyncio.get_running_loop()

        try:
            vectors: np.ndarray = await loop.run_in_executor(
                self._executor,
                lambda: model.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                ),
            )
            return vectors.tolist()
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed batch of {len(texts)} texts", cause=exc
            ) from exc

    async def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        async with self._lock:
            if self._model is not None:
                return self._model

            loop = asyncio.get_running_loop()
            logger.info("loading_embedding_model", model=self._model_name, device=self._device)

            try:
                from sentence_transformers import SentenceTransformer

                model = await loop.run_in_executor(
                    self._executor,
                    lambda: SentenceTransformer(self._model_name, device=self._device),
                )
                self._model = model
                probe: np.ndarray = await loop.run_in_executor(
                    self._executor,
                    lambda: model.encode(["probe"], normalize_embeddings=True)[0],
                )
                self._dimension = len(probe)
                logger.info(
                    "embedding_model_loaded",
                    model=self._model_name,
                    dimension=self._dimension,
                )
                return self._model
            except ImportError as exc:
                raise ModelNotLoadedError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers",
                    cause=exc,
                ) from exc
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load model '{self._model_name}'", cause=exc
                ) from exc

    async def close(self) -> None:
        self._model = None
        self._executor.shutdown(wait=False)
        logger.info("embedding_provider_closed")
