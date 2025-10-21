"""Embedding provider implementations."""

from .base import EmbeddingProvider
from .sentence_transformer import SentenceTransformerProvider

__all__ = ["EmbeddingProvider", "SentenceTransformerProvider"]
