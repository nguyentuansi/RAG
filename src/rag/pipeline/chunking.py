"""Advanced text chunking strategies."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable
from uuid import uuid4

from src.rag.core.logging import get_logger
from src.rag.domain.chunks import ChunkingConfig, ChunkingStrategy, ChunkMetadata, TextChunk
from src.rag.domain.documents import Document

logger = get_logger(__name__)


class BaseChunker(ABC):
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    @abstractmethod
    def chunk(self, text: str, document_id) -> list[TextChunk]:
        ...

    def _make_chunk(self, text: str, doc_id, index: int, start: int, end: int, **meta) -> TextChunk:
        return TextChunk(
            chunk_id=str(uuid4()),
            document_id=doc_id,
            content=text,
            chunk_index=index,
            start_char=start,
            end_char=end,
            token_estimate=len(text.split()),
            chunk_metadata=ChunkMetadata(**meta),
        )


class RecursiveCharacterChunker(BaseChunker):
    """Split on separators from coarsest to finest until chunks are small enough."""

    def chunk(self, text: str, document_id) -> list[TextChunk]:
        raw_chunks = self._split(text, self.config.separators)
        results, char_offset = [], 0
        for i, chunk_text in enumerate(raw_chunks):
            start = text.find(chunk_text, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(chunk_text)
            char_offset = max(char_offset, end - self.config.chunk_overlap)
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                results.append(self._make_chunk(chunk_text.strip(), document_id, i, start, end))
        return results

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._fixed_split(text)
        sep = separators[0]
        if not sep:
            return self._fixed_split(text)
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part
            if len(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.config.chunk_size:
                    chunks.extend(self._split(part, separators[1:]))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    def _fixed_split(self, text: str) -> list[str]:
        size, overlap = self.config.chunk_size, self.config.chunk_overlap
        return [text[i : i + size] for i in range(0, len(text), size - overlap) if text[i : i + size].strip()]


class SemanticChunker(BaseChunker):
    """Chunk at sentence boundaries, grouping sentences until size limit."""

    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\.)\n+(?=[A-Z\d\-•])")

    def chunk(self, text: str, document_id) -> list[TextChunk]:
        sentences = [s.strip() for s in self._SENTENCE_END.split(text) if s.strip()]
        results: list[TextChunk] = []
        current: list[str] = []
        current_len = 0
        char_offset = 0
        chunk_idx = 0

        for sentence in sentences:
            if current_len + len(sentence) > self.config.chunk_size and current:
                body = " ".join(current)
                start = text.find(body, char_offset)
                end = start + len(body)
                results.append(self._make_chunk(body, document_id, chunk_idx, start, end))
                chunk_idx += 1
                # Overlap: keep last sentence(s) whose total fits overlap budget
                overlap_buf: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.config.chunk_overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap_buf
                current_len = overlap_len
                char_offset = max(0, end - self.config.chunk_overlap)
            current.append(sentence)
            current_len += len(sentence)

        if current:
            body = " ".join(current)
            start = text.find(body, char_offset)
            end = start + len(body) if start != -1 else char_offset + len(body)
            results.append(self._make_chunk(body, document_id, chunk_idx, max(0, start), end))
        return results


class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window with configurable stride."""

    def chunk(self, text: str, document_id) -> list[TextChunk]:
        size = self.config.chunk_size
        stride = size - self.config.chunk_overlap
        results = []
        i = 0
        idx = 0
        while i < len(text):
            window = text[i : i + size]
            if len(window.strip()) >= self.config.min_chunk_size:
                results.append(self._make_chunk(window.strip(), document_id, idx, i, i + len(window)))
                idx += 1
            i += stride
        return results


class MarkdownHeaderChunker(BaseChunker):
    """Split markdown at heading boundaries and sub-chunk oversized sections."""

    _HEADER = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def chunk(self, text: str, document_id) -> list[TextChunk]:
        boundaries = [(m.start(), m.group(1), m.group(2)) for m in self._HEADER.finditer(text)]
        if not boundaries:
            return RecursiveCharacterChunker(self.config).chunk(text, document_id)

        sections: list[tuple[str, str, str]] = []
        for i, (start, level, heading) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            sections.append((heading, level, text[start:end]))

        results: list[TextChunk] = []
        char_offset = 0
        for idx, (heading, level, body) in enumerate(sections):
            if len(body) <= self.config.chunk_size:
                start = text.find(body, char_offset)
                end = start + len(body)
                results.append(
                    self._make_chunk(body.strip(), document_id, idx, start, end, heading=heading, section=level)
                )
                char_offset = end
            else:
                sub = RecursiveCharacterChunker(self.config).chunk(body, document_id)
                results.extend(sub)
        return results


class ChunkingStrategyFactory:
    _registry: dict[ChunkingStrategy, type[BaseChunker]] = {
        ChunkingStrategy.RECURSIVE: RecursiveCharacterChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
        ChunkingStrategy.MARKDOWN_HEADER: MarkdownHeaderChunker,
    }

    @classmethod
    def create(cls, config: ChunkingConfig) -> BaseChunker:
        klass = cls._registry.get(config.strategy, RecursiveCharacterChunker)
        return klass(config)


class ChunkingPipeline:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self._chunker = ChunkingStrategyFactory.create(config)

    def chunk_document(self, document: Document) -> list[TextChunk]:
        chunks = self._chunker.chunk(document.content, document.id)
        logger.info(
            "document_chunked",
            doc_id=str(document.id),
            strategy=self.config.strategy,
            chunks=len(chunks),
        )
        return chunks
