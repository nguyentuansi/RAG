"""Text chunking strategies: recursive, semantic, sliding window, markdown-header."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from uuid import uuid4

from src.rag.core.logging import get_logger
from src.rag.domain.chunks import ChunkingConfig, ChunkingStrategy, ChunkMetadata, TextChunk
from src.rag.domain.documents import Document

logger = get_logger(__name__)


class BaseChunker(ABC):
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    @abstractmethod
    def split(self, text: str, document_id: str) -> list[TextChunk]:
        ...

    def _make_chunk(
        self,
        content: str,
        index: int,
        start: int,
        end: int,
        document_id: str,
        metadata: ChunkMetadata | None = None,
    ) -> TextChunk:
        return TextChunk(
            chunk_id=str(uuid4()),
            document_id=document_id,  # type: ignore[arg-type]
            content=content,
            chunk_index=index,
            start_char=start,
            end_char=end,
            chunk_metadata=metadata or ChunkMetadata(),
            token_estimate=len(content.split()),
        )


class RecursiveCharacterChunker(BaseChunker):
    """
    LangChain-style recursive character splitter.

    Tries each separator in order, preferring the one that keeps chunks
    within config.chunk_size while maintaining context with overlap.
    """

    def split(self, text: str, document_id: str) -> list[TextChunk]:
        raw_chunks = self._recursive_split(text, self.config.separators)
        chunks: list[TextChunk] = []
        pos = 0

        for i, chunk_text in enumerate(raw_chunks):
            start = text.find(chunk_text, pos)
            if start == -1:
                start = pos
            end = start + len(chunk_text)
            pos = max(pos, start + 1)

            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(self._make_chunk(chunk_text, len(chunks), start, end, document_id))

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._split_by_size(text)

        sep = separators[0]
        if sep:
            splits = text.split(sep)
        else:
            splits = list(text)

        good: list[str] = []
        pending = ""

        for fragment in splits:
            candidate = (pending + sep + fragment).strip() if pending else fragment.strip()
            if len(candidate) <= self.config.chunk_size:
                pending = candidate
            else:
                if pending:
                    good.append(pending)
                    overlap_words = pending.split()[-self.config.chunk_overlap // 5 :]
                    pending = " ".join(overlap_words) + " " + fragment.strip()
                else:
                    sub = self._recursive_split(fragment, separators[1:])
                    good.extend(sub)
                    pending = ""

        if pending:
            good.append(pending)

        return [g for g in good if g.strip()]

    def _split_by_size(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
            chunks.append(text[i : i + self.config.chunk_size])
        return chunks


class SemanticChunker(BaseChunker):
    """
    Sentence-boundary aware chunker.

    Groups sentences into chunks that don't exceed chunk_size while
    respecting paragraph and section boundaries.
    """

    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*\n")

    def split(self, text: str, document_id: str) -> list[TextChunk]:
        sentences = self._SENTENCE_END.split(text)
        chunks: list[TextChunk] = []
        current: list[str] = []
        current_len = 0
        pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            s_len = len(sentence)
            if current_len + s_len > self.config.chunk_size and current:
                content = " ".join(current)
                start = text.find(current[0], pos)
                end = start + len(content)
                chunks.append(self._make_chunk(content, len(chunks), start, end, document_id))
                # carry overlap sentences
                overlap = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.config.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap
                current_len = overlap_len
                pos = max(pos, start + 1)

            current.append(sentence)
            current_len += s_len

        if current:
            content = " ".join(current)
            start = text.find(current[0], pos)
            end = start + len(content)
            chunks.append(self._make_chunk(content, len(chunks), start, end, document_id))

        return chunks


class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window with configurable overlap."""

    def split(self, text: str, document_id: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        step = self.config.chunk_size - self.config.chunk_overlap

        for i, start in enumerate(range(0, len(text), step)):
            end = min(start + self.config.chunk_size, len(text))
            content = text[start:end].strip()
            if len(content) >= self.config.min_chunk_size:
                chunks.append(self._make_chunk(content, i, start, end, document_id))

        return chunks


class MarkdownHeaderChunker(BaseChunker):
    """
    Splits Markdown by headers, keeping each section as a chunk.
    Falls back to recursive splitting for sections that exceed chunk_size.
    """

    _HEADER = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def split(self, text: str, document_id: str) -> list[TextChunk]:
        header_positions = [(m.start(), m.end(), m.group(2)) for m in self._HEADER.finditer(text)]
        if not header_positions:
            return RecursiveCharacterChunker(self.config).split(text, document_id)

        sections: list[tuple[int, int, str]] = []
        for i, (start, end, heading) in enumerate(header_positions):
            section_end = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(text)
            sections.append((start, section_end, heading))

        chunks: list[TextChunk] = []
        fallback = RecursiveCharacterChunker(self.config)

        for section_start, section_end, heading in sections:
            content = text[section_start:section_end].strip()
            if not content:
                continue
            if len(content) <= self.config.chunk_size:
                chunks.append(
                    self._make_chunk(
                        content,
                        len(chunks),
                        section_start,
                        section_end,
                        document_id,
                        metadata=ChunkMetadata(heading=heading),
                    )
                )
            else:
                sub = fallback.split(content, document_id)
                for sc in sub:
                    chunks.append(
                        TextChunk(
                            chunk_id=sc.chunk_id,
                            document_id=sc.document_id,
                            content=sc.content,
                            chunk_index=len(chunks),
                            start_char=section_start + sc.start_char,
                            end_char=section_start + sc.end_char,
                            chunk_metadata=ChunkMetadata(heading=heading),
                            token_estimate=sc.token_estimate,
                        )
                    )

        return chunks


class ChunkingPipeline:
    """Entry point: selects and applies the appropriate chunking strategy."""

    _CHUNKERS: dict[ChunkingStrategy, type[BaseChunker]] = {
        ChunkingStrategy.RECURSIVE: RecursiveCharacterChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
        ChunkingStrategy.MARKDOWN_HEADER: MarkdownHeaderChunker,
    }

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        chunker_cls = self._CHUNKERS.get(config.strategy, RecursiveCharacterChunker)
        self._chunker = chunker_cls(config)

    def chunk_document(self, document: Document) -> list[TextChunk]:
        chunks = self._chunker.split(document.content, str(document.id))
        logger.info(
            "document_chunked",
            doc_id=str(document.id),
            strategy=self.config.strategy,
            chunks=len(chunks),
        )
        return chunks
