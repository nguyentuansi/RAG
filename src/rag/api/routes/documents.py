"""Document management API endpoints."""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from src.rag.api.dependencies import CurrentUserDep, EmbeddingProviderDep, SettingsDep, VectorStoreDep
from src.rag.core.exceptions import DocumentNotFoundError, DocumentTooLargeError, UnsupportedDocumentFormatError
from src.rag.core.logging import get_logger
from src.rag.domain.documents import Document, DocumentFormat, DocumentMetadata, DocumentStatus

router = APIRouter()
logger = get_logger(__name__)


class DocumentResponse(BaseModel):
    id: str
    status: DocumentStatus
    format: DocumentFormat
    chunk_count: int
    word_count: int
    metadata: dict
    created_at: str


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class DeleteResponse(BaseModel):
    deleted: bool
    document_id: str
    chunks_removed: int


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and index a document",
)
async def upload_document(
    file: UploadFile,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
    title: str | None = Query(default=None),
    tags: str | None = Query(default=None, description="Comma-separated tags"),
) -> DocumentResponse:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content_bytes = await file.read()

    if len(content_bytes) > max_bytes:
        raise DocumentTooLargeError(
            f"File exceeds {settings.max_upload_size_mb} MB limit",
            details={"size_bytes": len(content_bytes), "limit_bytes": max_bytes},
        )

    filename = file.filename or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".txt"
    try:
        fmt = DocumentFormat.from_extension(ext)
    except ValueError:
        raise UnsupportedDocumentFormatError(f"Unsupported file type: {ext}")

    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    metadata = DocumentMetadata(
        source=filename,
        title=title or filename,
        tags=tag_list,
        custom={"uploaded_by": current_user.get("sub", "unknown")},
    )

    doc = Document(
        content=content_bytes.decode("utf-8", errors="replace"),
        format=fmt,
        metadata=metadata,
    )

    # Kick off pipeline asynchronously (fire-and-forget in background task)
    import asyncio
    asyncio.create_task(_index_document(doc, vector_store, embedding_provider, settings))

    logger.info("document_upload_accepted", doc_id=str(doc.id), filename=filename)

    return DocumentResponse(
        id=str(doc.id),
        status=doc.status,
        format=doc.format,
        chunk_count=doc.chunk_count,
        word_count=doc.word_count,
        metadata=doc.metadata.model_dump(),
        created_at=doc.created_at.isoformat(),
    )


async def _index_document(doc, vector_store, embedding_provider, settings) -> None:
    from src.rag.pipeline.chunking import ChunkingPipeline
    from src.rag.pipeline.embedding import AsyncEmbeddingPipeline
    from src.rag.domain.chunks import ChunkingConfig, ChunkingStrategy
    from src.rag.infrastructure.vector_store.base import VectorRecord

    try:
        doc.mark_processing()

        chunker = ChunkingPipeline(ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
        ))
        chunks = chunker.chunk_document(doc)

        embedder = AsyncEmbeddingPipeline(embedding_provider)
        embedded = await embedder.embed_chunks(chunks)

        records = [
            VectorRecord(
                id=ec.chunk.chunk_id,
                vector=ec.embedding,
                payload=ec.to_vector_payload(),
            )
            for ec in embedded
        ]

        await vector_store.upsert_vectors(settings.collection_name, records)
        doc.mark_indexed(chunk_count=len(chunks))
        logger.info("document_indexed", doc_id=str(doc.id), chunks=len(chunks))
    except Exception as exc:
        doc.mark_failed(str(exc))
        logger.error("document_indexing_failed", doc_id=str(doc.id), error=str(exc))


@router.get("/{document_id}", summary="Get document by ID")
async def get_document(
    document_id: str,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> dict:
    # Retrieve a sample vector to confirm the doc exists
    sample = await vector_store.search(
        settings.collection_name,
        query_vector=[0.0] * 768,
        top_k=1,
        filters={"document_id": document_id},
    )
    if not sample:
        raise DocumentNotFoundError(f"Document '{document_id}' not found")

    payload = sample[0].payload
    return {
        "document_id": document_id,
        "source": payload.get("source"),
        "chunk_count": payload.get("chunk_index", 0) + 1,
        "metadata": {k: v for k, v in payload.items() if k not in {"content", "embedding"}},
    }


@router.delete("/{document_id}", summary="Delete document and its vectors")
async def delete_document(
    document_id: str,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> DeleteResponse:
    results = await vector_store.search(
        settings.collection_name,
        query_vector=[0.0] * 768,
        top_k=1000,
        filters={"document_id": document_id},
    )
    if not results:
        raise DocumentNotFoundError(f"Document '{document_id}' not found")

    chunk_ids = [r.id for r in results]
    deleted = await vector_store.delete_vectors(settings.collection_name, chunk_ids)
    logger.info("document_deleted", doc_id=document_id, chunks_removed=deleted)

    return DeleteResponse(deleted=True, document_id=document_id, chunks_removed=deleted)


@router.get("", summary="List documents (paginated)")
async def list_documents(
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    info = await vector_store.get_collection_info(settings.collection_name)
    return {
        "total_vectors": info.vector_count,
        "collection": settings.collection_name,
        "page": page,
        "page_size": page_size,
        "note": "Use /search to query documents by content",
    }
