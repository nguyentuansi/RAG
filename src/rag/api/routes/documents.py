"""Document management API endpoints."""

from __future__ import annotations

import time
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from src.rag.api.dependencies import (
    CacheDep,
    CurrentUserDep,
    EmbeddingProviderDep,
    SettingsDep,
    VectorStoreDep,
)
from src.rag.core.exceptions import DocumentNotFoundError, DocumentTooLargeError, UnsupportedDocumentFormatError
from src.rag.core.logging import get_logger
from src.rag.domain.documents import Document, DocumentFormat, DocumentMetadata, DocumentStatus

router = APIRouter()
logger = get_logger(__name__)


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    message: str


class DocumentListResponse(BaseModel):
    items: list[dict]
    total: int
    page: int
    page_size: int


class BatchIngestRequest(BaseModel):
    texts: list[str]
    sources: list[str] | None = None
    collection_name: str | None = None


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document file to ingest")],
    background_tasks: BackgroundTasks,
    settings: SettingsDep,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    cache: CacheDep,
    current_user: CurrentUserDep,
    collection_name: str = Query(default=None),
) -> DocumentUploadResponse:
    """Upload and asynchronously ingest a document into the vector store."""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()

    if len(content) > max_bytes:
        raise DocumentTooLargeError(
            f"File exceeds {settings.max_upload_size_mb} MB limit",
            details={"size_bytes": len(content), "max_bytes": max_bytes},
        )

    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    try:
        doc_format = DocumentFormat.from_extension(ext)
    except ValueError:
        raise UnsupportedDocumentFormatError(
            f"Unsupported file type: {ext}",
            details={"filename": filename},
        )

    doc = Document(
        content=content.decode("utf-8", errors="replace"),
        format=doc_format,
        metadata=DocumentMetadata(
            source=filename,
            title=filename,
        ),
    )

    target_collection = collection_name or settings.collection_name

    background_tasks.add_task(
        _ingest_document_task,
        doc=doc,
        collection_name=target_collection,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    logger.info(
        "document_upload_accepted",
        document_id=str(doc.id),
        filename=filename,
        size_bytes=len(content),
        user=current_user.get("sub"),
    )

    return DocumentUploadResponse(
        document_id=str(doc.id),
        status="accepted",
        message=f"Document queued for ingestion into collection '{target_collection}'",
    )


@router.get("/documents/{document_id}")
async def get_document(
    document_id: UUID,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> dict:
    """Retrieve a document's metadata by ID."""
    results = await vector_store.search(
        collection_name=settings.collection_name,
        query_vector=[0.0] * 768,
        filters={"document_id": str(document_id)},
        top_k=1,
    )
    if not results:
        raise DocumentNotFoundError(
            f"Document '{document_id}' not found",
            details={"document_id": str(document_id)},
        )
    payload = results[0].payload
    return {
        "document_id": str(document_id),
        "source": payload.get("source"),
        "chunk_count": payload.get("chunk_count", 0),
        "metadata": payload,
    }


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: UUID,
    vector_store: VectorStoreDep,
    settings: SettingsDep,
    current_user: CurrentUserDep,
) -> None:
    """Delete all chunks belonging to a document."""
    # Search for all chunks of this document
    results = await vector_store.search(
        collection_name=settings.collection_name,
        query_vector=[0.0] * 768,
        filters={"document_id": str(document_id)},
        top_k=1000,
    )
    if not results:
        raise DocumentNotFoundError(f"Document '{document_id}' not found")

    chunk_ids = [r.id for r in results]
    deleted = await vector_store.delete_vectors(settings.collection_name, chunk_ids)

    logger.info(
        "document_deleted",
        document_id=str(document_id),
        chunks_deleted=deleted,
        user=current_user.get("sub"),
    )


@router.post("/documents/batch", response_model=DocumentUploadResponse, status_code=202)
async def batch_ingest(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    settings: SettingsDep,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
    current_user: CurrentUserDep,
) -> DocumentUploadResponse:
    """Ingest multiple raw text strings in one request."""
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts list cannot be empty")

    collection_name = request.collection_name or settings.collection_name

    background_tasks.add_task(
        _batch_ingest_task,
        texts=request.texts,
        sources=request.sources or [f"batch_{i}" for i in range(len(request.texts))],
        collection_name=collection_name,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    return DocumentUploadResponse(
        document_id="batch",
        status="accepted",
        message=f"{len(request.texts)} texts queued for ingestion",
    )


async def _ingest_document_task(
    doc: Document,
    collection_name: str,
    vector_store,
    embedding_provider,
) -> None:
    """Background task: chunk → embed → index."""
    from src.rag.pipeline.chunking import ChunkingPipeline
    from src.rag.pipeline.embedding import AsyncEmbeddingPipeline
    from src.rag.infrastructure.vector_store.base import VectorRecord
    from src.rag.domain.chunks import ChunkingConfig

    t0 = time.monotonic()
    try:
        doc.mark_processing()
        chunker = ChunkingPipeline(ChunkingConfig())
        chunks = chunker.chunk(doc)

        embedder = AsyncEmbeddingPipeline(embedding_provider)
        embedded = await embedder.embed_chunks(chunks)

        await vector_store.create_collection(
            collection_name,
            vector_size=embedding_provider.dimension,
        )

        records = [
            VectorRecord(
                id=ec.chunk.chunk_id,
                vector=ec.embedding,
                payload=ec.to_vector_payload(),
            )
            for ec in embedded
        ]
        await vector_store.upsert_vectors(collection_name, records)
        doc.mark_indexed(len(records))

        logger.info(
            "document_ingested",
            document_id=str(doc.id),
            chunks=len(records),
            elapsed_ms=round((time.monotonic() - t0) * 1000, 2),
        )
    except Exception as exc:
        doc.mark_failed(str(exc))
        logger.error("document_ingestion_failed", document_id=str(doc.id), error=str(exc))


async def _batch_ingest_task(
    texts: list[str],
    sources: list[str],
    collection_name: str,
    vector_store,
    embedding_provider,
) -> None:
    from src.rag.infrastructure.vector_store.base import VectorRecord
    from uuid import uuid4

    await vector_store.create_collection(collection_name, vector_size=embedding_provider.dimension)
    vectors = await embedding_provider.embed_batch(texts)
    records = [
        VectorRecord(
            id=str(uuid4()),
            vector=v,
            payload={"content": texts[i], "source": sources[i], "chunk_index": i},
        )
        for i, v in enumerate(vectors)
    ]
    await vector_store.upsert_vectors(collection_name, records)
    logger.info("batch_ingested", count=len(records), collection=collection_name)
