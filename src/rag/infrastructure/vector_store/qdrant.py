"""Qdrant vector store implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.rag.core.exceptions import CollectionNotFoundError, VectorStoreConnectionError, VectorStoreError
from src.rag.core.logging import get_logger

from .base import CollectionInfo, VectorRecord, VectorSearchResult, VectorStore

logger = get_logger(__name__)

_DISTANCE_MAP: dict[str, qmodels.Distance] = {
    "cosine": qmodels.Distance.COSINE,
    "euclid": qmodels.Distance.EUCLID,
    "dot": qmodels.Distance.DOT,
    "manhattan": qmodels.Distance.MANHATTAN,
}


class QdrantVectorStore(VectorStore):
    """Async Qdrant vector store with connection pooling and batch operations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        api_key: str | None = None,
        prefer_grpc: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self._client: AsyncQdrantClient | None = None
        self._host = host
        self._port = port
        self._grpc_port = grpc_port
        self._api_key = api_key
        self._prefer_grpc = prefer_grpc
        self._timeout = timeout
        self._lock = asyncio.Lock()

    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    try:
                        self._client = AsyncQdrantClient(
                            host=self._host,
                            port=self._port,
                            grpc_port=self._grpc_port,
                            api_key=self._api_key,
                            prefer_grpc=self._prefer_grpc,
                            timeout=self._timeout,
                        )
                        logger.info(
                            "qdrant_connected",
                            host=self._host,
                            port=self._port,
                        )
                    except Exception as exc:
                        raise VectorStoreConnectionError(
                            f"Cannot connect to Qdrant at {self._host}:{self._port}",
                            cause=exc,
                        ) from exc
        return self._client

    async def create_collection(
        self,
        name: str,
        vector_size: int,
        *,
        distance: str = "cosine",
        on_disk: bool = False,
    ) -> None:
        client = await self._get_client()
        qdrant_distance = _DISTANCE_MAP.get(distance.lower(), qmodels.Distance.COSINE)

        try:
            exists = await self.collection_exists(name)
            if exists:
                logger.debug("collection_already_exists", collection=name)
                return

            await client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qdrant_distance,
                    on_disk=on_disk,
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=200),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=20000,
                ),
            )
            logger.info("collection_created", collection=name, vector_size=vector_size)
        except Exception as exc:
            raise VectorStoreError(f"Failed to create collection '{name}'", cause=exc) from exc

    async def delete_collection(self, name: str) -> None:
        client = await self._get_client()
        try:
            await client.delete_collection(collection_name=name)
            logger.info("collection_deleted", collection=name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete collection '{name}'", cause=exc) from exc

    async def collection_exists(self, name: str) -> bool:
        client = await self._get_client()
        try:
            collections = await client.get_collections()
            return any(c.name == name for c in collections.collections)
        except Exception as exc:
            raise VectorStoreError("Failed to list collections", cause=exc) from exc

    async def get_collection_info(self, name: str) -> CollectionInfo:
        client = await self._get_client()
        try:
            info = await client.get_collection(collection_name=name)
            return CollectionInfo(
                name=name,
                vector_size=info.config.params.vectors.size,  # type: ignore[union-attr]
                distance_metric=str(info.config.params.vectors.distance),  # type: ignore[union-attr]
                vector_count=info.vectors_count or 0,
                indexed_vector_count=info.indexed_vectors_count or 0,
                status=str(info.status),
            )
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                raise CollectionNotFoundError(f"Collection '{name}' not found") from exc
            raise VectorStoreError(f"Failed to get collection info for '{name}'", cause=exc) from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to get collection info for '{name}'", cause=exc) from exc

    async def upsert_vectors(
        self,
        collection_name: str,
        records: list[VectorRecord],
        *,
        batch_size: int = 100,
    ) -> int:
        if not records:
            return 0

        client = await self._get_client()
        total_upserted = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            points = [
                qmodels.PointStruct(id=r.id, vector=r.vector, payload=r.payload)
                for r in batch
            ]
            try:
                await client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                total_upserted += len(batch)
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to upsert batch {i // batch_size} into '{collection_name}'",
                    cause=exc,
                ) from exc

        logger.info(
            "vectors_upserted",
            collection=collection_name,
            count=total_upserted,
        )
        return total_upserted

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        client = await self._get_client()

        qdrant_filter: qmodels.Filter | None = None
        if filters:
            must_conditions = [
                qmodels.FieldCondition(
                    key=k,
                    match=qmodels.MatchValue(value=v),
                )
                for k, v in filters.items()
            ]
            qdrant_filter = qmodels.Filter(must=must_conditions)

        try:
            results = await client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return [
                VectorSearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results
            ]
        except Exception as exc:
            raise VectorStoreError(
                f"Search failed on collection '{collection_name}'", cause=exc
            ) from exc

    async def delete_vectors(self, collection_name: str, ids: list[str]) -> int:
        if not ids:
            return 0
        client = await self._get_client()
        try:
            await client.delete(
                collection_name=collection_name,
                points_selector=qmodels.PointIdsList(points=ids),  # type: ignore[arg-type]
                wait=True,
            )
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete vectors from '{collection_name}'", cause=exc
            ) from exc

    async def get_vector(self, collection_name: str, vector_id: str) -> VectorRecord | None:
        client = await self._get_client()
        try:
            results = await client.retrieve(
                collection_name=collection_name,
                ids=[vector_id],
                with_vectors=True,
                with_payload=True,
            )
            if not results:
                return None
            r = results[0]
            return VectorRecord(
                id=str(r.id),
                vector=r.vector or [],  # type: ignore[arg-type]
                payload=r.payload or {},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to get vector '{vector_id}' from '{collection_name}'", cause=exc
            ) from exc

    async def count(self, collection_name: str) -> int:
        client = await self._get_client()
        try:
            result = await client.count(collection_name=collection_name, exact=True)
            return result.count
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to count vectors in '{collection_name}'", cause=exc
            ) from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("qdrant_connection_closed")
