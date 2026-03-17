"""Orchestrator and specialized agents."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.rag.agents.base import AgentCapability, AgentMessage, BaseAgent, MessageBus, MessageType
from src.rag.core.logging import get_logger

logger = get_logger(__name__)


class SearchAgent(BaseAgent):
    """Handles vector search and retrieval tasks."""

    def __init__(self, agent_id: str, bus: MessageBus, vector_store, embedding_provider) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        super().__init__(agent_id, bus)

    def _setup(self) -> None:
        self._capabilities = [
            AgentCapability(
                name="search",
                description="Execute semantic search over the vector store",
                input_schema={"query": "str", "top_k": "int", "collection": "str"},
                output_schema={"results": "list[SearchResult]", "latency_ms": "float"},
            )
        ]
        self._bus.subscribe("search.request", self.handle)

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        query = message.payload.get("query", "")
        top_k = message.payload.get("top_k", 5)
        collection = message.payload.get("collection", "rag_documents")

        t0 = time.monotonic()
        try:
            query_vector = await self._embedding_provider.embed_text(query)
            raw = await self._vector_store.search(collection, query_vector, top_k=top_k)
            results = [
                {"id": r.id, "score": r.score, "content": r.payload.get("content", "")}
                for r in raw
            ]
            reply = message.reply({
                "results": results,
                "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })
            self._bus.resolve_reply(reply)
            return reply
        except Exception as exc:
            err = message.error(str(exc))
            self._bus.resolve_reply(err)
            return err


class DocumentAgent(BaseAgent):
    """Handles document ingestion and management."""

    def __init__(self, agent_id: str, bus: MessageBus, ingestion_pipeline, chunking_pipeline, embedding_pipeline, vector_store) -> None:
        self._ingestor = ingestion_pipeline
        self._chunker = chunking_pipeline
        self._embedder = embedding_pipeline
        self._vector_store = vector_store
        super().__init__(agent_id, bus)

    def _setup(self) -> None:
        self._capabilities = [
            AgentCapability(
                name="ingest",
                description="Ingest a document from a file path or URL",
                input_schema={"source": "str", "collection": "str"},
                output_schema={"document_id": "str", "chunks": "int"},
            )
        ]
        self._bus.subscribe("document.ingest", self.handle)

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        source = message.payload.get("source", "")
        collection = message.payload.get("collection", "rag_documents")

        try:
            if source.startswith("http"):
                doc = await self._ingestor.ingest_url(source)
            else:
                from pathlib import Path
                doc = await self._ingestor.ingest_file(Path(source))

            chunks = self._chunker.chunk_document(doc)
            embedded = await self._embedder.embed_chunks(chunks)

            from src.rag.infrastructure.vector_store.base import VectorRecord
            records = [
                VectorRecord(id=ec.chunk.chunk_id, vector=ec.embedding, payload=ec.to_vector_payload())
                for ec in embedded
            ]
            await self._vector_store.upsert_vectors(collection, records)

            reply = message.reply({"document_id": str(doc.id), "chunks": len(records)})
            self._bus.resolve_reply(reply)
            return reply
        except Exception as exc:
            err = message.error(str(exc))
            self._bus.resolve_reply(err)
            return err


class AnalyticsAgent(BaseAgent):
    """Provides collection statistics and usage analytics."""

    def __init__(self, agent_id: str, bus: MessageBus, vector_store) -> None:
        self._vector_store = vector_store
        super().__init__(agent_id, bus)

    def _setup(self) -> None:
        self._capabilities = [
            AgentCapability(
                name="stats",
                description="Return collection statistics",
                input_schema={"collection": "str"},
                output_schema={"vector_count": "int", "indexed_count": "int"},
            )
        ]
        self._bus.subscribe("analytics.stats", self.handle)

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        collection = message.payload.get("collection", "rag_documents")
        try:
            info = await self._vector_store.get_collection_info(collection)
            reply = message.reply({
                "collection": info.name,
                "vector_count": info.vector_count,
                "indexed_count": info.indexed_vector_count,
                "status": info.status,
            })
            self._bus.resolve_reply(reply)
            return reply
        except Exception as exc:
            err = message.error(str(exc))
            self._bus.resolve_reply(err)
            return err


class OrchestratorAgent(BaseAgent):
    """
    Routes incoming requests to the appropriate specialized agent.

    Maintains an agent registry, handles timeouts, and aggregates responses
    for multi-step tasks.
    """

    def __init__(self, agent_id: str, bus: MessageBus) -> None:
        self._registry: dict[str, BaseAgent] = {}
        self._default_timeout = 30.0
        super().__init__(agent_id, bus)

    def _setup(self) -> None:
        self._bus.subscribe("orchestrator.request", self.handle)

    def register(self, agent: BaseAgent) -> None:
        self._registry[agent.agent_id] = agent
        logger.info("agent_registered", agent_id=agent.agent_id, capabilities=[c.name for c in agent.capabilities])

    async def handle(self, message: AgentMessage) -> AgentMessage | None:
        task = message.payload.get("task", "")
        target_topic = self._route(task)

        if not target_topic:
            return message.error(f"No agent capable of handling task: {task}")

        routed = AgentMessage(
            type=MessageType.REQUEST,
            sender=self.agent_id,
            recipient=target_topic,
            topic=target_topic,
            payload=message.payload,
            correlation_id=message.id,
        )

        try:
            reply = await self._bus.request(routed, timeout=self._default_timeout)
            return reply
        except asyncio.TimeoutError:
            return message.error(f"Task '{task}' timed out after {self._default_timeout}s")
        except Exception as exc:
            return message.error(f"Task '{task}' failed: {exc}")

    def _route(self, task: str) -> str | None:
        routing = {
            "search": "search.request",
            "ingest": "document.ingest",
            "stats": "analytics.stats",
        }
        for keyword, topic in routing.items():
            if keyword in task.lower():
                return topic
        return None

    async def execute(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Convenience method for direct task execution without raw message construction."""
        msg = AgentMessage(
            type=MessageType.REQUEST,
            sender="api",
            recipient=self.agent_id,
            topic="orchestrator.request",
            payload={"task": task, **payload},
        )
        reply = await self._bus.request(msg, timeout=self._default_timeout)
        if reply.type == MessageType.ERROR:
            raise RuntimeError(reply.payload.get("error", "Agent error"))
        return reply.payload
