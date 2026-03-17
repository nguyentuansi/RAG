"""Orchestrator and specialized agents."""

from __future__ import annotations

import asyncio
from typing import Any

from src.rag.agents.base import AgentCapability, AgentMessage, BaseAgent, MessageBus, MessageType
from src.rag.core.logging import get_logger

logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent):
    """Routes tasks to specialized sub-agents and aggregates results."""

    def __init__(self, bus: MessageBus) -> None:
        super().__init__("orchestrator", bus)
        self._registry: dict[str, BaseAgent] = {}

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="route",
                description="Route a task to the appropriate specialized agent",
            )
        ]

    def register(self, agent: BaseAgent) -> None:
        self._registry[agent.name] = agent

    async def on_start(self) -> None:
        for agent in self._registry.values():
            await agent.start()
        logger.info("orchestrator_started", agents=list(self._registry))

    async def on_stop(self) -> None:
        await asyncio.gather(*[a.stop() for a in self._registry.values()])

    async def handle(self, message: AgentMessage) -> AgentMessage:
        intent = message.content.get("intent", "")
        target = self._resolve_target(intent)

        if not target:
            return message.reply(
                {"error": f"No agent handles intent '{intent}'"}, error=True
            )

        try:
            response = await asyncio.wait_for(
                self.bus.request(target, message), timeout=30.0
            )
            return response
        except asyncio.TimeoutError:
            return message.reply({"error": f"Agent '{target}' timed out"}, error=True)
        except Exception as exc:
            logger.error("orchestrator_dispatch_error", target=target, error=str(exc))
            return message.reply({"error": str(exc)}, error=True)

    def _resolve_target(self, intent: str) -> str | None:
        mapping = {
            "search": "search_agent",
            "ingest": "document_agent",
            "stats": "analytics_agent",
            "explain": "analytics_agent",
        }
        for keyword, agent_name in mapping.items():
            if keyword in intent.lower() and agent_name in self._registry:
                return agent_name
        return None


class SearchAgent(BaseAgent):
    """Handles document search and retrieval tasks."""

    def __init__(self, bus: MessageBus, vector_store, embedding_provider) -> None:
        super().__init__("search_agent", bus)
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="search",
                description="Search indexed documents by natural language query",
                input_schema={"query": "str", "top_k": "int", "collection": "str"},
                output_schema={"results": "list[SearchResult]", "latency_ms": "float"},
            )
        ]

    async def handle(self, message: AgentMessage) -> AgentMessage:
        query = message.content.get("query", "")
        top_k = message.content.get("top_k", 5)
        collection = message.content.get("collection", "rag_documents")

        try:
            query_vector = await self.embedding_provider.embed_text(query)
            raw = await self.vector_store.search(collection, query_vector, top_k=top_k)
            results = [
                {
                    "id": r.id,
                    "score": r.score,
                    "content": r.payload.get("content", "")[:500],
                    "source": r.payload.get("source"),
                }
                for r in raw
            ]
            return message.reply({"results": results, "total": len(results)})
        except Exception as exc:
            return message.reply({"error": str(exc)}, error=True)


class DocumentAgent(BaseAgent):
    """Manages document ingestion tasks."""

    def __init__(self, bus: MessageBus, ingestion_pipeline) -> None:
        super().__init__("document_agent", bus)
        self.ingestion = ingestion_pipeline

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="ingest",
                description="Ingest a document file or URL into the vector store",
                input_schema={"path": "str | None", "url": "str | None", "format": "str"},
            )
        ]

    async def handle(self, message: AgentMessage) -> AgentMessage:
        path = message.content.get("path")
        url = message.content.get("url")

        try:
            if path:
                from pathlib import Path
                doc = await self.ingestion.ingest_file(Path(path))
            elif url:
                doc = await self.ingestion.ingest_url(url)
            else:
                return message.reply({"error": "Neither path nor url provided"}, error=True)

            return message.reply({
                "document_id": str(doc.document.id),
                "chunks": doc.chunk_count,
                "status": doc.document.status.value,
            })
        except Exception as exc:
            return message.reply({"error": str(exc)}, error=True)


class AnalyticsAgent(BaseAgent):
    """Provides collection statistics and index health reports."""

    def __init__(self, bus: MessageBus, vector_store) -> None:
        super().__init__("analytics_agent", bus)
        self.vector_store = vector_store

    @property
    def capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="stats",
                description="Return vector collection statistics",
                input_schema={"collection": "str"},
            )
        ]

    async def handle(self, message: AgentMessage) -> AgentMessage:
        collection = message.content.get("collection", "rag_documents")
        try:
            info = await self.vector_store.get_collection_info(collection)
            return message.reply({
                "collection": info.name,
                "vector_count": info.vector_count,
                "indexed_count": info.indexed_vector_count,
                "status": info.status,
                "distance_metric": info.distance_metric,
            })
        except Exception as exc:
            return message.reply({"error": str(exc)}, error=True)
