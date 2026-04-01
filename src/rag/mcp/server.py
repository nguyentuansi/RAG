"""Production MCP server with tool registration and JSON-RPC 2.0 dispatch."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from src.rag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable
    tags: list[str] = field(default_factory=list)


class MCPServer:
    """
    JSON-RPC 2.0 MCP server.

    Tools are registered via @server.tool() and exposed through
    the standard MCP protocol over stdio.
    """

    def __init__(self, name: str = "rag-mcp-server", version: str = "1.0.0") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, ToolDefinition] = {}
        self._vector_store = None
        self._embedding_provider = None
        self._settings = None

    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        tags: list[str] | None = None,
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=func,
                tags=tags or [],
            )
            return func
        return decorator

    async def configure(self, vector_store, embedding_provider, settings) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._settings = settings
        self._register_default_tools()
        logger.info("mcp_server_configured", tools=list(self._tools))

    def _register_default_tools(self) -> None:
        @self.tool(
            name="search_documents",
            description="Search indexed documents using natural language similarity",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                    "collection_name": {"type": "string"},
                    "score_threshold": {"type": "number", "default": 0.0},
                },
                "required": ["query"],
            },
            tags=["search"],
        )
        async def search_documents(query: str, top_k: int = 5, collection_name: str | None = None, score_threshold: float = 0.0) -> dict:
            collection = collection_name or self._settings.collection_name
            query_vector = await self._embedding_provider.embed_text(query)
            results = await self._vector_store.search(
                collection, query_vector, top_k=top_k, score_threshold=score_threshold
            )
            return {
                "results": [
                    {
                        "id": r.id,
                        "score": r.score,
                        "content": r.payload.get("content", ""),
                        "source": r.payload.get("source"),
                        "document_id": r.payload.get("document_id"),
                    }
                    for r in results
                ],
                "total": len(results),
            }

        @self.tool(
            name="get_collection_stats",
            description="Return statistics about a vector collection",
            input_schema={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string"},
                },
                "required": [],
            },
            tags=["admin"],
        )
        async def get_collection_stats(collection_name: str | None = None) -> dict:
            collection = collection_name or self._settings.collection_name
            info = await self._vector_store.get_collection_info(collection)
            return {
                "collection": info.name,
                "vector_count": info.vector_count,
                "indexed_vectors": info.indexed_vector_count,
                "distance_metric": info.distance_metric,
                "status": info.status,
            }

        @self.tool(
            name="list_collections",
            description="List all available vector collections",
            input_schema={"type": "object", "properties": {}, "required": []},
            tags=["admin"],
        )
        async def list_collections() -> dict:
            return {"tools": list(self._tools), "collections": [self._settings.collection_name]}

    async def _handle_request(self, request: dict) -> dict:
        rpc_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": self.version},
                },
            }

        if method == "tools/list":
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in self._tools.values()
            ]
            return {"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": tools}}

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            tool = self._tools.get(tool_name)
            if not tool:
                return self._error(rpc_id, -32601, f"Tool '{tool_name}' not found")
            try:
                result = await tool.handler(**tool_args)
                return {
                    "jsonrpc": "2.0", "id": rpc_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
                }
            except Exception as exc:
                logger.error("mcp_tool_error", tool=tool_name, error=str(exc))
                return self._error(rpc_id, -32603, str(exc))

        return self._error(rpc_id, -32601, f"Unknown method: {method}")

    def _error(self, rpc_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}

    async def run_stdio(self) -> None:
        """Run the server over stdio (standard MCP transport)."""
        logger.info("mcp_server_starting", name=self.name)
        reader = asyncio.StreamReader()
        await asyncio.get_event_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

        async for line in reader:
            raw = line.decode().strip()
            if not raw:
                continue
            try:
                request = json.loads(raw)
                response = await self._handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError as exc:
                err = self._error(None, -32700, f"Parse error: {exc}")
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()


server = MCPServer()
