"""
MCP Server Module

Model Context Protocol server implementation for the Visual RAG System.
Exposes RAG functionality as tools and resources for LLM integration.

Features:
- Document search tools
- Pipeline management
- Real-time status monitoring
- Beginner-friendly explanations

Usage:
    # Run the server
    python -m mcp_server.server
    
    # Or import for integration
    from mcp_server.server import RAGMCPServer
"""

from .server import RAGMCPServer

__version__ = "0.1.0"
__all__ = ["RAGMCPServer"]