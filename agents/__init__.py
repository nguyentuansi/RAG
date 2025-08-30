"""
Multi-Agent System Module

Demonstrates agent-to-agent (A2A) communication patterns for complex RAG tasks.
Includes specialized agents for search, document processing, analytics, and explanations.

Agents:
- SearchAgent: Document search and retrieval
- DocumentAgent: File processing and management  
- AnalyticsAgent: Performance monitoring and insights
- ExplainerAgent: Beginner-friendly explanations
- AgentOrchestrator: Coordinates multi-agent workflows

Usage:
    from agents.orchestrator import AgentOrchestrator
    from pipeline.main_pipeline import VisualRAGPipeline
    
    pipeline = VisualRAGPipeline()
    orchestrator = AgentOrchestrator(pipeline)
    
    response = await orchestrator.process_user_request("Explain embeddings")
"""

from .orchestrator import (
    AgentOrchestrator,
    SearchAgent,
    DocumentAgent,
    AnalyticsAgent,
    ExplainerAgent,
    AgentMessage,
    MessageType,
    AgentRole
)

__version__ = "0.1.0"
__all__ = [
    "AgentOrchestrator",
    "SearchAgent", 
    "DocumentAgent",
    "AnalyticsAgent",
    "ExplainerAgent",
    "AgentMessage",
    "MessageType", 
    "AgentRole"
]