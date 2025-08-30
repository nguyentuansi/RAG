"""
RAG Pipeline Module

Core pipeline implementation for the Visual RAG System.
Provides a single, orchestrated pipeline that's easy to understand and monitor.

Main Components:
- VisualRAGPipeline: The main pipeline orchestrator
- Pipeline stages for each processing step
- State management and visual feedback

Example Usage:
    from pipeline import VisualRAGPipeline
    
    # Initialize pipeline
    pipeline = VisualRAGPipeline()
    
    # Process documents
    results = await pipeline.ingest_documents(['doc1.txt', 'doc2.pdf'])
    
    # Search and generate response
    result = await pipeline.search_and_generate('What is the main topic?')
"""

from .main_pipeline import VisualRAGPipeline, PipelineState, PipelineMetrics
from .stages import (
    DocumentIngestion,
    TextChunking, 
    EmbeddingGeneration,
    VectorIndexing,
    SearchRetrieval,
    ResponseGeneration
)

__version__ = "0.1.0"
__all__ = [
    "VisualRAGPipeline",
    "PipelineState", 
    "PipelineMetrics",
    "DocumentIngestion",
    "TextChunking",
    "EmbeddingGeneration", 
    "VectorIndexing",
    "SearchRetrieval",
    "ResponseGeneration"
]