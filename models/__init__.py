"""
Open-Source Embedding Models Module

This module provides implementations for various open-source embedding models
optimized for RAG (Retrieval-Augmented Generation) systems.

Available Models:
- BGE-M3: Best for multilingual, high-accuracy applications
- Nomic Embed V2: MoE model with flexible dimensions  
- E5 Base V2: Fast and reliable general-purpose model
- MiniLM L6 V2: Fastest option for speed-critical applications

Example Usage:
    from models.embeddings import EmbeddingModelFactory
    
    # Create a model
    model = EmbeddingModelFactory.create_model('bge-m3')
    
    # Generate embeddings
    embeddings = model.encode(["Hello world", "Bonjour monde"])
    
    # List available models
    models = EmbeddingModelFactory.list_models()
    print(models)

For beginners:
    Vector embeddings are like "fingerprints" for text - they capture
    the meaning of sentences as numbers that computers can compare.
    Similar sentences get similar numbers!
"""

from .embeddings import (
    EmbeddingModelInterface,
    BGEModel,
    NomicModel, 
    E5Model,
    MiniLMModel,
    EmbeddingModelFactory,
    EmbeddingBenchmark
)

__version__ = "0.1.0"
__all__ = [
    "EmbeddingModelInterface",
    "BGEModel",
    "NomicModel", 
    "E5Model",
    "MiniLMModel",
    "EmbeddingModelFactory",
    "EmbeddingBenchmark"
]

# Model selection guide for beginners
MODEL_SELECTION_GUIDE = {
    "I need the most accurate results": "bge-m3",
    "I have multilingual content": "bge-m3", 
    "I need flexible embedding sizes": "nomic-v2",
    "I want good balance of speed and accuracy": "e5-base",
    "Speed is my top priority": "minilm",
    "I'm just starting/prototyping": "minilm",
    "I have limited computing resources": "minilm",
    "I'm deploying on mobile/edge devices": "minilm"
}

def get_model_recommendation(priority: str) -> str:
    """Get model recommendation based on priority
    
    Args:
        priority: One of the keys from MODEL_SELECTION_GUIDE
        
    Returns:
        Recommended model name
        
    Example:
        model_name = get_model_recommendation("Speed is my top priority")
        model = EmbeddingModelFactory.create_model(model_name)
    """
    return MODEL_SELECTION_GUIDE.get(priority, "e5-base")