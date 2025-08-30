import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import yaml

logger = logging.getLogger(__name__)


class EmbeddingModelInterface(ABC):
    """Interface for all embedding models"""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode text(s) into embedding vector(s)"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @abstractmethod
    def get_max_length(self) -> int:
        """Get maximum input sequence length"""
        pass


class BGEModel(EmbeddingModelInterface):
    """BGE (BAAI General Embedding) Model Implementation
    
    BGE-M3 supports:
    - Multi-lingual (100+ languages)
    - Multi-functionality (dense, sparse, multi-vector retrieval)
    - Multi-granularity (sentence to 8192 tokens)
    """
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load BGE model using sentence-transformers"""
        try:
            logger.info(f"Loading BGE model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"BGE model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode texts using BGE model"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
        
        return embeddings[0] if len(texts) == 1 else embeddings
    
    def get_dimension(self) -> int:
        """BGE-base uses 768 dimensions"""
        return 768
    
    def get_max_length(self) -> int:
        """BGE-base supports up to 512 tokens"""
        return 512
    
    def encode_sparse(self, texts: Union[str, List[str]]) -> Dict:
        """Encode using sparse representation (if supported)"""
        # BGE-M3 supports sparse encoding for hybrid search
        # This would require the FlagEmbedding library for full BGE-M3 features
        logger.warning("Sparse encoding not implemented. Install FlagEmbedding for full BGE-M3 features")
        return {}


class NomicModel(EmbeddingModelInterface):
    """Nomic Embed V2 Model Implementation
    
    Features:
    - Mixture of Experts (MoE) architecture
    - Matryoshka representation learning (flexible dimensions)
    - 100+ languages support
    - Efficient inference with dynamic routing
    """
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._supported_dimensions = [768, 512, 256, 128]  # Matryoshka dimensions
        self._load_model()
    
    def _load_model(self):
        """Load Nomic model"""
        try:
            logger.info(f"Loading Nomic model: {self.model_name}")
            # Nomic models work with sentence-transformers
            self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
            logger.info(f"Nomic model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Nomic model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], dimensionality: int = 768, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode texts with flexible dimensionality"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Add task prefix for better performance
        prefixed_texts = [f"search_document: {text}" for text in texts]
        
        embeddings = self.model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            **kwargs
        )
        
        # Truncate to desired dimensionality if using Matryoshka
        if dimensionality in self._supported_dimensions and dimensionality < embeddings.shape[-1]:
            embeddings = embeddings[..., :dimensionality]
        
        return embeddings[0] if len(texts) == 1 else embeddings
    
    def get_dimension(self) -> int:
        """Default Nomic dimension is 768"""
        return 768
    
    def get_max_length(self) -> int:
        """Nomic supports up to 8192 tokens"""
        return 8192
    
    def get_supported_dimensions(self) -> List[int]:
        """Get supported Matryoshka dimensions"""
        return self._supported_dimensions


class E5Model(EmbeddingModelInterface):
    """E5 (Microsoft) Model Implementation
    
    Features:
    - Reliable general-purpose embeddings
    - Good balance of speed and accuracy
    - No special prefixes required (unlike E5 v1)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load E5 model"""
        try:
            logger.info(f"Loading E5 model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"E5 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load E5 model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode texts using E5 model"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            **kwargs
        )
        
        return embeddings[0] if len(texts) == 1 else embeddings
    
    def get_dimension(self) -> int:
        """E5-base uses 768 dimensions"""
        return 768
    
    def get_max_length(self) -> int:
        """MPNet supports up to 384 tokens"""
        return 384


class MiniLMModel(EmbeddingModelInterface):
    """MiniLM Model Implementation
    
    Features:
    - Very fast inference
    - Smaller model size (good for resource-constrained environments)
    - Lower dimensional embeddings (384D)
    - Good for speed-critical applications
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load MiniLM model"""
        try:
            logger.info(f"Loading MiniLM model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"MiniLM model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load MiniLM model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode texts using MiniLM model"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            **kwargs
        )
        
        return embeddings[0] if len(texts) == 1 else embeddings
    
    def get_dimension(self) -> int:
        """MiniLM uses 384 dimensions"""
        return 384
    
    def get_max_length(self) -> int:
        """MiniLM supports up to 256 tokens"""
        return 256


class EmbeddingModelFactory:
    """Factory for creating embedding models"""
    
    _models = {
        'all-MiniLM-L6-v2': MiniLMModel,
        'all-mpnet-base-v2': E5Model,
        'bge-small-en-v1.5': BGEModel,
        'e5-base-v2': E5Model,
        'bge-base-en-v1.5': BGEModel
    }
    
    _model_configs = {
        'all-MiniLM-L6-v2': {
            'model_path': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimensions': 384,
            'max_length': 256,
            'speed_optimized': True,
            'description': '🚀 **Fastest** - Great for speed, good accuracy (384D)',
            'use_case': 'Speed-critical applications, real-time search',
            'pros': ['Very fast inference', 'Small memory footprint', 'Reliable'],
            'cons': ['Lower dimensional embeddings', 'Shorter context length']
        },
        'all-mpnet-base-v2': {
            'model_path': 'sentence-transformers/all-mpnet-base-v2',
            'dimensions': 768,
            'max_length': 384,
            'description': '⚖️ **Balanced** - Excellent accuracy-speed balance (768D)',
            'use_case': 'General-purpose RAG, most document types',
            'pros': ['High accuracy', 'Good performance', 'Well-tested'],
            'cons': ['Moderate context length', 'English-focused']
        },
        'bge-small-en-v1.5': {
            'model_path': 'BAAI/bge-small-en-v1.5',
            'dimensions': 384,
            'max_length': 512,
            'description': '🎯 **Efficient** - High accuracy with small size (384D)',
            'use_case': 'Resource-constrained environments, edge deployment',
            'pros': ['High accuracy for size', 'Fast inference', 'Good context length'],
            'cons': ['Lower dimensions', 'English-only']
        },
        'e5-base-v2': {
            'model_path': 'intfloat/e5-base-v2',
            'dimensions': 768,
            'max_length': 512,
            'description': '🧠 **Accurate** - Microsoft E5, very reliable (768D)',
            'use_case': 'High-accuracy requirements, enterprise applications',
            'pros': ['High accuracy', 'Stable performance', 'Good context length'],
            'cons': ['Slower than MiniLM', 'English-focused']
        },
        'bge-base-en-v1.5': {
            'model_path': 'BAAI/bge-base-en-v1.5',
            'dimensions': 768,
            'max_length': 512,
            'description': '🏆 **Premium** - SOTA accuracy for English (768D)',
            'use_case': 'Best accuracy requirements, research applications',
            'pros': ['State-of-the-art accuracy', 'Good context length', 'Reliable'],
            'cons': ['Slower inference', 'Larger memory usage']
        }
    }
    
    @classmethod
    def create_model(cls, model_name: str, device: Optional[str] = None, **kwargs) -> EmbeddingModelInterface:
        """Create an embedding model by name"""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._models.keys())}")
        
        config = cls._model_configs[model_name]
        model_path = config['model_path']
        
        model_class = cls._models[model_name]
        return model_class(model_name=model_path, device=device, **kwargs)
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """List available models with their configurations"""
        return cls._model_configs.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """Get information about a specific model"""
        if model_name not in cls._model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._model_configs[model_name].copy()
    
    @classmethod
    def recommend_model(cls, criteria: Dict[str, Any]) -> str:
        """Recommend a model based on criteria"""
        speed_priority = criteria.get('speed_priority', False)
        multilingual = criteria.get('multilingual', False)
        max_dimensions = criteria.get('max_dimensions', 1024)
        
        if speed_priority:
            return 'minilm'
        elif multilingual:
            return 'bge-m3' if max_dimensions >= 1024 else 'nomic-v2'
        else:
            return 'e5-base'


class EmbeddingBenchmark:
    """Benchmark different embedding models"""
    
    def __init__(self, models_to_test: Optional[List[str]] = None):
        self.models_to_test = models_to_test or ['bge-m3', 'nomic-v2', 'e5-base', 'minilm']
        self.results = {}
    
    async def run_benchmark(self, test_texts: List[str], num_runs: int = 3) -> Dict[str, Dict]:
        """Run benchmark on multiple models"""
        import time
        
        results = {}
        
        for model_name in self.models_to_test:
            try:
                logger.info(f"Benchmarking {model_name}")
                model = EmbeddingModelFactory.create_model(model_name)
                
                # Warmup
                model.encode(test_texts[:1])
                
                # Timing runs
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    embeddings = model.encode(test_texts)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                results[model_name] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'dimensions': model.get_dimension(),
                    'max_length': model.get_max_length(),
                    'texts_per_second': len(test_texts) / (sum(times) / len(times))
                }
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def get_fastest_model(self) -> str:
        """Get the fastest model from benchmark results"""
        if not self.results:
            raise ValueError("No benchmark results available")
        
        fastest = min(
            self.results.items(),
            key=lambda x: x[1].get('avg_time', float('inf'))
        )
        return fastest[0]
    
    def get_most_accurate_model(self) -> str:
        """Get recommended most accurate model (based on general knowledge)"""
        # Based on research, BGE-M3 generally performs best on accuracy
        return 'bge-m3'
    
    def print_results(self):
        """Print benchmark results in a readable format"""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n🚀 Embedding Model Benchmark Results")
        print("=" * 60)
        
        for model_name, metrics in self.results.items():
            if 'error' in metrics:
                print(f"❌ {model_name}: {metrics['error']}")
                continue
            
            print(f"\n📊 {model_name}:")
            print(f"  ⏱️  Average Time: {metrics['avg_time']:.3f}s")
            print(f"  🚄 Texts/Second: {metrics['texts_per_second']:.1f}")
            print(f"  📏 Dimensions: {metrics['dimensions']}")
            print(f"  📄 Max Length: {metrics['max_length']} tokens")
        
        print(f"\n🏆 Fastest Model: {self.get_fastest_model()}")
        print(f"🎯 Most Accurate: {self.get_most_accurate_model()}")


# Export main classes
__all__ = [
    'EmbeddingModelInterface',
    'BGEModel', 
    'NomicModel',
    'E5Model',
    'MiniLMModel',
    'EmbeddingModelFactory',
    'EmbeddingBenchmark'
]