import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    # Qdrant not available - create placeholders
    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass
    
    class Distance:
        COSINE = "cosine"
    
    class VectorParams:
        def __init__(self, *args, **kwargs):
            pass
    
    class PointStruct:
        def __init__(self, *args, **kwargs):
            pass

from rich.console import Console

# Import improved chunking
try:
    from .improved_chunking import ImprovedTextChunking
    IMPROVED_CHUNKING_AVAILABLE = True
except ImportError:
    IMPROVED_CHUNKING_AVAILABLE = False

console = Console()
logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Base class for all pipeline stages"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stage_config = config.get('pipeline', {}).get('stages', {})
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Process input data and return results"""
        pass


class DocumentIngestion(PipelineStage):
    """Stage 1: Load and validate documents from various formats"""
    
    async def process(
        self, 
        file_paths: List[Union[str, Path]],
        progress_hook: Optional[Callable] = None,
        data_hook: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Load documents from files and extract text content.
        
        Args:
            file_paths: List of file paths to process
            progress_hook: Callback for progress updates
            data_hook: Callback for data previews
            
        Returns:
            List of document dictionaries with text content and metadata
        """
        documents = []
        supported_formats = self.stage_config.get('document_ingestion', {}).get('supported_formats', [])
        
        for i, file_path in enumerate(file_paths):
            if progress_hook:
                progress_hook(i, len(file_paths))
            
            try:
                file_path = Path(file_path)
                
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Extract text based on file format
                if file_path.suffix.lower() == '.txt':
                    text_content = self._extract_txt(file_path)
                elif file_path.suffix.lower() == '.md':
                    text_content = self._extract_markdown(file_path)
                elif file_path.suffix.lower() == '.pdf':
                    text_content = self._extract_pdf(file_path)
                elif file_path.suffix.lower() == '.json':
                    text_content = self._extract_json(file_path)
                else:
                    logger.warning(f"Unsupported format: {file_path.suffix}")
                    continue
                
                document = {
                    'id': f"doc_{i}_{file_path.stem}",
                    'title': file_path.stem,
                    'content': text_content,
                    'source': str(file_path),
                    'format': file_path.suffix.lower(),
                    'size': len(text_content),
                    'metadata': {
                        'file_name': file_path.name,
                        'file_size': file_path.stat().st_size,
                        'created_at': file_path.stat().st_ctime
                    }
                }
                
                documents.append(document)
                
                # Send preview to UI
                if data_hook:
                    preview = {
                        'title': document['title'],
                        'content_preview': text_content[:200] + "..." if len(text_content) > 200 else text_content,
                        'size': document['size'],
                        'format': document['format']
                    }
                    data_hook(preview)
                    
                logger.info(f"Loaded document: {file_path.name} ({len(text_content)} chars)")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if progress_hook:
            progress_hook(len(file_paths), len(file_paths))
        
        console.print(f"[green]Loaded {len(documents)} documents")
        return documents
    
    def _extract_txt(self, file_path: Path) -> str:
        """Extract text from plain text files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _extract_markdown(self, file_path: Path) -> str:
        """Extract text from markdown files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except ImportError:
            logger.warning("PyPDF2 not installed, skipping PDF extraction")
            return f"PDF content from {file_path.name} (install PyPDF2 to extract text)"
    
    def _extract_json(self, file_path: Path) -> str:
        """Extract text from JSON files"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'content' in data:
                return data['content']
            elif isinstance(data, dict) and 'text' in data:
                return data['text']
            else:
                return json.dumps(data, indent=2)


class TextChunking(PipelineStage):
    """Stage 2: Split documents into searchable chunks with overlap"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Check which chunking method to use
        chunking_method = config.get('chunking', {}).get('method', 'semantic')
        self.use_improved = chunking_method == 'semantic' and IMPROVED_CHUNKING_AVAILABLE
        
        if self.use_improved:
            logger.info("Using improved semantic chunking")
            self.improved_chunker = ImprovedTextChunking(config)
        else:
            if chunking_method == 'semantic' and not IMPROVED_CHUNKING_AVAILABLE:
                logger.warning("Semantic chunking requested but not available, falling back to current method")
            logger.info("Using current character-based chunking")
    
    async def process(
        self, 
        documents: List[Dict],
        progress_hook: Optional[Callable] = None,
        data_hook: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of document dictionaries
            progress_hook: Callback for progress updates
            data_hook: Callback for data previews
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Use improved chunking if enabled
        if self.use_improved:
            # Convert input format for improved chunker
            input_data = {'documents': documents}
            result = await self.improved_chunker.process(input_data, progress_hook, data_hook)
            return result['chunks']
        
        # Fall back to current chunking method
        chunks = []
        chunk_config = self.config['chunking']
        chunk_size = chunk_config['chunk_size']
        overlap = chunk_config['overlap']
        separators = chunk_config['separators']
        
        for doc_idx, document in enumerate(documents):
            if progress_hook:
                progress_hook(doc_idx, len(documents))
            
            text = document['content']
            doc_chunks = self._split_text_recursive(
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
                separators=separators
            )
            
            # Create chunk objects
            for chunk_idx, chunk_text in enumerate(doc_chunks):
                chunk = {
                    'id': f"chunk_{document['id']}_{chunk_idx}",
                    'content': chunk_text,
                    'document_id': document['id'],
                    'document_title': document['title'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(doc_chunks),
                    'size': len(chunk_text),
                    'metadata': {
                        'source': document['source'],
                        'format': document['format'],
                        'chunk_start': chunk_idx * (chunk_size - overlap),
                        'chunk_end': chunk_idx * (chunk_size - overlap) + len(chunk_text)
                    }
                }
                
                chunks.append(chunk)
                
                # Send preview to UI (first few chunks)
                if data_hook and chunk_idx < 3:
                    preview = {
                        'chunk_id': chunk['id'],
                        'content_preview': chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text,
                        'size': chunk['size'],
                        'chunk_index': chunk_idx,
                        'document_title': document['title']
                    }
                    data_hook(preview)
            
            logger.info(f"Split '{document['title']}' into {len(doc_chunks)} chunks")
        
        if progress_hook:
            progress_hook(len(documents), len(documents))
        
        console.print(f"[green]Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _split_text_recursive(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int,
        separators: List[str]
    ) -> List[str]:
        """Recursively split text using different separators"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        
        # Try each separator in order
        for separator in separators:
            if separator in text:
                splits = text.split(separator)
                current_chunk = ""
                
                for split in splits:
                    # If adding this split would exceed chunk_size
                    if len(current_chunk) + len(split) + len(separator) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Add overlap from end of current chunk
                            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
                            current_chunk = overlap_text + separator + split
                        else:
                            # Single split is too long, recurse with next separator
                            sub_chunks = self._split_text_recursive(
                                split, chunk_size, overlap, separators[1:]
                            )
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                    else:
                        if current_chunk:
                            current_chunk += separator + split
                        else:
                            current_chunk = split
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        # If no separators worked, split by character count
        result = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                result.append(chunk.strip())
        
        return result


class EmbeddingGeneration(PipelineStage):
    """Stage 3: Generate vector embeddings using open-source models"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = None
        self.model_name = config['embedding_models']['default']
        self.model_config = config['embedding_models'][self.model_name]
    
    async def process(
        self, 
        chunks: List[Dict],
        progress_hook: Optional[Callable] = None,
        data_hook: Optional[Callable] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of chunk dictionaries
            progress_hook: Callback for progress updates
            data_hook: Callback for data previews
            
        Returns:
            List of embedding vectors (numpy arrays)
        """
        if self.model is None:
            console.print(f"[yellow]Loading embedding model: {self.model_name}")
            await self._load_model()
        
        embeddings = []
        texts = [chunk['content'] for chunk in chunks]
        batch_size = self.stage_config.get('embedding_generation', {}).get('batch_size', 32)
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]
            
            if progress_hook:
                progress_hook(i, len(texts))
            
            try:
                # Generate embeddings for batch
                batch_embeddings = await self._generate_batch_embeddings(batch_texts)
                embeddings.extend(batch_embeddings)
                
                # Send preview to UI (first few embeddings)
                if data_hook and i == 0:
                    for j, (embedding, chunk) in enumerate(zip(batch_embeddings[:3], batch_chunks[:3])):
                        preview = {
                            'chunk_id': chunk['id'],
                            'embedding_shape': embedding.shape,
                            'embedding_sample': embedding[:10].tolist(),  # First 10 dimensions
                            'model': self.model_name,
                            'dimensions': len(embedding)
                        }
                        data_hook(preview)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Add placeholder embeddings to maintain alignment
                placeholder_dim = self.model_config['dimensions']
                batch_embeddings = [np.zeros(placeholder_dim) for _ in batch_texts]
                embeddings.extend(batch_embeddings)
        
        if progress_hook:
            progress_hook(len(texts), len(texts))
        
        console.print(f"[green]Generated {len(embeddings)} embeddings using {self.model_name}")
        return embeddings
    
    async def _load_model(self):
        """Load the embedding model"""
        try:
            model_path = self.model_config['model_path']
            
            # Check if we should use GPU
            device = 'cuda' if torch.cuda.is_available() and not self.config.get('development', {}).get('disable_gpu', False) else 'cpu'
            
            self.model = SentenceTransformer(model_path, device=device)
            logger.info(f"Loaded {self.model_name} on {device}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(texts, convert_to_numpy=True)
        )
        
        return list(embeddings)


class VectorIndexing(PipelineStage):
    """Stage 4: Store vectors in Qdrant database with metadata"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = None
        self.collection_name = config['vectordb']['collection_name']
    
    async def process(
        self, 
        input_data: Dict,
        progress_hook: Optional[Callable] = None,
        data_hook: Optional[Callable] = None
    ) -> Dict:
        """
        Index vectors in Qdrant database.
        
        Args:
            input_data: Dictionary with 'chunks' and 'embeddings'
            progress_hook: Callback for progress updates
            data_hook: Callback for data previews
            
        Returns:
            Dictionary with indexing results
        """
        chunks = input_data['chunks']
        embeddings = input_data['embeddings']
        
        if self.client is None:
            await self._connect_qdrant()
        
        # Ensure collection exists
        await self._ensure_collection_exists(embeddings[0].shape[0])
        
        # Prepare points for insertion
        points = []
        batch_size = self.stage_config.get('vector_indexing', {}).get('index_batch_size', 100)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if progress_hook and i % 10 == 0:
                progress_hook(i, len(chunks))
            
            point = PointStruct(
                id=hash(chunk['id']) % (2**31),  # Convert to positive int
                vector=embedding.tolist(),
                payload={
                    'chunk_id': chunk['id'],
                    'content': chunk['content'],
                    'document_id': chunk['document_id'],
                    'document_title': chunk['document_title'],
                    'chunk_index': chunk['chunk_index'],
                    'source': chunk['metadata']['source'],
                    'format': chunk['metadata']['format'],
                    'size': chunk['size']
                }
            )
            
            points.append(point)
            
            # Send preview to UI (first few points)
            if data_hook and i < 3:
                preview = {
                    'point_id': point.id,
                    'chunk_id': chunk['id'],
                    'vector_dim': len(embedding),
                    'payload_keys': list(point.payload.keys()),
                    'content_preview': chunk['content'][:100] + "..."
                }
                data_hook(preview)
        
        # Insert points in batches
        indexed_count = 0
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points
                    )
                )
                indexed_count += len(batch_points)
                
                if progress_hook:
                    progress_hook(i + len(batch_points), len(points))
                    
            except Exception as e:
                logger.error(f"Error indexing batch {i}: {e}")
        
        console.print(f"[green]Indexed {indexed_count} vectors in collection '{self.collection_name}'")
        
        return {
            'success': True,
            'indexed_count': indexed_count,
            'collection_name': self.collection_name,
            'total_points': len(points)
        }
    
    async def _connect_qdrant(self):
        """Connect to Qdrant database"""
        try:
            # Use environment variables or config
            host = self.config.get('qdrant', {}).get('host', 'localhost')
            port = self.config.get('qdrant', {}).get('port', 6333)
            
            self.client = QdrantClient(host=host, port=port, check_compatibility=False)
            logger.info(f"Connected to Qdrant at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
    
    async def _ensure_collection_exists(self, vector_dim: int):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection - get distance metric from config
                metric_name = self.config.get('vectordb', {}).get('distance_metric', 'cosine')
                distance_metric = self._get_distance_metric(metric_name)
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_dim,
                        distance=distance_metric
                    )
                )
                
                logger.info(f"Created collection '{self.collection_name}' with {vector_dim}D vectors")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error managing collection: {e}")
            raise
    
    def _get_distance_metric(self, metric_name: str) -> Distance:
        """Map configuration metric name to Qdrant Distance enum"""
        metric_map = {
            'cosine': Distance.COSINE,
            'euclidean': Distance.EUCLID,
        }
        
        if metric_name not in metric_map:
            logger.warning(f"Unknown distance metric '{metric_name}', defaulting to cosine")
            return Distance.COSINE
            
        logger.info(f"Using {metric_name} distance metric for vector similarity")
        return metric_map[metric_name]


class SearchRetrieval(PipelineStage):
    """Stage 5: Search for similar documents using vector similarity"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = None
        self.collection_name = config['vectordb']['collection_name']
        self.embedding_stage = EmbeddingGeneration(config)  # Reuse embedding logic
    
    async def process(self, input_data: Dict) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            input_data: Dictionary with 'query', 'search_type', 'top_k'
            
        Returns:
            List of search results with scores
        """
        query = input_data['query']
        search_type = input_data.get('search_type', 'dense')
        top_k = input_data.get('top_k', 5)
        
        if self.client is None:
            await self._connect_qdrant()
        
        # Check if collection exists and has data
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                raise Exception(f"Collection '{self.collection_name}' does not exist. Please process some documents first.")
            
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            
            if points_count == 0:
                raise Exception(f"Collection '{self.collection_name}' exists but has no vectors. Please process some documents first.")
                
        except Exception as e:
            raise
        
        # Generate query embedding with preprocessing
        query_embedding = await self._embed_query_enhanced(query)
        
        # First try with score threshold
        score_threshold = self.config['search'].get('score_threshold', 0.3)
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # If no results with threshold, try without threshold
        if len(search_results) == 0:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
                # No score_threshold - get best matches regardless of score
            )
        
        # Format results with better field access
        results = []
        for result in search_results:
            # Handle different possible field names for content
            content = (
                result.payload.get('content') or 
                result.payload.get('text') or 
                result.payload.get('chunk_text') or
                result.payload.get('chunk_content') or
                "Content not available - Available fields: " + str(list(result.payload.keys()))
            )
            
            results.append({
                'id': result.payload.get('chunk_id', result.payload.get('id', 'unknown')),
                'content': content,
                'title': result.payload.get('document_title', result.payload.get('title', 'Unknown Document')),
                'source': result.payload.get('source', result.payload.get('document_path', 'Unknown Source')),
                'score': result.score,
                'metadata': {
                    'document_id': result.payload.get('document_id', 'unknown'),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'format': result.payload.get('format', 'unknown')
                }
            })
        
        console.print(f"[green]Found {len(results)} relevant chunks for query")
        return results
    
    async def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for search query"""
        # Ensure embedding model is loaded
        if self.embedding_stage.model is None:
            await self.embedding_stage._load_model()
        
        # Generate single embedding
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.embedding_stage.model.encode([query], convert_to_numpy=True)
        )
        
        return embedding[0]
    
    async def _embed_query_enhanced(self, query: str) -> np.ndarray:
        """Generate enhanced embedding for search query with preprocessing"""
        # Query preprocessing and enhancement
        enhanced_query = self._preprocess_query(query)
        
        # Ensure embedding model is loaded
        if self.embedding_stage.model is None:
            await self.embedding_stage._load_model()
        
        # Generate enhanced embedding with normalization
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.embedding_stage.model.encode([enhanced_query], convert_to_numpy=True, normalize_embeddings=True)
        )
        
        return embedding[0]
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and enhance query for better search results"""
        # Basic preprocessing
        query = query.strip().lower()
        
        # Add instruction prefix for better embedding alignment
        if query.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            # Question format - add instruction prefix
            enhanced_query = f"Query: {query}"
        else:
            # Keyword search - expand with context
            enhanced_query = f"Find information about: {query}"
        
        # Add domain context for RAG-specific queries
        if any(term in query for term in ['rag', 'retrieval', 'augmented', 'generation', 'vector', 'embedding']):
            enhanced_query += " in the context of retrieval-augmented generation systems"
        
        return enhanced_query
    
    async def _connect_qdrant(self):
        """Connect to Qdrant database for search"""
        try:
            # Use environment variables or config
            host = self.config.get('qdrant', {}).get('host', 'localhost')
            port = self.config.get('qdrant', {}).get('port', 6333)
            
            self.client = QdrantClient(host=host, port=port, check_compatibility=False)
            
            # Test connection by getting collections
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant for search at {host}:{port}")
            logger.info(f"Available collections: {[col.name for col in collections.collections]}")
            
        except Exception as e:
            logger.error(f"Error connecting to Qdrant for search: {e}")
            logger.error("Make sure Qdrant is running: docker run -d -p 6333:6333 qdrant/qdrant:v1.12.5")
            raise Exception(f"Cannot connect to Qdrant database at {host}:{port}. Please ensure Qdrant is running. Error: {e}")


class ResponseGeneration(PipelineStage):
    """Stage 6: Generate responses using retrieved context"""
    
    async def process(self, input_data: Dict) -> Dict:
        """
        Generate response using retrieved context.
        
        Args:
            input_data: Dictionary with 'query' and 'search_results'
            
        Returns:
            Dictionary with generated response and metadata
        """
        query = input_data['query']
        search_results = input_data['search_results']
        
        # For now, return a simple concatenation of top results
        # In a full implementation, this would use an LLM
        
        if not search_results:
            return {
                'response': "I couldn't find any relevant information for your query.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context from search results
        context_pieces = []
        sources = []
        
        for result in search_results:
            title = result.get('title', result.get('document_title', 'Unknown Document'))
            content = result.get('content', 'Content not available')
            context_pieces.append(f"From '{title}': {content}")
            sources.append({
                'title': title,
                'content': content,  # Include content in sources for dashboard
                'source': result['source'],
                'score': result['score']
            })
        
        context = "\n\n".join(context_pieces)
        
        # Simple response generation (replace with LLM in production)
        response = f"""Based on the available information:

{context}

Query: {query}

This information was found in {len(search_results)} relevant documents with an average relevance score of {sum(r['score'] for r in search_results) / len(search_results):.2f}."""
        
        return {
            'response': response,
            'sources': sources,
            'confidence': sum(r['score'] for r in search_results) / len(search_results),
            'context_length': len(context)
        }