import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, TaskID

from .stages import (
    DocumentIngestion,
    EmbeddingGeneration,
    ResponseGeneration,
    SearchRetrieval,
    TextChunking,
    VectorIndexing,
)

console = Console()
logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = "⚪ Idle"
    LOADING = "🔄 Loading Documents"
    CHUNKING = "✂️ Chunking Text"  
    EMBEDDING = "🔢 Generating Embeddings"
    INDEXING = "📊 Indexing Vectors"
    READY = "✅ Ready for Search"
    SEARCHING = "🔍 Searching"
    GENERATING = "💬 Generating Response"
    ERROR = "❌ Error"
    PAUSED = "⏸️ Paused"


@dataclass
class PipelineMetrics:
    documents_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    vectors_indexed: int = 0
    searches_performed: int = 0
    total_processing_time: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    def update_timestamp(self):
        self.last_update = time.time()


@dataclass
class StageResult:
    stage_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class VisualRAGPipeline:
    """
    Single orchestrated RAG pipeline with real-time visual feedback.
    
    This is the HEART of the system - all data flows through here,
    making it easy for beginners to understand the complete process.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.pipeline_id = str(uuid4())[:8]
        self.config = self._load_config(config_path)
        self.state = PipelineState.IDLE
        self.metrics = PipelineMetrics()
        
        # Initialize pipeline stages
        try:
            self.stages = {
                'ingestion': DocumentIngestion(self.config),
                'chunking': TextChunking(self.config),
                'embedding': EmbeddingGeneration(self.config),
                'indexing': VectorIndexing(self.config),
                'retrieval': SearchRetrieval(self.config),
                'generation': ResponseGeneration(self.config)
            }
            logger.info("All pipeline stages initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline stages: {e}")
            # Initialize with placeholder stages that will show appropriate errors
            self.stages = {}
            raise Exception(f"Pipeline stage initialization failed: {e}")
        
        # Visual feedback callbacks
        self._status_callbacks = []
        self._progress_callbacks = []
        self._data_callbacks = []
        
        # Current processing data (for visualization)
        self.current_documents = []
        self.current_chunks = []
        self.current_embeddings = []
        self.current_search_results = []
        
        logger.info(f"Pipeline {self.pipeline_id} initialized")
    
    def _load_config(self, config_path: Optional[Path] = None) -> Dict:
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def add_status_callback(self, callback):
        """Add callback for state changes (for UI updates)"""
        self._status_callbacks.append(callback)
    
    def add_progress_callback(self, callback):
        """Add callback for progress updates (for progress bars)"""
        self._progress_callbacks.append(callback)
    
    def add_data_callback(self, callback):
        """Add callback for data updates (for live previews)"""
        self._data_callbacks.append(callback)
    
    def _notify_status_change(self, new_state: PipelineState):
        """Notify all UI components about state changes"""
        self.state = new_state
        for callback in self._status_callbacks:
            try:
                callback(new_state, self.metrics)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def _notify_progress(self, stage: str, current: int, total: int):
        """Notify UI about progress within a stage"""
        for callback in self._progress_callbacks:
            try:
                callback(stage, current, total)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _notify_data_update(self, stage: str, data_sample: Any):
        """Notify UI about intermediate data (for live previews)"""
        for callback in self._data_callbacks:
            try:
                callback(stage, data_sample)
            except Exception as e:
                logger.error(f"Data callback error: {e}")
    
    async def ingest_documents(
        self, 
        file_paths: List[Union[str, Path]],
        step_by_step: bool = False
    ) -> List[StageResult]:
        """
        Complete document ingestion pipeline with visual feedback.
        
        This method processes documents through ALL stages:
        Document Loading → Chunking → Embedding → Indexing
        
        Args:
            file_paths: List of document file paths
            step_by_step: If True, pauses after each stage for inspection
            
        Returns:
            List of results from each pipeline stage
        """
        results = []
        start_time = time.time()
        
        try:
            # Stage 1: Document Ingestion
            self._notify_status_change(PipelineState.LOADING)
            console.print(f"[blue]Starting pipeline {self.pipeline_id}")
            
            ingestion_result = await self._run_stage(
                'ingestion',
                self.stages['ingestion'].process,
                file_paths
            )
            results.append(ingestion_result)
            
            if not ingestion_result.success:
                self._notify_status_change(PipelineState.ERROR)
                return results
                
            self.current_documents = ingestion_result.data
            self.metrics.documents_processed = len(self.current_documents)
            self.metrics.update_timestamp()
            
            if step_by_step:
                await self._wait_for_user_continue("Document loading complete")
            
            # Stage 2: Text Chunking
            self._notify_status_change(PipelineState.CHUNKING)
            
            chunking_result = await self._run_stage(
                'chunking',
                self.stages['chunking'].process,
                self.current_documents
            )
            results.append(chunking_result)
            
            if not chunking_result.success:
                self._notify_status_change(PipelineState.ERROR)
                return results
                
            self.current_chunks = chunking_result.data
            self.metrics.chunks_created = len(self.current_chunks)
            self.metrics.update_timestamp()
            
            if step_by_step:
                await self._wait_for_user_continue("Text chunking complete")
            
            # Stage 3: Embedding Generation
            self._notify_status_change(PipelineState.EMBEDDING)
            
            embedding_result = await self._run_stage(
                'embedding',
                self.stages['embedding'].process,
                self.current_chunks
            )
            results.append(embedding_result)
            
            if not embedding_result.success:
                self._notify_status_change(PipelineState.ERROR)
                return results
                
            self.current_embeddings = embedding_result.data
            self.metrics.embeddings_generated = len(self.current_embeddings)
            self.metrics.update_timestamp()
            
            if step_by_step:
                await self._wait_for_user_continue("Embedding generation complete")
            
            # Stage 4: Vector Indexing
            self._notify_status_change(PipelineState.INDEXING)
            
            indexing_result = await self._run_stage(
                'indexing',
                self.stages['indexing'].process,
                {
                    'chunks': self.current_chunks,
                    'embeddings': self.current_embeddings
                }
            )
            results.append(indexing_result)
            
            if not indexing_result.success:
                self._notify_status_change(PipelineState.ERROR)
                return results
                
            self.metrics.vectors_indexed = indexing_result.metadata.get('indexed_count', 0)
            self.metrics.update_timestamp()
            
            # Pipeline Complete
            self.metrics.total_processing_time = time.time() - start_time
            self._notify_status_change(PipelineState.READY)
            
            console.print(f"[green]Pipeline {self.pipeline_id} completed successfully!")
            console.print(f"[green]  📄 Documents: {self.metrics.documents_processed}")
            console.print(f"[green]  ✂️ Chunks: {self.metrics.chunks_created}")
            console.print(f"[green]  🔢 Embeddings: {self.metrics.embeddings_generated}")
            console.print(f"[green]  📊 Indexed: {self.metrics.vectors_indexed}")
            console.print(f"[green]  ⏱️ Total time: {self.metrics.total_processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._notify_status_change(PipelineState.ERROR)
            results.append(StageResult(
                stage_name="pipeline",
                success=False,
                error=str(e)
            ))
        
        return results
    
    async def search_and_generate(
        self, 
        query: str, 
        search_type: str = "dense",
        top_k: int = 5
    ) -> StageResult:
        """
        Search pipeline: Query → Retrieval → Generation
        
        Args:
            query: Search query
            search_type: "dense", "hybrid", or "mmr"
            top_k: Number of results to return
            
        Returns:
            Generation result with response and sources
        """
        if self.state != PipelineState.READY:
            return StageResult(
                stage_name="search",
                success=False,
                error=f"Pipeline not ready. Current state: {self.state.value}"
            )
        
        try:
            # Stage 1: Search Retrieval
            self._notify_status_change(PipelineState.SEARCHING)
            
            search_result = await self._run_stage(
                'retrieval',
                self.stages['retrieval'].process,
                {
                    'query': query,
                    'search_type': search_type,
                    'top_k': top_k
                }
            )
            
            if not search_result.success:
                self._notify_status_change(PipelineState.ERROR)
                return search_result
            
            self.current_search_results = search_result.data
            self.metrics.searches_performed += 1
            
            # Stage 2: Response Generation
            self._notify_status_change(PipelineState.GENERATING)
            
            generation_result = await self._run_stage(
                'generation',
                self.stages['generation'].process,
                {
                    'query': query,
                    'search_results': self.current_search_results
                }
            )
            
            # Back to ready state
            self._notify_status_change(PipelineState.READY)
            
            return generation_result
            
        except Exception as e:
            logger.error(f"Search pipeline error: {e}")
            self._notify_status_change(PipelineState.ERROR)
            return StageResult(
                stage_name="search_pipeline",
                success=False,
                error=str(e)
            )
    
    async def _run_stage(
        self, 
        stage_name: str, 
        stage_func, 
        input_data: Any
    ) -> StageResult:
        """Run a single pipeline stage with error handling and timing"""
        start_time = time.time()
        
        try:
            console.print(f"[yellow]Running stage: {stage_name}")
            
            # Progress tracking hook
            def progress_hook(current: int, total: int):
                self._notify_progress(stage_name, current, total)
            
            # Data preview hook
            def data_hook(sample_data: Any):
                self._notify_data_update(stage_name, sample_data)
            
            # Run the stage with hooks
            if hasattr(stage_func, '__code__') and 'progress_hook' in stage_func.__code__.co_varnames:
                result = await stage_func(
                    input_data,
                    progress_hook=progress_hook,
                    data_hook=data_hook
                )
            else:
                result = await stage_func(input_data)
            
            processing_time = time.time() - start_time
            
            console.print(f"[green]✅ Stage {stage_name} completed in {processing_time:.2f}s")
            
            # Extract metadata for metrics
            metadata = {}
            if isinstance(result, dict):
                # For indexing stage, capture indexed_count
                if stage_name == 'indexing' and 'indexed_count' in result:
                    metadata['indexed_count'] = result['indexed_count']
                # For other stages, capture relevant metadata
                metadata.update({k: v for k, v in result.items() if k not in ['success', 'data']})
            
            return StageResult(
                stage_name=stage_name,
                success=True,
                data=result,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            console.print(f"[red]❌ Stage {stage_name} failed: {e}")
            
            return StageResult(
                stage_name=stage_name,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def _wait_for_user_continue(self, message: str):
        """Pause pipeline for step-by-step mode"""
        self._notify_status_change(PipelineState.PAUSED)
        console.print(f"[yellow]⏸️  {message} - Press Enter to continue...")
        # In real implementation, this would wait for UI input
        await asyncio.sleep(0.1)  # Placeholder
        
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status for UI"""
        return {
            'pipeline_id': self.pipeline_id,
            'state': self.state.value,
            'metrics': {
                'documents_processed': self.metrics.documents_processed,
                'chunks_created': self.metrics.chunks_created,
                'embeddings_generated': self.metrics.embeddings_generated,
                'vectors_indexed': self.metrics.vectors_indexed,
                'searches_performed': self.metrics.searches_performed,
                'total_processing_time': self.metrics.total_processing_time,
                'last_update': self.metrics.last_update
            },
            'config': {
                'embedding_model': self.config['embedding_models']['default'],
                'chunk_size': self.config['chunking']['chunk_size'],
                'collection_name': self.config['vectordb']['collection_name']
            }
        }
    
    def get_data_samples(self) -> Dict[str, Any]:
        """Get current data samples for UI previews"""
        return {
            'documents': self.current_documents[:3] if self.current_documents else [],
            'chunks': self.current_chunks[:5] if self.current_chunks else [],
            'embeddings': {
                'count': len(self.current_embeddings),
                'dimensions': len(self.current_embeddings[0]) if self.current_embeddings else 0,
                'sample_vector': self.current_embeddings[0][:10] if self.current_embeddings else []
            },
            'search_results': self.current_search_results[:3] if self.current_search_results else []
        }