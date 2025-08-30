"""
MCP (Model Context Protocol) Server Implementation

This server exposes our RAG system as tools and resources that Claude and other
LLMs can use. It demonstrates real-world production patterns for MCP integration.

Features:
- Vector search tools
- Document upload tools  
- Collection management resources
- Real-time pipeline monitoring
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.types import (
        Resource,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource
    )
    MCP_AVAILABLE = True
except ImportError:
    # MCP not available, create placeholder classes
    MCP_AVAILABLE = False
    
    class Server:
        def __init__(self, name: str, version: str):
            self.name = name
            self.version = version
    
    class Resource:
        def __init__(self, **kwargs):
            pass
    
    class Tool:
        def __init__(self, **kwargs):
            pass

from pipeline.main_pipeline import VisualRAGPipeline
from models.embeddings import EmbeddingModelFactory

logger = logging.getLogger(__name__)

class RAGMCPServer:
    """MCP Server for the Visual RAG System"""
    
    def __init__(self):
        self.server = Server("visual-rag-system", "0.1.0")
        self.pipeline: Optional[VisualRAGPipeline] = None
        self.setup_handlers()
    
    def setup_handlers(self):
        """Set up MCP server handlers"""
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, server will run in demo mode")
            return
        
        # Set up tool handlers by assigning methods
        self.server.list_tools = self.list_tools
        self.server.call_tool = self.call_tool
        
        # Set up resource handlers
        self.server.list_resources = self.list_resources
        self.server.read_resource = self.read_resource
    
    async def initialize_server(self, arguments: InitializationOptions) -> Dict[str, Any]:
        """Initialize the RAG pipeline when MCP server starts"""
        try:
            logger.info("Initializing RAG pipeline for MCP server...")
            self.pipeline = VisualRAGPipeline()
            logger.info("RAG pipeline initialized successfully")
            
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "visual-rag-system",
                    "version": "0.1.0"
                }
            }
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def list_tools(self):
        """List available tools for the MCP client"""
        if not MCP_AVAILABLE:
            return []
        
        return [
            Tool(
                name="search_documents",
                description="Search through uploaded documents using semantic similarity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant documents"
                        },
                        "top_k": {
                            "type": "integer", 
                            "description": "Number of results to return (default: 5)",
                            "default": 5
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["dense", "hybrid", "mmr"],
                            "description": "Type of search to perform (default: dense)",
                            "default": "dense"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="upload_documents",
                description="Upload and process documents into the vector database",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to upload and process"
                        },
                        "step_by_step": {
                            "type": "boolean",
                            "description": "Whether to process in step-by-step mode (default: false)",
                            "default": False
                        }
                    },
                    "required": ["file_paths"]
                }
            ),
            Tool(
                name="get_pipeline_status", 
                description="Get current status and metrics of the RAG pipeline",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            ),
            Tool(
                name="benchmark_models",
                description="Benchmark different embedding models for performance comparison",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of model names to benchmark (default: all models)"
                        },
                        "test_texts": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Custom texts to use for benchmarking"
                        }
                    }
                }
            ),
            Tool(
                name="explain_embeddings",
                description="Generate a beginner-friendly explanation of how embeddings work",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text_example": {
                            "type": "string",
                            "description": "Example text to demonstrate embedding concepts"
                        },
                        "model": {
                            "type": "string", 
                            "description": "Embedding model to use for demonstration",
                            "default": "minilm"
                        }
                    }
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        """Handle tool calls from MCP clients"""
        
        if not self.pipeline:
            return [TextContent(
                type="text",
                text="❌ RAG pipeline not initialized. Please restart the server."
            )]
        
        try:
            if name == "search_documents":
                return await self._handle_search_documents(arguments)
            elif name == "upload_documents":
                return await self._handle_upload_documents(arguments)
            elif name == "get_pipeline_status":
                return await self._handle_get_pipeline_status(arguments)
            elif name == "benchmark_models":
                return await self._handle_benchmark_models(arguments)
            elif name == "explain_embeddings":
                return await self._handle_explain_embeddings(arguments)
            else:
                return [TextContent(
                    type="text",
                    text=f"❌ Unknown tool: {name}"
                )]
                
        except Exception as e:
            logger.error(f"Tool call error for {name}: {e}")
            return [TextContent(
                type="text", 
                text=f"❌ Error executing {name}: {str(e)}"
            )]
    
    async def _handle_search_documents(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle document search tool call"""
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        search_type = arguments.get("search_type", "dense")
        
        # Check if documents are indexed
        status = self.pipeline.get_status()
        if status['metrics']['vectors_indexed'] == 0:
            return [TextContent(
                type="text",
                text="⚠️ No documents have been uploaded and indexed yet. Please use the 'upload_documents' tool first."
            )]
        
        # Perform search
        result = await self.pipeline.search_and_generate(
            query=query,
            search_type=search_type, 
            top_k=top_k
        )
        
        if not result.success:
            return [TextContent(
                type="text",
                text=f"❌ Search failed: {result.error}"
            )]
        
        # Format response
        response_data = result.data
        response_text = f"## 🔍 Search Results for: '{query}'\n\n"
        response_text += f"**Generated Response:**\n{response_data['response']}\n\n"
        response_text += f"**Sources ({len(response_data['sources'])}):**\n"
        
        for i, source in enumerate(response_data['sources'], 1):
            response_text += f"{i}. **{source['title']}** (Score: {source['score']:.3f})\n"
            response_text += f"   Source: {source['source']}\n\n"
        
        response_text += f"**Search Details:**\n"
        response_text += f"- Search Type: {search_type}\n"
        response_text += f"- Results Requested: {top_k}\n"
        response_text += f"- Results Found: {len(response_data['sources'])}\n"
        response_text += f"- Average Score: {sum(s['score'] for s in response_data['sources']) / len(response_data['sources']):.3f}\n"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_upload_documents(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle document upload tool call"""
        file_paths = arguments["file_paths"]
        step_by_step = arguments.get("step_by_step", False)
        
        # Validate file paths
        valid_paths = []
        invalid_paths = []
        
        for path_str in file_paths:
            path = Path(path_str)
            if path.exists() and path.is_file():
                valid_paths.append(path)
            else:
                invalid_paths.append(path_str)
        
        if not valid_paths:
            return [TextContent(
                type="text",
                text=f"❌ No valid files found. Invalid paths: {invalid_paths}"
            )]
        
        # Process documents
        response_text = f"📄 Processing {len(valid_paths)} documents...\n\n"
        
        if invalid_paths:
            response_text += f"⚠️ Skipped invalid paths: {invalid_paths}\n\n"
        
        results = await self.pipeline.ingest_documents(
            valid_paths, 
            step_by_step=step_by_step
        )
        
        # Check results
        if results and results[-1].success:
            status = self.pipeline.get_status()
            metrics = status['metrics']
            
            response_text += "✅ Documents processed successfully!\n\n"
            response_text += "**Processing Summary:**\n"
            response_text += f"- Documents Processed: {metrics['documents_processed']}\n"
            response_text += f"- Chunks Created: {metrics['chunks_created']}\n"
            response_text += f"- Embeddings Generated: {metrics['embeddings_generated']}\n"
            response_text += f"- Vectors Indexed: {metrics['vectors_indexed']}\n"
            response_text += f"- Total Processing Time: {metrics['total_processing_time']:.2f}s\n\n"
            
            response_text += "**Stage Results:**\n"
            for result in results:
                status_icon = "✅" if result.success else "❌"
                response_text += f"{status_icon} {result.stage_name.title()}: "
                if result.success:
                    response_text += f"Completed in {result.processing_time:.2f}s\n"
                else:
                    response_text += f"Failed - {result.error}\n"
            
            response_text += f"\n🎉 Your documents are now ready for search!"
            
        else:
            response_text += "❌ Document processing failed!\n\n"
            if results:
                for result in results:
                    if not result.success:
                        response_text += f"❌ {result.stage_name}: {result.error}\n"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_get_pipeline_status(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle pipeline status tool call"""
        status = self.pipeline.get_status()
        metrics = status['metrics']
        config = status['config']
        
        response_text = f"## 📊 RAG Pipeline Status\n\n"
        response_text += f"**Current State:** {status['state']}\n"
        response_text += f"**Pipeline ID:** {status['pipeline_id']}\n\n"
        
        response_text += f"**Processing Metrics:**\n"
        response_text += f"- 📄 Documents Processed: {metrics['documents_processed']}\n"
        response_text += f"- ✂️ Chunks Created: {metrics['chunks_created']}\n"
        response_text += f"- 🔢 Embeddings Generated: {metrics['embeddings_generated']}\n"
        response_text += f"- 📊 Vectors Indexed: {metrics['vectors_indexed']}\n"
        response_text += f"- 🔍 Searches Performed: {metrics['searches_performed']}\n"
        response_text += f"- ⏱️ Total Processing Time: {metrics['total_processing_time']:.2f}s\n\n"
        
        response_text += f"**Configuration:**\n"
        response_text += f"- 🧠 Embedding Model: {config['embedding_model']}\n"
        response_text += f"- ✂️ Chunk Size: {config['chunk_size']}\n"
        response_text += f"- 🗄️ Collection Name: {config['collection_name']}\n\n"
        
        # System readiness
        if metrics['vectors_indexed'] > 0:
            response_text += "✅ **System Status:** Ready for search operations\n"
        elif metrics['documents_processed'] > 0:
            response_text += "🔄 **System Status:** Documents processed, indexing in progress\n"
        else:
            response_text += "⏸️ **System Status:** Waiting for documents to be uploaded\n"
        
        # Data samples preview
        data_samples = self.pipeline.get_data_samples()
        if data_samples['documents']:
            response_text += f"\n**Recent Documents:**\n"
            for doc in data_samples['documents']:
                response_text += f"- {doc.get('title', 'Untitled')} ({doc.get('size', 0)} chars)\n"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_benchmark_models(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle model benchmarking tool call"""
        from models.embeddings import EmbeddingBenchmark
        
        models_to_test = arguments.get("models")
        test_texts = arguments.get("test_texts", [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning enables computers to learn from data",
            "Vector embeddings represent text as numerical vectors",
            "Natural language processing helps understand human language",
            "Similarity search finds the most relevant documents"
        ])
        
        # Extend test texts for better benchmarking
        test_texts = test_texts * 10  # 50 texts total
        
        response_text = f"## 🏃‍♂️ Embedding Model Benchmark\n\n"
        response_text += f"Testing {len(test_texts)} texts across models...\n\n"
        
        benchmark = EmbeddingBenchmark(models_to_test)
        
        try:
            results = await benchmark.run_benchmark(test_texts, num_runs=3)
            
            response_text += "**Benchmark Results:**\n\n"
            
            # Sort by speed
            sorted_results = sorted(
                results.items(),
                key=lambda x: x[1].get('avg_time', float('inf'))
            )
            
            for model_name, metrics in sorted_results:
                if 'error' in metrics:
                    response_text += f"❌ **{model_name}**: {metrics['error']}\n\n"
                    continue
                
                response_text += f"🧠 **{model_name}**:\n"
                response_text += f"  - ⏱️ Average Time: {metrics['avg_time']:.3f}s\n"
                response_text += f"  - 🚄 Texts/Second: {metrics['texts_per_second']:.1f}\n"
                response_text += f"  - 📏 Dimensions: {metrics['dimensions']}\n"
                response_text += f"  - 📄 Max Length: {metrics['max_length']} tokens\n\n"
            
            # Recommendations
            fastest_model = benchmark.get_fastest_model()
            most_accurate = benchmark.get_most_accurate_model()
            
            response_text += "**Recommendations:**\n"
            response_text += f"🏆 **Fastest Model:** {fastest_model}\n"
            response_text += f"🎯 **Most Accurate:** {most_accurate}\n\n"
            
            response_text += "**Selection Guide:**\n"
            response_text += "- For speed-critical apps: Choose the fastest model\n"
            response_text += "- For accuracy-critical apps: Choose BGE-M3 or Nomic v2\n" 
            response_text += "- For balanced performance: Choose E5-base\n"
            response_text += "- For multilingual content: Choose BGE-M3 or Nomic v2\n"
            
        except Exception as e:
            response_text += f"❌ Benchmark failed: {str(e)}\n"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_explain_embeddings(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle embedding explanation tool call"""
        text_example = arguments.get("text_example", "The quick brown fox jumps over the lazy dog")
        model_name = arguments.get("model", "minilm")
        
        response_text = f"## 🔢 Understanding Embeddings\n\n"
        response_text += f"Let me explain embeddings using the example: *\"{text_example}\"*\n\n"
        
        try:
            # Create model and generate embedding
            model = EmbeddingModelFactory.create_model(model_name)
            embedding = model.encode(text_example)
            
            response_text += f"### What are embeddings?\n"
            response_text += f"Embeddings are like \"fingerprints\" for text - they capture meaning as numbers.\n\n"
            
            response_text += f"### Your Example:\n"
            response_text += f"**Text:** \"{text_example}\"\n"
            response_text += f"**Model:** {model_name}\n"
            response_text += f"**Embedding Size:** {len(embedding)} numbers\n\n"
            
            response_text += f"**First 10 numbers:**\n"
            for i, value in enumerate(embedding[:10]):
                response_text += f"  {i+1:2d}. {value:.6f}\n"
            response_text += f"  ... and {len(embedding)-10} more numbers\n\n"
            
            response_text += f"### What do these numbers mean?\n"
            response_text += f"Each number represents how much your text relates to different concepts:\n"
            response_text += f"- **Positive numbers**: Strong positive association with a concept\n"
            response_text += f"- **Negative numbers**: Strong negative association  \n"
            response_text += f"- **Near zero**: Weak or no association\n\n"
            
            response_text += f"### How similarity works:\n"
            response_text += f"Similar texts get similar patterns of numbers. For example:\n"
            response_text += f"- \"The fast brown fox\" → Very similar numbers\n"
            response_text += f"- \"A quick animal runs\" → Somewhat similar numbers\n"
            response_text += f"- \"Blue sky and clouds\" → Very different numbers\n\n"
            
            response_text += f"### Why this matters for search:\n"
            response_text += f"When you search, we:\n"
            response_text += f"1. Convert your question to numbers (embedding)\n"
            response_text += f"2. Find documents with the most similar numbers\n"
            response_text += f"3. Return the best matches\n\n"
            
            # Model-specific information
            model_config = EmbeddingModelFactory.get_model_info(model_name)
            response_text += f"### About the {model_name} model:\n"
            response_text += f"- **Dimensions:** {model_config['dimensions']}\n"
            response_text += f"- **Max text length:** {model_config['max_length']} tokens\n"
            
            if model_config.get('supports_multilingual'):
                response_text += f"- **Languages:** Supports 100+ languages\n"
            else:
                response_text += f"- **Languages:** Primarily English\n"
            
            if model_config.get('speed_optimized'):
                response_text += f"- **Speed:** Optimized for fast inference\n"
                
        except Exception as e:
            response_text += f"❌ Failed to generate example: {str(e)}\n"
        
        return [TextContent(type="text", text=response_text)]
    
    def list_resources(self):
        """List available resources for MCP clients"""
        if not MCP_AVAILABLE:
            return []
        
        resources = [
            Resource(
                uri="rag://pipeline/status",
                name="Pipeline Status",
                description="Real-time status of the RAG pipeline",
                mimeType="application/json"
            ),
            Resource(
                uri="rag://models/available",
                name="Available Models", 
                description="List of available embedding models",
                mimeType="application/json"
            ),
            Resource(
                uri="rag://documents/summary",
                name="Document Summary",
                description="Summary of processed documents",
                mimeType="application/json"
            ),
            Resource(
                uri="rag://config/current",
                name="Current Configuration",
                description="Current pipeline configuration", 
                mimeType="application/json"
            )
        ]
        
        return resources
    
    def read_resource(self, uri: str):
        """Read resource content"""
        if not self.pipeline:
            return json.dumps({"error": "Pipeline not initialized"})
        
        try:
            if uri == "rag://pipeline/status":
                return json.dumps(self.pipeline.get_status(), indent=2)
            
            elif uri == "rag://models/available":
                models = EmbeddingModelFactory.list_models()
                return json.dumps({
                    "available_models": models,
                    "model_count": len(models),
                    "default_model": "minilm"
                }, indent=2)
            
            elif uri == "rag://documents/summary":
                data_samples = self.pipeline.get_data_samples()
                summary = {
                    "document_count": len(data_samples.get('documents', [])),
                    "chunk_count": len(data_samples.get('chunks', [])),
                    "embedding_info": data_samples.get('embeddings', {}),
                    "recent_searches": len(data_samples.get('search_results', []))
                }
                return json.dumps(summary, indent=2)
            
            elif uri == "rag://config/current":
                status = self.pipeline.get_status()
                return json.dumps(status['config'], indent=2)
            
            else:
                return json.dumps({"error": f"Unknown resource: {uri}"})
                
        except Exception as e:
            return json.dumps({"error": f"Failed to read resource: {str(e)}"})
    
    async def run(self):
        """Run the MCP server"""
        if not MCP_AVAILABLE:
            logger.error("MCP library not available. Install with: pip install mcp")
            logger.info("Running in demo mode - server functionality limited")
            
            # Keep server alive for demonstration
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Server shutdown")
            return
        
        try:
            from mcp.server.stdio import stdio_server
            
            logger.info("Starting RAG MCP Server...")
            logger.info("Server will be available for MCP client connections")
            
            # Use the correct MCP server startup pattern
            async with stdio_server() as (read_stream, write_stream):
                logger.info("RAG MCP Server running on stdio")
                
                # Initialize the server with the correct initialization options
                from mcp.server.models import InitializationOptions
                
                # Create proper initialization options
                init_options = InitializationOptions(
                    server_name="visual-rag-system",
                    server_version="0.1.0",
                    capabilities={}
                )
                
                # Initialize the RAG pipeline
                try:
                    logger.info("Initializing RAG pipeline for MCP server...")
                    self.pipeline = VisualRAGPipeline()
                    logger.info("RAG pipeline initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize RAG pipeline: {e}")
                    # Continue anyway for demo purposes
                
                # Run the server with the streams
                await self.server.run(
                    read_stream, 
                    write_stream, 
                    init_options
                )
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            logger.info("RAG MCP Server shutting down")


async def main():
    """Main server entry point"""
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Visual RAG MCP Server...")
    
    server = RAGMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())