"""
Agent Orchestrator - Multi-Agent RAG System

Demonstrates agent-to-agent (A2A) communication patterns for complex RAG tasks.
This is a foundation for implementing Google's A2A protocol when available.

Agents:
- SearchAgent: Handles document search and retrieval
- DocumentAgent: Manages document processing and indexing  
- AnalyticsAgent: Provides insights and statistics
- ExplainerAgent: Creates beginner-friendly explanations

Communication Pattern:
User Request → Orchestrator → Specialized Agents → Orchestrator → User Response
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.main_pipeline import VisualRAGPipeline
from models.embeddings import EmbeddingModelFactory, EmbeddingBenchmark

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    SEARCH = "search"
    DOCUMENT = "document"
    ANALYTICS = "analytics"
    EXPLAINER = "explainer"


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response" 
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    from_agent: str = ""
    to_agent: str = ""
    message_type: MessageType = MessageType.REQUEST
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp
        }


class AgentInterface(ABC):
    """Base interface for all agents"""
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.message_queue: List[AgentMessage] = []
        self.capabilities: List[str] = []
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and return response if needed"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        pass
    
    def create_message(
        self, 
        to_agent: str, 
        message_type: MessageType,
        content: Dict[str, Any]
    ) -> AgentMessage:
        """Create a new message"""
        return AgentMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content
        )


class SearchAgent(AgentInterface):
    """Agent specialized in document search and retrieval"""
    
    def __init__(self, pipeline: VisualRAGPipeline):
        super().__init__("search-agent", AgentRole.SEARCH)
        self.pipeline = pipeline
        self.capabilities = [
            "semantic_search",
            "hybrid_search", 
            "result_ranking",
            "similarity_scoring"
        ]
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process search-related messages"""
        try:
            action = message.content.get("action")
            
            if action == "search_documents":
                return await self._handle_search_request(message)
            elif action == "get_search_stats":
                return await self._handle_stats_request(message)
            else:
                return self.create_message(
                    message.from_agent,
                    MessageType.ERROR,
                    {"error": f"Unknown search action: {action}"}
                )
                
        except Exception as e:
            logger.error(f"SearchAgent error: {e}")
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": str(e)}
            )
    
    async def _handle_search_request(self, message: AgentMessage) -> AgentMessage:
        """Handle document search request"""
        query = message.content.get("query")
        top_k = message.content.get("top_k", 5)
        search_type = message.content.get("search_type", "dense")
        
        if not query:
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": "Query parameter required"}
            )
        
        # Check if system is ready
        status = self.pipeline.get_status()
        if status['metrics']['vectors_indexed'] == 0:
            return self.create_message(
                message.from_agent,
                MessageType.RESPONSE,
                {
                    "results": [],
                    "message": "No documents indexed yet",
                    "ready": False
                }
            )
        
        # Perform search
        result = await self.pipeline.search_and_generate(
            query=query,
            search_type=search_type,
            top_k=top_k
        )
        
        if result.success:
            return self.create_message(
                message.from_agent,
                MessageType.RESPONSE,
                {
                    "results": result.data['sources'],
                    "response": result.data['response'],
                    "confidence": result.data.get('confidence', 0.0),
                    "search_type": search_type,
                    "query": query
                }
            )
        else:
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": f"Search failed: {result.error}"}
            )
    
    async def _handle_stats_request(self, message: AgentMessage) -> AgentMessage:
        """Handle search statistics request"""
        status = self.pipeline.get_status()
        metrics = status['metrics']
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {
                "total_searches": metrics['searches_performed'],
                "indexed_documents": metrics['vectors_indexed'],
                "system_ready": metrics['vectors_indexed'] > 0
            }
        )
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities


class DocumentAgent(AgentInterface):
    """Agent specialized in document processing and management"""
    
    def __init__(self, pipeline: VisualRAGPipeline):
        super().__init__("document-agent", AgentRole.DOCUMENT)
        self.pipeline = pipeline
        self.capabilities = [
            "document_upload",
            "text_chunking",
            "embedding_generation",
            "document_analysis"
        ]
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process document-related messages"""
        try:
            action = message.content.get("action")
            
            if action == "upload_documents":
                return await self._handle_upload_request(message)
            elif action == "get_document_stats":
                return await self._handle_document_stats(message)
            elif action == "analyze_document":
                return await self._handle_document_analysis(message)
            else:
                return self.create_message(
                    message.from_agent,
                    MessageType.ERROR,
                    {"error": f"Unknown document action: {action}"}
                )
                
        except Exception as e:
            logger.error(f"DocumentAgent error: {e}")
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": str(e)}
            )
    
    async def _handle_upload_request(self, message: AgentMessage) -> AgentMessage:
        """Handle document upload request"""
        file_paths = message.content.get("file_paths", [])
        step_by_step = message.content.get("step_by_step", False)
        
        if not file_paths:
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": "No file paths provided"}
            )
        
        # Validate paths
        valid_paths = []
        for path_str in file_paths:
            path = Path(path_str)
            if path.exists() and path.is_file():
                valid_paths.append(path)
        
        if not valid_paths:
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": "No valid files found"}
            )
        
        # Process documents
        results = await self.pipeline.ingest_documents(valid_paths, step_by_step)
        
        if results and results[-1].success:
            status = self.pipeline.get_status()
            return self.create_message(
                message.from_agent,
                MessageType.RESPONSE,
                {
                    "success": True,
                    "documents_processed": status['metrics']['documents_processed'],
                    "chunks_created": status['metrics']['chunks_created'],
                    "embeddings_generated": status['metrics']['embeddings_generated'],
                    "processing_time": status['metrics']['total_processing_time']
                }
            )
        else:
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": "Document processing failed"}
            )
    
    async def _handle_document_stats(self, message: AgentMessage) -> AgentMessage:
        """Handle document statistics request"""
        status = self.pipeline.get_status()
        metrics = status['metrics']
        data_samples = self.pipeline.get_data_samples()
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {
                "total_documents": metrics['documents_processed'],
                "total_chunks": metrics['chunks_created'],
                "total_embeddings": metrics['embeddings_generated'],
                "recent_documents": [
                    {"title": doc.get('title', 'Untitled'), "size": doc.get('size', 0)}
                    for doc in data_samples.get('documents', [])
                ]
            }
        )
    
    async def _handle_document_analysis(self, message: AgentMessage) -> AgentMessage:
        """Analyze document characteristics"""
        data_samples = self.pipeline.get_data_samples()
        chunks = data_samples.get('chunks', [])
        
        if not chunks:
            return self.create_message(
                message.from_agent,
                MessageType.RESPONSE,
                {"analysis": "No documents to analyze"}
            )
        
        # Simple analysis
        chunk_sizes = [chunk.get('size', 0) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        analysis = {
            "total_chunks": len(chunks),
            "average_chunk_size": round(avg_size, 2),
            "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "document_types": list(set(
                chunk.get('metadata', {}).get('format', 'unknown') 
                for chunk in chunks
            ))
        }
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {"analysis": analysis}
        )
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities


class AnalyticsAgent(AgentInterface):
    """Agent specialized in system analytics and performance monitoring"""
    
    def __init__(self, pipeline: VisualRAGPipeline):
        super().__init__("analytics-agent", AgentRole.ANALYTICS)
        self.pipeline = pipeline
        self.capabilities = [
            "performance_monitoring",
            "usage_analytics",
            "model_comparison", 
            "system_diagnostics"
        ]
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process analytics-related messages"""
        try:
            action = message.content.get("action")
            
            if action == "get_performance_metrics":
                return await self._handle_performance_request(message)
            elif action == "benchmark_models":
                return await self._handle_benchmark_request(message)
            elif action == "system_health":
                return await self._handle_health_check(message)
            else:
                return self.create_message(
                    message.from_agent,
                    MessageType.ERROR,
                    {"error": f"Unknown analytics action: {action}"}
                )
                
        except Exception as e:
            logger.error(f"AnalyticsAgent error: {e}")
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": str(e)}
            )
    
    async def _handle_performance_request(self, message: AgentMessage) -> AgentMessage:
        """Handle performance metrics request"""
        status = self.pipeline.get_status()
        metrics = status['metrics']
        
        # Calculate performance indicators
        total_time = metrics['total_processing_time']
        docs_processed = metrics['documents_processed']
        
        performance = {
            "total_processing_time": total_time,
            "documents_processed": docs_processed,
            "chunks_created": metrics['chunks_created'],
            "embeddings_generated": metrics['embeddings_generated'],
            "searches_performed": metrics['searches_performed']
        }
        
        if total_time > 0 and docs_processed > 0:
            performance.update({
                "docs_per_second": round(docs_processed / total_time, 3),
                "chunks_per_doc": round(metrics['chunks_created'] / docs_processed, 1),
                "embeddings_per_second": round(metrics['embeddings_generated'] / total_time, 1)
            })
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {"performance": performance}
        )
    
    async def _handle_benchmark_request(self, message: AgentMessage) -> AgentMessage:
        """Handle model benchmarking request"""
        models_to_test = message.content.get("models")
        
        test_texts = [
            "Machine learning enables intelligent systems",
            "Natural language processing understands text",
            "Vector embeddings capture semantic meaning",
            "Information retrieval finds relevant documents",
            "Artificial intelligence transforms technology"
        ] * 4  # 20 texts total for quick benchmark
        
        benchmark = EmbeddingBenchmark(models_to_test)
        results = await benchmark.run_benchmark(test_texts, num_runs=2)
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {
                "benchmark_results": results,
                "fastest_model": benchmark.get_fastest_model(),
                "most_accurate_model": benchmark.get_most_accurate_model()
            }
        )
    
    async def _handle_health_check(self, message: AgentMessage) -> AgentMessage:
        """Handle system health check"""
        status = self.pipeline.get_status()
        metrics = status['metrics']
        
        health = {
            "system_status": "healthy" if metrics['vectors_indexed'] > 0 else "needs_documents",
            "pipeline_state": status['state'],
            "documents_ready": metrics['documents_processed'] > 0,
            "embeddings_ready": metrics['embeddings_generated'] > 0,
            "search_ready": metrics['vectors_indexed'] > 0,
            "last_activity": metrics['last_update']
        }
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {"health": health}
        )
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities


class ExplainerAgent(AgentInterface):
    """Agent specialized in creating beginner-friendly explanations"""
    
    def __init__(self):
        super().__init__("explainer-agent", AgentRole.EXPLAINER)
        self.capabilities = [
            "concept_explanation",
            "technical_translation",
            "tutorial_generation",
            "example_creation"
        ]
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process explanation requests"""
        try:
            action = message.content.get("action")
            
            if action == "explain_concept":
                return await self._handle_concept_explanation(message)
            elif action == "explain_embeddings":
                return await self._handle_embeddings_explanation(message)
            elif action == "explain_rag":
                return await self._handle_rag_explanation(message)
            else:
                return self.create_message(
                    message.from_agent,
                    MessageType.ERROR,
                    {"error": f"Unknown explanation request: {action}"}
                )
                
        except Exception as e:
            logger.error(f"ExplainerAgent error: {e}")
            return self.create_message(
                message.from_agent,
                MessageType.ERROR,
                {"error": str(e)}
            )
    
    async def _handle_concept_explanation(self, message: AgentMessage) -> AgentMessage:
        """Explain a technical concept in simple terms"""
        concept = message.content.get("concept", "").lower()
        
        explanations = {
            "embeddings": "Embeddings are like fingerprints for text - they turn words into numbers that capture meaning, so computers can understand that 'cat' and 'kitten' are similar.",
            
            "vector database": "A vector database is like a super-smart filing cabinet that stores these number-fingerprints and can instantly find similar documents, even if they use different words.",
            
            "semantic search": "Semantic search understands what you mean, not just the words you use. If you search for 'car', it also finds documents about 'automobile' and 'vehicle'.",
            
            "chunking": "Chunking is like cutting a long book into readable chapters. We split big documents into smaller pieces so the computer can better understand and find specific information.",
            
            "similarity score": "A similarity score is like a percentage showing how well two things match. 0.9 means very similar, 0.1 means very different.",
            
            "rag": "RAG (Retrieval-Augmented Generation) is like having a smart assistant that looks up information in your documents before answering questions, making sure the answers are based on your actual data."
        }
        
        explanation = explanations.get(concept, f"I don't have a simple explanation for '{concept}' yet. Could you ask about embeddings, vector database, semantic search, chunking, similarity score, or RAG?")
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {
                "concept": concept,
                "explanation": explanation,
                "beginner_friendly": True
            }
        )
    
    async def _handle_embeddings_explanation(self, message: AgentMessage) -> AgentMessage:
        """Detailed embedding explanation with example"""
        text_example = message.content.get("text_example", "The cat sat on the mat")
        
        explanation = {
            "what_are_embeddings": "Embeddings convert text into lists of numbers that represent meaning. Think of them as coordinates in a 'meaning space' - similar texts get similar coordinates.",
            
            "simple_analogy": "Imagine every word and sentence has a location in a huge city called 'Meaning City'. Words with similar meanings live in the same neighborhood. Embeddings are like the street addresses.",
            
            "your_example": f"For '{text_example}', an embedding model would create a list of numbers like [0.23, -0.45, 0.78, ...] that captures what this sentence means.",
            
            "why_useful": "These numbers let computers compare meanings mathematically. If two sentences have similar numbers, they probably mean similar things!",
            
            "search_connection": "When you search, we convert your question to numbers, then find documents with the most similar numbers. It's like finding houses in the same neighborhood of Meaning City."
        }
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {
                "explanation": explanation,
                "example_text": text_example,
                "complexity_level": "beginner"
            }
        )
    
    async def _handle_rag_explanation(self, message: AgentMessage) -> AgentMessage:
        """Explain how RAG works in simple terms"""
        explanation = {
            "what_is_rag": "RAG stands for Retrieval-Augmented Generation. It's like having a research assistant that looks up information before answering your questions.",
            
            "the_process": [
                "1. 📚 First, we read all your documents and remember them (like studying for a test)",
                "2. 🔍 When you ask a question, we search for the most relevant information", 
                "3. 💭 We use that information to create a smart, accurate answer",
                "4. 📝 We show you both the answer AND where it came from"
            ],
            
            "why_better": "Regular AI might make up answers or use outdated information. RAG always checks your actual documents first, so answers are accurate and up-to-date.",
            
            "analogy": "It's like the difference between answering from memory (might be wrong) versus looking it up in the book (accurate and current).",
            
            "benefits": [
                "✅ Answers based on YOUR documents",
                "✅ Shows sources so you can verify", 
                "✅ Always uses latest information",
                "✅ Reduces AI 'hallucination' (making things up)"
            ]
        }
        
        return self.create_message(
            message.from_agent,
            MessageType.RESPONSE,
            {
                "explanation": explanation,
                "topic": "RAG System",
                "audience": "beginners"
            }
        )
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities


class AgentOrchestrator:
    """Orchestrates communication between specialized agents"""
    
    def __init__(self, pipeline: VisualRAGPipeline):
        self.agent_id = "orchestrator"
        self.pipeline = pipeline
        
        # Initialize specialized agents
        self.agents: Dict[str, AgentInterface] = {
            "search": SearchAgent(pipeline),
            "document": DocumentAgent(pipeline),
            "analytics": AnalyticsAgent(pipeline),
            "explainer": ExplainerAgent()
        }
        
        # Message routing and history
        self.message_history: List[AgentMessage] = []
        self.active_conversations: Dict[str, List[AgentMessage]] = {}
        
        logger.info("Agent Orchestrator initialized with agents: " + ", ".join(self.agents.keys()))
    
    async def process_user_request(self, request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a user request by routing to appropriate agents"""
        conversation_id = str(uuid4())[:8]
        
        try:
            # Analyze request and determine which agents to involve
            agent_plan = self._analyze_request(request)
            
            response = {
                "conversation_id": conversation_id,
                "request": request,
                "agent_plan": agent_plan,
                "responses": {},
                "success": True
            }
            
            # Execute agent plan
            for agent_name, task in agent_plan.items():
                agent_response = await self._communicate_with_agent(
                    agent_name, 
                    task,
                    conversation_id
                )
                response["responses"][agent_name] = agent_response
            
            # Store conversation history
            self.active_conversations[conversation_id] = self.message_history[-10:]  # Keep last 10 messages
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return {
                "conversation_id": conversation_id,
                "request": request,
                "error": str(e),
                "success": False
            }
    
    def _analyze_request(self, request: str) -> Dict[str, Dict]:
        """Analyze user request and create execution plan"""
        request_lower = request.lower()
        plan = {}
        
        # Search-related requests
        if any(word in request_lower for word in ["search", "find", "look for", "query", "what", "how", "where"]):
            plan["search"] = {
                "action": "search_documents",
                "query": request,
                "top_k": 5
            }
        
        # Document-related requests  
        if any(word in request_lower for word in ["upload", "add", "process", "document", "file"]):
            plan["document"] = {
                "action": "get_document_stats"
            }
        
        # Analytics requests
        if any(word in request_lower for word in ["performance", "metrics", "benchmark", "stats", "analytics"]):
            plan["analytics"] = {
                "action": "get_performance_metrics"
            }
        
        # Explanation requests
        if any(word in request_lower for word in ["explain", "what is", "how does", "help me understand"]):
            # Determine what concept to explain
            if "embedding" in request_lower:
                plan["explainer"] = {
                    "action": "explain_embeddings",
                    "text_example": request
                }
            elif "rag" in request_lower:
                plan["explainer"] = {
                    "action": "explain_rag"
                }
            else:
                # Extract concept from request
                for concept in ["vector database", "semantic search", "chunking", "similarity"]:
                    if concept in request_lower:
                        plan["explainer"] = {
                            "action": "explain_concept",
                            "concept": concept
                        }
                        break
        
        # Default: if no specific plan, use search as fallback
        if not plan:
            plan["search"] = {
                "action": "search_documents", 
                "query": request,
                "top_k": 3
            }
        
        return plan
    
    async def _communicate_with_agent(
        self, 
        agent_name: str, 
        task: Dict,
        conversation_id: str
    ) -> Dict[str, Any]:
        """Send message to agent and get response"""
        
        if agent_name not in self.agents:
            return {"error": f"Unknown agent: {agent_name}"}
        
        agent = self.agents[agent_name]
        
        # Create message
        message = AgentMessage(
            from_agent=self.agent_id,
            to_agent=agent_name,
            message_type=MessageType.REQUEST,
            content=task
        )
        
        # Store in history
        self.message_history.append(message)
        
        # Process with agent
        response_message = await agent.process_message(message)
        
        if response_message:
            self.message_history.append(response_message)
            return {
                "agent": agent_name,
                "success": response_message.message_type != MessageType.ERROR,
                "content": response_message.content,
                "message_id": response_message.id
            }
        else:
            return {
                "agent": agent_name,
                "success": False,
                "error": "No response from agent"
            }
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents"""
        return {
            name: agent.get_capabilities() 
            for name, agent in self.agents.items()
        }
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        if conversation_id in self.active_conversations:
            return [msg.to_dict() for msg in self.active_conversations[conversation_id]]
        return []
    
    async def explain_system(self) -> Dict[str, Any]:
        """Generate a beginner-friendly explanation of the agent system"""
        explanation = {
            "what_are_agents": "Agents are like specialized team members, each expert in different tasks. They work together to help you with complex requests.",
            
            "our_team": {
                "🔍 Search Agent": "Finds relevant information in your documents",
                "📄 Document Agent": "Handles uploading and processing files",
                "📊 Analytics Agent": "Provides performance metrics and insights", 
                "🎓 Explainer Agent": "Makes technical concepts easy to understand",
                "🎯 Orchestrator": "Coordinates the team and routes your requests"
            },
            
            "how_it_works": [
                "1. You ask a question or make a request",
                "2. The Orchestrator analyzes what you need",
                "3. Appropriate agents are called to help",
                "4. Agents work on their specialized tasks",
                "5. The Orchestrator combines their responses",
                "6. You get a comprehensive answer!"
            ],
            
            "why_agents": "By splitting work among specialists, we can handle complex requests more effectively than a single system trying to do everything.",
            
            "agent_count": len(self.agents),
            "total_capabilities": sum(len(agent.get_capabilities()) for agent in self.agents.values())
        }
        
        return explanation


# Example usage and testing
async def demo_agent_system():
    """Demo the agent system with example requests"""
    from pipeline.main_pipeline import VisualRAGPipeline
    
    logger.info("Starting Agent System Demo")
    
    # Initialize pipeline and orchestrator
    pipeline = VisualRAGPipeline()
    orchestrator = AgentOrchestrator(pipeline)
    
    # Demo requests
    demo_requests = [
        "What are embeddings and how do they work?",
        "Show me the current system performance metrics",
        "Search for information about machine learning", 
        "Explain how RAG systems work for beginners"
    ]
    
    for request in demo_requests:
        logger.info(f"\nProcessing request: {request}")
        response = await orchestrator.process_user_request(request)
        logger.info(f"Response: {response}")
    
    # Show system explanation
    system_explanation = await orchestrator.explain_system()
    logger.info(f"\nSystem Explanation: {system_explanation}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_agent_system())