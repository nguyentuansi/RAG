"""
Visual RAG Pipeline Dashboard - Main Streamlit Application

This is the heart of our beginner-friendly RAG system visualization.
Users can see every step of the pipeline in real-time!
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our pipeline components with fallback handling
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from pipeline.main_pipeline import VisualRAGPipeline, PipelineState
    from models.embeddings import EmbeddingModelFactory
    from visualizer.components.pipeline_flow import PipelineFlowDiagram
    from visualizer.components.stage_monitor import StageMonitor
    from visualizer.components.data_preview import DataPreview
    from visualizer.components.vectordb_inspector import render_vectordb_inspector
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    PIPELINE_ERROR = str(e)
    
    # Create placeholder classes
    class VisualRAGPipeline:
        def __init__(self):
            pass
    
    class PipelineState:
        pass
    
    class EmbeddingModelFactory:
        @staticmethod
        def list_models():
            return {
                "bge-m3": {"dimensions": 1024, "model_path": "BAAI/bge-m3"},
                "nomic-v2": {"dimensions": 768, "model_path": "nomic-ai/nomic-embed-text-v2"},
                "e5-base": {"dimensions": 768, "model_path": "intfloat/e5-base-v2"},
                "minilm": {"dimensions": 384, "model_path": "sentence-transformers/all-MiniLM-L6-v2"}
            }
    
    class PipelineFlowDiagram:
        def render(self, status):
            st.error("Pipeline components not available. Install dependencies to see flow diagram.")
    
    class StageMonitor:
        def render(self, status):
            st.error("Pipeline components not available. Install dependencies to see stage monitor.")
    
    class DataPreview:
        def render(self, data):
            st.error("Pipeline components not available. Install dependencies to see data preview.")
    
    def render_vectordb_inspector(config=None, pipeline_status=None):
        st.error("Vector database inspector not available. Install dependencies to enable vector database inspection.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Visual RAG System Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .pipeline-stage {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        background-color: #f9f9f9;
    }
    
    .stage-active {
        border-color: #ff6b6b;
        background-color: #fff5f5;
    }
    
    .stage-completed {
        border-color: #51cf66;
        background-color: #f3fff3;
    }
    
    .stage-idle {
        border-color: #868e96;
        background-color: #f8f9fa;
    }
    
    .metrics-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .model-info-card {
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


class RAGDashboard:
    """Main dashboard controller for the Visual RAG System"""
    
    def __init__(self):
        self.initialize_session_state()
        
    @property
    def pipeline(self):
        """Get pipeline from session state"""
        return st.session_state.get('pipeline_instance', None)
        
    @pipeline.setter
    def pipeline(self, value):
        """Store pipeline in session state"""
        st.session_state.pipeline_instance = value
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'pipeline_initialized' not in st.session_state:
            st.session_state.pipeline_initialized = False
            st.session_state.pipeline_instance = None
            st.session_state.pipeline_status = {}
            st.session_state.pipeline_data_samples = {}
            st.session_state.data_previews = {}
            st.session_state.processing_logs = []
            st.session_state.current_embeddings = []
            st.session_state.search_history = []
            st.session_state.uploaded_files = []
    
    def run(self):
        """Main dashboard application"""
        st.title("🔍 Visual RAG System Demo")
        st.markdown("*Learn how RAG works by watching your data flow through each stage!*")
        
        # Show system status banner
        if not PIPELINE_AVAILABLE:
            st.warning("""
            ⚠️ **Demo Mode**: You're viewing the dashboard structure, but pipeline functionality is limited.
            """)
            
            with st.expander("🚀 **Click here to enable full functionality**", expanded=False):
                st.markdown("""
                ### 📦 Installation Steps
                
                **Option 1: Quick Install (Recommended)**
                ```bash
                # Install core dependencies
                pip install pydantic sentence-transformers torch plotly
                
                # Then restart the dashboard
                python3 run.py --mode visual
                ```
                
                **Option 2: Full Installation**
                ```bash
                # Install everything including vector database
                pip install -e .
                docker run -d -p 6333:6333 qdrant/qdrant:v1.12.5
                python3 run.py --mode dev
                ```
                
                ### ✅ What You'll Get
                - 📄 **Document Processing**: Upload PDFs, text files, markdown
                - 🔢 **Embedding Generation**: 4 open-source models (BGE, Nomic, E5, MiniLM)  
                - 🔍 **Semantic Search**: Find information using natural language
                - 📊 **Vector Database**: Store and index your documents
                - 🎯 **Interactive Pipeline**: Watch data transform in real-time
                
                ### 🐛 Current Error
                ```
                {PIPELINE_ERROR}
                ```
                """)
        else:
            st.success("✅ **Full Mode**: All pipeline components are available!")
        
        # Sidebar for configuration
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Pipeline Flow", 
            "📄 Document Upload", 
            "🔍 Search & Query",
            "🗂️ Vector Database",
            "📈 Analytics",
            "🎓 Learn RAG"
        ])
        
        with tab1:
            self.render_pipeline_flow()
        
        with tab2:
            self.render_document_upload()
            
        with tab3:
            self.render_search_interface()
            
        with tab4:
            self.render_vectordb_inspector()
            
        with tab5:
            self.render_analytics()
            
        with tab6:
            self.render_learning_section()
            
        # Add demo content section when pipeline not available
        if not PIPELINE_AVAILABLE:
            st.markdown("---")
            st.subheader("📚 Demo Content (Works without full pipeline)")
            
            demo_tab1, demo_tab2, demo_tab3 = st.tabs([
                "🎯 System Overview",
                "📊 Model Comparison", 
                "🔧 Architecture"
            ])
            
            with demo_tab1:
                self.render_system_overview()
            
            with demo_tab2:
                self.render_model_comparison()
            
            with demo_tab3:
                self.render_architecture_demo()
    
    def render_sidebar(self):
        """Render the configuration sidebar"""
        st.sidebar.title("⚙️ Configuration")
        
        # Model Selection
        st.sidebar.subheader("🧠 Embedding Model")
        
        available_models = EmbeddingModelFactory.list_models()
        model_options = list(available_models.keys())
        model_descriptions = {
            name: f"{config['dimensions']}D - {config.get('model_path', '').split('/')[-1]}"
            for name, config in available_models.items()
        }
        
        selected_model = st.sidebar.selectbox(
            "Choose Model:",
            options=model_options,
            format_func=lambda x: f"{x} ({model_descriptions[x]})",
            help="Different models offer tradeoffs between speed and accuracy"
        )
        
        # Show model info
        if selected_model:
            model_config = available_models[selected_model]
            st.sidebar.markdown(f"""
            <div class="model-info-card">
                <strong>📋 Model Info:</strong><br>
                📏 Dimensions: {model_config['dimensions']}<br>
                📄 Max Length: {model_config['max_length']} tokens<br>
                🌍 Multilingual: {'✅' if model_config.get('supports_multilingual', False) else '❌'}
            </div>
            """, unsafe_allow_html=True)
        
        # Pipeline Settings
        st.sidebar.subheader("⚙️ Pipeline Settings")
        
        chunk_size = st.sidebar.slider(
            "Chunk Size (tokens)", 
            min_value=128, 
            max_value=2048, 
            value=512,
            help="Smaller chunks = more precise, larger chunks = more context"
        )
        
        chunk_overlap = st.sidebar.slider(
            "Chunk Overlap", 
            min_value=0, 
            max_value=200, 
            value=50,
            help="How much chunks should overlap to maintain context"
        )
        
        top_k = st.sidebar.slider(
            "Search Results (Top K)", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="Number of similar chunks to retrieve"
        )
        
        # Distance Metric Selection
        distance_options = {
            'cosine': {
                'name': 'Cosine Similarity',
                'description': 'Measures angle between vectors (0-1 scale). Best for semantic similarity.',
                'emoji': '📐'
            },
            'euclidean': {
                'name': 'Euclidean Distance', 
                'description': 'Straight-line distance in space. Best for magnitude-sensitive comparisons.',
                'emoji': '📏'
            }
        }
        
        distance_metric = st.sidebar.selectbox(
            "📊 Similarity Method:",
            options=list(distance_options.keys()),
            format_func=lambda x: f"{distance_options[x]['emoji']} {distance_options[x]['name']}",
            help="Choose how to measure similarity between text chunks"
        )
        
        # Show distance metric info
        metric_info = distance_options[distance_metric]
        st.sidebar.caption(f"💡 {metric_info['description']}")
        
        # Initialize Pipeline Button
        if st.sidebar.button("🚀 Initialize Pipeline", type="primary"):
            self.initialize_pipeline(selected_model, chunk_size, chunk_overlap, distance_metric)
        
        # Pipeline Status
        if st.session_state.pipeline_initialized:
            st.sidebar.success("✅ Pipeline Ready!")
            
            # Database Status
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient("localhost", port=6333)
                info = client.get_collection('rag_documents')
                st.sidebar.success(f"🗂️ DB: {info.points_count} vectors")
            except:
                st.sidebar.error("🗂️ DB: Offline")
            if st.session_state.pipeline_status:
                status = st.session_state.pipeline_status
                st.sidebar.markdown(f"""
                <div class="metrics-card">
                    <strong>📊 Pipeline Stats:</strong><br>
                    📄 Documents: {status.get('metrics', {}).get('documents_processed', 0)}<br>
                    ✂️ Chunks: {status.get('metrics', {}).get('chunks_created', 0)}<br>
                    🔢 Embeddings: {status.get('metrics', {}).get('embeddings_generated', 0)}<br>
                    🔍 Searches: {status.get('metrics', {}).get('searches_performed', 0)}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.sidebar.info("👆 Initialize pipeline to get started")
    
    def initialize_pipeline(self, model_name: str, chunk_size: int, chunk_overlap: int, distance_metric: str = 'cosine'):
        """Initialize the RAG pipeline with selected configuration"""
        if not PIPELINE_AVAILABLE:
            st.error(f"❌ Cannot initialize pipeline: {PIPELINE_ERROR}")
            st.info("💡 **To enable full pipeline functionality, install dependencies:**")
            st.code("pip install pydantic sentence-transformers torch", language="bash")
            return
        
        try:
            with st.spinner("Initializing pipeline..."):
                # Create pipeline with custom config
                config_path = Path(__file__).parent.parent / "pipeline" / "config.yaml"
                
                # Load and modify config
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                config['embedding_models']['default'] = model_name
                config['chunking']['chunk_size'] = chunk_size
                config['chunking']['overlap'] = chunk_overlap
                config['vectordb']['distance_metric'] = distance_metric
                
                # Initialize pipeline
                st.info(f"Creating pipeline with config: {config_path}")
                self.pipeline = VisualRAGPipeline(config_path)
                
                # Verify pipeline was created
                if self.pipeline is None:
                    raise Exception("Pipeline object is None after creation")
                
                st.info("Pipeline object created, setting up callbacks...")
                
                # Set up callbacks for real-time updates
                self.pipeline.add_status_callback(self.update_pipeline_status)
                self.pipeline.add_progress_callback(self.update_progress)
                self.pipeline.add_data_callback(self.update_data_preview)
                
                # Get initial status
                initial_status = self.pipeline.get_status()
                st.info(f"Initial pipeline status: {initial_status}")
                
                st.session_state.pipeline_initialized = True
                st.session_state.pipeline_status = initial_status
                
                st.success(f"✅ Pipeline initialized with {model_name}!")
                st.success(f"Pipeline ID: {self.pipeline.pipeline_id if hasattr(self.pipeline, 'pipeline_id') else 'Unknown'}")
                
        except Exception as e:
            st.error(f"❌ Error initializing pipeline: {e}")
            logger.error(f"Pipeline initialization error: {e}")
            
            # Show helpful error messages
            if "No module named" in str(e):
                missing_module = str(e).split("'")[1]
                st.info(f"💡 **Missing dependency: {missing_module}**")
                st.code(f"pip install {missing_module}", language="bash")
    
    def update_pipeline_status(self, state: PipelineState, metrics):
        """Callback for pipeline status updates"""
        st.session_state.pipeline_status = {
            'state': state.value,
            'metrics': {
                'documents_processed': metrics.documents_processed,
                'chunks_created': metrics.chunks_created,
                'embeddings_generated': metrics.embeddings_generated,
                'vectors_indexed': metrics.vectors_indexed,
                'searches_performed': metrics.searches_performed,
                'total_processing_time': metrics.total_processing_time,
                'last_update': metrics.last_update
            }
        }
    
    def update_progress(self, stage: str, current: int, total: int):
        """Callback for progress updates"""
        progress = current / total if total > 0 else 0
        st.session_state[f"{stage}_progress"] = progress
    
    def update_data_preview(self, stage: str, data_sample):
        """Callback for data preview updates"""
        if 'data_previews' not in st.session_state:
            st.session_state.data_previews = {}
        st.session_state.data_previews[stage] = data_sample
        
        # Also update the pipeline data samples in session state
        if self.pipeline:
            try:
                pipeline_data = self.pipeline.get_data_samples()
                st.session_state.pipeline_data_samples = pipeline_data
            except Exception as e:
                logger.error(f"Error updating pipeline data samples: {e}")
    
    def render_pipeline_flow(self):
        """Render the main pipeline flow visualization"""
        st.subheader("🔄 RAG Pipeline Flow")
        st.markdown("*Watch your documents transform through each stage of the RAG pipeline*")
        
        if not st.session_state.pipeline_initialized:
            st.info("👈 Initialize the pipeline from the sidebar to see the flow diagram")
            return
        
        # Pipeline flow diagram
        flow_diagram = PipelineFlowDiagram()
        flow_diagram.render(st.session_state.pipeline_status)
        
        # Stage details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Stage Monitor")
            stage_monitor = StageMonitor()
            stage_monitor.render(st.session_state.pipeline_status)
        
        with col2:
            st.subheader("👀 Data Preview")
            
            # Add refresh button for data samples
            if st.button("🔄 Refresh Data", help="Update data preview with latest pipeline data"):
                if self.pipeline:
                    try:
                        fresh_data = self.pipeline.get_data_samples()
                        st.session_state.pipeline_data_samples = fresh_data
                        st.success("✅ Data refreshed!")
                    except Exception as e:
                        st.error(f"Error refreshing data: {e}")
            
            data_preview = DataPreview()
            # Get data samples from session state instead of pipeline directly
            data_samples = st.session_state.get('pipeline_data_samples', {})
            if self.pipeline and not data_samples:
                # Fallback to pipeline if session state is empty
                try:
                    data_samples = self.pipeline.get_data_samples()
                    st.session_state.pipeline_data_samples = data_samples
                except Exception as e:
                    st.error(f"Error getting data samples: {e}")
                    data_samples = {}
            data_preview.render(data_samples)
    
    def render_document_upload(self):
        """Render document upload interface"""
        st.subheader("📄 Document Upload & Processing")
        st.markdown("*Upload documents to build your knowledge base*")
        
        if not st.session_state.pipeline_initialized:
            st.warning("⚠️ Please initialize the pipeline first")
            return
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'md', 'pdf', 'json'],
            accept_multiple_files=True,
            help="Supported formats: TXT, Markdown, PDF, JSON"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            step_by_step = st.checkbox("Step-by-step mode", help="Pause after each stage to see results")
        
        with col2:
            process_button = st.button("🚀 Process Documents", type="primary")
        
        # Process documents
        if process_button and uploaded_files:
            self.process_uploaded_documents(uploaded_files, step_by_step)
        
        # Show uploaded files
        if uploaded_files:
            st.subheader("📋 Uploaded Files")
            file_data = []
            for file in uploaded_files:
                file_data.append({
                    "Name": file.name,
                    "Type": file.type,
                    "Size": f"{file.size / 1024:.1f} KB"
                })
            
            df = pd.DataFrame(file_data)
            st.dataframe(df, use_container_width=True)
    
    def process_uploaded_documents(self, uploaded_files, step_by_step: bool):
        """Process uploaded documents through the pipeline"""
        # Debug information
        st.info(f"Debug: PIPELINE_AVAILABLE = {PIPELINE_AVAILABLE}")
        st.info(f"Debug: self.pipeline = {self.pipeline}")
        st.info(f"Debug: pipeline_initialized = {st.session_state.get('pipeline_initialized', False)}")
        st.info(f"Debug: pipeline_instance in session_state = {st.session_state.get('pipeline_instance', 'Not found')}")
        
        if not PIPELINE_AVAILABLE:
            st.error("❌ Cannot process documents: Pipeline modules not available")
            st.info("💡 Install missing dependencies first")
            return
            
        if not self.pipeline:
            st.error("❌ Cannot process documents: Pipeline object is None")
            st.info("💡 Try clicking 'Initialize Pipeline' button first")
            return
            
        if not hasattr(self.pipeline, 'ingest_documents'):
            st.error("❌ Cannot process documents: Pipeline missing ingest_documents method")
            return
        
        try:
            # Save uploaded files temporarily
            temp_dir = Path("/tmp/rag_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_paths.append(file_path)
            
            # Process with visual feedback
            with st.container():
                st.info("🔄 Processing documents through pipeline...")
                
                # Create placeholders for real-time updates
                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                # Run processing
                async def process_docs():
                    results = await self.pipeline.ingest_documents(
                        file_paths, 
                        step_by_step=step_by_step
                    )
                    return results
                
                # Run async processing
                results = asyncio.run(process_docs())
                
                # Show results
                if results and results[-1].success:
                    st.success("✅ Documents processed successfully!")
                    
                    # Update session state with latest pipeline data
                    st.session_state.pipeline_status = self.pipeline.get_status()
                    st.session_state.pipeline_data_samples = self.pipeline.get_data_samples()
                    st.session_state.uploaded_files.extend([f.name for f in uploaded_files])
                    
                    st.info("🔄 Data synchronized with dashboard components")
                    
                    # Force auto-refresh of the page to update all components
                    st.rerun()
                    
                    # Show processing summary
                    self.show_processing_summary(results)
                else:
                    st.error("❌ Document processing failed")
                    if results:
                        st.error(f"Error: {results[-1].error}")
                        
        except Exception as e:
            st.error(f"❌ Error processing documents: {e}")
            logger.error(f"Document processing error: {e}")
    
    def show_processing_summary(self, results: List):
        """Show summary of document processing results"""
        st.subheader("📋 Processing Summary")
        
        summary_data = []
        for result in results:
            summary_data.append({
                "Stage": result.stage_name.title(),
                "Status": "✅ Success" if result.success else "❌ Failed",
                "Time": f"{result.processing_time:.2f}s",
                "Details": result.error if result.error else "Completed"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
    
    def render_search_interface(self):
        """Render search and query interface"""
        st.subheader("🔍 Search & Query")
        st.markdown("*Search through your processed documents*")
        
        if not st.session_state.pipeline_initialized:
            st.warning("⚠️ Please initialize the pipeline and process some documents first")
            return
        
        # Check if documents are processed
        status = st.session_state.pipeline_status
        vectors_indexed = status.get('metrics', {}).get('vectors_indexed', 0)
        if vectors_indexed == 0:
            st.info("📄 Please upload and process some documents first")
            return
            
        # Show detailed indexing status
        st.subheader("📊 Vector Database Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📄 Documents Processed", 
                status.get('metrics', {}).get('documents_processed', 0),
                help="Total documents loaded and processed"
            )
        
        with col2:
            st.metric(
                "✂️ Chunks Created", 
                status.get('metrics', {}).get('chunks_created', 0),
                help="Text chunks ready for embedding"
            )
            
        with col3:
            embeddings_count = status.get('metrics', {}).get('embeddings_generated', 0)
            st.metric(
                "🔢 Embeddings Generated", 
                embeddings_count,
                help="Vector embeddings created from text chunks"
            )
        
        # Indexing status with detailed explanation
        st.markdown("### 🗂️ Vector Indexing Status")
        
        if vectors_indexed > 0:
            # Calculate indexing efficiency
            embedding_ratio = (vectors_indexed / embeddings_count * 100) if embeddings_count > 0 else 0
            
            if embedding_ratio >= 100:
                st.success(f"✅ **Fully Indexed**: {vectors_indexed} vectors indexed and ready for fast search")
                st.info("🚀 **Search Mode**: Optimized ANN (Approximate Nearest Neighbor) search available")
            elif embedding_ratio > 0:
                st.warning(f"⚠️ **Partially Indexed**: {vectors_indexed}/{embeddings_count} vectors indexed ({embedding_ratio:.1f}%)")
                st.info("🐌 **Search Mode**: May fall back to slower linear search for non-indexed vectors")
            else:
                st.error("❌ **Indexing Error**: Embeddings exist but indexing failed")
        else:
            st.error("❌ **Not Indexed**: No vectors in searchable index")
            st.info("📝 **Next Steps**: Upload documents and complete the full processing pipeline")
            
        # Show what indexing means
        with st.expander("🤔 What does 'Indexed' mean?"):
            st.markdown("""
            **Non-Indexed Vectors** (Slow):
            - Vectors stored as raw arrays
            - Search requires checking EVERY vector one by one
            - Time: Linear O(n) - gets slower as you add more documents
            - Like searching a book page by page
            
            **Indexed Vectors** (Fast):
            - Vectors organized in HNSW (Hierarchical Navigable Small World) index
            - Search jumps directly to similar vectors
            - Time: Logarithmic O(log n) - stays fast even with millions of vectors  
            - Like using a book's index to find exactly what you need
            
            **Your Database Uses**: Qdrant with HNSW indexing for optimal search performance
            """)
            
        # Show indexing status with database details
        if vectors_indexed > 0:
            st.info(f"✅ Ready to search {vectors_indexed} indexed vectors")
            
            # Quick database status
            with st.expander("🗂️ Quick Database Status"):
                try:
                    from qdrant_client import QdrantClient
                    client = QdrantClient("localhost", port=6333)
                    info = client.get_collection('rag_documents')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Total Chunks", info.points_count)
                    with col2:
                        st.metric("🔢 Vector Dimensions", info.config.params.vectors.size)
                    with col3:
                        approx_size = (info.points_count * info.config.params.vectors.size * 4) / (1024 * 1024)
                        st.metric("💾 Index Size", f"{approx_size:.1f} MB")
                    
                    st.info("💡 Go to 'Vector Database' tab for detailed analysis")
                    
                    # Quick actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 View Distribution", key="view_dist"):
                            st.info("👆 Click on 'Vector Database' tab above to see data distribution charts")
                    with col2:
                        if st.button("🔍 Find Missing Files", key="find_missing"):
                            st.info("👆 Click on 'Vector Database' tab and go to 'Gap Analysis' to find missing files")
                    
                except Exception as e:
                    st.warning("⚠️ Cannot get database details. Vector Database tab has full diagnostics.")
        else:
            st.info(f"⚠️ No vectors indexed yet. Process some documents first.")
        
        # Add database inspection button for troubleshooting
        if st.button("🔧 Database Diagnostics", help="Run database health check"):
            self.inspect_qdrant_status()
        
        # Search interface with tips
        query = st.text_input(
            "Ask a question:",
            placeholder="What would you like to know?",
            help="Ask anything about your uploaded documents"
        )
        
        # Search tips
        with st.expander("💡 Tips for Better Search Results"):
            st.markdown("""
            **🎯 Query Tips:**
            - Use specific keywords from your documents
            - Ask complete questions: "What is RAG?" instead of just "RAG"
            - Be descriptive: "How does vector similarity work?" 
            
            **📊 Score Guide:**
            - 🟢 **0.8+**: Excellent match - highly relevant content
            - 🟡 **0.6-0.8**: Good match - relevant but may need refinement  
            - 🔴 **<0.6**: Poor match - try different keywords
            
            **🚀 Pro Tips:**
            - Try different phrasings if results are poor
            - Use domain-specific terms when available
            - Longer, more specific queries often work better
            """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_type = st.selectbox(
                "Search Type:", 
                ["dense", "hybrid", "mmr"],
                help="Different search strategies"
            )
        
        with col2:
            top_k = st.slider("Results to show:", 1, 15, 5)
        
        with col3:
            search_button = st.button("🔍 Search", type="primary")
        
        # Perform search
        if search_button and query:
            self.perform_search(query, search_type, top_k)
        
        # Show search history
        if st.session_state.search_history:
            st.subheader("📚 Search History")
            for i, search in enumerate(reversed(st.session_state.search_history[-5:])):
                with st.expander(f"🔍 {search['query'][:50]}{'...' if len(search['query']) > 50 else ''}"):
                    st.write(f"**Query:** {search['query']}")
                    st.write(f"**Results:** {len(search['results'])} found")
                    st.write(f"**Time:** {search['timestamp']}")
    
    def perform_search(self, query: str, search_type: str, top_k: int):
        """Perform search through the pipeline"""
        if not PIPELINE_AVAILABLE or not self.pipeline:
            st.error("❌ Cannot perform search: Pipeline not available")
            st.info("💡 Initialize the pipeline first or install missing dependencies")
            return
        
        try:
            with st.spinner("🔍 Searching..."):
                # Run search
                async def search():
                    result = await self.pipeline.search_and_generate(
                        query=query,
                        search_type=search_type,
                        top_k=top_k
                    )
                    return result
                
                result = asyncio.run(search())
                
                if result.success:
                    # Display search quality metrics
                    sources = result.data['sources']
                    if sources:
                        avg_score = sum(s['score'] for s in sources) / len(sources)
                        best_score = max(s['score'] for s in sources)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Average Score", f"{avg_score:.3f}")
                        with col2:
                            st.metric("🎯 Best Score", f"{best_score:.3f}")
                        with col3:
                            quality = "🟢 Excellent" if best_score > 0.8 else "🟡 Good" if best_score > 0.6 else "🔴 Poor"
                            st.metric("🏆 Search Quality", quality)
                    
                    # Display response
                    st.subheader("💬 Generated Response")
                    st.write(result.data['response'])
                    
                    # Display sources
                    st.subheader("📚 Sources")
                    sources = result.data['sources']
                    
                    for i, source in enumerate(sources, 1):
                        title = source.get('title', source.get('document_title', 'Unknown Document'))
                        score = source.get('score', 0)
                        
                        # Score color coding
                        score_color = "🟢" if score >= 0.7 else "🟡" if score >= 0.6 else "🔴"
                        
                        with st.expander(f"{score_color} Source {i}: {title} (Score: {score:.3f})"):
                            content = source.get('content', 'Content not available')
                            
                            if content and content != 'Content not available':
                                # Add source metadata
                                col1, col2 = st.columns([3, 1])
                                with col2:
                                    st.metric("Relevance", f"{score:.3f}")
                                    source_path = source.get('source', 'Unknown')
                                    if source_path != 'Unknown':
                                        st.caption(f"📁 {Path(source_path).name}")
                                
                                with col1:
                                    # Display content with better formatting
                                    if len(content) > 1000:
                                        st.markdown("**Preview:**")
                                        st.write(content[:1000] + "...")
                                        if st.button(f"Show full content {i}", key=f"full_content_{i}"):
                                            st.markdown("**Full Content:**")
                                            st.text_area("", content, height=300, key=f"full_text_{i}")
                                    else:
                                        st.write(content)
                            else:
                                st.error("❌ Content not available")
                                st.info("This might indicate an issue with document processing or field mapping.")
                    
                    # Add to search history
                    search_record = {
                        'query': query,
                        'results': sources,
                        'response': result.data['response'],
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.search_history.append(search_record)
                    
                    # Update metrics
                    st.session_state.pipeline_status = self.pipeline.get_status()
                    
                else:
                    st.error(f"❌ Search failed: {result.error}")
                    
                    # Show helpful tips for common issues
                    if "Cannot connect to Qdrant" in str(result.error):
                        st.error("🚨 **Qdrant Database Connection Error**")
                        st.info("💡 **To fix this:**")
                        st.code("docker run -d -p 6333:6333 qdrant/qdrant:v1.12.5", language="bash")
                        st.markdown("Or check if Qdrant is already running on port 6333")
                    
        except Exception as e:
            st.error(f"❌ Search error: {e}")
            logger.error(f"Search error: {e}")
            
            # Show helpful tips for common issues
            if "Cannot connect to Qdrant" in str(e):
                st.error("🚨 **Qdrant Database Connection Error**")
                st.info("💡 **To fix this:**")
                st.code("docker run -d -p 6333:6333 qdrant/qdrant:v1.12.5", language="bash")
                st.markdown("Or check if Qdrant is already running on port 6333")
    
    def inspect_qdrant_status(self):
        """Inspect the actual Qdrant database status"""
        try:
            if not PIPELINE_AVAILABLE or not self.pipeline:
                st.error("❌ Pipeline not available")
                return
                
            # Try to connect and get collection info
            from qdrant_client import QdrantClient
            
            config = self.pipeline.config
            host = config.get('qdrant', {}).get('host', 'localhost')
            port = config.get('qdrant', {}).get('port', 6333)
            collection_name = config['vectordb']['collection_name']
            
            with st.spinner("Connecting to Qdrant database..."):
                client = QdrantClient(host=host, port=port)
                
                # Get collections
                collections = client.get_collections()
                st.success(f"✅ Connected to Qdrant at {host}:{port}")
                
                collection_names = [col.name for col in collections.collections]
                st.info(f"📊 Available collections: {collection_names}")
                
                if collection_name in collection_names:
                    # Get collection info
                    collection_info = client.get_collection(collection_name)
                    
                    st.subheader(f"🗂️ Collection: '{collection_name}'")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Vectors", 
                            collection_info.points_count,
                            help="Actual vectors stored in Qdrant"
                        )
                    
                    with col2:
                        st.metric(
                            "Vector Dimensions", 
                            collection_info.config.params.vectors.size,
                            help="Size of each vector"
                        )
                    
                    with col3:
                        st.metric(
                            "Distance Metric", 
                            collection_info.config.params.vectors.distance.value,
                            help="How similarity is calculated"
                        )
                    
                    # Index status
                    index_info = collection_info.config.hnsw_config
                    if index_info:
                        st.success("🚀 **HNSW Index Active** - Fast search enabled")
                        st.json({
                            "m": index_info.m,
                            "ef_construct": index_info.ef_construct,
                            "full_scan_threshold": index_info.full_scan_threshold
                        })
                    else:
                        st.warning("⚠️ **No Index Found** - Search will be slower")
                        
                else:
                    st.warning(f"⚠️ Collection '{collection_name}' not found")
                    st.info("This means no documents have been indexed yet")
                    
        except Exception as e:
            st.error(f"❌ Cannot inspect Qdrant database: {e}")
            
            if "Connection refused" in str(e) or "ConnectError" in str(e):
                st.error("🚨 **Qdrant is not running**")
                st.info("💡 **Start Qdrant with:**")
                st.code("docker run -d -p 6333:6333 qdrant/qdrant:v1.12.5", language="bash")
                st.info("Then restart your dashboard")
    
    def render_vectordb_inspector(self):
        """Render vector database inspector"""
        if not PIPELINE_AVAILABLE:
            st.error("❌ Vector database inspector not available")
            st.info("💡 Install dependencies to enable vector database inspection")
            return
            
        # Get current pipeline status for comparison
        pipeline_status = st.session_state.get('pipeline_status', {})
        
        # Get pipeline config if available
        config = self.pipeline.config if self.pipeline else {}
        
        render_vectordb_inspector(config, pipeline_status)
    
    def render_analytics(self):
        """Render analytics and metrics"""
        st.subheader("📈 System Analytics")
        st.markdown("*Monitor your RAG system performance*")
        
        if not st.session_state.pipeline_initialized:
            st.info("Initialize the pipeline to see analytics")
            return
        
        status = st.session_state.pipeline_status
        metrics = status.get('metrics', {})
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📄 Documents", 
                metrics.get('documents_processed', 0),
                help="Total documents processed"
            )
        
        with col2:
            st.metric(
                "✂️ Chunks", 
                metrics.get('chunks_created', 0),
                help="Text chunks created"
            )
        
        with col3:
            st.metric(
                "🔢 Embeddings", 
                metrics.get('embeddings_generated', 0),
                help="Vector embeddings generated"
            )
        
        with col4:
            st.metric(
                "🔍 Searches", 
                metrics.get('searches_performed', 0),
                help="Search queries performed"
            )
        
        # Performance chart (placeholder)
        if metrics.get('total_processing_time', 0) > 0:
            st.subheader("⏱️ Processing Performance")
            
            # Create a simple performance visualization
            stages = ["Loading", "Chunking", "Embedding", "Indexing"]
            times = [0.5, 1.2, 3.1, 0.8]  # Placeholder data
            
            fig = px.bar(x=stages, y=times, title="Processing Time by Stage")
            fig.update_layout(
                xaxis_title="Pipeline Stage",
                yaxis_title="Time (seconds)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_learning_section(self):
        """Render educational content about RAG"""
        st.subheader("🎓 Learn About RAG Systems")
        st.markdown("*Understanding how Retrieval-Augmented Generation works*")
        
        # Educational tabs
        learn_tab1, learn_tab2, learn_tab3, learn_tab4 = st.tabs([
            "🤔 What is RAG?",
            "🔢 Vector Embeddings", 
            "🔍 How Search Works",
            "🏗️ System Architecture"
        ])
        
        with learn_tab1:
            self.render_rag_explanation()
        
        with learn_tab2:
            self.render_embeddings_explanation()
        
        with learn_tab3:
            self.render_search_explanation()
        
        with learn_tab4:
            self.render_architecture_explanation()
    
    def render_rag_explanation(self):
        """Explain RAG in beginner-friendly terms"""
        st.markdown("""
        ## 🤔 What is RAG?
        
        **RAG** stands for **Retrieval-Augmented Generation**. Think of it like a super-smart assistant that:
        
        1. 📚 **Reads your documents** (like a librarian organizing books)
        2. 🧠 **Remembers everything** by converting text to numbers (embeddings)
        3. 🔍 **Finds relevant information** when you ask questions
        4. 💬 **Gives you answers** based on what it found
        
        ### Why is RAG useful?
        - ✅ **Accurate**: Answers based on YOUR documents
        - ✅ **Up-to-date**: Uses your latest information
        - ✅ **Traceable**: Shows you where answers come from
        - ✅ **Private**: Your data stays with you
        
        ### RAG vs Regular AI
        | Regular AI | RAG System |
        |------------|------------|
        | Uses only training data | Uses YOUR documents |
        | Can "hallucinate" | Bases answers on real sources |
        | Fixed knowledge | Updates with new documents |
        | Hard to verify | Shows sources |
        """)
    
    def render_embeddings_explanation(self):
        """Explain embeddings in simple terms"""
        st.markdown("""
        ## 🔢 Vector Embeddings Explained
        
        ### What are embeddings?
        Think of embeddings as **"fingerprints for text"**. Just like your fingerprint uniquely identifies you, 
        embeddings uniquely identify the meaning of text.
        
        ### The Fruit Example 🍎🍊🍌
        Imagine we have these fruits:
        - Apple 🍎
        - Orange 🍊  
        - Banana 🍌
        - Granny Smith Apple 🍏
        
        In embedding space:
        - 🍎 Apple and 🍏 Granny Smith are **close** (similar meaning)
        - 🍊 Orange is **medium distance** (both fruits, different type)
        - 🍌 Banana is **far** (very different fruit)
        """)
        
        # Simple embedding visualization
        if st.button("🎯 Show Example Embeddings"):
            import numpy as np
            
            # Create example 2D embeddings for visualization
            fruits = ["Apple", "Orange", "Banana", "Granny Smith"]
            # Simulate embeddings (in reality these would be 384-1024 dimensions)
            embeddings_2d = np.array([
                [0.8, 0.9],   # Apple
                [0.3, 0.7],   # Orange  
                [-0.5, -0.2], # Banana
                [0.9, 0.8]    # Granny Smith (close to Apple)
            ])
            
            fig = px.scatter(
                x=embeddings_2d[:, 0], 
                y=embeddings_2d[:, 1], 
                text=fruits,
                title="Fruit Embeddings in 2D Space"
            )
            fig.update_traces(textposition="middle right")
            st.plotly_chart(fig)
    
    def render_search_explanation(self):
        """Explain how similarity search works"""
        st.markdown("""
        ## 🔍 How Search Works
        
        ### The Similarity Game
        When you ask a question, the system:
        
        1. **Converts your question** to an embedding vector
        2. **Compares** it to all document embeddings
        3. **Finds the closest matches** (highest similarity scores)
        4. **Returns relevant documents** to answer your question
        
        ### Distance Methods
        
        **📐 Cosine Similarity** (Recommended)
        - Measures the angle between vectors (0-1 scale)
        - **1.0**: Perfect match (identical meaning)
        - **0.8**: Very similar (related concepts)
        - **0.6**: Somewhat related (same domain)
        - **0.4**: Barely related (distant connection)
        - **0.0**: Completely different (no relationship)
        
        **📏 Euclidean Distance** (Alternative)
        - Measures straight-line distance in vector space
        - **0.0**: Perfect match (identical vectors)
        - **Lower scores**: More similar content
        - **Higher scores**: More different content
        - Best for magnitude-sensitive comparisons
        
        ### Search Types
        - **🎯 Dense Search**: Pure semantic similarity
        - **🔄 Hybrid Search**: Combines semantic + keyword matching
        - **📊 MMR Search**: Diverse results (avoids duplicates)
        """)
    
    def render_architecture_explanation(self):
        """Explain the system architecture"""
        st.markdown("""
        ## 🏗️ System Architecture
        
        ### The RAG Pipeline
        Our system follows this flow:
        
        ```
        📄 Documents → ✂️ Chunks → 🔢 Embeddings → 📊 Vector DB → 🔍 Search → 💬 Response
        ```
        
        ### Component Breakdown
        
        **1. 📄 Document Loader**
        - Reads PDF, TXT, Markdown files
        - Extracts clean text content
        - Preserves metadata (title, source, etc.)
        
        **2. ✂️ Text Chunker**  
        - Splits large documents into smaller pieces
        - Maintains context with overlap
        - Optimizes chunk size for search
        
        **3. 🔢 Embedding Generator**
        - Converts text to vector numbers
        - Uses open-source models (BGE, Nomic, E5)
        - Processes in batches for efficiency
        
        **4. 📊 Vector Database (Qdrant)**
        - Stores embeddings with metadata
        - Enables fast similarity search
        - Handles millions of vectors
        
        **5. 🔍 Search Engine**
        - Finds most relevant chunks
        - Ranks by similarity score
        - Supports different search strategies
        
        **6. 💬 Response Generator**
        - Combines search results
        - Creates coherent answers
        - Cites sources for verification
        """)
    
    def render_system_overview(self):
        """Render system overview that works without pipeline"""
        st.markdown("""
        ## 🔍 Visual RAG System Overview
        
        ### What This System Does
        This is a **production-ready RAG (Retrieval-Augmented Generation)** system that:
        
        1. **📄 Processes Documents** - Upload PDFs, text files, markdown
        2. **✂️ Chunks Text** - Splits documents into searchable pieces
        3. **🔢 Creates Embeddings** - Converts text to numerical vectors
        4. **📊 Stores in Vector DB** - Uses Qdrant for fast similarity search
        5. **🔍 Enables Search** - Find information with natural language queries
        6. **💬 Generates Answers** - Creates responses based on your documents
        
        ### Key Features
        - **🎨 Visual Pipeline**: See your data transform in real-time
        - **🧠 Open-Source Models**: 4 embedding models, no API keys needed
        - **🔧 MCP Integration**: Connect to Claude Desktop and other LLMs
        - **🤖 Agent System**: Multi-agent coordination for complex tasks
        - **📚 Educational**: Explains concepts like you're 5 years old
        """)
    
    def render_model_comparison(self):
        """Render model comparison table"""
        st.markdown("## 🧠 Embedding Models Comparison")
        
        models_data = {
            "Model": ["BGE-M3", "Nomic Embed V2", "E5 Base V2", "MiniLM L6 V2"],
            "Dimensions": [1024, 768, 768, 384],
            "Speed": ["⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
            "Best For": [
                "High accuracy, multilingual",
                "Flexible dimensions, research", 
                "Balanced speed & accuracy",
                "Speed critical, mobile apps"
            ]
        }
        
        df = pd.DataFrame(models_data)
        st.dataframe(df, use_container_width=True)
    
    def render_architecture_demo(self):
        """Render architecture demonstration"""
        st.markdown("""
        ## 🏗️ System Architecture
        
        ### 📁 Core Modules
        - **`pipeline/`** - Single orchestrated pipeline
        - **`models/`** - 4 open-source embedding models
        - **`visualizer/`** - Interactive dashboard
        - **`mcp_server/`** - LLM integration
        - **`agents/`** - Multi-agent system
        """)


def main():
    """Main application entry point"""
    dashboard = RAGDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()