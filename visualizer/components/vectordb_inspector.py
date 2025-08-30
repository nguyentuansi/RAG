"""
Vector Database Inspector Component
Visualizes what data is indexed in Qdrant and what's missing
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Any
import time
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance
    import numpy as np
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class VectorDatabaseInspector:
    """Comprehensive vector database inspection and visualization"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.client = None
        
    def render(self, pipeline_status: Dict):
        """Render the vector database inspector dashboard"""
        
        st.subheader("🗂️ Vector Database Inspector")
        st.markdown("*See exactly what data is indexed and what's missing*")
        
        if not QDRANT_AVAILABLE:
            st.error("❌ Qdrant client not available.")
            st.info("💡 To enable full vector database inspection:")
            st.code("pip install qdrant-client numpy pandas scikit-learn", language="bash")
            
            # Show basic info that doesn't require Qdrant
            st.subheader("📋 Basic Configuration")
            if pipeline_status:
                metrics = pipeline_status.get('metrics', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Documents", metrics.get('documents_processed', 0))
                with col2:
                    st.metric("✂️ Chunks", metrics.get('chunks_created', 0))  
                with col3:
                    st.metric("🔢 Embeddings", metrics.get('embeddings_generated', 0))
            return
            
        # Connection and basic info
        connection_status = self._check_connection()
        
        if not connection_status['connected']:
            st.error(f"❌ Cannot connect to Qdrant: {connection_status['error']}")
            st.info("💡 Make sure Qdrant is running: `docker run -d -p 6333:6333 qdrant/qdrant:v1.12.5`")
            return
        
        # Collection overview
        self._render_collection_overview()
        
        # Detailed data inspection
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_indexed_data()
            
        with col2:
            self._render_missing_data(pipeline_status)
        
        # Visualization tabs
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Index Statistics",
            "🔍 Data Distribution", 
            "📈 Vector Analysis",
            "🔍 Gap Analysis"
        ])
        
        with viz_tab1:
            self._render_index_statistics()
            
        with viz_tab2:
            self._render_data_distribution()
            
        with viz_tab3:
            self._render_vector_analysis()
            
        with viz_tab4:
            self._render_gap_analysis()
    
    def _check_connection(self) -> Dict:
        """Check Qdrant connection status"""
        try:
            self.client = QdrantClient("localhost", port=6333, check_compatibility=False)
            collections = self.client.get_collections()
            return {
                'connected': True,
                'collections': [c.name for c in collections.collections]
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def _render_collection_overview(self):
        """Render collection overview with key metrics"""
        
        st.markdown("### 📊 Collection Overview")
        
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if 'rag_documents' in collection_names:
                info = self.client.get_collection('rag_documents')
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📄 Total Vectors",
                        info.points_count,
                        help="Total number of indexed document chunks"
                    )
                
                with col2:
                    st.metric(
                        "🔢 Dimensions", 
                        info.config.params.vectors.size,
                        help="Size of each vector embedding"
                    )
                    
                with col3:
                    distance = info.config.params.vectors.distance.value
                    # Map technical names to user-friendly names
                    distance_names = {
                        'Cosine': 'Cosine Similarity',
                        'Euclid': 'Euclidean Distance'
                    }
                    friendly_name = distance_names.get(distance.title(), distance.title())
                    
                    # Add helpful descriptions
                    distance_help = {
                        'Cosine Similarity': 'Measures angle between vectors (0-1 scale, higher = more similar). Best for semantic similarity.',
                        'Euclidean Distance': 'Straight-line distance in vector space (lower = more similar). Best for magnitude-sensitive comparisons.'
                    }.get(friendly_name, "How similarity is calculated")
                    
                    st.metric(
                        "📏 Distance Method",
                        friendly_name,
                        help=distance_help
                    )
                
                with col4:
                    # Calculate approximate index size
                    approx_size_mb = (info.points_count * info.config.params.vectors.size * 4) / (1024 * 1024)
                    st.metric(
                        "💾 Index Size",
                        f"{approx_size_mb:.1f} MB",
                        help="Approximate memory usage"
                    )
                
                # Collection status
                status_color = "🟢" if info.status == "green" else "🟡" if info.status == "yellow" else "🔴"
                st.info(f"{status_color} **Collection Status**: {info.status}")
                
            else:
                st.warning("⚠️ 'rag_documents' collection not found")
                st.info("Process some documents to create the collection")
                
        except Exception as e:
            st.error(f"❌ Error getting collection info: {e}")
    
    def _render_indexed_data(self):
        """Show what data is currently indexed"""
        
        st.markdown("### ✅ Indexed Data")
        
        try:
            # Get sample of indexed data
            sample_results = self.client.scroll(
                collection_name="rag_documents",
                limit=50,
                with_payload=True
            )[0]  # Get points only
            
            if sample_results:
                # Create DataFrame for analysis
                data_rows = []
                for point in sample_results:
                    payload = point.payload
                    data_rows.append({
                        'Document': payload.get('document_title', 'Unknown'),
                        'Source': Path(payload.get('source', '')).name,
                        'Chunk ID': payload.get('chunk_id', ''),
                        'Chunk Index': payload.get('chunk_index', 0),
                        'Size': len(payload.get('content', '')),
                        'Format': payload.get('format', 'unknown')
                    })
                
                df = pd.DataFrame(data_rows)
                
                # Summary statistics
                st.write("📈 **Indexed Documents:**")
                doc_stats = df.groupby('Document').agg({
                    'Chunk Index': 'count',
                    'Size': 'sum'
                }).rename(columns={'Chunk Index': 'Chunks', 'Size': 'Total Size'})
                
                try:
                    st.dataframe(doc_stats, use_container_width=True)
                except Exception as e:
                    st.write(doc_stats.to_string())
                
                # Recent indexing activity
                st.write("📋 **Recent Chunks:**")
                recent_df = df.head(10)[['Document', 'Chunk Index', 'Size']].copy()
                recent_df['Size'] = recent_df['Size'].apply(lambda x: f"{x} chars")
                try:
                    st.dataframe(recent_df, use_container_width=True)
                except Exception as e:
                    st.write(recent_df.to_string())
                
            else:
                st.info("No indexed data found")
                
        except Exception as e:
            st.error(f"❌ Error retrieving indexed data: {e}")
    
    def _render_missing_data(self, pipeline_status: Dict):
        """Show what data might be missing from index"""
        
        st.markdown("### ❓ Missing/Unprocessed Data")
        
        # Compare pipeline metrics with actual indexed data
        metrics = pipeline_status.get('metrics', {})
        
        try:
            collection_info = self.client.get_collection('rag_documents')
            actual_vectors = collection_info.points_count
            
            expected_embeddings = metrics.get('embeddings_generated', 0)
            expected_chunks = metrics.get('chunks_created', 0)
            
            # Calculate gaps
            missing_vectors = max(0, expected_embeddings - actual_vectors)
            
            if missing_vectors > 0:
                st.warning(f"⚠️ **{missing_vectors} embeddings** not indexed!")
                
                gap_reasons = []
                if expected_embeddings > expected_chunks:
                    gap_reasons.append("• Some chunks failed embedding generation")
                if expected_chunks > actual_vectors:
                    gap_reasons.append("• Some embeddings failed to index")
                
                if gap_reasons:
                    st.write("**Possible reasons:**")
                    for reason in gap_reasons:
                        st.write(reason)
                
                st.info("💡 **Solution**: Reprocess documents to fix indexing gaps")
                
            else:
                st.success("✅ All generated embeddings are properly indexed!")
                
            # Show processing pipeline status
            st.write("🔄 **Processing Pipeline Status:**")
            
            pipeline_data = {
                'Stage': ['Documents', 'Chunks', 'Embeddings', 'Indexed'],
                'Count': [
                    metrics.get('documents_processed', 0),
                    expected_chunks,
                    expected_embeddings, 
                    actual_vectors
                ],
                'Status': ['✅', '✅', '✅' if expected_embeddings > 0 else '❌', '✅' if actual_vectors > 0 else '❌']
            }
            
            pipeline_df = pd.DataFrame(pipeline_data)
            try:
                st.dataframe(pipeline_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.write(pipeline_df.to_string())
            
        except Exception as e:
            st.error(f"❌ Error analyzing missing data: {e}")
    
    def _render_index_statistics(self):
        """Render index statistics visualization"""
        
        st.markdown("### 📊 Index Statistics")
        
        try:
            # Get collection statistics
            info = self.client.get_collection('rag_documents')
            
            # HNSW configuration
            hnsw_config = info.config.hnsw_config
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**HNSW Index Configuration:**")
                config_data = {
                    'Parameter': ['M (connectivity)', 'EF Construct', 'Full Scan Threshold'],
                    'Value': [
                        str(hnsw_config.m),
                        str(hnsw_config.ef_construct),
                        str(hnsw_config.full_scan_threshold)
                    ],
                    'Description': [
                        'Number of bi-directional links',
                        'Size of dynamic candidate list',
                        'Threshold for exact search'
                    ]
                }
                try:
                    st.dataframe(pd.DataFrame(config_data), hide_index=True)
                except Exception as e:
                    st.write(pd.DataFrame(config_data).to_string())
            
            with col2:
                st.write("**Index Health:**")
                
                # Calculate index health metrics
                total_points = info.points_count
                vector_size = info.config.params.vectors.size
                
                # Estimate index efficiency
                if total_points < 1000:
                    efficiency = "Excellent (Small dataset)"
                    efficiency_color = "🟢"
                elif total_points < 10000:
                    efficiency = "Good"
                    efficiency_color = "🟡"
                else:
                    efficiency = "Normal (Large dataset)"
                    efficiency_color = "🟡"
                
                health_metrics = pd.DataFrame({
                    'Metric': ['Total Points', 'Vector Dimensions', 'Efficiency', 'Status'],
                    'Value': [str(total_points), str(vector_size), efficiency, f"{efficiency_color} Healthy"]
                })
                try:
                    st.dataframe(health_metrics, hide_index=True)
                except Exception as e:
                    st.write(health_metrics.to_string())
            
        except Exception as e:
            st.error(f"❌ Error generating statistics: {e}")
    
    def _render_data_distribution(self):
        """Render data distribution visualizations"""
        
        st.markdown("### 🔍 Data Distribution")
        
        try:
            # Get sample data for analysis
            results = self.client.scroll(
                collection_name="rag_documents",
                limit=100,
                with_payload=True
            )[0]
            
            if not results:
                st.info("No data to visualize")
                return
            
            # Prepare data for visualization
            data = []
            for point in results:
                payload = point.payload
                data.append({
                    'document': payload.get('document_title', 'Unknown'),
                    'chunk_size': len(payload.get('content', '')),
                    'chunk_index': payload.get('chunk_index', 0),
                    'format': payload.get('format', 'unknown')
                })
            
            df = pd.DataFrame(data)
            
            # Document distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Chunks per Document:**")
                doc_counts = df['document'].value_counts()
                fig = px.bar(
                    x=doc_counts.index,
                    y=doc_counts.values,
                    labels={'x': 'Document', 'y': 'Number of Chunks'},
                    title="Document Chunk Distribution"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Chunk Size Distribution:**")
                fig = px.histogram(
                    df,
                    x='chunk_size',
                    nbins=20,
                    title="Chunk Size Distribution",
                    labels={'chunk_size': 'Chunk Size (characters)', 'count': 'Frequency'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Format distribution
            st.write("**File Format Distribution:**")
            format_counts = df['format'].value_counts()
            fig = px.pie(
                values=format_counts.values,
                names=format_counts.index,
                title="Indexed File Formats"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error generating distribution charts: {e}")
    
    def _render_vector_analysis(self):
        """Render vector space analysis"""
        
        st.markdown("### 📈 Vector Analysis")
        
        try:
            # Get sample vectors for analysis
            results = self.client.scroll(
                collection_name="rag_documents",
                limit=50,
                with_payload=True,
                with_vectors=True
            )[0]
            
            if not results:
                st.info("No vector data available for analysis")
                return
            
            # Extract vectors and metadata
            vectors = []
            documents = []
            chunk_indices = []
            
            for point in results:
                vectors.append(point.vector)
                documents.append(point.payload.get('document_title', 'Unknown'))
                chunk_indices.append(point.payload.get('chunk_index', 0))
            
            if QDRANT_AVAILABLE:
                vectors = np.array(vectors)
            else:
                st.error("NumPy not available for vector analysis")
                return
            
            st.write(f"**Analyzing {len(vectors)} vectors with {vectors.shape[1]} dimensions**")
            
            # Vector statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Vector Statistics:**")
                
                # Calculate basic statistics
                vector_norms = np.linalg.norm(vectors, axis=1)
                
                stats_data = {
                    'Metric': [
                        'Mean Vector Norm',
                        'Std Vector Norm', 
                        'Min Vector Norm',
                        'Max Vector Norm',
                        'Dimensions'
                    ],
                    'Value': [
                        f"{np.mean(vector_norms):.4f}",
                        f"{np.std(vector_norms):.4f}",
                        f"{np.min(vector_norms):.4f}",
                        f"{np.max(vector_norms):.4f}",
                        str(vectors.shape[1])
                    ]
                }
                try:
                    st.dataframe(pd.DataFrame(stats_data), hide_index=True)
                except Exception as e:
                    st.write(pd.DataFrame(stats_data).to_string())
            
            with col2:
                st.write("**Vector Norm Distribution:**")
                fig = px.histogram(
                    x=vector_norms,
                    nbins=20,
                    title="Vector Norm Distribution",
                    labels={'x': 'Vector Norm', 'y': 'Frequency'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Dimensionality reduction visualization (if we have enough vectors)
            if len(vectors) >= 10:
                st.write("**Vector Space Visualization (2D Projection):**")
                
                try:
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import TSNE
                    
                    # Use PCA for dimensionality reduction
                    if vectors.shape[1] > 2:
                        pca = PCA(n_components=2)
                        vectors_2d = pca.fit_transform(vectors)
                    else:
                        vectors_2d = vectors
                    
                    # Create scatter plot
                    fig = px.scatter(
                        x=vectors_2d[:, 0],
                        y=vectors_2d[:, 1],
                        color=documents,
                        title="Vector Space Projection (PCA)",
                        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Document'},
                        hover_data={'Chunk Index': chunk_indices}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if vectors.shape[1] > 2:
                        explained_ratio = pca.explained_variance_ratio_
                        st.info(f"📊 PCA explains {(explained_ratio[0] + explained_ratio[1])*100:.1f}% of variance")
                    
                except ImportError:
                    st.info("💡 Install scikit-learn for vector space visualization: `pip install scikit-learn`")
                
        except Exception as e:
            st.error(f"❌ Error in vector analysis: {e}")
    
    def _render_gap_analysis(self):
        """Render gap analysis using the specialized component"""
        try:
            from .index_gap_analyzer import render_index_gap_analysis
            render_index_gap_analysis()
        except ImportError:
            st.error("❌ Gap analysis component not available")
        except Exception as e:
            st.error(f"❌ Error in gap analysis: {e}")


def render_vectordb_inspector(config: Dict = None, pipeline_status: Dict = None):
    """Convenience function to render the vector database inspector"""
    inspector = VectorDatabaseInspector(config)
    inspector.render(pipeline_status or {})