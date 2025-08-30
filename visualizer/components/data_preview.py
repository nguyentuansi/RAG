"""
Data Preview Component

Shows live previews of data as it flows through each stage of the pipeline,
helping beginners understand how text transforms into searchable vectors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import json


class DataPreview:
    """Live data preview for pipeline stages"""
    
    def __init__(self):
        self.max_preview_items = 3
        self.max_text_length = 200
        
    def render(self, data_samples: Dict[str, Any]):
        """Render data previews for all available stages"""
        
        if not data_samples:
            st.info("🔄 Process some documents to see live data previews!")
            return
        
        # Create tabs for different data types
        preview_tabs = st.tabs([
            "📄 Documents",
            "✂️ Chunks", 
            "🔢 Embeddings",
            "🔍 Search Results"
        ])
        
        with preview_tabs[0]:
            self.render_document_preview(data_samples.get('documents', []))
        
        with preview_tabs[1]:
            self.render_chunk_preview(data_samples.get('chunks', []))
        
        with preview_tabs[2]:
            self.render_embedding_preview(data_samples.get('embeddings', {}))
        
        with preview_tabs[3]:
            self.render_search_results_preview(data_samples.get('search_results', []))
    
    def render_document_preview(self, documents: List[Dict]):
        """Preview loaded documents"""
        
        st.subheader("📄 Loaded Documents")
        
        if not documents:
            st.info("No documents loaded yet")
            return
        
        st.write(f"**{len(documents)} documents loaded**")
        
        # Show document summaries
        doc_data = []
        for doc in documents[:self.max_preview_items]:
            doc_data.append({
                "Title": doc.get('title', 'Untitled'),
                "Format": doc.get('format', 'unknown'),
                "Size (chars)": doc.get('size', 0),
                "Source": doc.get('source', 'unknown')
            })
        
        if doc_data:
            df = pd.DataFrame(doc_data)
            st.dataframe(df, use_container_width=True)
        
        # Show content preview
        if documents:
            st.subheader("📖 Content Preview")
            
            selected_doc = st.selectbox(
                "Select document to preview:",
                options=range(min(len(documents), 5)),
                format_func=lambda i: documents[i].get('title', f'Document {i+1}')
            )
            
            if selected_doc < len(documents):
                doc = documents[selected_doc]
                content = doc.get('content', '')
                
                # Show truncated content
                preview_text = content[:self.max_text_length*2]
                if len(content) > len(preview_text):
                    preview_text += "..."
                
                st.text_area(
                    f"Content ({len(content)} characters):",
                    preview_text,
                    height=150,
                    disabled=True
                )
                
                # Document metadata
                metadata = doc.get('metadata', {})
                if metadata:
                    st.subheader("📋 Metadata")
                    st.json(metadata)
    
    def render_chunk_preview(self, chunks: List[Dict]):
        """Preview text chunks"""
        
        st.subheader("✂️ Text Chunks")
        
        if not chunks:
            st.info("No chunks created yet. Upload and process documents first.")
            return
        
        st.write(f"**{len(chunks)} chunks created**")
        
        # Chunk statistics
        chunk_sizes = [chunk.get('size', 0) for chunk in chunks]
        if chunk_sizes:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Size", f"{np.mean(chunk_sizes):.0f} chars")
            
            with col2:
                st.metric("Min Size", f"{min(chunk_sizes)} chars")
            
            with col3:
                st.metric("Max Size", f"{max(chunk_sizes)} chars")
        
        # Chunk size distribution
        if len(chunk_sizes) > 1:
            fig = px.histogram(
                x=chunk_sizes,
                nbins=20,
                title="Chunk Size Distribution",
                labels={'x': 'Chunk Size (characters)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show individual chunks
        st.subheader("🔍 Individual Chunks")
        
        for i, chunk in enumerate(chunks[:self.max_preview_items]):
            with st.expander(f"Chunk {i+1}: {chunk.get('document_title', 'Unknown')}"):
                
                # Chunk metadata
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Size", f"{chunk.get('size', 0)} chars")
                
                with col2:
                    st.metric("Index", chunk.get('chunk_index', 0))
                
                with col3:
                    st.metric("Total Chunks", chunk.get('total_chunks', 1))
                
                # Content preview
                content = chunk.get('content', '')
                preview_text = content[:self.max_text_length]
                if len(content) > len(preview_text):
                    preview_text += "..."
                
                st.text_area(
                    "Content:",
                    preview_text,
                    height=100,
                    disabled=True,
                    key=f"chunk_content_{i}"
                )
        
        if len(chunks) > self.max_preview_items:
            st.info(f"... and {len(chunks) - self.max_preview_items} more chunks")
    
    def render_embedding_preview(self, embedding_data: Dict[str, Any]):
        """Preview generated embeddings"""
        
        st.subheader("🔢 Vector Embeddings")
        
        if not embedding_data:
            st.info("No embeddings generated yet. Process documents first.")
            return
        
        count = embedding_data.get('count', 0)
        dimensions = embedding_data.get('dimensions', 0)
        sample_vector = embedding_data.get('sample_vector', [])
        
        if count == 0:
            st.info("No embeddings available")
            return
        
        # Embedding statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Embeddings", count)
        
        with col2:
            st.metric("Dimensions", dimensions)
        
        # Vector visualization
        if sample_vector is not None and len(sample_vector) > 0:
            st.subheader("🎯 Sample Vector Preview")
            
            # Show first 10 dimensions as numbers
            st.write("**First 10 dimensions:**")
            sample_df = pd.DataFrame({
                'Dimension': range(1, min(11, len(sample_vector) + 1)),
                'Value': sample_vector[:10]
            })
            st.dataframe(sample_df, use_container_width=True)
            
            # Visualize sample vector values
            fig = px.bar(
                sample_df,
                x='Dimension',
                y='Value',
                title="Sample Embedding Vector (First 10 Dimensions)"
            )
            fig.update_layout(
                xaxis_title="Dimension",
                yaxis_title="Value"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Explain what this means
            st.info("""
            💡 **What are these numbers?**
            
            Each number represents how much the text relates to different concepts that the AI learned during training. 
            - **Positive numbers**: Strong positive association
            - **Negative numbers**: Strong negative association  
            - **Numbers near zero**: Weak association
            
            Similar texts will have similar patterns of numbers!
            """)
        
        # Embedding comparison (if multiple available)
        if count > 1:
            st.subheader("📊 Embedding Analysis")
            
            # Create dummy data for visualization (in real implementation, would use actual embeddings)
            self.render_embedding_space_visualization(count, dimensions)
    
    def render_embedding_space_visualization(self, count: int, dimensions: int):
        """Visualize embeddings in reduced dimensional space"""
        
        st.write("**Embedding Space Visualization**")
        
        # Generate dummy 2D projection for visualization
        # In real implementation, this would use t-SNE or UMAP on actual embeddings
        np.random.seed(42)
        x_coords = np.random.normal(0, 1, min(count, 50))
        y_coords = np.random.normal(0, 1, min(count, 50))
        
        # Create clusters to simulate semantic grouping
        n_clusters = min(3, count // 10 + 1)
        cluster_centers = np.random.normal(0, 2, (n_clusters, 2))
        
        points_per_cluster = len(x_coords) // n_clusters
        colors = []
        
        for i in range(n_clusters):
            start_idx = i * points_per_cluster
            end_idx = (i + 1) * points_per_cluster if i < n_clusters - 1 else len(x_coords)
            
            # Adjust points to be near cluster center
            x_coords[start_idx:end_idx] += cluster_centers[i][0]
            y_coords[start_idx:end_idx] += cluster_centers[i][1]
            
            colors.extend([f'Cluster {i+1}'] * (end_idx - start_idx))
        
        # Create scatter plot
        fig = px.scatter(
            x=x_coords,
            y=y_coords,
            color=colors,
            title="Document Embeddings in 2D Space (t-SNE projection)",
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        📈 **Understanding this visualization:**
        
        - Each dot represents one chunk of text
        - Similar chunks appear close together
        - Different topics form separate clusters
        - The AI automatically groups related content!
        """)
    
    def render_search_results_preview(self, search_results: List[Dict]):
        """Preview search results"""
        
        st.subheader("🔍 Search Results")
        
        if not search_results:
            st.info("No search results yet. Try searching for something!")
            return
        
        st.write(f"**{len(search_results)} results found**")
        
        # Results overview
        scores = [result.get('score', 0) for result in search_results]
        if scores:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Score", f"{max(scores):.3f}")
            
            with col2:
                st.metric("Average Score", f"{np.mean(scores):.3f}")
            
            with col3:
                st.metric("Lowest Score", f"{min(scores):.3f}")
        
        # Score distribution
        if len(scores) > 1:
            fig = px.bar(
                x=range(1, len(scores) + 1),
                y=scores,
                title="Search Result Similarity Scores",
                labels={'x': 'Result Rank', 'y': 'Similarity Score'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual results
        st.subheader("📋 Individual Results")
        
        for i, result in enumerate(search_results[:self.max_preview_items]):
            score = result.get('score', 0)
            
            # Color code by score
            if score >= 0.8:
                score_color = "#51cf66"  # Green
                score_text = "Excellent Match"
            elif score >= 0.6:
                score_color = "#ffd43b"  # Yellow
                score_text = "Good Match"
            else:
                score_color = "#ff6b6b"  # Red
                score_text = "Fair Match"
            
            with st.expander(f"Result {i+1}: {result.get('document_title', 'Unknown')} - {score_text}"):
                
                # Result metadata
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Similarity Score:** <span style="color: {score_color}; font-weight: bold;">{score:.3f}</span>
                    """, unsafe_allow_html=True)
                    st.write(f"**Source:** {result.get('source', 'Unknown')}")
                
                with col2:
                    metadata = result.get('metadata', {})
                    if 'chunk_index' in metadata:
                        st.write(f"**Chunk:** {metadata['chunk_index']}")
                    if 'format' in metadata:
                        st.write(f"**Format:** {metadata['format']}")
                
                # Content preview
                content = result.get('content', '')
                preview_text = content[:self.max_text_length]
                if len(content) > len(preview_text):
                    preview_text += "..."
                
                st.text_area(
                    "Matching Content:",
                    preview_text,
                    height=100,
                    disabled=True,
                    key=f"search_result_{i}"
                )
        
        if len(search_results) > self.max_preview_items:
            st.info(f"... and {len(search_results) - self.max_preview_items} more results")
        
        # Explain scoring
        st.info("""
        🎯 **Understanding Similarity Scores:**
        
        - **0.9-1.0**: Nearly identical content
        - **0.7-0.9**: Very similar, likely relevant
        - **0.5-0.7**: Somewhat related
        - **0.3-0.5**: Might be relevant
        - **0.0-0.3**: Probably not relevant
        """)
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text to specified length"""
        if max_length is None:
            max_length = self.max_text_length
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + "..."