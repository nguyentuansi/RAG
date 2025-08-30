"""
Stage Monitor Component

Provides real-time monitoring of pipeline stage execution,
including progress bars, timing information, and status updates.
"""

import streamlit as st
import time
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px


class StageMonitor:
    """Real-time pipeline stage monitoring"""
    
    def __init__(self):
        self.stage_colors = {
            'idle': '#868e96',
            'active': '#ff6b6b', 
            'completed': '#51cf66',
            'error': '#ff8787'
        }
        
        self.stage_descriptions = {
            'ingestion': 'Loading documents and extracting text content',
            'chunking': 'Splitting text into searchable chunks with overlap',
            'embedding': 'Converting text to numerical vector representations',
            'indexing': 'Storing vectors in searchable database',
            'retrieval': 'Finding most relevant chunks for query',
            'generation': 'Creating response from retrieved context'
        }
    
    def render(self, pipeline_status: Dict[str, Any]):
        """Render the stage monitoring interface"""
        
        if not pipeline_status:
            st.info("Pipeline not initialized")
            return
        
        current_state = pipeline_status.get('state', 'Idle')
        metrics = pipeline_status.get('metrics', {})
        
        # Current status overview
        self.render_status_overview(current_state, metrics)
        
        # Detailed stage breakdown
        self.render_stage_details(metrics)
        
        # Performance metrics
        self.render_performance_metrics(metrics)
    
    def render_status_overview(self, current_state: str, metrics: Dict):
        """Render current pipeline status overview"""
        
        st.subheader("📊 Current Status")
        
        # Status indicator
        status_color = self.get_status_color(current_state)
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {status_color} 0%, {status_color}80 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        ">
            <h3 style="margin: 0;">{current_state}</h3>
            <p style="margin: 0.5rem 0 0 0;">Last updated: {time.strftime("%H:%M:%S")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress indicators
        if self.is_processing_state(current_state):
            # Show progress bar for active processing
            progress_value = self.get_current_progress(current_state)
            st.progress(progress_value)
            st.caption(f"Processing... {progress_value*100:.1f}% complete")
    
    def render_stage_details(self, metrics: Dict):
        """Render detailed information for each stage"""
        
        st.subheader("🔍 Stage Details")
        
        stages = [
            ('📄 Document Loading', 'documents_processed', 'docs'),
            ('✂️ Text Chunking', 'chunks_created', 'chunks'),
            ('🔢 Embedding Generation', 'embeddings_generated', 'vectors'),
            ('📊 Vector Indexing', 'vectors_indexed', 'indexed'),
            ('🔍 Search Operations', 'searches_performed', 'searches')
        ]
        
        for stage_name, metric_key, unit in stages:
            value = metrics.get(metric_key, 0)
            status = "✅ Complete" if value > 0 else "⏸️ Pending"
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{stage_name}**")
                st.caption(self.stage_descriptions.get(metric_key.split('_')[0], ""))
            
            with col2:
                st.metric(
                    label="Count",
                    value=value,
                    help=f"Number of {unit} processed"
                )
            
            with col3:
                color = self.stage_colors['completed'] if value > 0 else self.stage_colors['idle']
                st.markdown(f"""
                <div style="
                    background-color: {color}20;
                    border-left: 4px solid {color};
                    padding: 0.5rem;
                    border-radius: 4px;
                    text-align: center;
                ">
                    <strong>{status}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
    
    def render_performance_metrics(self, metrics: Dict):
        """Render performance metrics and timing information"""
        
        st.subheader("⏱️ Performance Metrics")
        
        total_time = metrics.get('total_processing_time', 0)
        
        if total_time > 0:
            # Time breakdown chart
            self.render_timing_chart(metrics)
            
            # Throughput metrics
            col1, col2 = st.columns(2)
            
            with col1:
                docs_processed = metrics.get('documents_processed', 0)
                if docs_processed > 0:
                    docs_per_second = docs_processed / total_time
                    st.metric(
                        "📄 Document Throughput",
                        f"{docs_per_second:.2f} docs/sec",
                        help="Documents processed per second"
                    )
            
            with col2:
                chunks_created = metrics.get('chunks_created', 0)
                if chunks_created > 0:
                    chunks_per_second = chunks_created / total_time
                    st.metric(
                        "✂️ Chunk Throughput", 
                        f"{chunks_per_second:.2f} chunks/sec",
                        help="Text chunks created per second"
                    )
                    
            # System efficiency indicators
            self.render_efficiency_indicators(metrics)
        else:
            st.info("No performance data available yet. Process some documents to see metrics!")
    
    def render_timing_chart(self, metrics: Dict):
        """Render a chart showing time breakdown by stage"""
        
        # Simulate stage timings (in a real implementation, these would be tracked)
        total_time = metrics.get('total_processing_time', 0)
        
        if total_time > 0:
            # Estimate stage timings based on typical ratios
            stage_times = {
                'Document Loading': total_time * 0.1,
                'Text Chunking': total_time * 0.15, 
                'Embedding Generation': total_time * 0.6,
                'Vector Indexing': total_time * 0.15
            }
            
            # Create pie chart
            fig = px.pie(
                values=list(stage_times.values()),
                names=list(stage_times.keys()),
                title="Processing Time Breakdown"
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_efficiency_indicators(self, metrics: Dict):
        """Render system efficiency indicators"""
        
        st.subheader("📈 System Efficiency")
        
        # Calculate efficiency metrics
        docs_processed = metrics.get('documents_processed', 0)
        chunks_created = metrics.get('chunks_created', 0)
        embeddings_generated = metrics.get('embeddings_generated', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if docs_processed > 0:
                avg_chunks_per_doc = chunks_created / docs_processed
                st.metric(
                    "📋 Avg Chunks/Doc",
                    f"{avg_chunks_per_doc:.1f}",
                    help="Average number of chunks per document"
                )
        
        with col2:
            if chunks_created > 0:
                embedding_success_rate = (embeddings_generated / chunks_created) * 100
                st.metric(
                    "🎯 Embedding Success",
                    f"{embedding_success_rate:.1f}%",
                    help="Percentage of chunks successfully embedded"
                )
        
        with col3:
            searches_performed = metrics.get('searches_performed', 0)
            if embeddings_generated > 0:
                search_efficiency = searches_performed / max(embeddings_generated, 1) * 100
                st.metric(
                    "🔍 Search Utilization",
                    f"{search_efficiency:.1f}%", 
                    help="How much of the indexed data is being searched"
                )
        
        # Progress indicators for different stages
        self.render_progress_indicators(metrics)
    
    def render_progress_indicators(self, metrics: Dict):
        """Render progress indicators for pipeline completion"""
        
        st.subheader("📊 Pipeline Progress")
        
        # Define pipeline milestones
        milestones = [
            ('Documents Loaded', 'documents_processed', 1),
            ('Text Chunked', 'chunks_created', 1),
            ('Embeddings Generated', 'embeddings_generated', 1),
            ('Vectors Indexed', 'vectors_indexed', 1),
            ('Ready for Search', 'vectors_indexed', 1)
        ]
        
        # Calculate overall progress
        completed_stages = 0
        total_stages = len(milestones)
        
        for milestone_name, metric_key, threshold in milestones:
            value = metrics.get(metric_key, 0)
            is_complete = value >= threshold
            
            if is_complete:
                completed_stages += 1
            
            # Visual progress indicator
            status_icon = "✅" if is_complete else "⏸️"
            color = self.stage_colors['completed'] if is_complete else self.stage_colors['idle']
            
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                padding: 0.5rem;
                margin: 0.25rem 0;
                background-color: {color}15;
                border-radius: 6px;
                border-left: 4px solid {color};
            ">
                <span style="font-size: 1.2em; margin-right: 0.5rem;">{status_icon}</span>
                <span><strong>{milestone_name}</strong></span>
                <span style="margin-left: auto; color: {color};">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall progress bar
        overall_progress = completed_stages / total_stages
        st.progress(overall_progress)
        st.caption(f"Overall Progress: {completed_stages}/{total_stages} stages complete")
    
    def get_status_color(self, state: str) -> str:
        """Get color for current status"""
        
        if 'Error' in state or '❌' in state:
            return self.stage_colors['error']
        elif 'Processing' in state or '🔄' in state:
            return self.stage_colors['active']
        elif 'Ready' in state or '✅' in state:
            return self.stage_colors['completed']
        else:
            return self.stage_colors['idle']
    
    def is_processing_state(self, state: str) -> bool:
        """Check if pipeline is currently processing"""
        processing_indicators = ['🔄', 'Loading', 'Chunking', 'Generating', 'Indexing', 'Searching']
        return any(indicator in state for indicator in processing_indicators)
    
    def get_current_progress(self, state: str) -> float:
        """Get progress value for current processing state"""
        
        # Simulate progress based on state
        # In a real implementation, this would track actual progress
        
        progress_map = {
            '🔄 Loading Documents': 0.2,
            '✂️ Chunking Text': 0.4,
            '🔢 Generating Embeddings': 0.7,
            '📊 Indexing Vectors': 0.9,
            '🔍 Searching': 0.8,
            '💬 Generating Response': 0.9
        }
        
        for state_key, progress in progress_map.items():
            if state_key in state:
                return progress
        
        return 0.0