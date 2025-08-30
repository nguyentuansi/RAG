"""
Pipeline Flow Diagram Component

This component creates an interactive visual representation of the RAG pipeline,
showing the current state of each stage and data flow between them.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional


class PipelineFlowDiagram:
    """Interactive pipeline flow visualization"""
    
    def __init__(self):
        self.stage_positions = {
            'ingestion': (1, 3),
            'chunking': (2, 3),
            'embedding': (3, 3),
            'indexing': (4, 3),
            'retrieval': (2, 1),
            'generation': (3, 1)
        }
        
        self.stage_info = {
            'ingestion': {
                'name': '📄 Document Loading',
                'description': 'Load and parse documents',
                'color': '#74c0fc',
                'icon': '📄'
            },
            'chunking': {
                'name': '✂️ Text Chunking', 
                'description': 'Split text into searchable pieces',
                'color': '#ffd43b',
                'icon': '✂️'
            },
            'embedding': {
                'name': '🔢 Generate Embeddings',
                'description': 'Convert text to vector numbers',
                'color': '#ff8cc8',
                'icon': '🔢'
            },
            'indexing': {
                'name': '📊 Vector Indexing',
                'description': 'Store in searchable database',
                'color': '#51cf66',
                'icon': '📊'
            },
            'retrieval': {
                'name': '🔍 Search & Retrieval',
                'description': 'Find relevant documents',
                'color': '#ff6b6b',
                'icon': '🔍'
            },
            'generation': {
                'name': '💬 Response Generation',
                'description': 'Create final answer',
                'color': '#9775fa',
                'icon': '💬'
            }
        }
    
    def render(self, pipeline_status: Dict[str, Any]):
        """Render the pipeline flow diagram"""
        
        # Create two sections: build phase and query phase
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.render_flow_diagram(pipeline_status)
        
        with col2:
            self.render_stage_legend(pipeline_status)
    
    def render_flow_diagram(self, pipeline_status: Dict[str, Any]):
        """Render the main flow diagram"""
        
        fig = go.Figure()
        
        # Get current state
        current_state = pipeline_status.get('state', 'Idle')
        metrics = pipeline_status.get('metrics', {})
        
        # Add nodes for each stage
        self.add_pipeline_nodes(fig, current_state, metrics)
        
        # Add connections between stages
        self.add_pipeline_connections(fig)
        
        # Configure layout
        fig.update_layout(
            title="🔄 RAG Pipeline Flow",
            showlegend=False,
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[0, 5]
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[0, 4]
            ),
            plot_bgcolor='white',
            height=400,
            annotations=self.create_stage_annotations(current_state, metrics)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def add_pipeline_nodes(self, fig: go.Figure, current_state: str, metrics: Dict):
        """Add nodes representing each pipeline stage"""
        
        for stage, (x, y) in self.stage_positions.items():
            stage_info = self.stage_info[stage]
            
            # Determine node color based on state
            color = self.get_stage_color(stage, current_state, metrics)
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=60,
                    color=color,
                    line=dict(width=3, color='white'),
                    symbol='circle'
                ),
                name=stage_info['name'],
                hovertemplate=f"<b>{stage_info['name']}</b><br>{stage_info['description']}<extra></extra>",
                showlegend=False
            ))
    
    def add_pipeline_connections(self, fig: go.Figure):
        """Add arrows showing data flow between stages"""
        
        # Build phase connections (left to right)
        build_flow = [
            ('ingestion', 'chunking'),
            ('chunking', 'embedding'),
            ('embedding', 'indexing')
        ]
        
        # Query phase connections
        query_flow = [
            ('retrieval', 'generation')
        ]
        
        # Add connection arrows
        for source, target in build_flow + query_flow:
            self.add_arrow(fig, source, target, 'lightgray')
    
    def add_arrow(self, fig: go.Figure, source: str, target: str, color: str):
        """Add an arrow between two stages"""
        
        source_pos = self.stage_positions[source]
        target_pos = self.stage_positions[target]
        
        # Calculate arrow direction
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        
        # Adjust start and end points to account for node size
        start_x = source_pos[0] + 0.15 * (1 if dx > 0 else -1 if dx < 0 else 0)
        start_y = source_pos[1] + 0.15 * (1 if dy > 0 else -1 if dy < 0 else 0)
        end_x = target_pos[0] - 0.15 * (1 if dx > 0 else -1 if dx < 0 else 0)
        end_y = target_pos[1] - 0.15 * (1 if dy > 0 else -1 if dy < 0 else 0)
        
        # Add arrow line
        fig.add_trace(go.Scatter(
            x=[start_x, end_x],
            y=[start_y, end_y],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def create_stage_annotations(self, current_state: str, metrics: Dict) -> list:
        """Create text annotations for each stage"""
        
        annotations = []
        
        for stage, (x, y) in self.stage_positions.items():
            stage_info = self.stage_info[stage]
            
            # Create annotation text
            text = f"{stage_info['icon']}<br>{stage_info['name'].replace(' ', '<br>')}"
            
            # Add metrics if available
            count = self.get_stage_count(stage, metrics)
            if count > 0:
                text += f"<br><b>{count}</b>"
            
            annotations.append(
                dict(
                    x=x, y=y,
                    text=text,
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    xanchor='center',
                    yanchor='middle'
                )
            )
        
        return annotations
    
    def get_stage_color(self, stage: str, current_state: str, metrics: Dict) -> str:
        """Get color for stage based on current state"""
        
        # Map stages to state indicators
        stage_states = {
            'ingestion': 'documents_processed',
            'chunking': 'chunks_created', 
            'embedding': 'embeddings_generated',
            'indexing': 'vectors_indexed',
            'retrieval': 'searches_performed',
            'generation': 'searches_performed'
        }
        
        # Check if stage has been completed
        if stage in stage_states:
            metric_key = stage_states[stage]
            if metrics.get(metric_key, 0) > 0:
                return '#51cf66'  # Green for completed
        
        # Check if currently active
        active_indicators = {
            '🔄 Loading Documents': 'ingestion',
            '✂️ Chunking Text': 'chunking',
            '🔢 Generating Embeddings': 'embedding', 
            '📊 Indexing Vectors': 'indexing',
            '🔍 Searching': 'retrieval',
            '💬 Generating Response': 'generation'
        }
        
        for state_text, active_stage in active_indicators.items():
            if state_text in current_state and stage == active_stage:
                return '#ff6b6b'  # Red for active
        
        # Default color
        return self.stage_info[stage]['color']
    
    def get_stage_count(self, stage: str, metrics: Dict) -> int:
        """Get count for stage from metrics"""
        
        count_mapping = {
            'ingestion': 'documents_processed',
            'chunking': 'chunks_created',
            'embedding': 'embeddings_generated', 
            'indexing': 'vectors_indexed',
            'retrieval': 'searches_performed',
            'generation': 'searches_performed'
        }
        
        if stage in count_mapping:
            return metrics.get(count_mapping[stage], 0)
        
        return 0
    
    def render_stage_legend(self, pipeline_status: Dict[str, Any]):
        """Render legend explaining the pipeline stages"""
        
        st.subheader("🎯 Pipeline Stages")
        
        current_state = pipeline_status.get('state', 'Idle')
        metrics = pipeline_status.get('metrics', {})
        
        # Build Phase
        st.markdown("**📚 Build Phase** *(Process documents once)*")
        build_stages = ['ingestion', 'chunking', 'embedding', 'indexing']
        
        for stage in build_stages:
            stage_info = self.stage_info[stage]
            count = self.get_stage_count(stage, metrics)
            color = self.get_stage_color(stage, current_state, metrics)
            
            # Create status indicator
            if color == '#51cf66':
                status = f"✅ Complete ({count})"
            elif color == '#ff6b6b':
                status = "🔄 Processing..."
            else:
                status = "⏸️ Pending"
            
            st.markdown(f"""
            <div style="
                border-left: 4px solid {color};
                padding: 8px 12px;
                margin: 4px 0;
                background-color: {color}20;
                border-radius: 4px;
            ">
                <strong>{stage_info['name']}</strong><br>
                <small>{stage_info['description']}</small><br>
                <em>{status}</em>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Query Phase  
        st.markdown("**🔍 Query Phase** *(Run for each search)*")
        query_stages = ['retrieval', 'generation']
        
        for stage in query_stages:
            stage_info = self.stage_info[stage]
            count = self.get_stage_count(stage, metrics)
            color = self.get_stage_color(stage, current_state, metrics)
            
            if color == '#51cf66':
                status = f"✅ Ready ({count} searches)"
            elif color == '#ff6b6b':
                status = "🔄 Processing..."
            else:
                status = "⏸️ Ready to search"
            
            st.markdown(f"""
            <div style="
                border-left: 4px solid {color};
                padding: 8px 12px;
                margin: 4px 0;
                background-color: {color}20;
                border-radius: 4px;
            ">
                <strong>{stage_info['name']}</strong><br>
                <small>{stage_info['description']}</small><br>
                <em>{status}</em>
            </div>
            """, unsafe_allow_html=True)