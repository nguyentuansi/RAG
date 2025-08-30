"""
Visualization Components Module

This module contains interactive components for visualizing the RAG pipeline,
making it easy for beginners to understand how data flows through the system.

Components:
- PipelineFlowDiagram: Interactive flow diagram showing pipeline stages
- StageMonitor: Real-time monitoring of stage execution
- DataPreview: Live preview of data transformations

Usage:
    from visualizer.components import PipelineFlowDiagram, StageMonitor, DataPreview
    
    # In Streamlit app
    flow_diagram = PipelineFlowDiagram()
    flow_diagram.render(pipeline_status)
"""

from .pipeline_flow import PipelineFlowDiagram
from .stage_monitor import StageMonitor  
from .data_preview import DataPreview

__all__ = [
    "PipelineFlowDiagram",
    "StageMonitor", 
    "DataPreview"
]