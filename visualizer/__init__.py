"""
Visualizer Module

Interactive dashboard and components for visualizing the RAG pipeline.
Built with Streamlit for beginner-friendly real-time monitoring.

Main Components:
- Streamlit dashboard app
- Interactive pipeline flow diagram
- Real-time stage monitoring
- Live data previews

Usage:
    # Run the dashboard
    streamlit run visualizer/app.py
    
    # Or import components directly
    from visualizer.components import PipelineFlowDiagram
"""

from .app import RAGDashboard

__version__ = "0.1.0"
__all__ = ["RAGDashboard"]