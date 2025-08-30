"""
Index Gap Analyzer - Identify what data should be indexed but isn't
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
import glob
import os

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class IndexGapAnalyzer:
    """Analyze gaps between uploaded documents and indexed vectors"""
    
    def __init__(self):
        self.client = None
        if QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient("localhost", port=6333, check_compatibility=False)
            except:
                pass
    
    def analyze_index_gaps(self) -> Dict:
        """Analyze what documents should be indexed vs what actually is"""
        
        analysis = {
            'indexed_files': set(),
            'uploaded_files': set(), 
            'missing_files': set(),
            'orphaned_indices': set(),
            'file_details': []
        }
        
        # Get uploaded files from typical upload directories
        upload_dirs = [
            '/tmp/rag_uploads',
            './uploads',
            './documents'
        ]
        
        for upload_dir in upload_dirs:
            if os.path.exists(upload_dir):
                for file_path in glob.glob(f"{upload_dir}/**/*", recursive=True):
                    if os.path.isfile(file_path):
                        analysis['uploaded_files'].add(file_path)
        
        # Get indexed files from Qdrant
        if self.client:
            try:
                results = self.client.scroll(
                    collection_name="rag_documents",
                    limit=1000,  # Get more data for comprehensive analysis
                    with_payload=True
                )[0]
                
                indexed_sources = set()
                for point in results:
                    source_path = point.payload.get('source', '')
                    if source_path:
                        indexed_sources.add(source_path)
                        analysis['indexed_files'].add(source_path)
                
                # Analyze each uploaded file
                for uploaded_file in analysis['uploaded_files']:
                    file_indexed = uploaded_file in indexed_sources
                    file_size = os.path.getsize(uploaded_file) if os.path.exists(uploaded_file) else 0
                    
                    analysis['file_details'].append({
                        'file_path': uploaded_file,
                        'file_name': Path(uploaded_file).name,
                        'indexed': file_indexed,
                        'size': file_size,
                        'status': '✅ Indexed' if file_indexed else '❌ Missing'
                    })
                    
                    if not file_indexed:
                        analysis['missing_files'].add(uploaded_file)
                
                # Find orphaned indices (indexed files that no longer exist)
                for indexed_file in indexed_sources:
                    if not os.path.exists(indexed_file):
                        analysis['orphaned_indices'].add(indexed_file)
                        
            except Exception as e:
                st.error(f"Error analyzing indexed files: {e}")
        
        return analysis
    
    def render_gap_analysis(self):
        """Render the index gap analysis in Streamlit"""
        
        st.subheader("🔍 Index Gap Analysis")
        st.markdown("*Find out exactly which files are indexed and which are missing*")
        
        if not QDRANT_AVAILABLE or not self.client:
            st.error("❌ Cannot connect to Qdrant for gap analysis")
            return
        
        with st.spinner("Analyzing index gaps..."):
            analysis = self.analyze_index_gaps()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📁 Files Uploaded",
                len(analysis['uploaded_files']),
                help="Total files found in upload directories"
            )
        
        with col2:
            st.metric(
                "✅ Files Indexed", 
                len(analysis['indexed_files']),
                help="Files that have been successfully indexed"
            )
        
        with col3:
            missing_count = len(analysis['missing_files'])
            st.metric(
                "❌ Missing Files",
                missing_count,
                delta=-missing_count if missing_count > 0 else None,
                help="Files uploaded but not indexed"
            )
        
        with col4:
            orphaned_count = len(analysis['orphaned_indices'])
            st.metric(
                "👻 Orphaned Indices",
                orphaned_count,
                delta=-orphaned_count if orphaned_count > 0 else None,
                help="Indexed files that no longer exist"
            )
        
        # Detailed file analysis
        if analysis['file_details']:
            st.subheader("📋 File-by-File Analysis")
            
            df = pd.DataFrame(analysis['file_details'])
            df['Size (KB)'] = (df['size'] / 1024).round(2)
            
            # Add filters
            col1, col2 = st.columns(2)
            
            with col1:
                status_filter = st.selectbox(
                    "Filter by Status:",
                    ["All", "✅ Indexed", "❌ Missing"]
                )
            
            with col2:
                show_paths = st.checkbox("Show Full Paths", False)
            
            # Apply filters
            filtered_df = df.copy()
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
            
            # Select columns to display
            display_cols = ['file_name', 'status', 'Size (KB)']
            if show_paths:
                display_cols.insert(1, 'file_path')
            
            # Color code the dataframe
            def color_status(val):
                if val == '✅ Indexed':
                    return 'background-color: #d4edda'
                elif val == '❌ Missing':
                    return 'background-color: #f8d7da'
                return ''
            
            styled_df = filtered_df[display_cols].style.map(
                color_status, subset=['status']
            )
            
            st.dataframe(styled_df, use_container_width=True)
        
        # Action recommendations
        self._render_recommendations(analysis)
    
    def _render_recommendations(self, analysis: Dict):
        """Render action recommendations based on gap analysis"""
        
        st.subheader("💡 Recommended Actions")
        
        if analysis['missing_files']:
            st.error(f"❌ **{len(analysis['missing_files'])} files need indexing**")
            
            with st.expander("🔧 How to fix missing files"):
                st.markdown("""
                **Option 1: Reprocess All Documents**
                1. Go to 'Document Upload' tab
                2. Upload the missing files again
                3. Click 'Process Documents'
                
                **Option 2: Batch Reprocessing**
                1. Select all files in the upload directory
                2. Use the bulk processing option
                3. Monitor processing status
                
                **Option 3: Individual File Processing**
                """)
                
                # List missing files
                st.write("**Missing Files:**")
                for missing_file in list(analysis['missing_files'])[:10]:  # Show first 10
                    st.write(f"• {Path(missing_file).name}")
                
                if len(analysis['missing_files']) > 10:
                    st.write(f"... and {len(analysis['missing_files']) - 10} more")
        
        if analysis['orphaned_indices']:
            st.warning(f"⚠️ **{len(analysis['orphaned_indices'])} orphaned indices found**")
            
            with st.expander("🧹 How to clean orphaned indices"):
                st.markdown("""
                **These indexed entries point to files that no longer exist.**
                
                **Recommended Action:**
                1. Clean up the vector database
                2. Remove stale entries
                3. Rebuild index from current files
                """)
                
                st.write("**Orphaned Indices:**")
                for orphaned in list(analysis['orphaned_indices'])[:5]:
                    st.write(f"• {Path(orphaned).name}")
        
        if not analysis['missing_files'] and not analysis['orphaned_indices']:
            st.success("✅ **Perfect Index Health!**")
            st.info("All uploaded files are properly indexed and no orphaned entries found.")


def render_index_gap_analysis():
    """Convenience function to render index gap analysis"""
    analyzer = IndexGapAnalyzer()
    analyzer.render_gap_analysis()