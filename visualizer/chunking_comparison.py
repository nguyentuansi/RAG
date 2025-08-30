"""
Chunking Comparison Tool - Compare Different Text Chunking Strategies
Shows side-by-side comparison of current vs improved chunking methods
"""

import streamlit as st
import re
from typing import List, Dict, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.stages import TextChunking

class ChunkingStrategies:
    """Different text chunking strategies for comparison"""
    
    @staticmethod
    def current_approach(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """Current approach - character-based with basic separators"""
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        
        chunks = []
        if len(text) <= chunk_size:
            return [{"content": text, "size": len(text), "method": "single"}]
        
        current_chunk = ""
        for separator in separators:
            if separator in text:
                splits = text.split(separator)
                for split in splits:
                    if len(current_chunk) + len(split) + len(separator) > chunk_size:
                        if current_chunk:
                            chunks.append({
                                "content": current_chunk.strip(),
                                "size": len(current_chunk.strip()),
                                "method": f"split_on_{repr(separator)}"
                            })
                            # Simple character overlap
                            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
                            current_chunk = overlap_text + separator + split
                        else:
                            current_chunk = split
                    else:
                        current_chunk += separator + split if current_chunk else split
                
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "size": len(current_chunk.strip()),
                        "method": f"split_on_{repr(separator)}"
                    })
                break
        
        return chunks or [{"content": text, "size": len(text), "method": "fallback"}]
    
    @staticmethod
    def semantic_approach(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """Improved approach - semantic-aware chunking"""
        chunks = []
        
        # Step 1: Split into semantic units (paragraphs, sections)
        sections = ChunkingStrategies._split_into_sections(text)
        
        current_chunk = ""
        current_sentences = []
        
        for section in sections:
            sentences = ChunkingStrategies._split_into_sentences(section)
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(test_chunk) > chunk_size and current_chunk:
                    # Finalize current chunk
                    chunks.append({
                        "content": current_chunk.strip(),
                        "size": len(current_chunk.strip()),
                        "method": "semantic_sentence",
                        "sentences": len(current_sentences)
                    })
                    
                    # Start new chunk with overlap
                    overlap_sentences = ChunkingStrategies._get_sentence_overlap(
                        current_sentences, overlap
                    )
                    current_chunk = " ".join(overlap_sentences + [sentence])
                    current_sentences = overlap_sentences + [sentence]
                else:
                    # Add sentence to current chunk
                    current_chunk = test_chunk
                    current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "size": len(current_chunk.strip()),
                "method": "semantic_sentence",
                "sentences": len(current_sentences)
            })
        
        return chunks
    
    @staticmethod
    def _split_into_sections(text: str) -> List[str]:
        """Split text into logical sections"""
        # Look for clear section breaks
        section_patterns = [
            r'\n\n#+\s+.+\n',  # Markdown headers
            r'\n\n\d+\.\s+',   # Numbered sections
            r'\n\n[A-Z][^.]+:\n',  # Labeled sections
            r'\n\n'  # Double newlines (paragraphs)
        ]
        
        for pattern in section_patterns:
            if re.search(pattern, text):
                sections = re.split(pattern, text)
                return [section.strip() for section in sections if section.strip()]
        
        return [text]
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences, preserving context"""
        # More sophisticated sentence splitting
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # Handle special cases
        text = re.sub(r'(\w\.)\s+([A-Z])', r'\1\n\2', text)  # "Dr. Smith" -> "Dr.\nSmith"
        
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    @staticmethod
    def _get_sentence_overlap(sentences: List[str], overlap_chars: int) -> List[str]:
        """Get sentence-based overlap instead of character-based"""
        if not sentences or overlap_chars <= 0:
            return []
        
        # Take last 1-2 sentences that fit within overlap limit
        overlap_text = ""
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            test_overlap = sentence + " " + overlap_text if overlap_text else sentence
            if len(test_overlap) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                overlap_text = test_overlap
            else:
                break
        
        return overlap_sentences

class ChunkingComparison:
    """Streamlit interface for comparing chunking strategies"""
    
    def __init__(self):
        self.sample_texts = {
            "Technical Document": """
# AI Agent Integration

## Agent Protocol

1. Identity Maintenance: Always operate as Mix AI Assistant, never claim to be Claude or any other AI.

2. Knowledge Integration: Seamlessly integrate MixDbData functionality with conversational responses.

3. Create module template 4. Query data with SearchMixDbRequestModel 5. Render in views

## Response Guidelines

- Provide accurate, helpful responses about Mix development
- Use appropriate technical terminology
- Include relevant code examples when helpful
- Maintain professional but friendly tone

## Error Handling

When errors occur:
1. Acknowledge the issue clearly
2. Provide specific troubleshooting steps  
3. Offer alternative approaches
4. Escalate to human support if needed
""",
            "Business Document": """
Our company's quarterly performance exceeded expectations. Revenue grew by 15% compared to last quarter. The marketing team's new campaign generated 2,500 leads. Customer satisfaction scores improved to 4.2/5.0. We launched three new products successfully.

The engineering team delivered all planned features on time. Technical debt was reduced by 20%. Code coverage increased to 85%. Performance improved significantly with page load times under 2 seconds.

Looking forward, we plan to expand into European markets. A new office will open in Berlin next month. Hiring goals include 50 new employees across all departments. Investment in R&D will increase by 30%.
""",
            "User's Example": """
MixDbData 3. Create module template 4. Query data with SearchMixDbRequestModel 5. Render in views

🤖 AI Agent Integration

Agent Protocol

1. Identity Maintenance: Always operate as Mix AI...
"""
        }
    
    def run(self):
        """Main interface"""
        st.set_page_config(
            page_title="Chunking Comparison Tool",
            page_icon="✂️",
            layout="wide"
        )
        
        st.title("✂️ Text Chunking Comparison Tool")
        st.markdown("*Compare current vs improved chunking strategies*")
        
        # Input section
        st.subheader("📝 Input Text")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sample text selector
            selected_sample = st.selectbox(
                "Choose sample text:",
                options=list(self.sample_texts.keys()),
                help="Select a pre-written sample or choose 'Custom' to write your own"
            )
        
        with col2:
            if st.button("📋 Use Your Example", help="Load the text from your screenshot"):
                selected_sample = "User's Example"
        
        # Text input area
        if selected_sample in self.sample_texts:
            default_text = self.sample_texts[selected_sample]
        else:
            default_text = ""
        
        text_input = st.text_area(
            "Text to chunk:",
            value=default_text,
            height=150,
            help="Paste your text here to see how different chunking methods handle it"
        )
        
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze")
            return
        
        # Configuration
        st.subheader("⚙️ Chunking Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=100,
                max_value=1000,
                value=300,  # Smaller for better comparison
                help="Maximum number of characters per chunk"
            )
        
        with col2:
            overlap = st.slider(
                "Overlap (characters)",
                min_value=0,
                max_value=200,
                value=50,
                help="Characters to overlap between chunks"
            )
        
        with col3:
            show_metadata = st.checkbox(
                "Show Metadata",
                value=True,
                help="Display chunk size, method, and other details"
            )
        
        # Generate chunks
        st.subheader("🔍 Chunking Results")
        
        current_chunks = ChunkingStrategies.current_approach(text_input, chunk_size, overlap)
        semantic_chunks = ChunkingStrategies.semantic_approach(text_input, chunk_size, overlap)
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_chunks(
                "❌ Current Approach (Character-Based)",
                current_chunks,
                show_metadata,
                "error"
            )
        
        with col2:
            self._render_chunks(
                "✅ Improved Approach (Semantic-Based)",
                semantic_chunks,
                show_metadata,
                "success"
            )
        
        # Analysis
        self._render_analysis(current_chunks, semantic_chunks)
    
    def _render_chunks(self, title: str, chunks: List[Dict], show_metadata: bool, style: str):
        """Render chunks in a column"""
        st.markdown(f"### {title}")
        
        total_chunks = len(chunks)
        avg_size = sum(chunk['size'] for chunk in chunks) / total_chunks if chunks else 0
        
        # Summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Total Chunks", total_chunks)
        with col2:
            st.metric("📏 Avg Size", f"{avg_size:.0f} chars")
        
        # Individual chunks
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f"Chunk {i} ({chunk['size']} chars)", expanded=i <= 2):
                
                # Content with highlighting for readability
                content = chunk['content']
                
                # Highlight potential issues in current approach
                if style == "error":
                    # Highlight broken sentences (text that starts/ends mid-sentence)
                    if not content.strip()[0].isupper() or not content.strip().endswith(('.', '!', '?')):
                        st.warning("⚠️ **Potential issue**: Chunk breaks mid-sentence")
                
                st.markdown(f"```\n{content}\n```")
                
                if show_metadata:
                    st.caption(f"Method: {chunk['method']} | Size: {chunk['size']} chars")
                    if 'sentences' in chunk:
                        st.caption(f"Sentences: {chunk['sentences']}")
    
    def _render_analysis(self, current_chunks: List[Dict], semantic_chunks: List[Dict]):
        """Render comparison analysis"""
        st.subheader("📊 Analysis & Recommendations")
        
        # Calculate quality metrics
        current_broken = sum(1 for chunk in current_chunks 
                           if not chunk['content'].strip()[0].isupper() or 
                           not chunk['content'].strip().endswith(('.', '!', '?')))
        
        semantic_broken = sum(1 for chunk in semantic_chunks 
                            if not chunk['content'].strip()[0].isupper() or 
                            not chunk['content'].strip().endswith(('.', '!', '?')))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔴 Current: Broken Chunks",
                current_broken,
                delta=f"{current_broken/len(current_chunks)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                "🟢 Semantic: Broken Chunks", 
                semantic_broken,
                delta=f"{semantic_broken/len(semantic_chunks)*100:.1f}%"
            )
        
        with col3:
            improvement = (current_broken - semantic_broken) / max(current_broken, 1) * 100
            st.metric(
                "📈 Improvement",
                f"{improvement:.1f}%",
                delta=f"{improvement:.1f}% better"
            )
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if current_broken > semantic_broken:
            st.success("""
            ✅ **The semantic approach is significantly better because:**
            - Preserves sentence boundaries
            - Maintains context and meaning
            - Uses intelligent overlap (whole sentences vs partial characters)
            - Better for AI comprehension and search quality
            """)
        else:
            st.info("Both approaches perform similarly for this text.")
        
        # Implementation guide
        with st.expander("🛠️ How to Implement Semantic Chunking"):
            st.markdown("""
            **Key Improvements Needed:**
            
            1. **Sentence-Aware Splitting**: Use NLP libraries or regex to detect sentence boundaries properly
            2. **Semantic Overlap**: Overlap by complete sentences, not character count
            3. **Context Preservation**: Never break in the middle of a sentence or concept
            4. **Hierarchical Splitting**: Split by sections/paragraphs first, then sentences
            5. **Quality Validation**: Check that each chunk starts/ends at logical boundaries
            
            **Implementation Steps:**
            ```python
            # 1. Replace character-based splitting with sentence-based
            # 2. Use semantic overlap (complete sentences)
            # 3. Add boundary validation
            # 4. Preserve document structure (headers, lists, etc.)
            ```
            """)

def main():
    """Run the chunking comparison tool"""
    app = ChunkingComparison()
    app.run()

if __name__ == "__main__":
    main()