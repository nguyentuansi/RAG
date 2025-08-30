"""
Improved Text Chunking Strategies
Semantic-aware chunking that preserves meaning and context
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies"""
    chunk_size: int = 512
    overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1024
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    use_semantic_overlap: bool = True

class SemanticChunker:
    """Advanced text chunker that preserves semantic meaning"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        
        # Comprehensive sentence boundary patterns
        self.sentence_endings = re.compile(
            r'(?<=[.!?:;…])\s+(?=[A-Z])|'  # Standard sentence endings + colons/semicolons/ellipses
            r'(?<=[.!?:;])\n+(?=[A-Z])|'   # With newlines
            r'(?<=[.!?])\s*\n\s*(?=[A-Z•\-\d])|'  # Line breaks to bullets/dashes/numbers
            r'\n\s*(?=\d+\.)|'             # Numbered lists
            r'\n\s*(?=[•\-\*])|'           # Bullet points
            r'(?<=:)\n+(?=[A-Z])|'         # Colons followed by newline and caps
            r'(?<=```)\n+|'                # Code blocks
            r'\n\s*#{1,6}\s|'              # Markdown headers
            r'\n\s*>\s|'                   # Blockquotes
            r'(?<=[.!?])\s+(?=\d+\.)'      # Before numbered items
        )
        
        # Section boundary patterns
        self.section_patterns = [
            re.compile(r'\n\s*#{1,6}\s+.+\n'),  # Markdown headers
            re.compile(r'\n\s*\d+\.\s+[A-Z]'),   # Numbered sections  
            re.compile(r'\n\s*[A-Z][^.]{2,30}:\s*\n'),  # Labeled sections
            re.compile(r'\n\s*\n\s*'),  # Double newlines (paragraphs)
        ]
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Main chunking method - preserves semantic meaning
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text.strip():
            return []
        
        # Step 1: Identify document structure
        sections = self._identify_sections(text)
        
        # Step 2: Process each section
        chunks = []
        for section_idx, section in enumerate(sections):
            section_chunks = self._chunk_section(section, section_idx)
            chunks.extend(section_chunks)
        
        # Step 3: Post-process chunks
        chunks = self._post_process_chunks(chunks)
        
        # Step 4: Add metadata
        for i, chunk in enumerate(chunks):
            chunk.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'quality_score': self._calculate_quality_score(chunk),
                'chunking_method': 'semantic'
            })
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[Dict]:
        """Identify logical sections in the document"""
        sections = []
        current_pos = 0
        
        # Look for section boundaries
        for pattern in self.section_patterns:
            matches = list(pattern.finditer(text))
            
            if matches:
                # Split by this pattern
                sections = []
                last_end = 0
                
                for match in matches:
                    # Add content before this boundary
                    if match.start() > last_end:
                        content = text[last_end:match.start()].strip()
                        if content:
                            sections.append({
                                'content': content,
                                'type': 'content',
                                'start': last_end,
                                'end': match.start()
                            })
                    
                    # Add the boundary itself (header, etc.)
                    boundary_content = text[match.start():match.end()].strip()
                    if boundary_content:
                        sections.append({
                            'content': boundary_content,
                            'type': 'boundary',
                            'start': match.start(),
                            'end': match.end()
                        })
                    
                    last_end = match.end()
                
                # Add remaining content
                if last_end < len(text):
                    content = text[last_end:].strip()
                    if content:
                        sections.append({
                            'content': content,
                            'type': 'content',
                            'start': last_end,
                            'end': len(text)
                        })
                
                return sections
        
        # No clear sections found, treat as single section
        return [{
            'content': text,
            'type': 'content',
            'start': 0,
            'end': len(text)
        }]
    
    def _chunk_section(self, section: Dict, section_idx: int) -> List[Dict]:
        """Chunk a single section while preserving meaning"""
        content = section['content']
        section_type = section.get('type', 'content')
        
        # Handle boundary sections (headers, etc.) - keep them intact
        if section_type == 'boundary':
            return [{
                'content': content.strip(),
                'size': len(content.strip()),
                'section_idx': section_idx,
                'section_type': section_type,
                'is_boundary': True
            }]
        
        # For content sections, use sentence-aware chunking
        sentences = self._split_into_sentences(content)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_size = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > self.config.chunk_size and current_chunk_sentences:
                # Finalize current chunk
                chunk_content = ' '.join(current_chunk_sentences).strip()
                chunks.append({
                    'content': chunk_content,
                    'size': len(chunk_content),
                    'sentences': len(current_chunk_sentences),
                    'section_idx': section_idx,
                    'section_type': section_type,
                    'is_boundary': False
                })
                
                # Start new chunk with semantic overlap
                if self.config.use_semantic_overlap:
                    overlap_sentences = self._get_semantic_overlap(
                        current_chunk_sentences, self.config.overlap
                    )
                    current_chunk_sentences = overlap_sentences
                    current_size = sum(len(s) for s in overlap_sentences)
                else:
                    current_chunk_sentences = []
                    current_size = 0
            
            # Add current sentence
            current_chunk_sentences.append(sentence)
            current_size += sentence_size + 1  # +1 for space
            i += 1
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_content = ' '.join(current_chunk_sentences).strip()
            chunks.append({
                'content': chunk_content,
                'size': len(chunk_content),
                'sentences': len(current_chunk_sentences),
                'section_idx': section_idx,
                'section_type': section_type,
                'is_boundary': False
            })
        
        # DEBUG: Print chunk info (removed for cleaner output)
        # print(f"[DEBUG] Created {len(chunks)} chunks for section {section_idx}")
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved boundary detection"""
        # Handle common abbreviations and technical terms
        abbreviations = {
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.',
            'etc.', 'vs.', 'e.g.', 'i.e.', 'Jr.', 'Sr.', 'Ph.D.', 'M.D.',
            'API', 'URL', 'HTTP', 'HTML', 'CSS', 'JS', 'SQL', 'JSON', 'XML',
            'CMS', 'MCP', 'UI', 'UX'  # Technical abbreviations
        }
        
        # DEBUG: Print info about sentence splitting (removed for cleaner output)
        # print(f"[DEBUG] _split_into_sentences called with {len(text)} chars")
        
        # Protect abbreviations from being split
        protected_text = text
        replacements = {}
        
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            protected_text = protected_text.replace(f"{abbrev}.", placeholder)
            replacements[placeholder] = f"{abbrev}."
        
        # First try regex-based splitting
        sentences = self.sentence_endings.split(protected_text)
        
        # If regex produces very few sentences, fall back to line-based splitting
        if len(sentences) < len(text.split('\n')) // 3:
            # For technical content, split on meaningful line breaks
            lines = text.split('\n')
            sentences = []
            current_sentence = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(' '.join(current_sentence))
                        current_sentence = []
                    continue
                
                # Check if this is a complete thought
                if (line.endswith(('.', '!', '?', ':', ';')) or 
                    line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                    line.startswith('#') or
                    '```' in line):
                    current_sentence.append(line)
                    sentences.append(' '.join(current_sentence))
                    current_sentence = []
                else:
                    current_sentence.append(line)
            
            if current_sentence:
                sentences.append(' '.join(current_sentence))
        
        # Restore abbreviations and clean up
        clean_sentences = []
        for sentence in sentences:
            # Restore abbreviations
            for placeholder, abbrev in replacements.items():
                sentence = sentence.replace(placeholder, abbrev)
            
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # Filter out very short fragments
                clean_sentences.append(sentence)
        
        # DEBUG: Print results (removed for cleaner output)
        # print(f"[DEBUG] Split into {len(clean_sentences)} sentences")
        
        return clean_sentences
    
    def _get_semantic_overlap(self, sentences: List[str], target_overlap_chars: int) -> List[str]:
        """Get sentence-based overlap that preserves meaning"""
        if not sentences or target_overlap_chars <= 0:
            return []
        
        # Take complete sentences from the end that fit within overlap limit
        overlap_sentences = []
        total_chars = 0
        
        for sentence in reversed(sentences):
            sentence_chars = len(sentence) + 1  # +1 for space
            
            if total_chars + sentence_chars <= target_overlap_chars:
                overlap_sentences.insert(0, sentence)
                total_chars += sentence_chars
            else:
                break
        
        return overlap_sentences
    
    def _validate_chunk_completeness(self, chunk_content: str) -> bool:
        """Validate that a chunk contains complete thoughts"""
        content = chunk_content.strip()
        
        if not content:
            return False
        
        # Minimum content length
        if len(content) < 20:
            return False
        
        # Allow markdown headers - these are always valid starts
        if content.startswith('#'):
            return True
        
        # Allow bullet points and lists
        if any(content.startswith(prefix) for prefix in ('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            return True
        
        # Allow technical content patterns
        if any(content.startswith(prefix) for prefix in ('**', '__', '`', 'Model Context Protocol', 'Mixcore CMS')):
            return True
        
        # Check if chunk ends with proper sentence ending or structural elements
        good_endings = ('.', '!', '?', ':', ';', '```', ')', ']', '}', '|', '*', '-', 'md)')
        if content.endswith(good_endings):
            return True
        
        # Allow table rows and markdown structure
        lines = content.split('\n')
        last_line = lines[-1].strip() if lines else ''
        if last_line.startswith('|') or last_line.endswith('|'):
            return True
        
        # Allow lists and structured content
        if any(last_line.startswith(prefix) for prefix in ('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            return True
        
        # Check if chunk starts properly (not mid-word)
        first_char = content[0]
        if first_char.islower() and not content.startswith(('and', 'or', 'but', 'the', 'a', 'an')):
            # Could be continuation of previous sentence - reject only if truly broken
            return False
        
        return True
    
    def _post_process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Post-process chunks to ensure quality"""
        processed_chunks = []
        
        for chunk in chunks:
            content = chunk['content']
            
            # Skip empty chunks
            if not content.strip():
                continue
            
            # Validate chunk completeness
            if not self._validate_chunk_completeness(content):
                logger.warning(f"Chunk failed validation: '{content[:50]}...'")
                # Try to fix by merging with previous chunk
                if processed_chunks:
                    processed_chunks[-1]['content'] += ' ' + content
                    processed_chunks[-1]['size'] = len(processed_chunks[-1]['content'])
                    continue
            
            # Handle chunks that are too small
            if len(content) < self.config.min_chunk_size and not chunk.get('is_boundary', False):
                # Try to merge with previous chunk
                if processed_chunks and len(processed_chunks[-1]['content']) + len(content) < self.config.max_chunk_size:
                    processed_chunks[-1]['content'] += ' ' + content
                    processed_chunks[-1]['size'] = len(processed_chunks[-1]['content'])
                    continue
            
            # Handle chunks that are too large
            if len(content) > self.config.max_chunk_size:
                # Re-chunk this content more aggressively
                sub_chunks = self._emergency_split(content, chunk)
                processed_chunks.extend(sub_chunks)
                continue
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _emergency_split(self, content: str, original_chunk: Dict) -> List[Dict]:
        """Emergency splitting for oversized chunks"""
        # Use more aggressive splitting while preserving some structure
        parts = []
        current_part = ""
        
        # Split on any reasonable boundary
        boundaries = ['. ', '! ', '? ', '; ', '\n', ', ']
        
        for boundary in boundaries:
            if boundary in content:
                splits = content.split(boundary)
                for split in splits[:-1]:  # All but last
                    test_part = current_part + boundary + split if current_part else split
                    
                    if len(test_part) < self.config.chunk_size:
                        current_part = test_part
                    else:
                        if current_part:
                            parts.append(current_part.strip())
                        current_part = split
                
                # Handle the last split
                if splits:
                    current_part += boundary + splits[-1]
                break
        
        # Add final part
        if current_part:
            parts.append(current_part.strip())
        
        # Convert to chunk format
        chunks = []
        for i, part in enumerate(parts):
            if part.strip():
                chunk = original_chunk.copy()
                chunk.update({
                    'content': part.strip(),
                    'size': len(part.strip()),
                    'emergency_split': True,
                    'part_index': i
                })
                chunks.append(chunk)
        
        return chunks
    
    def _calculate_quality_score(self, chunk: Dict) -> float:
        """Calculate a quality score for the chunk"""
        content = chunk['content']
        score = 1.0
        
        # Penalize broken sentences (doesn't start with capital or end with punctuation)
        if content and not content[0].isupper():
            score -= 0.3
        
        if content and not content.rstrip()[-1] in '.!?':
            score -= 0.3
        
        # Reward good size
        size = chunk['size']
        if self.config.min_chunk_size <= size <= self.config.chunk_size:
            score += 0.2
        elif size > self.config.max_chunk_size:
            score -= 0.4
        
        # Bonus for sentence boundaries
        if chunk.get('sentences', 0) > 0:
            score += 0.1
        
        # Penalty for emergency splits
        if chunk.get('emergency_split', False):
            score -= 0.5
        
        return max(0.0, min(1.0, score))

class ImprovedTextChunking:
    """Drop-in replacement for the original TextChunking class"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Create chunking configuration
        chunk_config = config.get('chunking', {})
        self.chunking_config = ChunkingConfig(
            chunk_size=chunk_config.get('chunk_size', 512),
            overlap=chunk_config.get('overlap', 50),
            min_chunk_size=chunk_config.get('min_chunk_size', 100),
            max_chunk_size=chunk_config.get('max_chunk_size', 1024),
            preserve_sentences=chunk_config.get('preserve_sentences', True),
            preserve_paragraphs=chunk_config.get('preserve_paragraphs', True),
            use_semantic_overlap=chunk_config.get('use_semantic_overlap', True)
        )
        
        self.chunker = SemanticChunker(self.chunking_config)
    
    async def process(
        self, 
        input_data: Dict,
        progress_hook: Optional[callable] = None,
        data_hook: Optional[callable] = None
    ) -> Dict:
        """Process documents using improved chunking"""
        documents = input_data['documents']
        chunks = []
        
        logger.info(f"Starting improved chunking for {len(documents)} documents")
        
        for doc_idx, document in enumerate(documents):
            if progress_hook:
                progress_hook(doc_idx, len(documents))
            
            text = document['content']
            
            # Use semantic chunker
            doc_chunks = self.chunker.chunk_text(text)
            
            # Convert to expected format
            for chunk_idx, chunk_data in enumerate(doc_chunks):
                chunk = {
                    'id': f"chunk_{document['id']}_{chunk_idx}",
                    'content': chunk_data['content'],
                    'document_id': document['id'],
                    'document_title': document['title'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(doc_chunks),
                    'size': chunk_data['size'],
                    'metadata': {
                        'source': document['source'],
                        'format': document['format'],
                        'quality_score': chunk_data.get('quality_score', 0.0),
                        'chunking_method': 'semantic',
                        'sentences': chunk_data.get('sentences', 0),
                        'is_boundary': chunk_data.get('is_boundary', False)
                    }
                }
                chunks.append(chunk)
                
                # Send preview to UI
                if data_hook and chunk_idx < 3:
                    preview = {
                        'chunk_id': chunk['id'],
                        'content_preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                        'size': chunk['size'],
                        'quality_score': chunk_data.get('quality_score', 0.0),
                        'sentences': chunk_data.get('sentences', 0)
                    }
                    data_hook(preview)
        
        logger.info(f"Improved chunking completed: {len(chunks)} chunks generated")
        
        return {
            'chunks': chunks,
            'total_chunks': len(chunks),
            'chunking_method': 'semantic',
            'average_quality': sum(chunk['metadata']['quality_score'] for chunk in chunks) / len(chunks) if chunks else 0
        }