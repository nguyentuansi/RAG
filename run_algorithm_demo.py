#!/usr/bin/env python3
"""
Algorithm Demonstration - Show exactly how chunking changes
Run this to see step-by-step how the algorithms differ
"""

import re
from typing import List

def current_algorithm_demo(text: str, chunk_size: int = 150) -> List[str]:
    """Demonstrate the current character-based algorithm"""
    print("🚨 CURRENT ALGORITHM (Character-Based) - THE PROBLEM")
    print("=" * 60)
    
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    chunks = []
    
    print(f"📋 Input text: {text}")
    print(f"📏 Chunk size limit: {chunk_size} characters")
    print()
    
    for separator in separators:
        if separator in text:
            print(f"🔍 Trying separator: {repr(separator)}")
            splits = text.split(separator)
            print(f"📊 Splits into {len(splits)} parts: {splits[:3]}{'...' if len(splits) > 3 else ''}")
            
            current_chunk = ""
            
            for i, split in enumerate(splits):
                test_chunk = current_chunk + separator + split if current_chunk else split
                print(f"   Step {i+1}: Adding '{split[:30]}{'...' if len(split) > 30 else ''}'")
                print(f"   Would be: '{test_chunk[:50]}{'...' if len(test_chunk) > 50 else ''}'")
                print(f"   Length would be: {len(test_chunk)} chars")
                
                if len(test_chunk) > chunk_size and current_chunk:
                    print(f"   🚨 EXCEEDS LIMIT! Cutting chunk at {len(current_chunk)} chars")
                    print(f"   ❌ Chunk ends: '...{current_chunk[-30:]}'")
                    chunks.append(current_chunk.strip())
                    current_chunk = split
                    print(f"   ❌ Next chunk starts: '{split[:30]}{'...' if len(split) > 30 else ''}'")
                    print(f"   ⚠️  BROKEN: Cut mid-context!")
                else:
                    current_chunk = test_chunk
                    print(f"   ✅ Added to chunk (total: {len(current_chunk)} chars)")
                print()
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            break
    
    print(f"📊 RESULT: {len(chunks)} chunks created")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars): '{chunk[:60]}{'...' if len(chunk) > 60 else ''}'")
        # Check if broken
        if chunk and (not chunk[0].isupper() or not chunk.rstrip()[-1] in '.!?'):
            print(f"   ⚠️  QUALITY ISSUE: Broken sentence boundaries!")
    print("\n" + "="*60 + "\n")
    
    return chunks

def semantic_algorithm_demo(text: str, chunk_size: int = 150) -> List[str]:
    """Demonstrate the semantic sentence-aware algorithm"""
    print("✅ SEMANTIC ALGORITHM (Sentence-Based) - THE SOLUTION")
    print("=" * 60)
    
    # Step 1: Split into sentences
    print("🎯 STEP 1: Identify complete sentences")
    
    # Simple sentence splitting for demo
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
    
    print(f"📋 Input text: {text}")
    print(f"📏 Chunk size limit: {chunk_size} characters")
    print(f"🔍 Found {len(sentences)} complete sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"   {i}. '{sentence}' ({len(sentence)} chars)")
    print()
    
    # Step 2: Build chunks by complete sentences
    print("🎯 STEP 2: Build chunks using complete sentences only")
    
    chunks = []
    current_chunk_sentences = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_size = len(sentence)
        print(f"   Processing sentence {i+1}: '{sentence[:40]}{'...' if len(sentence) > 40 else ''}'")
        print(f"   Sentence size: {sentence_size} chars")
        print(f"   Current chunk size: {current_size} chars")
        print(f"   Would total: {current_size + sentence_size} chars")
        
        if current_size + sentence_size > chunk_size and current_chunk_sentences:
            print(f"   🎯 WOULD EXCEED LIMIT! Finalizing current chunk with complete sentences")
            chunk_content = ' '.join(current_chunk_sentences)
            chunks.append(chunk_content)
            print(f"   ✅ Finalized chunk: '{chunk_content[:50]}{'...' if len(chunk_content) > 50 else ''}'")
            print(f"   ✅ QUALITY: Ends with complete sentence!")
            
            # Start new chunk
            current_chunk_sentences = [sentence]
            current_size = sentence_size
            print(f"   🎯 Starting new chunk with: '{sentence[:40]}{'...' if len(sentence) > 40 else ''}'")
            print(f"   ✅ QUALITY: Starts with complete sentence!")
        else:
            current_chunk_sentences.append(sentence)
            current_size += sentence_size + 1  # +1 for space
            print(f"   ✅ Added to current chunk (total: {current_size} chars)")
        print()
    
    # Add final chunk
    if current_chunk_sentences:
        chunk_content = ' '.join(current_chunk_sentences)
        chunks.append(chunk_content)
        print(f"   ✅ Final chunk: '{chunk_content[:50]}{'...' if len(chunk_content) > 50 else ''}'")
    
    print(f"📊 RESULT: {len(chunks)} chunks created")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars): '{chunk[:60]}{'...' if len(chunk) > 60 else ''}'")
        # Check quality
        if chunk and chunk[0].isupper() and chunk.rstrip()[-1] in '.!?':
            print(f"   ✅ QUALITY: Perfect sentence boundaries!")
        else:
            print(f"   ⚠️  QUALITY ISSUE: Check boundaries")
    print("\n" + "="*60 + "\n")
    
    return chunks

def compare_results(current_chunks: List[str], semantic_chunks: List[str]):
    """Compare the results of both algorithms"""
    print("📊 COMPARISON ANALYSIS")
    print("=" * 60)
    
    print(f"Current method: {len(current_chunks)} chunks")
    print(f"Semantic method: {len(semantic_chunks)} chunks")
    print()
    
    # Quality analysis
    current_broken = 0
    semantic_broken = 0
    
    for chunk in current_chunks:
        if chunk and (not chunk[0].isupper() or not chunk.rstrip()[-1] in '.!?'):
            current_broken += 1
    
    for chunk in semantic_chunks:
        if chunk and (not chunk[0].isupper() or not chunk.rstrip()[-1] in '.!?'):
            semantic_broken += 1
    
    print(f"❌ Current method broken chunks: {current_broken}/{len(current_chunks)} ({current_broken/len(current_chunks)*100:.1f}%)")
    print(f"✅ Semantic method broken chunks: {semantic_broken}/{len(semantic_chunks)} ({semantic_broken/len(semantic_chunks)*100:.1f}%)")
    
    improvement = (current_broken - semantic_broken) / max(current_broken, 1) * 100
    print(f"📈 Quality improvement: {improvement:.1f}%")
    print()
    
    print("🎯 KEY INSIGHT:")
    print("The semantic method preserves complete thoughts by:")
    print("1. 🧠 Understanding sentence boundaries (not just character counts)")
    print("2. ✅ Never cutting mid-sentence")  
    print("3. 🔄 Using semantic overlap (complete sentences)")
    print("4. 📊 Validating chunk quality")
    
def main():
    """Run the algorithm demonstration"""
    
    # Your example from the screenshot
    test_text = """MixDbData 3. Create module template 4. Query data with SearchMixDbRequestModel 5. Render in views. AI Agent Integration. Agent Protocol. 1. Identity Maintenance: Always operate as Mix AI Assistant, never claim to be Claude or any other AI."""
    
    print("🔬 CHUNKING ALGORITHM DEMONSTRATION")
    print("=" * 80)
    print("This shows exactly HOW the algorithm changes to fix cut-off sentences")
    print("=" * 80)
    print()
    
    # Demonstrate both algorithms
    current_chunks = current_algorithm_demo(test_text, chunk_size=150)  # Smaller size for demo
    semantic_chunks = semantic_algorithm_demo(test_text, chunk_size=150)
    
    # Compare results
    compare_results(current_chunks, semantic_chunks)
    
    print("\n🧪 TO TEST WITH YOUR OWN TEXT:")
    print("1. python run_chunking_comparison.py  # Full comparison tool")
    print("2. python run.py --mode simple        # Main app with chunking options")

if __name__ == "__main__":
    main()