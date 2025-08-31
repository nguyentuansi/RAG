#!/usr/bin/env python3
"""
Download a single embedding model to test the process
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import time
from sentence_transformers import SentenceTransformer

def download_default_model():
    """Download the default model (all-mpnet-base-v2) for testing"""
    
    model_name = "all-mpnet-base-v2"
    model_path = "sentence-transformers/all-mpnet-base-v2"
    
    print(f"🔄 Downloading default model: {model_name}")
    print(f"📍 Path: {model_path}")
    print(f"💡 This model will be used as the default in the RAG system")
    
    try:
        start_time = time.time()
        
        print("📥 Starting download...")
        model = SentenceTransformer(model_path)
        
        print("🧪 Testing model...")
        test_embedding = model.encode("This is a test sentence.", convert_to_numpy=True)
        
        download_time = time.time() - start_time
        
        print(f"✅ SUCCESS!")
        print(f"⏱️  Download time: {download_time:.2f} seconds")
        print(f"📊 Embedding dimensions: {test_embedding.shape[0]}")
        print(f"💾 Model cached locally for future use")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    print("🤖 Single Model Download Test")
    print("=" * 40)
    
    success = download_default_model()
    
    if success:
        print("\n🎉 Model download successful!")
        print("✅ The RAG system can now use this model offline")
        print("🚀 Subsequent loads will be much faster")
    else:
        print("\n❌ Download failed. Check your internet connection.")
        sys.exit(1)