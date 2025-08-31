#!/usr/bin/env python3
"""
Download and cache embedding models locally for faster loading
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import time
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_models():
    """Download all embedding models to local cache"""
    
    # Models to download (from our updated configuration)
    models_to_download = {
        "all-MiniLM-L6-v2": {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "🚀 Fastest - Great for speed, good accuracy (384D)"
        },
        "all-mpnet-base-v2": {
            "path": "sentence-transformers/all-mpnet-base-v2", 
            "description": "⚖️ Balanced - Excellent accuracy-speed balance (768D)"
        },
        "bge-small-en-v1.5": {
            "path": "BAAI/bge-small-en-v1.5",
            "description": "🎯 Efficient - High accuracy with small size (384D)"
        },
        "e5-base-v2": {
            "path": "intfloat/e5-base-v2",
            "description": "🧠 Accurate - Microsoft E5, very reliable (768D)"
        },
        "bge-base-en-v1.5": {
            "path": "BAAI/bge-base-en-v1.5",
            "description": "🏆 Premium - SOTA accuracy for English (768D)"
        }
    }
    
    print("🔄 Starting model download process...")
    print(f"📁 Models will be cached in: {os.path.expanduser('~/.cache/huggingface/hub')}")
    print("=" * 70)
    
    successful_downloads = []
    failed_downloads = []
    total_start_time = time.time()
    
    for model_name, info in models_to_download.items():
        print(f"\n📦 Downloading {model_name}")
        print(f"   {info['description']}")
        print(f"   Path: {info['path']}")
        
        try:
            start_time = time.time()
            
            # Download and cache the model
            model = SentenceTransformer(info['path'])
            
            # Test encoding to ensure model works
            test_embedding = model.encode("This is a test sentence.", convert_to_numpy=True)
            
            download_time = time.time() - start_time
            
            print(f"   ✅ SUCCESS - Downloaded in {download_time:.2f}s")
            print(f"   📊 Embedding shape: {test_embedding.shape}")
            print(f"   💾 Model cached locally")
            
            successful_downloads.append({
                'name': model_name,
                'path': info['path'],
                'time': download_time,
                'dimensions': test_embedding.shape[0]
            })
            
        except Exception as e:
            print(f"   ❌ FAILED - {str(e)}")
            failed_downloads.append({
                'name': model_name,
                'path': info['path'],
                'error': str(e)
            })
    
    total_time = time.time() - total_start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 70)
    
    if successful_downloads:
        print(f"\n✅ Successfully downloaded {len(successful_downloads)} models:")
        for model in successful_downloads:
            print(f"   • {model['name']} ({model['dimensions']}D) - {model['time']:.2f}s")
    
    if failed_downloads:
        print(f"\n❌ Failed to download {len(failed_downloads)} models:")
        for model in failed_downloads:
            print(f"   • {model['name']}: {model['error']}")
    
    print(f"\n⏱️  Total time: {total_time:.2f} seconds")
    print(f"💾 All models cached in: {os.path.expanduser('~/.cache/huggingface/hub')}")
    
    # Performance comparison
    if successful_downloads:
        print(f"\n🚀 Model Performance Ranking (by download time):")
        sorted_models = sorted(successful_downloads, key=lambda x: x['time'])
        for i, model in enumerate(sorted_models, 1):
            speed_emoji = "🚀" if model['time'] < 30 else "⚡" if model['time'] < 60 else "🐌"
            print(f"   {i}. {speed_emoji} {model['name']} - {model['time']:.2f}s ({model['dimensions']}D)")
    
    print(f"\n🎉 Download process complete!")
    
    if len(successful_downloads) == len(models_to_download):
        print("✅ All models ready for offline use!")
        return True
    else:
        print(f"⚠️  {len(failed_downloads)} models failed - check network connection and try again")
        return False

def check_cached_models():
    """Check which models are already cached locally"""
    print("🔍 Checking for already cached models...")
    
    models = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2", 
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "e5-base-v2": "intfloat/e5-base-v2",
        "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5"
    }
    
    cached_models = []
    
    for model_name, model_path in models.items():
        try:
            # Try to load without downloading (use_auth_token=False, cache_folder check)
            model = SentenceTransformer(model_path, cache_folder=os.path.expanduser('~/.cache/huggingface/hub'))
            cached_models.append(model_name)
            print(f"   ✅ {model_name} - Already cached")
        except:
            print(f"   📥 {model_name} - Not cached, will download")
    
    print(f"\n📊 {len(cached_models)}/{len(models)} models already cached")
    return cached_models

if __name__ == "__main__":
    print("🤖 Embedding Models Download Tool")
    print("=" * 50)
    
    # Check what's already cached
    cached = check_cached_models()
    
    print("\n" + "=" * 50)
    user_input = input("Continue with download? (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        success = download_models()
        
        if success:
            print("\n🎯 Next steps:")
            print("1. Models are now cached locally for faster loading")
            print("2. The RAG system will use cached models automatically") 
            print("3. No internet connection needed for model loading")
            print("4. Start your Streamlit app and enjoy faster startup!")
        else:
            print("\n⚠️  Some downloads failed. Check your internet connection and try again.")
            sys.exit(1)
    else:
        print("❌ Download cancelled by user")
        sys.exit(0)