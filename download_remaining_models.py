#!/usr/bin/env python3
"""
Download the remaining embedding models that aren't cached yet
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import time
from sentence_transformers import SentenceTransformer

def download_remaining_models():
    """Download the remaining models that need caching"""
    
    # Models that still need downloading
    remaining_models = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5"
    }
    
    print("🔄 Downloading remaining models...")
    print(f"📊 Models to download: {len(remaining_models)}")
    print("=" * 50)
    
    successful_downloads = []
    failed_downloads = []
    total_start_time = time.time()
    
    for model_name, model_path in remaining_models.items():
        print(f"\n📦 Downloading {model_name}")
        print(f"   Path: {model_path}")
        
        try:
            start_time = time.time()
            
            # Download and cache the model
            print("   📥 Starting download...")
            model = SentenceTransformer(model_path)
            
            # Test encoding to ensure model works
            test_embedding = model.encode("Test sentence", convert_to_numpy=True)
            
            download_time = time.time() - start_time
            
            print(f"   ✅ SUCCESS - Downloaded in {download_time:.2f}s")
            print(f"   📊 Embedding shape: {test_embedding.shape}")
            print(f"   💾 Model cached locally")
            
            successful_downloads.append({
                'name': model_name,
                'path': model_path,
                'time': download_time,
                'dimensions': test_embedding.shape[0]
            })
            
        except Exception as e:
            print(f"   ❌ FAILED - {str(e)}")
            failed_downloads.append({
                'name': model_name,
                'path': model_path,
                'error': str(e)
            })
    
    total_time = time.time() - total_start_time
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 50)
    
    if successful_downloads:
        print(f"\n✅ Successfully downloaded {len(successful_downloads)} models:")
        for model in successful_downloads:
            print(f"   • {model['name']} ({model['dimensions']}D) - {model['time']:.2f}s")
    
    if failed_downloads:
        print(f"\n❌ Failed to download {len(failed_downloads)} models:")
        for model in failed_downloads:
            print(f"   • {model['name']}: {model['error']}")
    
    print(f"\n⏱️  Total time: {total_time:.2f} seconds")
    
    # Final status
    if len(successful_downloads) == len(remaining_models):
        print("\n🎉 All remaining models downloaded successfully!")
        print("✅ Your RAG system now has all models cached locally")
        print("🚀 Startup will be much faster from now on!")
        return True
    else:
        print(f"\n⚠️  {len(failed_downloads)} models failed to download")
        return False

if __name__ == "__main__":
    print("🤖 Remaining Models Download")
    print("=" * 40)
    
    success = download_remaining_models()
    
    if success:
        print("\n🎯 All done! Your embedding models are ready for offline use.")
    else:
        print("\n⚠️  Some downloads failed. Check your internet connection.")
        sys.exit(1)