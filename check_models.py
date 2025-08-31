#!/usr/bin/env python3
"""
Check which embedding models are already cached locally
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def check_huggingface_cache():
    """Check what models are cached in HuggingFace cache directory"""
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    print(f"🔍 Checking HuggingFace cache: {cache_dir}")
    
    if not cache_dir.exists():
        print("❌ HuggingFace cache directory does not exist")
        print("💡 No models have been downloaded yet")
        return []
    
    print(f"📁 Cache directory exists: {cache_dir}")
    
    # List all model directories
    model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith('models--')]
    
    print(f"📊 Found {len(model_dirs)} cached model directories:")
    
    our_models = {
        'sentence-transformers--all-MiniLM-L6-v2': 'all-MiniLM-L6-v2',
        'sentence-transformers--all-mpnet-base-v2': 'all-mpnet-base-v2',
        'models--BAAI--bge-small-en-v1.5': 'bge-small-en-v1.5', 
        'models--intfloat--e5-base-v2': 'e5-base-v2',
        'models--BAAI--bge-base-en-v1.5': 'bge-base-en-v1.5'
    }
    
    cached_models = []
    
    for model_dir in sorted(model_dirs):
        size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        
        # Check if it's one of our models
        if model_dir.name in our_models:
            model_name = our_models[model_dir.name]
            print(f"   ✅ {model_name} - {size_mb:.1f} MB")
            cached_models.append(model_name)
        else:
            print(f"   📦 {model_dir.name} - {size_mb:.1f} MB")
    
    print(f"\n🎯 Our RAG models cached: {len(cached_models)}/5")
    
    missing_models = []
    for cache_name, model_name in our_models.items():
        if model_name not in cached_models:
            missing_models.append(model_name)
    
    if missing_models:
        print(f"📥 Models still need downloading:")
        for model in missing_models:
            print(f"   • {model}")
    else:
        print("🎉 All RAG models are cached locally!")
    
    return cached_models

def get_cache_size():
    """Get total size of HuggingFace cache"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    if not cache_dir.exists():
        return 0
    
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    return total_size / (1024 * 1024)  # Convert to MB

if __name__ == "__main__":
    print("🤖 Model Cache Inspector")
    print("=" * 40)
    
    cached_models = check_huggingface_cache()
    cache_size = get_cache_size()
    
    print(f"\n💾 Total cache size: {cache_size:.1f} MB")
    
    if len(cached_models) == 5:
        print("\n🚀 All models ready! Your RAG system will start faster.")
    elif len(cached_models) > 0:
        print(f"\n⚡ {len(cached_models)} models cached. Run download script for the rest.")
    else:
        print(f"\n📥 No models cached yet. Run download script to cache them locally.")
    
    print(f"\n💡 To download models:")
    print(f"   python download_models.py")
    print(f"   python download_single_model.py  # (for testing)")