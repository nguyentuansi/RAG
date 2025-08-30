#!/usr/bin/env python3
"""
Chunking Comparison Tool Launcher
Launch the side-by-side chunking comparison interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the chunking comparison tool"""
    
    # Get the path to the comparison app
    app_path = Path(__file__).parent / "visualizer" / "chunking_comparison.py"
    
    if not app_path.exists():
        print(f"❌ Error: {app_path} not found")
        sys.exit(1)
    
    print("✂️ Starting Chunking Comparison Tool...")
    print(f"📁 App location: {app_path}")
    print("🌐 Opening at: http://localhost:8502")
    print("\n💡 This tool shows:")
    print("   ❌ Current approach: Character-based chunking (breaks meaning)")
    print("   ✅ Improved approach: Semantic-aware chunking (preserves context)")
    print("   📊 Side-by-side comparison with quality metrics")
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        # Launch streamlit on a different port to avoid conflicts
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8502",  # Different port from main app
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chunking comparison tool closed!")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()