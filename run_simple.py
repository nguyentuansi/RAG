#!/usr/bin/env python3
"""
Simple RAG System Launcher
Launches the simplified single-page Streamlit interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the simplified RAG system interface"""
    
    # Get the path to the simple app
    app_path = Path(__file__).parent / "visualizer" / "simple_app.py"
    
    if not app_path.exists():
        print(f"❌ Error: {app_path} not found")
        sys.exit(1)
    
    print("🚀 Starting Simple RAG System...")
    print(f"📁 App location: {app_path}")
    print("🌐 Opening at: http://localhost:8501")
    print("\n💡 This interface guides you step-by-step:")
    print("   Step 1: Setup your RAG system")
    print("   Step 2: Upload documents (or use existing data)")
    print("   Step 3: Search and get answers")
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()