# RAG System

## How to Run

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start vector database (Qdrant)
docker run -d -p 6333:6333 qdrant/qdrant:v1.12.5

# Run the main application
python run.py

# Alternative run options:
python run_simple.py              # Simple interface
python run_chunking_comparison.py # Chunking comparison
python run_algorithm_demo.py      # Algorithm demo
```

## Requirements

- Python 3.8+
- Docker (for Qdrant vector database)
- Dependencies listed in requirements.txt