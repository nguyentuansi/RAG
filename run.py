#!/usr/bin/env python3
"""
Visual RAG System - Main Entry Point

This script provides multiple ways to run the RAG system:
- Interactive dashboard (Streamlit)
- Command-line interface
- Development mode with all services

Usage:
    python run.py --mode simple      # Simple step-by-step interface (RECOMMENDED)
    python run.py --mode visual      # Advanced Streamlit dashboard
    python run.py --mode cli         # Command-line interface  
    python run.py --mode dev         # Development mode (all services)
    python run.py --mode benchmark   # Benchmark embedding models
"""

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
    console = Console()
    app = typer.Typer(help="Visual RAG System - Production-Ready Demo")
except ImportError:
    RICH_AVAILABLE = False
    console = None
    
    # Create a simple console mock
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    
    console = MockConsole()

# Import our modules - with fallback if not available
try:
    from pipeline.main_pipeline import VisualRAGPipeline
    from models.embeddings import EmbeddingModelFactory, EmbeddingBenchmark
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"Warning: Pipeline modules not available: {e}")

# Configure logging
if RICH_AVAILABLE:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'qdrant-client', 'sentence-transformers', 
        'torch', 'transformers', 'plotly', 'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        console.print(Panel.fit(
            f"❌ Missing required packages: {', '.join(missing_packages)}\n\n"
            f"Install with: pip install {' '.join(missing_packages)}",
            title="Dependency Error",
            border_style="red"
        ))
        return False
    
    return True


def check_qdrant_connection() -> bool:
    """Check if Qdrant is running and accessible"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        return True
    except Exception as e:
        logger.warning(f"Qdrant not accessible: {e}")
        return False


def visual_func(port: int = 8501, host: str = "localhost"):
    """Launch the interactive visual dashboard"""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "🚀 Launching Visual RAG Dashboard\n\n"
            "This will open an interactive web interface where you can:\n"
            "• See the complete RAG pipeline flow\n"
            "• Upload and process documents\n"  
            "• Search your knowledge base\n"
            "• Learn how RAG systems work\n\n"
            f"Dashboard will be available at: http://{host}:{port}",
            title="Visual Mode",
            border_style="blue"
        ))
    else:
        print("🚀 Launching Visual RAG Dashboard")
        print("This will open an interactive web interface where you can:")
        print("• See the complete RAG pipeline flow")
        print("• Upload and process documents")
        print("• Search your knowledge base")
        print("• Learn how RAG systems work")
        print(f"Dashboard will be available at: http://{host}:{port}")
    
    # Check dependencies
    if not check_dependencies():
        if RICH_AVAILABLE:
            raise typer.Exit(1)
        else:
            sys.exit(1)
    
    # Check Qdrant
    if not check_qdrant_connection():
        if RICH_AVAILABLE:
            console.print("[yellow]⚠️  Qdrant not running. Starting with Docker...[/yellow]")
        else:
            print("⚠️  Qdrant not running. Starting with Docker...")
        start_qdrant()
    
    # Launch Streamlit
    app_path = Path(__file__).parent / "visualizer" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", host,
        "--server.port", str(port),
        "--theme.primaryColor", "#ff6b6b",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[green]✅ Dashboard shut down gracefully[/green]")
        else:
            print("\n✅ Dashboard shut down gracefully")


def cli_func(model: str = "minilm", interactive: bool = True):
    """Command-line interface for the RAG system"""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            f"💻 RAG System Command Line Interface\n\n"
            f"Using embedding model: {model}\n"
            f"Interactive mode: {'Enabled' if interactive else 'Disabled'}",
            title="CLI Mode",
            border_style="green"
        ))
    else:
        print("💻 RAG System Command Line Interface")
        print(f"Using embedding model: {model}")
        print(f"Interactive mode: {'Enabled' if interactive else 'Disabled'}")
    
    # Check dependencies
    if not check_dependencies():
        if RICH_AVAILABLE:
            raise typer.Exit(1)
        else:
            sys.exit(1)
    
    # Run CLI
    asyncio.run(run_cli(model, interactive))


async def run_cli(model_name: str, interactive: bool):
    """Run the command-line interface"""
    if not PIPELINE_AVAILABLE:
        print("❌ Pipeline modules not available. Install dependencies:")
        print("pip install -e .")
        return
    
    try:
        # Initialize pipeline
        if RICH_AVAILABLE:
            console.print("[blue]🔄 Initializing RAG pipeline...[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading pipeline...", total=None)
                
                pipeline = VisualRAGPipeline()
                progress.update(task, description="Pipeline ready!")
        else:
            print("🔄 Initializing RAG pipeline...")
            pipeline = VisualRAGPipeline()
            print("✅ Pipeline initialized successfully!")
        
        if RICH_AVAILABLE:
            console.print("[green]✅ Pipeline initialized successfully![/green]")
        
        if interactive:
            await interactive_cli_session(pipeline)
        else:
            print("Non-interactive mode - pipeline ready for API calls")
    
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"❌ Error: {e}")
        logger.exception("CLI error")


async def interactive_cli_session(pipeline):
    """Run interactive CLI session"""
    if RICH_AVAILABLE:
        console.print("\n[bold blue]🎉 Welcome to the Visual RAG System![/bold blue]")
        console.print("Type 'help' for commands or 'quit' to exit\n")
    else:
        print("\n🎉 Welcome to the Visual RAG System!")
        print("Type 'help' for commands or 'quit' to exit\n")
    
    while True:
        try:
            if RICH_AVAILABLE:
                command = console.input("[bold cyan]rag> [/bold cyan]")
            else:
                command = input("rag> ")
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.lower() == 'help':
                show_cli_help()
            elif command.startswith('upload'):
                await handle_upload_command(pipeline, command)
            elif command.startswith('search'):
                await handle_search_command(pipeline, command)  
            elif command.lower() == 'status':
                show_pipeline_status(pipeline)
            elif command.lower() == 'models':
                show_available_models()
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[green]👋 Thanks for using Visual RAG System![/green]")


def show_cli_help():
    """Show CLI help"""
    help_text = """
[bold]Available Commands:[/bold]

📄 [cyan]upload <file1> <file2> ...[/cyan] - Upload and process documents
🔍 [cyan]search <query>[/cyan]            - Search your knowledge base  
📊 [cyan]status[/cyan]                    - Show pipeline status
🧠 [cyan]models[/cyan]                    - List available embedding models
❓ [cyan]help[/cyan]                      - Show this help message
🚪 [cyan]quit[/cyan]                      - Exit the program

[bold]Examples:[/bold]
  upload document1.pdf document2.txt
  search "What is machine learning?"
  status
"""
    console.print(Panel(help_text, title="Help", border_style="yellow"))


async def handle_upload_command(pipeline, command: str):
    """Handle document upload command"""
    parts = command.split()[1:]  # Remove 'upload'
    
    if not parts:
        console.print("[red]Please specify files to upload[/red]")
        return
    
    file_paths = [Path(p) for p in parts]
    existing_files = [p for p in file_paths if p.exists()]
    
    if not existing_files:
        console.print("[red]No valid files found[/red]")
        return
    
    console.print(f"[blue]📄 Processing {len(existing_files)} documents...[/blue]")
    
    with Progress(console=console) as progress:
        task = progress.add_task("Processing documents...", total=100)
        
        results = await pipeline.ingest_documents(existing_files)
        
        progress.update(task, completed=100)
    
    if results and results[-1].success:
        console.print("[green]✅ Documents processed successfully![/green]")
        show_pipeline_status(pipeline)
    else:
        console.print("[red]❌ Document processing failed[/red]")


async def handle_search_command(pipeline, command: str):
    """Handle search command"""
    query = command[6:].strip()  # Remove 'search'
    
    if not query:
        console.print("[red]Please provide a search query[/red]")
        return
    
    console.print(f"[blue]🔍 Searching for: '{query}'[/blue]")
    
    result = await pipeline.search_and_generate(query)
    
    if result.success:
        response_data = result.data
        
        console.print("\n[bold green]💬 Generated Response:[/bold green]")
        console.print(Panel(response_data['response'], border_style="green"))
        
        console.print(f"\n[bold blue]📚 Sources ({len(response_data['sources'])}):[/bold blue]")
        for i, source in enumerate(response_data['sources'], 1):
            console.print(f"  {i}. {source['title']} (Score: {source['score']:.3f})")
    else:
        console.print(f"[red]❌ Search failed: {result.error}[/red]")


def show_pipeline_status(pipeline):
    """Show current pipeline status"""
    status = pipeline.get_status()
    metrics = status['metrics']
    
    status_text = f"""
[bold]Pipeline Status:[/bold] {status['state']}
[bold]Pipeline ID:[/bold] {status['pipeline_id']}

[bold]Metrics:[/bold]
📄 Documents Processed: {metrics['documents_processed']}
✂️ Chunks Created: {metrics['chunks_created']}  
🔢 Embeddings Generated: {metrics['embeddings_generated']}
📊 Vectors Indexed: {metrics['vectors_indexed']}
🔍 Searches Performed: {metrics['searches_performed']}
⏱️ Total Processing Time: {metrics['total_processing_time']:.2f}s

[bold]Configuration:[/bold]
🧠 Embedding Model: {status['config']['embedding_model']}
✂️ Chunk Size: {status['config']['chunk_size']}
📏 Distance Method: {status['config'].get('distance_metric', 'cosine').title()} Similarity
🗄️ Collection Name: {status['config']['collection_name']}
"""
    console.print(Panel(status_text, title="Pipeline Status", border_style="blue"))


def show_available_models():
    """Show available embedding models"""
    models = EmbeddingModelFactory.list_models()
    
    model_text = "[bold]Available Embedding Models:[/bold]\n\n"
    
    for name, config in models.items():
        model_text += f"🧠 [cyan]{name}[/cyan]\n"
        model_text += f"   📏 Dimensions: {config['dimensions']}\n"
        model_text += f"   📄 Max Length: {config['max_length']} tokens\n"
        model_text += f"   🌍 Multilingual: {'✅' if config.get('supports_multilingual', False) else '❌'}\n\n"
    
    console.print(Panel(model_text, title="Embedding Models", border_style="magenta"))


@app.command()
def benchmark(
    models: Optional[str] = typer.Option(None, help="Comma-separated model names to benchmark"),
    runs: int = typer.Option(3, help="Number of benchmark runs")
):
    """Benchmark embedding models"""
    console.print(Panel.fit(
        "🏃‍♂️ Benchmarking Embedding Models\n\n"
        "This will test the speed and performance of different\n"
        "open-source embedding models on your hardware.",
        title="Benchmark Mode",
        border_style="magenta"
    ))
    
    models_to_test = models.split(',') if models else None
    asyncio.run(run_benchmark(models_to_test, runs))


async def run_benchmark(models_to_test: Optional[list], runs: int):
    """Run embedding model benchmark"""
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Embeddings convert text into numerical vectors",
        "Natural language processing helps computers understand human language",
        "Vector databases enable fast similarity search"
    ] * 10  # 50 texts total
    
    benchmark = EmbeddingBenchmark(models_to_test)
    
    console.print(f"[blue]🔄 Running benchmark with {len(test_texts)} texts...[/blue]")
    
    with Progress(console=console) as progress:
        task = progress.add_task("Benchmarking models...", total=100)
        
        results = await benchmark.run_benchmark(test_texts, runs)
        
        progress.update(task, completed=100)
    
    benchmark.print_results()


def start_qdrant():
    """Start Qdrant using Docker"""
    try:
        console.print("[blue]🐳 Starting Qdrant with Docker...[/blue]")
        subprocess.run([
            "docker", "run", "-d", 
            "--name", "qdrant-demo",
            "-p", "6333:6333",
            "-p", "6334:6334", 
            "qdrant/qdrant:v1.12.5"
        ], check=True)
        
        # Wait for Qdrant to be ready
        time.sleep(5)
        console.print("[green]✅ Qdrant started successfully![/green]")
        
    except subprocess.CalledProcessError:
        console.print("[red]❌ Failed to start Qdrant. Please start it manually:[/red]")
        console.print("docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.12.5")


@app.command()
def dev():
    """Development mode - start all services"""
    console.print(Panel.fit(
        "🛠️  Development Mode\n\n"
        "Starting all services:\n"
        "• Qdrant vector database\n"
        "• Streamlit dashboard\n"
        "• MCP server (when implemented)\n\n"
        "Perfect for development and testing!",
        title="Development Mode",
        border_style="yellow"
    ))
    
    # Start Qdrant
    if not check_qdrant_connection():
        start_qdrant()
    
    # Start Streamlit dashboard
    console.print("[blue]🚀 Starting dashboard...[/blue]")
    visual()


def run_visual_simple():
    """Simple visual mode without typer"""
    print("🚀 Launching Visual RAG Dashboard")
    print("This will open an interactive web interface")
    print("Dashboard will be available at: http://localhost:8501")
    
    # Check if streamlit is available
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        return
    
    # Launch Streamlit
    app_path = Path(__file__).parent / "visualizer" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", "localhost",
        "--server.port", "8501"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n✅ Dashboard shut down gracefully")


def run_simple_app():
    """Launch the simplified single-page interface"""
    simple_app_path = Path(__file__).parent / "visualizer" / "simple_app.py"
    
    if not simple_app_path.exists():
        print(f"❌ Error: Simple app not found at {simple_app_path}")
        return
    
    print("🚀 Starting Simple RAG System...")
    print("🌐 Opening at: http://localhost:8501")
    print("\n💡 This interface guides you step-by-step:")
    print("   Step 1: Setup your RAG system")
    print("   Step 2: Upload documents (or use existing data)")
    print("   Step 3: Search and get answers")
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(simple_app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n✅ Simple RAG System shut down gracefully")


def run_simple_mode(mode: str):
    """Simple mode runner without typer"""
    if mode == "simple":
        run_simple_app()
    elif mode == "visual":
        run_visual_simple()
    elif mode == "cli":
        print("💻 CLI mode would start here")
        print("This requires the full pipeline dependencies to be installed")
    elif mode == "dev":
        print("🛠️ Development mode")
        print("Starting Streamlit dashboard...")
        run_visual_simple()
    elif mode == "benchmark":
        print("🏃‍♂️ Benchmark mode")
        print("This requires the models module to benchmark embedding models")
    else:
        print(f"Unknown mode: {mode}")


def main():
    """Main entry point for the application"""
    if RICH_AVAILABLE and "--mode" not in sys.argv:
        # Use typer for direct app() calls
        app()
    else:
        # Use argparse for --mode style calls
        parser = argparse.ArgumentParser(description="Visual RAG System")
        parser.add_argument(
            "--mode", 
            choices=["simple", "visual", "cli", "dev", "benchmark"],
            default="simple",
            help="Run mode"
        )
        args = parser.parse_args()
        run_simple_mode(args.mode)


if __name__ == "__main__":
    # Check if using --mode style arguments
    if "--mode" in sys.argv:
        parser = argparse.ArgumentParser(description="Visual RAG System")
        parser.add_argument(
            "--mode", 
            choices=["simple", "visual", "cli", "dev", "benchmark"],
            default="simple",
            help="Run mode"
        )
        args = parser.parse_args()
        run_simple_mode(args.mode)
    elif RICH_AVAILABLE:
        # Use typer for command-style arguments
        app()
    else:
        # Fallback to simple mode
        run_simple_mode("visual")


# Add typer decorators if available
if RICH_AVAILABLE:
    app.command()(visual_func)
    app.command()(cli_func)
    
    # Add the other commands as well
    @app.command()
    def simple():
        """Launch the simplified step-by-step RAG interface (RECOMMENDED)"""
        run_simple_app()
    
    @app.command()
    def visual(
        port: int = typer.Option(8501, help="Port for Streamlit dashboard"),
        host: str = typer.Option("localhost", help="Host for Streamlit dashboard")
    ):
        """Launch the interactive visual dashboard"""
        visual_func(port, host)
    
    @app.command() 
    def cli(
        model: str = typer.Option("minilm", help="Embedding model to use"),
        interactive: bool = typer.Option(True, help="Interactive mode")
    ):
        """Command-line interface for the RAG system"""
        cli_func(model, interactive)