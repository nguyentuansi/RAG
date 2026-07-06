# RAG Platform

A production-grade **Retrieval-Augmented Generation** platform built on open-source models. It ships a REST API, a multi-agent orchestration layer, an MCP server for LLM tool use, hybrid search, and a full observability stack — all deployable via Docker Compose.

---

## Architecture

```
                          ┌─────────────────────────────────────────────────┐
                          │                   Clients                        │
                          │   REST API │ MCP (LLM tools) │ Streamlit UI     │
                          └───────────────────┬─────────────────────────────┘
                                              │  HTTPS
                                     ┌────────▼────────┐
                                     │  Nginx (reverse  │
                                     │  proxy + TLS)    │
                                     └────────┬─────────┘
                                              │
                          ┌───────────────────▼───────────────────────────┐
                          │              FastAPI  (src/rag/api)            │
                          │                                                │
                          │  ┌──────────────┐   ┌──────────────────────┐  │
                          │  │  Middleware   │   │       Routes         │  │
                          │  │  • JWT auth  │   │  /documents          │  │
                          │  │  • API keys  │   │  /search             │  │
                          │  │  • RBAC      │   │  /health             │  │
                          │  │  • Rate limit│   │  /metrics            │  │
                          │  └──────────────┘   └──────────────────────┘  │
                          └────────────┬──────────────────────────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
     ┌─────────▼──────────┐  ┌─────────▼──────────┐  ┌────────▼─────────┐
     │   Pipeline Layer   │  │   Agent Layer      │  │  MCP Server      │
     │   src/rag/pipeline │  │  src/rag/agents    │  │  src/rag/mcp     │
     │                    │  │                    │  │                  │
     │  • Ingestion       │  │  • Orchestrator    │  │  • JSON-RPC 2.0  │
     │  • Chunking        │  │  • SearchAgent     │  │  • Tool registry │
     │  • Embedding       │  │  • DocumentAgent   │  │  • Schema gen    │
     │  • Retrieval       │  │  • AnalyticsAgent  │  │  • LLM-callable  │
     └─────────┬──────────┘  │  • MessageBus      │  └──────────────────┘
               │             └────────────────────┘
               │
     ┌─────────▼──────────────────────────────────────────────────────────┐
     │                    Infrastructure Layer                            │
     │                                                                    │
     │  ┌─────────────────────┐    ┌──────────────────┐                  │
     │  │  Vector Store       │    │  Embedding        │                  │
     │  │  (abstract + Qdrant)│    │  (abstract + ST)  │                  │
     │  └──────────┬──────────┘    └────────┬─────────┘                  │
     │             │                        │                             │
     │  ┌──────────▼──────────────────────────────────────────────────┐  │
     │  │  External Services                                          │  │
     │  │   Qdrant (v1.13)  │  Redis (v7.4)  │  SQLite / PostgreSQL  │  │
     │  └─────────────────────────────────────────────────────────────┘  │
     └────────────────────────────────────────────────────────────────────┘
               │
     ┌─────────▼──────────────────────────────┐
     │       Observability Stack              │
     │  Prometheus → Grafana │ OpenTelemetry  │
     └────────────────────────────────────────┘
```

---

## Directory Structure

```
.
├── src/rag/                      # Main application package
│   ├── core/
│   │   ├── config.py             # Pydantic-settings: all env vars
│   │   ├── exceptions.py         # Custom exception hierarchy
│   │   ├── logging.py            # Structured JSON logging (structlog)
│   │   └── telemetry.py          # OpenTelemetry tracer setup
│   │
│   ├── domain/                   # Pure domain models (no I/O)
│   │   ├── documents.py          # Document, DocumentMetadata, ProcessedDocument
│   │   ├── chunks.py             # TextChunk, EmbeddedChunk, ChunkingConfig
│   │   └── search.py             # SearchQuery, SearchResult, HybridSearchConfig
│   │
│   ├── infrastructure/           # Pluggable external adapters
│   │   ├── vector_store/
│   │   │   ├── base.py           # Abstract VectorStore (ABC)
│   │   │   └── qdrant.py         # Qdrant implementation
│   │   ├── embeddings/
│   │   │   ├── base.py           # Abstract EmbeddingProvider (ABC)
│   │   │   └── sentence_transformer.py
│   │   └── cache/
│   │       └── redis.py          # Async Redis with cache_aside decorator
│   │
│   ├── api/                      # FastAPI application
│   │   ├── app.py                # App factory, lifespan, CORS, mounts
│   │   ├── dependencies.py       # DI: vector store, embeddings, auth
│   │   ├── middleware/
│   │   │   ├── auth.py           # JWT bearer + API key verification
│   │   │   └── rate_limit.py     # Sliding-window limiter (Redis-backed)
│   │   └── routes/
│   │       ├── documents.py      # Upload, list, get, delete documents
│   │       ├── search.py         # Dense, hybrid, MMR search endpoints
│   │       ├── health.py         # /health, /health/ready, /health/live
│   │       └── metrics.py        # Prometheus /metrics endpoint
│   │
│   ├── pipeline/                 # Data processing stages
│   │   ├── ingestion.py          # Async multi-format loader (PDF/DOCX/HTML/MD/TXT)
│   │   ├── chunking.py           # Recursive, semantic, sliding-window, markdown
│   │   ├── embedding.py          # Batched async embedding with retry
│   │   └── retrieval.py          # Dense, BM25, RRF hybrid fusion, MMR
│   │
│   ├── agents/                   # Multi-agent system
│   │   ├── base.py               # AgentMessage, MessageBus, BaseAgent ABC
│   │   └── orchestrator.py       # OrchestratorAgent + SearchAgent / DocumentAgent
│   │
│   ├── mcp/
│   │   └── server.py             # JSON-RPC 2.0 MCP server, tool registration
│   │
│   ├── security/
│   │   ├── auth.py               # JWT creation/validation, API key hashing
│   │   ├── rbac.py               # Role/Permission enums, require_permission
│   │   └── prompt_injection.py   # Pattern-based injection + jailbreak detection
│   │
│   └── evaluation/
│       ├── metrics.py            # Faithfulness, relevancy, precision, recall, NDCG
│       └── pipeline.py           # EvaluationPipeline: run, compare, report
│
├── pipeline/                     # Legacy visualizer pipeline (Streamlit)
├── agents/                       # Legacy agent orchestrator
├── visualizer/                   # Streamlit dashboard
├── mcp_server/                   # Legacy MCP server
├── security/                     # Legacy prompt shield
│
├── docker/
│   ├── Dockerfile                # Multi-stage production build
│   └── Dockerfile.dev            # Dev image with hot-reload
├── docker-compose.yml            # Development stack
├── docker-compose.prod.yml       # Production stack (nginx, prometheus, grafana)
│
├── .github/workflows/
│   ├── ci.yml                    # Lint, type-check, Docker build
│   └── deploy.yml                # GHCR push, staging/prod deploy
│
├── scripts/
│   ├── seed_data.py              # Index sample documents
│   └── health_check.py           # Standalone readiness probe
│
├── Makefile                      # Developer commands
├── pyproject.toml                # Project metadata and tooling config
├── .env.example                  # All environment variables documented
└── CHANGELOG.md
```

---

## Tech Stack

### Core
| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | ≥ 3.11 |
| API framework | FastAPI + Uvicorn | ≥ 0.115 |
| Data validation | Pydantic v2 | ≥ 2.9 |
| Configuration | pydantic-settings | ≥ 2.5 |

### AI / ML
| Component | Technology |
|-----------|-----------|
| Embedding models | sentence-transformers (all-mpnet-base-v2, BGE, E5, MiniLM) |
| Model runtime | PyTorch + HuggingFace Transformers |
| Sparse search | rank-bm25 |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) |
| Diversity rerank | Maximal Marginal Relevance (MMR) |

### Storage
| Component | Technology | Version |
|-----------|-----------|---------|
| Vector database | Qdrant | 1.13 |
| Cache | Redis (with hiredis) | 7.4 |
| Metadata DB | SQLite / PostgreSQL (via SQLAlchemy) | — |

### Document Processing
| Format | Library |
|--------|---------|
| PDF | pypdf |
| DOCX | python-docx |
| HTML | BeautifulSoup4 + lxml |
| Markdown, plain text | built-in |

### Security
| Feature | Implementation |
|---------|---------------|
| Authentication | JWT (HS256) via python-jose + API key support |
| Authorization | Role-Based Access Control (ADMIN / OPERATOR / READER) |
| Rate limiting | Sliding-window counter stored in Redis |
| Prompt safety | Pattern-based injection + jailbreak detection with severity levels |

### Observability
| Component | Technology |
|-----------|-----------|
| Structured logging | structlog (JSON output) |
| Distributed tracing | OpenTelemetry SDK + OTLP exporter |
| Metrics | Prometheus client (counters, histograms, gauges) |
| Dashboards | Grafana |

### DevOps
| Component | Technology |
|-----------|-----------|
| Containerization | Docker multi-stage build (non-root, minimal runtime image) |
| Orchestration | Docker Compose (dev + prod profiles) |
| Reverse proxy | Nginx (SSL termination, upstream to API) |
| CI | GitHub Actions — ruff lint, mypy type-check, Docker build |
| CD | GitHub Actions — GHCR push, staging gate, prod on tag |

### LLM Integration
| Feature | Technology |
|---------|-----------|
| Tool protocol | MCP (Model Context Protocol), JSON-RPC 2.0 |
| Agent messaging | In-process message bus with pub/sub |
| RAG evaluation | RAGAS-style metrics (faithfulness, relevancy, precision, recall, NDCG@k) |

---

## Quick Start

### Development

```bash
# 1. Clone and install
git clone <repo>
cd RAG
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# 2. Copy environment config
cp .env.example .env
# Edit .env: set JWT_SECRET_KEY, model name, etc.

# 3. Start backing services
docker compose up -d qdrant redis

# 4. Seed sample data (optional)
python scripts/seed_data.py

# 5. Run the API
uvicorn src.rag.api.app:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

### Production

```bash
# Build and start the full stack (API + Qdrant + Redis + Nginx + Prometheus + Grafana)
docker compose -f docker-compose.prod.yml up -d

# Check readiness
python scripts/health_check.py
```

### Makefile shortcuts

```bash
make install       # Install all deps
make dev           # Start dev server with hot-reload
make lint          # Run ruff
make type-check    # Run mypy
make build         # Build production Docker image
make seed          # Index sample documents
make docker-up     # Start full prod stack
make docker-down   # Stop and remove containers
```

---

## API Overview

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents/upload` | Upload and index a document (PDF, DOCX, MD, HTML, TXT) |
| `GET` | `/documents` | List documents (paginated) |
| `GET` | `/documents/{id}` | Retrieve a document by ID |
| `DELETE` | `/documents/{id}` | Remove document and its vectors |
| `POST` | `/search` | Dense vector search |
| `POST` | `/search/hybrid` | Hybrid search (BM25 + dense, fused via RRF) |
| `POST` | `/search/mmr` | MMR search for diverse results |
| `GET` | `/collections/{name}/stats` | Collection statistics |
| `GET` | `/health` | Full health report |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

All endpoints (except health + metrics) require a `Authorization: Bearer <token>` header or `X-API-Key` header.

---

## Configuration

All settings are read from environment variables (or `.env`). See `.env.example` for the full reference. Key variables:

```bash
ENVIRONMENT=production          # development | production
JWT_SECRET_KEY=<256-bit key>    # Required in production
QDRANT_HOST=localhost
REDIS_URL=redis://localhost:6379/0
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DEVICE=cpu            # cpu | cuda | mps
COLLECTION_NAME=rag_documents
OTLP_ENDPOINT=http://otel-collector:4317   # optional
```

---

## Evaluation

The evaluation module computes RAGAS-style metrics against a labelled dataset:

```python
from src.rag.evaluation.pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(retriever=retriever, generator=generator)
results = await pipeline.run_evaluation("data/eval_set.jsonl")
report  = pipeline.generate_report(results)
# { "faithfulness": 0.91, "answer_relevancy": 0.87,
#   "context_precision": 0.83, "context_recall": 0.79 }
```

---

## Chunking Strategies

| Strategy | Best for |
|----------|----------|
| `recursive` | General prose and mixed content |
| `semantic` | Sentence-boundary-aware splits, Q&A |
| `sliding_window` | Dense overlap for high-recall retrieval |
| `markdown_header` | Structured docs and wikis |

Strategy and parameters are configurable per-request or globally via `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP`.

---

## License

MIT
