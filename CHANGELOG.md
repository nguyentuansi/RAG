# Changelog

All notable changes to the RAG Platform are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-06-18

### Added

#### Core Platform
- `src/rag/` — production `src` layout replacing root-level demo modules
- `Settings` via `pydantic-settings` covering all runtime configuration (Qdrant, Redis, JWT, CORS, rate limits, OTLP, Prometheus)
- Typed exception hierarchy rooted at `RAGException` with HTTP status mapping
- Structured JSON logging with per-request correlation ID via `structlog`
- OpenTelemetry tracing: `TracerProvider`, `configure_telemetry()`, `trace_span` context manager, OTLP exporter

#### Domain Models
- Pydantic v2 models: `Document`, `DocumentChunk`, `ProcessedDocument`, `DocumentMetadata`
- `DocumentStatus` and `DocumentFormat` enums with extension-based resolution
- `TextChunk`, `EmbeddedChunk` with vector payload serialization
- `SearchQuery`, `SearchResult`, `SearchResponse` with hybrid search config and RRF support

#### Infrastructure
- `VectorStore` ABC with create_collection, upsert_vectors, search, delete, count
- `QdrantVectorStore`: async client, batch upsert, filter translation, HNSW config
- `EmbeddingProvider` ABC with embed_text, embed_batch, warm_up lifecycle
- `SentenceTransformerProvider`: lazy loading, batching, GPU support via thread pool
- `RedisCache`: async Redis with JSON serialization, TTL management, `cache_aside` decorator

#### REST API (FastAPI)
- `POST /documents/upload` — multipart upload with background async indexing (202 Accepted)
- `GET /documents/{id}` — document lookup by ID
- `DELETE /documents/{id}` — bulk chunk deletion
- `POST /search` — dense similarity search with Redis result caching
- `POST /search/hybrid` — RRF fusion of dense + BM25 ranked lists
- `POST /search/mmr` — Maximal Marginal Relevance diversified retrieval
- `GET /collections/{name}/stats` — live collection metadata
- `GET /health`, `/health/ready`, `/health/live` — component health checks
- `GET /metrics` — Prometheus exposition format

#### Security
- JWT bearer token authentication (create, decode, refresh) via `python-jose`
- API key management: SHA-256 hashed storage, Redis-backed verification
- RBAC: `Role` (ADMIN/OPERATOR/READER), `Permission` enum, `require_permission()` FastAPI dependency
- `PromptInjectionDetector`: 12 regex patterns covering instruction override, jailbreak, shell/template/HTML injection; risk scoring with severity levels
- `RateLimitMiddleware`: Redis sorted-set sliding window, per-user identity (JWT/API key/IP), burst allowance

#### Processing Pipeline
- `DocumentIngestionPipeline`: async multi-format loading (PDF via pypdf, DOCX via python-docx, HTML via BeautifulSoup, Markdown, plain text), batch ingestion with progress callbacks
- `RecursiveCharacterChunker`: separator cascade from paragraph to character with configurable overlap
- `SemanticChunker`: sentence-boundary grouping with overlap preserved as sentence tail
- `SlidingWindowChunker`: fixed stride with configurable overlap
- `MarkdownHeaderChunker`: splits at heading boundaries, sub-chunks oversized sections
- `AsyncEmbeddingPipeline`: batched encoding with exponential-backoff retry and progress callbacks
- `HybridRetriever`: dense search, BM25 sparse search, RRF fusion, MMR reranking

#### Multi-Agent System
- `MessageBus`: async in-memory pub/sub with request/reply pattern and correlation ID
- `BaseAgent` ABC with start/stop lifecycle, capability declaration, auto-dispatch
- `OrchestratorAgent`: intent-based routing with 30-second timeout
- `SearchAgent`, `DocumentAgent`, `AnalyticsAgent`: concrete specialised implementations

#### MCP Server
- Production JSON-RPC 2.0 MCP server over stdio
- `@server.tool()` decorator for declarative tool registration with JSON Schema
- Tools: `search_documents`, `get_collection_stats`, `list_collections`
- Proper error codes (-32601, -32603, -32700)

#### Evaluation Framework
- `RAGEvaluator`: faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity (cosine), NDCG@k
- `EvaluationPipeline`: JSONL dataset loading, bulk async evaluation, configuration comparison, JSONL results export
- `EvaluationReport` with ASCII score bar summary

#### DevOps & Observability
- Multi-stage `Dockerfile`: base→deps→builder→runtime with non-root `rag:1001` user, HEALTHCHECK
- `docker-compose.prod.yml`: rag-api, Qdrant v1.13, Redis 7.4, Nginx, Prometheus, Grafana with resource limits
- GitHub Actions CI: ruff lint+format, mypy, bandit/safety, pytest on Python 3.11+3.12 with live Qdrant+Redis services
- GitHub Actions CD: semver tag → build+push GHCR → staging smoke test → production gate
- `Makefile` with 15 targets: install, dev, test, lint, format, type-check, build, push, clean, seed, migrate, docker-up/down/logs
- Prometheus `RAGMetrics`: requests_total, request_duration_seconds, documents_processed_total, search_latency_seconds, embedding_duration_seconds, vector_store_size
- `scripts/seed_data.py`: 4 sample documents indexed through the full pipeline
- `scripts/health_check.py`: standalone liveness/readiness probe, exits 0/1

### Changed
- Restructured from root-level demo modules to `src/` layout
- All models migrated to Pydantic v2 (`ConfigDict`, `model_config`, `computed_field`)
- Pipeline is now fully async, no blocking I/O in the event loop
- Vector store operations use async Qdrant gRPC client

### Fixed
- Prompt injection detector now covers encoding-bypass and fictional-framing patterns
- Embedding loading is deferred and thread-pooled to avoid blocking API startup
- Rate limiter never blocks a request on Redis failure (fail-open with logging)

---

## [0.1.0] — 2025-08-30

### Added
- Initial Streamlit dashboard demo with visual pipeline
- Basic Qdrant integration
- Sentence-transformers embedding support
- MCP server prototype
- Multi-agent orchestrator demo
- Docker Compose for local development
