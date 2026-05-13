#!/usr/bin/env python3
"""Seed the vector store with sample documents for development and demo purposes."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to RAG Systems",
        "source": "docs/intro-to-rag.md",
        "content": """
# Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is an AI framework that combines information
retrieval with language model generation. Instead of relying solely on the knowledge
baked into a language model's weights, RAG systems first retrieve relevant documents
from an external knowledge base, then use those documents as context to generate
accurate, up-to-date responses.

## Why RAG?

Traditional language models have several limitations:
- Knowledge cutoff: They cannot access information after their training date
- Hallucinations: They may generate plausible-sounding but incorrect information
- Source attribution: It's difficult to trace which training data influenced a response

RAG addresses these limitations by grounding generation in retrieved evidence.

## Core Components

A production RAG pipeline consists of:
1. **Document Ingestion**: Load, parse, and preprocess source documents
2. **Text Chunking**: Split documents into semantically coherent chunks
3. **Embedding Generation**: Convert chunks to dense vector representations
4. **Vector Indexing**: Store vectors in a searchable index (e.g. Qdrant)
5. **Query Processing**: Embed the user query using the same model
6. **Retrieval**: Find the most similar chunks via ANN search
7. **Generation**: Pass retrieved context to an LLM to generate the final answer
""".strip(),
    },
    {
        "title": "Vector Databases Explained",
        "source": "docs/vector-databases.md",
        "content": """
# Vector Databases

A vector database is a specialized database designed to store and query high-dimensional
vector embeddings efficiently. Unlike traditional relational databases optimized for
exact matches, vector databases excel at approximate nearest-neighbor (ANN) search.

## How They Work

When you embed a piece of text using a language model, you get a dense float vector
(e.g., 768 or 1536 dimensions). Similar texts produce vectors that are close together
in this high-dimensional space. Vector databases build special index structures —
such as HNSW (Hierarchical Navigable Small World) graphs — that allow you to find
the k nearest vectors to a query vector in milliseconds, even with millions of stored
vectors.

## Popular Options

- **Qdrant**: Rust-based, open source, Kubernetes-native, excellent performance
- **Weaviate**: GraphQL API, built-in vectorization, good for hybrid search
- **Pinecone**: Fully managed cloud service, simplest to get started with
- **Chroma**: Lightweight, perfect for prototyping and local development
- **pgvector**: PostgreSQL extension for teams that want to stay in SQL

## Choosing a Distance Metric

- **Cosine similarity**: Best for text embeddings (normalized vectors)
- **Euclidean (L2)**: Better when vector magnitudes carry meaning
- **Dot product**: Fastest but sensitive to magnitude; use with normalized embeddings
""".strip(),
    },
    {
        "title": "Chunking Strategies for RAG",
        "source": "docs/chunking-strategies.md",
        "content": """
# Text Chunking Strategies

How you split documents into chunks has a significant impact on RAG retrieval quality.
Chunks that are too large waste context window space; chunks that are too small lose
necessary context.

## Fixed-Size Chunking

The simplest approach: split every N characters with an M-character overlap.

Pros: Fast, predictable chunk sizes, easy to reason about
Cons: May split in the middle of sentences or concepts

## Recursive Character Chunking

Try multiple separator characters in order (double newline → newline → period → space)
and pick the first that keeps chunks under the size limit.

This is the LangChain default and works well for most prose documents.

## Semantic Chunking

Split at sentence boundaries, grouping sentences into chunks until a size limit is
reached. More expensive but produces chunks that always end at sentence boundaries,
improving retrieval precision.

## Document-Aware Chunking

For structured documents (Markdown, HTML, PDFs with headers), respect the document
structure. Split at header boundaries first, then apply recursive splitting within
oversized sections. This preserves the semantic hierarchy and allows you to attach
heading metadata to each chunk.

## Practical Recommendations

- Start with chunk_size=512, chunk_overlap=100 and iterate
- Always include overlapping context to avoid losing information at boundaries
- For code: split at function/class boundaries, not arbitrary character positions
- For tables: keep the entire table in one chunk; never split across rows
""".strip(),
    },
    {
        "title": "Production Deployment Best Practices",
        "source": "docs/production-best-practices.md",
        "content": """
# Production Deployment Checklist

## Security

- [ ] JWT tokens have short expiry (≤ 1 hour for access tokens)
- [ ] API keys stored as SHA-256 hashes, never plaintext
- [ ] Input sanitized against prompt injection before hitting the LLM
- [ ] Rate limiting enabled (per user and global)
- [ ] CORS restricted to known origins
- [ ] Container runs as non-root user

## Observability

- [ ] Structured JSON logging with correlation IDs on every request
- [ ] OpenTelemetry traces exported to your APM tool
- [ ] Prometheus metrics scraped and visualized in Grafana
- [ ] /health/live and /health/ready probes wired to Kubernetes
- [ ] Alerts on P95 latency > 2s, error rate > 1%, cache hit rate < 50%

## Reliability

- [ ] Qdrant deployed with replication (≥ 2 replicas in prod)
- [ ] Redis configured with AOF persistence or a replica
- [ ] Embedding model pre-loaded at startup (warm_up() in lifespan)
- [ ] Graceful shutdown: finish in-flight requests before stopping
- [ ] Retry with exponential back-off on transient embedding/search failures

## Performance

- [ ] Embedding results cached in Redis (TTL ≥ 1h for static content)
- [ ] Search results cached with query-hash key (TTL 5 min)
- [ ] Qdrant HNSW ef_construct ≥ 200 for production index quality
- [ ] Batch size ≥ 32 for embedding generation throughput
""".strip(),
    },
]


async def seed() -> None:
    from src.rag.core.config import get_settings
    from src.rag.infrastructure.vector_store.qdrant import QdrantVectorStore
    from src.rag.infrastructure.embeddings.sentence_transformer import SentenceTransformerProvider
    from src.rag.pipeline.chunking import ChunkingPipeline
    from src.rag.pipeline.embedding import AsyncEmbeddingPipeline
    from src.rag.domain.documents import Document, DocumentFormat, DocumentMetadata
    from src.rag.domain.chunks import ChunkingConfig
    from src.rag.infrastructure.vector_store.base import VectorRecord

    settings = get_settings()

    print(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}...")
    vector_store = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)

    print(f"Loading embedding model: {settings.embedding_model}...")
    embedding_provider = SentenceTransformerProvider(model_name=settings.embedding_model)
    await embedding_provider.warm_up()

    chunker = ChunkingPipeline(ChunkingConfig())
    embedder = AsyncEmbeddingPipeline(embedding_provider)

    total_chunks = 0
    for sample in SAMPLE_DOCUMENTS:
        doc = Document(
            content=sample["content"],
            format=DocumentFormat.MARKDOWN,
            metadata=DocumentMetadata(source=sample["source"], title=sample["title"]),
        )

        chunks = chunker.chunk_document(doc)
        embedded = await embedder.embed_chunks(chunks)

        await vector_store.create_collection(
            settings.collection_name,
            vector_size=embedding_provider.dimension,
        )

        records = [
            VectorRecord(
                id=ec.chunk.chunk_id,
                vector=ec.embedding,
                payload=ec.to_vector_payload(),
            )
            for ec in embedded
        ]
        await vector_store.upsert_vectors(settings.collection_name, records)
        total_chunks += len(records)
        print(f"  ✓ '{sample['title']}' → {len(records)} chunks")

    print(f"\nSeeding complete: {len(SAMPLE_DOCUMENTS)} documents, {total_chunks} chunks in '{settings.collection_name}'")
    await vector_store.close()
    await embedding_provider.close()


if __name__ == "__main__":
    asyncio.run(seed())
