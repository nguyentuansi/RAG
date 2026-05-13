#!/usr/bin/env python3
"""Seed sample documents into the RAG vector store."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to Retrieval-Augmented Generation",
        "source": "docs/rag-intro.md",
        "content": """
# Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of
retrieval-based systems with generative language models. Instead of relying solely
on a language model's parametric knowledge, RAG dynamically retrieves relevant
documents from an external knowledge base before generating a response.

## How RAG Works

1. **Query encoding**: The user's query is converted into a dense vector using an
   embedding model such as sentence-transformers.

2. **Semantic retrieval**: The query vector is compared against stored document
   vectors using cosine similarity or other distance metrics.

3. **Context injection**: The top-k most relevant documents are injected as context
   into the prompt sent to the language model.

4. **Grounded generation**: The LLM generates a response grounded in the retrieved
   facts, reducing hallucination significantly.

## Key Benefits

- **Reduced hallucination**: Responses are grounded in retrieved facts.
- **Up-to-date knowledge**: The knowledge base can be updated without retraining.
- **Source attribution**: Retrieved documents can be cited as sources.
- **Domain specificity**: Works well for niche domains not covered by LLM training.

## Common Architectures

### Naive RAG
The simplest form: embed, retrieve, generate. Works for straightforward Q&A.

### Advanced RAG
Includes query rewriting, re-ranking, and iterative retrieval for better results.

### Modular RAG
Composable pipeline stages that can be swapped independently.
        """.strip(),
        "tags": ["rag", "introduction", "tutorial"],
    },
    {
        "title": "Vector Databases: A Comparison",
        "source": "docs/vector-databases.md",
        "content": """
# Vector Databases: A Comparison Guide

Vector databases are specialized storage systems optimised for high-dimensional
vector operations, particularly approximate nearest-neighbour (ANN) search.

## Popular Options

### Qdrant
- Written in Rust — high performance and memory safety
- Supports scalar and product quantization
- Payload filtering with strong consistency guarantees
- HNSW index with configurable M and ef_construct parameters
- Cloud-hosted and self-hosted options

### Pinecone
- Fully managed SaaS — zero operational overhead
- Metadata filtering at query time
- Serverless tier with usage-based pricing
- Limited to cloud deployment

### Weaviate
- GraphQL query interface
- Built-in vectorization modules
- Multi-modal support
- HNSW-based approximate search

### Chroma
- Lightweight, embedded, great for prototyping
- Python-native API
- Not recommended for production at scale

## Choosing the Right Vector Store

| Criterion         | Qdrant | Pinecone | Weaviate | Chroma |
|-------------------|--------|----------|----------|--------|
| Self-hosted       | ✅     | ❌       | ✅       | ✅     |
| Filtering         | ✅     | ✅       | ✅       | Partial|
| Quantization      | ✅     | ✅       | ❌       | ❌     |
| Production-ready  | ✅     | ✅       | ✅       | ❌     |

## Qdrant Configuration Tips

For production workloads, tune HNSW parameters:
- `m=16` to `m=32`: higher M increases recall but uses more memory
- `ef_construct=200` to `ef_construct=400`: better index quality at build time
- Enable on-disk vectors for large collections to reduce RAM usage
        """.strip(),
        "tags": ["vector-database", "qdrant", "comparison"],
    },
    {
        "title": "Embedding Models for Semantic Search",
        "source": "docs/embedding-models.md",
        "content": """
# Embedding Models for Semantic Search

Embedding models transform text into fixed-length numerical vectors that capture
semantic meaning. Choosing the right model significantly affects retrieval quality.

## Key Metrics

- **Dimension**: Higher dimensions capture more nuance but require more storage.
- **Max sequence length**: The maximum number of tokens the model can encode.
- **Speed**: Inference time per batch — critical at scale.
- **MTEB Score**: Benchmark across multiple embedding tasks.

## Top Open-Source Models (2025)

### all-mpnet-base-v2 (768d)
A robust general-purpose model from sentence-transformers. Excellent balance of
speed and accuracy. Recommended starting point for English text.

### BAAI/bge-m3 (1024d)
Multi-lingual model from BAAI. State-of-the-art performance on BEIR benchmark.
Supports dense, sparse, and multi-vector retrieval.

### intfloat/e5-large-v2 (1024d)
Microsoft's E5 model family. Particularly strong on question-answering tasks.
Requires "query:" / "passage:" prefix for optimal performance.

### nomic-ai/nomic-embed-text-v2 (768d)
Open-source competitor to OpenAI's text-embedding-3-small. Strong on long-document
embedding due to extended context window (8192 tokens).

## Batch Embedding at Scale

For production systems, use batched embedding to maximise GPU utilisation:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
```

Always normalise embeddings before storing (`normalize_embeddings=True`) to enable
cosine similarity via dot product, which is faster than computing full cosine sim.
        """.strip(),
        "tags": ["embeddings", "models", "semantic-search"],
    },
    {
        "title": "Production RAG System Design Patterns",
        "source": "docs/production-patterns.md",
        "content": """
# Production RAG System Design Patterns

Moving a RAG prototype to production requires addressing reliability, latency,
scalability, and security concerns not present in demos.

## Caching Strategies

### Embedding Cache
Cache query embeddings in Redis to avoid re-encoding identical queries:
- Key: SHA-256 of the query text
- TTL: 1 hour for typical workloads
- Hit rate: typically 40-60% in production

### Result Cache
Cache search results for common queries:
- Invalidate when new documents are indexed
- Use shorter TTL for rapidly changing knowledge bases

## Async Indexing

Never index documents synchronously in the API request path. Use a background task
queue (Celery, ARQ, or asyncio background tasks) to:
1. Accept the document upload (return 202 Accepted)
2. Process chunking and embedding asynchronously
3. Update document status in the database
4. Notify via webhook when complete

## Chunking Strategy Selection

| Document Type  | Recommended Strategy |
|----------------|---------------------|
| Markdown docs  | MarkdownHeaderChunker|
| Code files     | AST-based chunking  |
| Academic PDFs  | SemanticChunker     |
| Support tickets| SlidingWindowChunker|

## Observability

Essential metrics to track:
- `search_latency_p99`: Target < 200ms
- `embedding_throughput`: Tokens/second
- `cache_hit_rate`: Target > 40%
- `indexing_queue_depth`: Alert if > 100

## Security Checklist

- [ ] Rate limit all public endpoints
- [ ] Validate prompt inputs for injection patterns
- [ ] Store API keys as hashes, never plaintext
- [ ] Use JWT with short expiry (60 min) + refresh tokens
- [ ] Implement RBAC before adding multi-tenant data
        """.strip(),
        "tags": ["production", "patterns", "architecture"],
    },
]


async def seed(host: str = "localhost", port: int = 6333, collection: str = "rag_documents") -> None:
    from rag.infrastructure.vector_store.qdrant import QdrantVectorStore
    from rag.infrastructure.embeddings.sentence_transformer import SentenceTransformerProvider
    from rag.infrastructure.vector_store.base import VectorRecord
    from rag.pipeline.chunking import ChunkingPipeline
    from rag.domain.documents import Document, DocumentFormat, DocumentMetadata
    from rag.domain.chunks import ChunkingConfig, ChunkingStrategy

    print(f"Connecting to Qdrant at {host}:{port}...")
    vs = QdrantVectorStore(host=host, port=port)
    ep = SentenceTransformerProvider()
    await ep.warm_up()

    await vs.create_collection(collection, vector_size=ep.dimension)
    chunker = ChunkingPipeline(ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC, chunk_size=512, chunk_overlap=100))

    total_chunks = 0
    for sample in SAMPLE_DOCUMENTS:
        doc = Document(
            content=sample["content"],
            format=DocumentFormat.MARKDOWN,
            metadata=DocumentMetadata(
                source=sample["source"],
                title=sample["title"],
                tags=sample["tags"],
            ),
        )
        chunks = chunker.chunk_document(doc)
        texts = [c.content for c in chunks]
        embeddings = await ep.embed_batch(texts)

        records = [
            VectorRecord(
                id=c.chunk_id,
                vector=emb,
                payload={
                    "content": c.content,
                    "document_id": str(doc.id),
                    "chunk_index": c.chunk_index,
                    "source": sample["source"],
                    "title": sample["title"],
                    "tags": sample["tags"],
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]
        count = await vs.upsert_vectors(collection, records)
        total_chunks += count
        print(f"  Indexed '{sample['title']}': {count} chunks")

    await ep.close()
    await vs.close()
    print(f"\nDone. Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    asyncio.run(seed())
