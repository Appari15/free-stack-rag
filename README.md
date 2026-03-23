# 🔍 RAG System — Local, Free, Production-Grade

A complete Retrieval-Augmented Generation system that runs **entirely on
your machine** with zero API costs. Built to learn and demonstrate every
concept behind production RAG: chunking strategies, hybrid search,
metadata filtering, prompt engineering, and full observability.

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│                   FastAPI                         │
│               Orchestration Layer                 │
└──────┬────────────────┬──────────────────┬───────┘
       │                │                  │
       ▼                ▼                  ▼
┌────────────┐  ┌──────────────┐  ┌──────────────┐
│  ChromaDB  │  │   BM25       │  │   Ollama     │
│  Semantic  │  │   Keyword    │  │   Local LLM  │
│  Search    │  │   Search     │  │  (Llama 3.1) │
└─────┬──────┘  └──────┬───────┘  └──────────────┘
      │                │
      └───────┬────────┘
              ▼
      ┌──────────────┐
      │  Reciprocal  │
      │  Rank Fusion │
      └──────────────┘
```

## Features

| Feature                  | Implementation                         |
|--------------------------|----------------------------------------|
| 🧠 Local LLM            | Ollama with Llama 3.1 8B               |
| 📐 Embeddings            | sentence-transformers (all-MiniLM-L6)  |
| 🗄️ Vector Store          | ChromaDB with cosine similarity        |
| 🔎 Hybrid Search         | BM25 + Semantic with RRF fusion        |
| 🏷️ Metadata Filtering    | Field-level filters on any attribute   |
| 📄 Multi-format Ingest   | PDF, Markdown, TXT, HTML                |
| 📁 Auto-ingest           | File watcher on a directory            |
| 📊 Observability         | Prometheus metrics + Grafana dashboard |
| 🧪 Evaluation Suite      | Precision, Recall, MRR, Faithfulness   |
| 🔑 Auth                  | API key middleware                     |
| 📤 File Upload           | Direct upload via multipart form       |
| ⚡ Cost                   | $0 — everything runs locally           |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/free-stack/free-stack-rag.git
cd rag-system

# 2. Start everything
make up

# 3. Wait for Ollama to pull the model (~5 min first time)
make logs

# 4. Open Swagger UI
open http://localhost:8080/docs

# 5. Open Grafana dashboard
open http://localhost:3000  # admin / admin
```

## Example Queries

```bash
# Ingest a document
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "X-API-Key: CHANGE_ME_IN_PRODUCTION" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_content": "RAG combines retrieval with generation...",
    "metadata": {
      "source": "notes.txt",
      "file_type": "txt",
      "title": "RAG Notes"
    }
  }'

# Query with hybrid search
curl -X POST http://localhost:8080/api/v1/query \
  -H "X-API-Key: CHANGE_ME_IN_PRODUCTION" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does RAG reduce hallucinations?",
    "search_mode": "hybrid",
    "top_k": 5
  }'

# Query with metadata filter
curl -X POST http://localhost:8080/api/v1/query \
  -H "X-API-Key: CHANGE_ME_IN_PRODUCTION" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vector databases",
    "metadata_filters": [
      {"field": "department", "value": "engineering", "operator": "eq"}
    ]
  }'
```

## Architecture Decisions

| Decision                | Rationale                                            |
|-------------------------|------------------------------------------------------|
| Recursive chunking      | Respects sentence/paragraph boundaries               |
| 512-token chunks        | Balances context richness vs embedding quality        |
| 50-token overlap        | Prevents information loss at chunk boundaries         |
| RRF over score fusion   | Rank-based; immune to different score scales          |
| 0.7/0.3 semantic/BM25   | Semantic dominates but BM25 catches exact matches     |
| Cosine similarity       | Normalized; works well with sentence-transformers     |

## AWS Production Equivalent

This local stack maps directly to production AWS services:

| Local                    | AWS Equivalent              |
|--------------------------|-----------------------------|
| Ollama (Llama 3.1)      | Amazon Bedrock (Claude 3.5) |
| sentence-transformers    | Titan Embeddings V2         |
| ChromaDB                 | OpenSearch Serverless        |
| Local filesystem         | S3                          |
| FastAPI                  | API Gateway + Lambda         |
| Prometheus + Grafana     | CloudWatch                  |
