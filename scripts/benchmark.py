"""
Benchmark the RAG system — measures latency, retrieval quality.
Run: python scripts/benchmark.py

Requires documents to be seeded first (python scripts/seed.py).
"""

import time
import statistics

import httpx

BASE_URL = "http://localhost:8080"
HEADERS = {"X-API-Key": "CHANGE_ME_IN_PRODUCTION", "Content-Type": "application/json"}

QUERIES = [
    {"query": "What is RAG?", "mode": "semantic"},
    {"query": "What is RAG?", "mode": "keyword"},
    {"query": "What is RAG?", "mode": "hybrid"},
    {"query": "How does vector search work?", "mode": "hybrid"},
    {"query": "What temperature should I use for RAG?", "mode": "hybrid"},
    {"query": "What are the benefits of RAG over fine-tuning?", "mode": "hybrid"},
    {"query": "Explain chunking strategies", "mode": "semantic"},
    {"query": "ChromaDB vs Pinecone", "mode": "keyword"},
]


def benchmark():
    client = httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=120)

    print("🏋️ RAG System Benchmark")
    print("=" * 70)

    latencies = []
    results = []

    for q in QUERIES:
        body = {
            "query": q["query"],
            "search_mode": q["mode"],
            "top_k": 5,
            "min_relevance_score": 0.2,
        }

        start = time.perf_counter()
        resp = client.post("/api/v1/query", json=body)
        elapsed = (time.perf_counter() - start) * 1000

        data = resp.json()
        latencies.append(elapsed)

        sources_count = len(data.get("sources", []))
        tokens = data.get("total_tokens_used", 0)
        top_score = data["sources"][0]["score"] if data.get("sources") else 0

        results.append(
            {
                "query": q["query"][:40],
                "mode": q["mode"],
                "latency_ms": round(elapsed),
                "sources": sources_count,
                "top_score": top_score,
                "tokens": tokens,
            }
        )

        print(
            f"  {q['mode']:8s} | {elapsed:6.0f}ms | "
            f"{sources_count} sources (top={top_score:.2f}) | "
            f"{tokens:4d} tokens | {q['query'][:45]}"
        )

    print("=" * 70)
    print(f"  Queries:     {len(QUERIES)}")
    print(f"  Avg latency: {statistics.mean(latencies):.0f}ms")
    print(f"  P50 latency: {statistics.median(latencies):.0f}ms")
    print(f"  P95 latency: {sorted(latencies)[int(0.95 * len(latencies))]:.0f}ms")
    print(f"  Min latency: {min(latencies):.0f}ms")
    print(f"  Max latency: {max(latencies):.0f}ms")

    # Also hit the metrics endpoint
    metrics = client.get("/api/v1/metrics").json()
    print("\n📊 System Metrics:")
    print(f"  Total queries:  {metrics['total_queries']}")
    print(f"  Total docs:     {metrics['total_documents']}")
    print(f"  Total chunks:   {metrics['total_chunks']}")
    print(f"  By mode:        {metrics['queries_by_mode']}")


if __name__ == "__main__":
    benchmark()
