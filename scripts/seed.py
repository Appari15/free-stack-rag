"""
Seed the RAG system with sample documents for demo purposes.
Run: python scripts/seed.py
"""

import httpx
import sys
from pathlib import Path

BASE_URL = "http://localhost:8080"
API_KEY = "CHANGE_ME_IN_PRODUCTION"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

SAMPLE_DIR = Path(__file__).parent.parent / "sample_docs"


def seed():
    client = httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=60)

    # Check health
    resp = client.get("/health")
    if resp.status_code != 200:
        print(f"❌ App not healthy: {resp.text}")
        sys.exit(1)
    print(f"✅ App healthy: {resp.json()['status']}")

    # Ingest each sample document
    for file_path in sorted(SAMPLE_DIR.glob("*.md")):
        content = file_path.read_text()
        body = {
            "raw_content": content,
            "metadata": {
                "source": file_path.name,
                "file_type": "markdown",
                "title": file_path.stem.replace("_", " ").title(),
                "tags": ["sample", "documentation"],
                "department": "engineering",
            },
        }

        resp = client.post("/api/v1/ingest", json=body)
        data = resp.json()
        print(
            f"📄 {file_path.name}: {data['chunks_created']} chunks ({data['processing_time_ms']:.0f}ms)"
        )

    # Show what we have
    docs = client.get("/api/v1/documents").json()
    print(f"\n📚 Total documents indexed: {len(docs)}")

    # Run a sample query
    print("\n🔍 Running sample query...")
    resp = client.post(
        "/api/v1/query",
        json={
            "query": "How does RAG reduce hallucinations?",
            "search_mode": "hybrid",
            "top_k": 3,
        },
    )
    data = resp.json()
    print(f"⏱️  Latency: {data['latency_ms']:.0f}ms")
    print(f"📎 Sources: {len(data['sources'])}")
    print(f"💬 Answer:\n{data['answer'][:500]}...")


if __name__ == "__main__":
    seed()
