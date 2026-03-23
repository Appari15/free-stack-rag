"""
End-to-end tests.  Require running services:
    make up
    pytest tests/test_e2e.py -v -m e2e

Marked with @pytest.mark.e2e so they're skipped in CI
unless services are available.
"""

import pytest
import httpx

BASE = "http://localhost:8080"
HEADERS = {"X-API-Key": "CHANGE_ME_IN_PRODUCTION", "Content-Type": "application/json"}


@pytest.fixture(scope="module")
def client():
    c = httpx.Client(base_url=BASE, headers=HEADERS, timeout=120)
    yield c
    c.close()


def _is_running():
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _is_running(), reason="Services not running")


@pytest.mark.e2e
class TestHealthAndDocs:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["vector_store_connected"] is True

    def test_swagger(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


@pytest.mark.e2e
class TestIngestionAndQuery:
    def test_full_flow(self, client):
        # 1. Ingest
        ingest_body = {
            "raw_content": (
                "Retrieval-Augmented Generation combines retrieval with "
                "language model generation to produce accurate, grounded "
                "answers. It reduces hallucinations by providing factual "
                "context from a knowledge base. RAG systems use vector "
                "databases for semantic similarity search."
            ),
            "metadata": {
                "source": "e2e_test.txt",
                "file_type": "txt",
                "title": "E2E Test Doc",
                "tags": ["e2e", "test"],
                "department": "testing",
            },
        }

        r = client.post("/api/v1/ingest", json=ingest_body)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "indexed"
        assert data["chunks_created"] >= 1
        doc_id = data["document_id"]

        # 2. Query — semantic
        r = client.post(
            "/api/v1/query",
            json={
                "query": "What is RAG and how does it work?",
                "search_mode": "semantic",
                "top_k": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["answer"]) > 10
        assert data["latency_ms"] > 0
        assert data["search_mode_used"] == "semantic"

        # 3. Query — hybrid
        r = client.post(
            "/api/v1/query",
            json={
                "query": "hallucinations",
                "search_mode": "hybrid",
                "top_k": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["answer"]) > 10

        # 4. Query — with metadata filter
        r = client.post(
            "/api/v1/query",
            json={
                "query": "RAG",
                "search_mode": "semantic",
                "top_k": 3,
                "metadata_filters": [
                    {"field": "department", "value": "testing", "operator": "eq"}
                ],
            },
        )
        assert r.status_code == 200

        # 5. List documents
        r = client.get("/api/v1/documents")
        assert r.status_code == 200
        docs = r.json()
        assert any(d["document_id"] == doc_id for d in docs)

        # 6. Metrics
        r = client.get("/api/v1/metrics")
        assert r.status_code == 200
        assert r.json()["total_queries"] >= 3

        # 7. Cleanup
        r = client.delete(f"/api/v1/documents/{doc_id}")
        assert r.status_code == 200
        assert r.json()["chunks_deleted"] >= 1


@pytest.mark.e2e
class TestFileUpload:
    def test_upload_txt(self, client):
        r = client.post(
            "/api/v1/ingest/upload",
            files={
                "file": (
                    "test.txt",
                    b"Vector databases store embeddings for fast similarity search.",
                    "text/plain",
                )
            },
            data={"title": "Upload Test", "tags": "upload,test"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["chunks_created"] >= 1

        # Cleanup
        client.delete(f"/api/v1/documents/{data['document_id']}")
