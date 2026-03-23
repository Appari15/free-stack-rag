"""
All API routes.

POST /api/v1/query           — ask a question
POST /api/v1/ingest          — ingest from raw content or file path
POST /api/v1/ingest/upload   — upload a file directly
GET  /api/v1/documents       — list indexed documents
DELETE /api/v1/documents/{id} — remove a document
GET  /api/v1/metrics         — system metrics summary
"""

from __future__ import annotations

import time
from typing import Annotated

import structlog
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    UploadFile,
    File,
    Form,
)

from core.models import (
    DocumentMetadata,
    DocumentStatus,
    FileType,
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    SystemMetrics,
)
from generation.llm_client import LLMClient
from ingestion.loader import DocumentLoader
from observability.metrics import MetricsCollector
from retrieval.hybrid_search import HybridSearchEngine
from retrieval.vector_store import VectorStore

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["RAG"])


# ── Dependencies ─────────────────────────────────────────────


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_search_engine(request: Request) -> HybridSearchEngine:
    return request.app.state.search_engine


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_metrics(request: Request) -> MetricsCollector:
    return request.app.state.metrics


# ── Query ────────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    search: Annotated[HybridSearchEngine, Depends(get_search_engine)],
    llm: Annotated[LLMClient, Depends(get_llm)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics)],
):
    """
    Main RAG endpoint.
    Retrieves relevant chunks and generates a grounded answer.
    """
    start = time.perf_counter()

    logger.info(
        "query_start",
        query=body.query,
        mode=body.search_mode.value,
        top_k=body.top_k,
        filters=len(body.metadata_filters),
    )

    # 1. Retrieve
    retrieved = await search.search(
        query=body.query,
        mode=body.search_mode,
        top_k=body.top_k,
        filters=body.metadata_filters,
        min_score=body.min_relevance_score,
    )

    if not retrieved:
        latency = (time.perf_counter() - start) * 1000
        metrics.record_query(latency, body.search_mode, 0)
        return QueryResponse(
            answer="I couldn't find any relevant documents to answer your question. "
            "Try broadening your search or lowering the minimum relevance score.",
            sources=[],
            search_mode_used=body.search_mode,
            model_used=llm.model,
            latency_ms=round(latency, 2),
        )

    # 2. Generate
    result = await llm.generate(
        query=body.query,
        retrieved_chunks=retrieved,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    # 3. Record metrics
    metrics.record_query(
        latency_ms=latency_ms,
        mode=body.search_mode,
        chunks_count=len(retrieved),
        tokens=result["tokens_used"],
    )

    response = QueryResponse(
        answer=result["answer"],
        sources=retrieved if body.include_sources else [],
        search_mode_used=body.search_mode,
        model_used=result["model"],
        total_tokens_used=result["tokens_used"],
        latency_ms=round(latency_ms, 2),
    )

    logger.info(
        "query_done",
        query_id=response.query_id,
        latency_ms=response.latency_ms,
        chunks=len(retrieved),
        tokens=result["tokens_used"],
    )

    return response


# ── Ingestion ────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestionResponse)
async def ingest(
    body: IngestionRequest,
    store: Annotated[VectorStore, Depends(get_vector_store)],
    search: Annotated[HybridSearchEngine, Depends(get_search_engine)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics)],
):
    """Ingest a document from raw content or a file path."""
    start = time.perf_counter()

    loader = DocumentLoader(body.chunk_size, body.chunk_overlap)

    if body.raw_content:
        chunks = loader.load_text(body.raw_content, body.metadata)
    elif body.file_path:
        chunks = loader.load_file(body.file_path)
    else:
        raise HTTPException(400, "Provide either raw_content or file_path")

    if not chunks:
        raise HTTPException(400, "No content could be extracted from the input")

    await store.upsert_chunks(chunks)
    await search.rebuild_bm25()

    elapsed = (time.perf_counter() - start) * 1000
    metrics.record_ingestion(elapsed, len(chunks))

    return IngestionResponse(
        document_id=chunks[0].document_id,
        status=DocumentStatus.INDEXED,
        chunks_created=len(chunks),
        processing_time_ms=round(elapsed, 2),
        message=f"Indexed {len(chunks)} chunks from '{body.metadata.source}'",
    )


@router.post("/ingest/upload", response_model=IngestionResponse)
async def upload_file(
    file: UploadFile = File(..., description="Document to ingest"),
    title: str = Form(None),
    tags: str = Form("", description="Comma-separated tags"),
    department: str = Form(None),
    store: VectorStore = Depends(get_vector_store),
    search: HybridSearchEngine = Depends(get_search_engine),
    metrics: MetricsCollector = Depends(get_metrics),
):
    """Upload a file for ingestion via multipart form."""
    start = time.perf_counter()

    suffix = (file.filename or "doc.txt").rsplit(".", 1)[-1].lower()
    type_map = {
        "pdf": FileType.PDF,
        "md": FileType.MARKDOWN,
        "markdown": FileType.MARKDOWN,
        "txt": FileType.TXT,
        "html": FileType.HTML,
        "htm": FileType.HTML,
    }
    file_type = type_map.get(suffix, FileType.TXT)

    metadata = DocumentMetadata(
        source=file.filename or "upload",
        file_type=file_type,
        title=title or file.filename,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        department=department,
    )

    content = await file.read()
    loader = DocumentLoader()
    chunks = loader.load_bytes(content, file.filename or "upload", metadata)

    if not chunks:
        raise HTTPException(400, "Could not extract content from the uploaded file")

    await store.upsert_chunks(chunks)
    await search.rebuild_bm25()

    elapsed = (time.perf_counter() - start) * 1000
    metrics.record_ingestion(elapsed, len(chunks))

    return IngestionResponse(
        document_id=chunks[0].document_id,
        status=DocumentStatus.INDEXED,
        chunks_created=len(chunks),
        processing_time_ms=round(elapsed, 2),
        message=f"Uploaded and indexed '{file.filename}' ({len(chunks)} chunks)",
    )


# ── Management ───────────────────────────────────────────────


@router.get("/documents")
async def list_documents(
    store: Annotated[VectorStore, Depends(get_vector_store)],
):
    """List all indexed documents."""
    return await store.list_documents()


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    store: Annotated[VectorStore, Depends(get_vector_store)],
    search: Annotated[HybridSearchEngine, Depends(get_search_engine)],
):
    """Delete a document and all its chunks."""
    deleted = await store.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(404, f"Document '{document_id}' not found")
    await search.rebuild_bm25()
    return {"document_id": document_id, "chunks_deleted": deleted}


@router.get("/metrics", response_model=SystemMetrics)
async def system_metrics(
    store: Annotated[VectorStore, Depends(get_vector_store)],
    metrics: Annotated[MetricsCollector, Depends(get_metrics)],
):
    """System metrics summary."""
    docs = await store.list_documents()
    count = await store.count()
    return metrics.get_summary(total_docs=len(docs), total_chunks=count)
