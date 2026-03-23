"""
FastAPI application — startup, lifespan, and top-level routing.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import AuthMiddleware
from api.routes import router
from config.settings import settings
from generation.llm_client import LLMClient
from ingestion.watcher import DocumentWatcher
from observability.logging import setup_logging
from observability.metrics import MetricsCollector
from retrieval.hybrid_search import HybridSearchEngine
from retrieval.vector_store import VectorStore

# ── Configure logging before anything else ───────────────────

setup_logging()
logger = structlog.get_logger()

# ── App state ────────────────────────────────────────────────

_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle:
      1. Connect to ChromaDB
      2. Verify Ollama is running
      3. Build BM25 index from existing data
      4. Start file watcher in background
    """
    global _startup_time
    _startup_time = time.time()

    logger.info("starting_rag_system", model=settings.ollama_model)

    # ── Initialize components ────────────────────────────
    vector_store = VectorStore()
    await vector_store.connect()

    llm = LLMClient()
    llm_ok = await llm.is_healthy()
    if not llm_ok:
        logger.warning("ollama_not_ready — LLM calls will retry until it's up")

    search_engine = HybridSearchEngine(vector_store)
    await search_engine.rebuild_bm25()

    metrics = MetricsCollector()

    # ── Store on app.state ───────────────────────────────
    app.state.vector_store = vector_store
    app.state.llm_client = llm
    app.state.search_engine = search_engine
    app.state.metrics = metrics

    # ── Background tasks ─────────────────────────────────
    watcher = DocumentWatcher(
        on_chunks_ready=_ingest_callback(vector_store, search_engine, metrics)
    )
    watcher_task = asyncio.create_task(watcher.start())

    logger.info(
        "rag_system_ready",
        watch_dir=settings.document_watch_dir,
        llm_healthy=llm_ok,
    )

    yield

    # ── Shutdown ─────────────────────────────────────────
    watcher_task.cancel()
    await llm.close()
    logger.info("rag_system_stopped")


def _ingest_callback(store, search, metrics):
    """Returns an async callable the watcher uses to store chunks."""

    async def callback(chunks):
        start = time.perf_counter()
        await store.upsert_chunks(chunks)
        await search.rebuild_bm25()
        elapsed = (time.perf_counter() - start) * 1000
        metrics.record_ingestion(elapsed, len(chunks))

    return callback


# ── App Factory ──────────────────────────────────────────────

app = FastAPI(
    title="RAG System",
    description=(
        "Local, free, production-grade Retrieval-Augmented Generation. "
        "Powered by Ollama, ChromaDB, BM25, and FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)

# ── Routes ───────────────────────────────────────────────────

app.include_router(router)


@app.get("/health", tags=["System"])
async def health():
    """System health check (no auth required)."""
    vs: VectorStore = app.state.vector_store
    llm: LLMClient = app.state.llm_client

    return {
        "status": "healthy",
        "version": "1.0.0",
        "vector_store_connected": vs.is_connected,
        "llm_connected": await llm.is_healthy(),
        "documents_indexed": await vs.count(),
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus scrape endpoint (no auth required)."""
    body, content_type = app.state.metrics.prometheus_export()
    return Response(content=body, media_type=content_type)
