"""
Prometheus metrics for request tracking and system health.

Exposes /metrics endpoint that Prometheus scrapes.
Grafana then visualizes the data.
"""

from __future__ import annotations

import time
from collections import defaultdict

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from core.models import SearchMode, SystemMetrics


class MetricsCollector:
    """
    Central metrics registry.  Singleton created at app startup.
    """

    def __init__(self):
        # ── Counters ─────────────────────────────────────
        self.queries_total = Counter(
            "rag_queries_total",
            "Total queries processed",
            ["search_mode"],
        )
        self.ingestions_total = Counter(
            "rag_ingestions_total",
            "Total documents ingested",
        )
        self.errors_total = Counter(
            "rag_errors_total",
            "Total errors by type",
            ["error_type"],
        )

        # ── Histograms ──────────────────────────────────
        self.query_latency = Histogram(
            "rag_query_latency_ms",
            "Query latency in milliseconds",
            buckets=[25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
        )
        self.chunks_retrieved = Histogram(
            "rag_chunks_retrieved",
            "Chunks retrieved per query",
            buckets=[1, 2, 3, 5, 10, 15, 20],
        )
        self.tokens_used = Histogram(
            "rag_tokens_used",
            "Tokens used per generation",
            buckets=[100, 250, 500, 1000, 2000, 4000],
        )
        self.ingestion_latency = Histogram(
            "rag_ingestion_latency_ms",
            "Ingestion latency in milliseconds",
            buckets=[100, 500, 1000, 5000, 10000, 30000],
        )

        # ── Gauges ─────────────────────────────────────
        self.documents_indexed = Gauge(
            "rag_documents_indexed",
            "Number of documents indexed",
        )

        # ── Info ─────────────────────────────────────────
        self.app_info = Info("rag_app", "Application metadata")
        self.app_info.info(
            {
                "version": "1.0.0",
                "llm": "ollama",
                "vector_store": "chromadb",
                "embedding_model": "all-MiniLM-L6-v2",
            }
        )

        # ── Internal tracking for /metrics endpoint ──────
        self._latencies: list[float] = []
        self._mode_counts: dict[str, int] = defaultdict(int)
        self._error_count = 0
        self._start_time = time.time()

    # ── Recording ────────────────────────────────────────

    def record_query(
        self,
        latency_ms: float,
        mode: SearchMode,
        chunks_count: int,
        tokens: int = 0,
    ):
        self.queries_total.labels(search_mode=mode.value).inc()
        self.query_latency.observe(latency_ms)
        self.chunks_retrieved.observe(chunks_count)
        if tokens:
            self.tokens_used.observe(tokens)

        self._latencies.append(latency_ms)
        self._mode_counts[mode.value] += 1

    def record_ingestion(self, latency_ms: float, chunks: int):
        self.ingestions_total.inc()
        self.ingestion_latency.observe(latency_ms)
        self.documents_indexed.inc()

    def record_error(self, error_type: str):
        self.errors_total.labels(error_type=error_type).inc()
        self._error_count += 1

    # ── Export ───────────────────────────────────────────

    def prometheus_export(self) -> tuple[bytes, str]:
        """Returns (body, content_type) for the /metrics endpoint."""
        return generate_latest(), CONTENT_TYPE_LATEST

    def get_summary(self, total_docs: int = 0, total_chunks: int = 0) -> SystemMetrics:
        """Human-readable summary via /api/v1/metrics."""
        total = len(self._latencies)

        if total == 0:
            return SystemMetrics(
                total_queries=0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                total_documents=total_docs,
                total_chunks=total_chunks,
                queries_by_mode={},
                errors=self._error_count,
            )

        sorted_lat = sorted(self._latencies)

        def percentile(p: float) -> float:
            idx = int(p * total)
            return round(sorted_lat[min(idx, total - 1)], 2)

        return SystemMetrics(
            total_queries=total,
            avg_latency_ms=round(sum(self._latencies) / total, 2),
            p50_latency_ms=percentile(0.50),
            p95_latency_ms=percentile(0.95),
            total_documents=total_docs,
            total_chunks=total_chunks,
            queries_by_mode=dict(self._mode_counts),
            errors=self._error_count,
        )
