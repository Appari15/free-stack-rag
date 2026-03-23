"""
Every Pydantic model used across the system.
Single file so the full data contract is readable in one place.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ───────────────────────── Enums ──────────────────────────────


class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class FileType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"


# ───────────────────────── Documents ──────────────────────────


class DocumentMetadata(BaseModel):
    """Attached to every document and each of its chunks."""

    source: str
    file_type: FileType
    title: str | None = None
    author: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)
    department: str | None = None
    version: str = "1.0"
    language: str = "en"
    custom: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Single chunk ready for embedding and indexing."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    total_chunks: int
    token_count: int = 0
    embedding: list[float] | None = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Chunk content must not be empty")
        return stripped


# ───────────────────────── Query ──────────────────────────────


class MetadataFilter(BaseModel):
    """A single filter clause for retrieval narrowing."""

    field: str
    value: Any
    operator: str = "eq"

    @field_validator("operator")
    @classmethod
    def check_operator(cls, v: str) -> str:
        allowed = {"eq", "neq", "gt", "lt", "gte", "lte", "in", "contains"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {allowed}")
        return v


class QueryRequest(BaseModel):
    query: str
    search_mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    metadata_filters: list[MetadataFilter] = Field(default_factory=list)
    min_relevance_score: float = Field(default=0.3, ge=0.0, le=1.0)
    include_sources: bool = True
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=64, le=4096)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class RetrievedChunk(BaseModel):
    """A chunk returned by the retrieval step, with its score."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: DocumentMetadata
    search_method: SearchMode


class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    search_mode_used: SearchMode
    model_used: str
    total_tokens_used: int = 0
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ───────────────────────── Ingestion ──────────────────────────


class IngestionRequest(BaseModel):
    file_path: str | None = None
    raw_content: str | None = None
    metadata: DocumentMetadata
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 512)
        if v >= size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class IngestionResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    chunks_created: int
    processing_time_ms: float
    message: str


# ───────────────────────── Health / Metrics ───────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    vector_store_connected: bool
    llm_connected: bool
    documents_indexed: int
    uptime_seconds: float


class SystemMetrics(BaseModel):
    total_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_documents: int
    total_chunks: int
    queries_by_mode: dict[str, int]
    errors: int
