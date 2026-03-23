"""
ChromaDB vector store — handles all vector CRUD operations.
"""

from __future__ import annotations

from typing import Any

import chromadb
import structlog

from config.settings import settings
from core.embeddings import embed_texts, embed_single
from core.models import (
    DocumentChunk,
    DocumentMetadata,
    FileType,
    MetadataFilter,
    RetrievedChunk,
    SearchMode,
)

logger = structlog.get_logger()


class VectorStore:
    """
    Wraps ChromaDB with:
      - Batch upsert with automatic embedding
      - Semantic search with metadata filtering
      - Document management (list, delete)
    """

    def __init__(self):
        self._client: chromadb.HttpClient | None = None
        self._collection = None
        self.is_connected = False

    async def connect(self):
        """Initialize connection and collection."""
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.is_connected = True
        count = self._collection.count()
        logger.info(
            "vector_store_connected",
            collection=settings.chroma_collection,
            existing_chunks=count,
        )

    # ── Write ────────────────────────────────────────────

    async def upsert_chunks(self, chunks: list[DocumentChunk]):
        """Embed and store a batch of chunks."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = embed_texts(documents)

        metadatas = []
        for c in chunks:
            meta: dict[str, Any] = {
                "document_id": c.document_id,
                "source": c.metadata.source,
                "file_type": c.metadata.file_type.value,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "token_count": c.token_count,
                "language": c.metadata.language,
                "version": c.metadata.version,
            }
            if c.metadata.title:
                meta["title"] = c.metadata.title
            if c.metadata.author:
                meta["author"] = c.metadata.author
            if c.metadata.department:
                meta["department"] = c.metadata.department
            if c.metadata.tags:
                meta["tags"] = ",".join(c.metadata.tags)
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("chunks_upserted", count=len(chunks))

    # ── Semantic Search ──────────────────────────────────

    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: list[MetadataFilter] | None = None,
    ) -> list[RetrievedChunk]:
        """
        kNN search using cosine similarity.
        """
        query_embedding = embed_single(query)
        where = _build_where(filters)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return _parse_results(results, SearchMode.SEMANTIC)

    # ── Full-text Search (ChromaDB built-in) ─────────────

    async def text_search(
        self,
        query: str,
        top_k: int = 5,
        filters: list[MetadataFilter] | None = None,
    ) -> list[RetrievedChunk]:
        """ChromaDB's built-in text search (not true BM25)."""
        where = _build_where(filters)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return _parse_results(results, SearchMode.KEYWORD)

    # ── Bulk Read (for BM25 index) ───────────────────────

    async def get_all_chunks(self) -> list[tuple[str, str, dict]]:
        """
        Returns (chunk_id, text, metadata) for every stored chunk.
        Used by the BM25 index builder.
        """
        data = self._collection.get(include=["documents", "metadatas"])
        results = []
        for i, doc in enumerate(data["documents"]):
            results.append((data["ids"][i], doc, data["metadatas"][i]))
        return results

    # ── Management ───────────────────────────────────────

    async def count(self) -> int:
        return self._collection.count() if self._collection else 0

    async def list_documents(self) -> list[dict]:
        """Unique documents with their metadata."""
        data = self._collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in data["metadatas"]:
            doc_id = meta.get("document_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "source": meta.get("source"),
                    "title": meta.get("title"),
                    "total_chunks": meta.get("total_chunks", 0),
                    "file_type": meta.get("file_type"),
                }
        return list(seen.values())

    async def delete_document(self, document_id: str) -> int:
        data = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        ids = data["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info("document_deleted", document_id=document_id, chunks=len(ids))
        return len(ids)

    async def clear(self):
        """Delete everything — useful for tests."""
        self._client.delete_collection(settings.chroma_collection)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vector_store_cleared")


# ── Internal Helpers ─────────────────────────────────────────


def _build_where(filters: list[MetadataFilter] | None) -> dict | None:
    """Convert our MetadataFilter list to ChromaDB where clause."""
    if not filters:
        return None

    op_map = {
        "eq": "$eq",
        "neq": "$ne",
        "gt": "$gt",
        "lt": "$lt",
        "gte": "$gte",
        "lte": "$lte",
        "in": "$in",
        "contains": "$eq",
    }

    conditions = []
    for f in filters:
        chroma_op = op_map.get(f.operator, "$eq")
        conditions.append({f.field: {chroma_op: f.value}})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _parse_results(raw: dict, mode: SearchMode) -> list[RetrievedChunk]:
    """Convert ChromaDB results to our RetrievedChunk model."""
    chunks: list[RetrievedChunk] = []

    if not raw["ids"] or not raw["ids"][0]:
        return chunks

    for i, chunk_id in enumerate(raw["ids"][0]):
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i] if raw.get("distances") else 0

        # ChromaDB cosine distance → similarity: score = 1 - distance
        score = max(0.0, min(1.0, 1.0 - distance))

        tags_str = meta.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                content=raw["documents"][0][i],
                score=round(score, 4),
                metadata=DocumentMetadata(
                    source=meta.get("source", "unknown"),
                    file_type=FileType(meta.get("file_type", "txt")),
                    title=meta.get("title"),
                    author=meta.get("author"),
                    department=meta.get("department"),
                    tags=tags,
                    language=meta.get("language", "en"),
                    version=meta.get("version", "1.0"),
                ),
                search_method=mode,
            )
        )

    return chunks
