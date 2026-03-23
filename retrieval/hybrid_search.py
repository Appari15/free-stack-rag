"""
Hybrid search engine: combines semantic search (ChromaDB kNN)
with keyword search (BM25) using Reciprocal Rank Fusion.

Why RRF?
  - Rank-based, so different score scales don't matter
  - Documents in both result sets get naturally boosted
  - Simple, no training needed, proven effective
  - The k constant (60) prevents top results from dominating

  RRF_score(doc) = Σ  weight_i / (k + rank_i)
"""

from __future__ import annotations

import structlog

from config.settings import settings
from core.models import (
    DocumentMetadata,
    FileType,
    MetadataFilter,
    RetrievedChunk,
    SearchMode,
)
from retrieval.bm25_index import BM25Index
from retrieval.vector_store import VectorStore

logger = structlog.get_logger()

# RRF constant — standard value from the original paper
RRF_K = 60


class HybridSearchEngine:
    """
    Single entry point for all search modes.
    Manages the BM25 index lifecycle and fusion logic.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25_index = BM25Index()
        self._semantic_weight = settings.semantic_weight
        self._keyword_weight = settings.keyword_weight

    async def rebuild_bm25(self):
        """Rebuild BM25 index from all chunks in vector store."""
        all_chunks = await self.vector_store.get_all_chunks()
        self.bm25_index.build(all_chunks)

    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 5,
        filters: list[MetadataFilter] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """
        Main search interface.  Dispatches to the right strategy.
        """
        if mode == SearchMode.SEMANTIC:
            results = await self._semantic(query, top_k, filters)
        elif mode == SearchMode.KEYWORD:
            results = await self._keyword(query, top_k)
        else:
            results = await self._hybrid(query, top_k, filters)

        # Filter by minimum relevance score
        filtered = [r for r in results if r.score >= min_score]

        logger.info(
            "search_done",
            mode=mode.value,
            raw_results=len(results),
            after_min_score=len(filtered),
            min_score=min_score,
        )
        return filtered[:top_k]

    # ── Strategies ───────────────────────────────────────────

    async def _semantic(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None,
    ) -> list[RetrievedChunk]:
        return await self.vector_store.semantic_search(query, top_k, filters)

    async def _keyword(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        # Rebuild BM25 if empty (lazy initialization)
        if not self.bm25_index.is_built:
            await self.rebuild_bm25()

        raw = self.bm25_index.search(query, top_k)
        return [_raw_to_chunk(r, SearchMode.KEYWORD) for r in raw]

    async def _hybrid(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None,
    ) -> list[RetrievedChunk]:
        """
        Fetch from both systems, fuse with RRF.
        """
        fetch_k = top_k * 3  # Over-fetch for better fusion

        # Run both searches
        semantic_results = await self._semantic(query, fetch_k, filters)
        keyword_results = await self._keyword(query, fetch_k)

        # ── Reciprocal Rank Fusion ───────────────────────
        fused_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(semantic_results):
            rrf_score = self._semantic_weight / (RRF_K + rank + 1)
            fused_scores[chunk.chunk_id] = (
                fused_scores.get(chunk.chunk_id, 0) + rrf_score
            )
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(keyword_results):
            rrf_score = self._keyword_weight / (RRF_K + rank + 1)
            fused_scores[chunk.chunk_id] = (
                fused_scores.get(chunk.chunk_id, 0) + rrf_score
            )
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        # Sort by fused score descending
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Normalize to [0, 1]
        max_score = ranked[0][1] if ranked else 1.0

        results: list[RetrievedChunk] = []
        for chunk_id, score in ranked[:top_k]:
            chunk = chunk_map[chunk_id]
            # Create new chunk with fused score
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=round(score / max_score, 4),
                    metadata=chunk.metadata,
                    search_method=SearchMode.HYBRID,
                )
            )

        return results


def _raw_to_chunk(raw: dict, mode: SearchMode) -> RetrievedChunk:
    """Convert a BM25 raw dict to a RetrievedChunk."""
    meta = raw.get("metadata", {})
    tags_str = meta.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

    return RetrievedChunk(
        chunk_id=raw["chunk_id"],
        document_id=meta.get("document_id", ""),
        content=raw["content"],
        score=raw["score"],
        metadata=DocumentMetadata(
            source=meta.get("source", "unknown"),
            file_type=FileType(meta.get("file_type", "txt")),
            title=meta.get("title"),
            department=meta.get("department"),
            tags=tags,
        ),
        search_method=mode,
    )
