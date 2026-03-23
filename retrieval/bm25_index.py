"""
BM25 keyword search index built on top of rank_bm25.

BM25 (Best Match 25) is a bag-of-words ranking function that scores
documents by term frequency, inverse document frequency, and document
length normalization.  It excels at finding exact keyword matches
that semantic search may miss.

This index is rebuilt when new documents are ingested.
"""

from __future__ import annotations

import re
import threading

import structlog
from rank_bm25 import BM25Plus

logger = structlog.get_logger()

# ── Stop words for lightweight tokenization ──────────────────

STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "it",
        "its",
        "this",
        "that",
        "and",
        "or",
        "not",
        "no",
        "but",
        "if",
        "so",
        "as",
        "up",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split, remove stopwords and short tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


class BM25Index:
    """
    Thread-safe BM25 index that can be rebuilt incrementally.

    Usage:
        index = BM25Index()
        index.build([(id1, text1, meta1), (id2, text2, meta2), ...])
        results = index.search("some query", top_k=5)
    """

    def __init__(self):
        self._bm25: BM25Plus | None = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metas: list[dict] = []
        self._lock = threading.Lock()
        self._doc_count = 0

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def build(self, chunk_data: list[tuple[str, str, dict]]):
        """
        Build (or rebuild) the entire BM25 index from scratch.

        Args:
            chunk_data: list of (chunk_id, text, metadata) tuples
        """
        if not chunk_data:
            return

        ids = [c[0] for c in chunk_data]
        texts = [c[1] for c in chunk_data]
        metas = [c[2] for c in chunk_data]

        tokenized = [tokenize(t) for t in texts]

        with self._lock:
            self._bm25 = BM25Plus(tokenized)
            self._chunk_ids = ids
            self._chunk_texts = texts
            self._chunk_metas = metas
            self._doc_count = len(ids)

        logger.info("bm25_index_built", documents=self._doc_count)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Score all documents against the query and return top-k.

        Returns list of dicts with keys:
          chunk_id, content, score (normalized 0-1), metadata
        """
        with self._lock:
            if self._bm25 is None:
                return []

            query_tokens = tokenize(query)
            if not query_tokens:
                return []

            scores = self._bm25.get_scores(query_tokens)

        # Pair and sort
        scored = sorted(
            zip(self._chunk_ids, self._chunk_texts, scores, self._chunk_metas),
            key=lambda x: x[2],
            reverse=True,
        )[:top_k]

        if not scored:
            return []

        # Normalize scores to [0, 1]
        max_score = max(s[2] for s in scored) if scored else 1.0
        if max_score <= 0:
            max_score = 1.0

        results = []
        for chunk_id, content, score, meta in scored:
            results.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "score": round(score / max_score, 4) if max_score > 0 else 0,
                    "metadata": meta,
                }
            )

        return results
