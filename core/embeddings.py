"""
Embedding manager — wraps sentence-transformers for consistent
embedding across ingestion and query time.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Fast (14k sentences/sec on CPU)
  - Great quality for its size
"""

from __future__ import annotations

import structlog
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = structlog.get_logger()

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("loading_embedding_model", model=settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info(
            "embedding_model_loaded",
            model=settings.embedding_model,
            dimension=_model.get_sentence_embedding_dimension(),
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """Convenience wrapper for a single text."""
    return embed_texts([text])[0]


def get_dimension() -> int:
    """Return the embedding dimension of the loaded model."""
    return _get_model().get_sentence_embedding_dimension()
