"""
Shared fixtures for tests.
"""

import pytest

from core.chunking import ChunkConfig
from core.models import DocumentMetadata, FileType


@pytest.fixture
def sample_metadata():
    return DocumentMetadata(
        source="test_doc.md",
        file_type=FileType.MARKDOWN,
        title="Test Document",
        tags=["test", "unit"],
        department="engineering",
    )


@pytest.fixture
def chunk_config():
    return ChunkConfig(chunk_size=100, chunk_overlap=10, min_chunk_size=10)


@pytest.fixture
def short_text():
    return "This is a short document that fits in one chunk."


@pytest.fixture
def long_text():
    return (
        """
# Retrieval-Augmented Generation

RAG is an AI framework that enhances LLMs by grounding their responses
in external knowledge retrieved at inference time.

## How It Works

The pipeline consists of three stages: ingestion, retrieval, and generation.

During ingestion, documents are chunked into smaller pieces, embedded into
vectors, and stored in a vector database. This happens offline before any
queries are processed.

During retrieval, the user's question is also embedded into a vector, and
the database performs approximate nearest neighbor search to find the most
similar chunks.

During generation, the retrieved chunks are inserted into the LLM prompt
as context. The model then generates an answer grounded in this evidence.

## Benefits

RAG reduces hallucinations because the model has factual context to reference.
It keeps knowledge up to date because new documents can be added without
retraining. It enables source attribution because we know exactly which
chunks contributed to the answer.

## Hybrid Search

Combining semantic search with keyword search produces better results than
either approach alone. Semantic search finds conceptually similar content
even when exact words don't match. Keyword search catches specific terms
and exact phrases that embedding models might miss.

Reciprocal Rank Fusion merges the two ranked lists into a single ranked
list that benefits from both approaches.
"""
        * 3
    )  # Repeat to ensure multiple chunks
