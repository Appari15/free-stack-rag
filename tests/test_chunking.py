"""
Tests for the chunking engine.

Validates:
  - Chunks respect size limits
  - Small text produces a single chunk
  - Empty text produces no chunks
  - Overlap is applied between consecutive chunks
  - Token counting is accurate
  - Metadata is preserved
"""

from core.chunking import (
    ChunkConfig,
    chunk_text,
    count_tokens,
    recursive_chunk,
    apply_overlap,
)
from core.models import FileType


class TestTokenCounting:
    def test_counts_tokens(self):
        assert count_tokens("hello world") > 0

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_returns_int(self):
        assert isinstance(count_tokens("test"), int)

    def test_longer_text_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("This is a much longer piece of text with many words.")
        assert long > short


class TestRecursiveChunking:
    def test_small_text_single_chunk(self, chunk_config):
        config = ChunkConfig(chunk_size=500, min_chunk_size=5)
        result = recursive_chunk("Small text.", config)
        assert len(result) == 1
        assert result[0] == "Small text."

    def test_splits_on_paragraphs(self, chunk_config):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        config = ChunkConfig(chunk_size=5, min_chunk_size=1)
        result = recursive_chunk(text, config)
        assert len(result) >= 2

    def test_respects_max_size(self, long_text, chunk_config):
        result = recursive_chunk(long_text, chunk_config)
        for chunk in result:
            tokens = count_tokens(chunk)
            # Allow 20% tolerance for boundary effects
            assert tokens <= chunk_config.chunk_size * 1.2, (
                f"Chunk has {tokens} tokens, max is {chunk_config.chunk_size}"
            )

    def test_empty_input(self, chunk_config):
        assert recursive_chunk("", chunk_config) == []

    def test_whitespace_only(self, chunk_config):
        assert recursive_chunk("   \n\n  ", chunk_config) == []

    def test_discards_tiny_chunks(self):
        config = ChunkConfig(chunk_size=5, min_chunk_size=3)
        text = "One. Two. Three. Four. Five. Six. Seven."
        result = recursive_chunk(text, config)
        for chunk in result:
            assert count_tokens(chunk) >= config.min_chunk_size


class TestOverlap:
    def test_no_overlap_when_zero(self):
        chunks = ["chunk one", "chunk two"]
        config = ChunkConfig(chunk_overlap=0)
        result = apply_overlap(chunks, config)
        assert result == chunks

    def test_overlap_adds_prefix(self):
        chunks = ["First chunk of text.", "Second chunk of text."]
        config = ChunkConfig(chunk_overlap=3)
        result = apply_overlap(chunks, config)
        assert len(result) == 2
        assert result[0] == chunks[0]  # First unchanged
        assert len(result[1]) > len(chunks[1])  # Second got prefix

    def test_single_chunk_unchanged(self):
        chunks = ["Only chunk"]
        config = ChunkConfig(chunk_overlap=5)
        result = apply_overlap(chunks, config)
        assert result == chunks


class TestChunkText:
    def test_produces_document_chunks(self, long_text, sample_metadata):
        chunks = chunk_text(long_text, sample_metadata)
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.document_id
            assert chunk.metadata.source == "test_doc.md"
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks == len(chunks)
            assert chunk.token_count > 0

    def test_all_same_document_id(self, long_text, sample_metadata):
        chunks = chunk_text(long_text, sample_metadata)
        doc_ids = {c.document_id for c in chunks}
        assert len(doc_ids) == 1

    def test_custom_document_id(self, short_text, sample_metadata):
        chunks = chunk_text(short_text, sample_metadata, document_id="my-id")
        assert all(c.document_id == "my-id" for c in chunks)

    def test_chunk_indices_sequential(self, long_text, sample_metadata):
        chunks = chunk_text(long_text, sample_metadata)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_preserved(self, long_text, sample_metadata):
        chunks = chunk_text(long_text, sample_metadata)
        for chunk in chunks:
            assert chunk.metadata.file_type == FileType.MARKDOWN
            assert "test" in chunk.metadata.tags
            assert chunk.metadata.department == "engineering"
