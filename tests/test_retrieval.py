"""
Tests for search components.

Validates:
  - BM25 index builds correctly
  - BM25 scoring is sane
  - Metadata filter construction
  - Model validation
"""

import pytest

from core.models import MetadataFilter, SearchMode
from retrieval.bm25_index import BM25Index, tokenize


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_removes_stopwords(self):
        tokens = tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens

    def test_removes_short_tokens(self):
        tokens = tokenize("I a am ok fine")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "fine" in tokens

    def test_handles_punctuation(self):
        tokens = tokenize("hello, world! how are you?")
        assert "hello" in tokens
        assert "world" in tokens


class TestBM25Index:
    @pytest.fixture
    def sample_data(self):
        return [
            (
                "id1",
                "Python is a programming language used for AI",
                {"source": "a.txt"},
            ),
            ("id2", "Machine learning models can classify images", {"source": "b.txt"}),
            ("id3", "JavaScript runs in web browsers", {"source": "c.txt"}),
            (
                "id4",
                "Python libraries like NumPy and pandas are popular for data science",
                {"source": "d.txt"},
            ),
        ]

    def test_build(self, sample_data):
        index = BM25Index()
        assert not index.is_built
        index.build(sample_data)
        assert index.is_built
        assert index.doc_count == 4

    def test_search_returns_results(self, sample_data):
        index = BM25Index()
        index.build(sample_data)
        results = index.search("Python programming", top_k=2)
        assert len(results) <= 2
        assert all(r["score"] > 0 for r in results)

    def test_search_relevance(self, sample_data):
        index = BM25Index()
        index.build(sample_data)

        results = index.search("Python", top_k=4)
        # Python docs should rank higher
        top_ids = [r["chunk_id"] for r in results[:2]]
        assert "id1" in top_ids or "id4" in top_ids

    def test_search_empty_index(self):
        index = BM25Index()
        results = index.search("anything")
        assert results == []

    def test_search_no_match(self, sample_data):
        index = BM25Index()
        index.build(sample_data)
        results = index.search("quantum physics")
        # May return results with score 0 (filtered out)
        for r in results:
            assert r["score"] >= 0

    def test_scores_normalized(self, sample_data):
        index = BM25Index()
        index.build(sample_data)
        results = index.search("Python machine learning", top_k=4)
        if results:
            assert results[0]["score"] == 1.0  # Top score normalized to 1
            for r in results:
                assert 0 <= r["score"] <= 1.0

    def test_rebuild(self, sample_data):
        index = BM25Index()
        index.build(sample_data[:2])
        assert index.doc_count == 2

        index.build(sample_data)
        assert index.doc_count == 4


class TestMetadataFilter:
    def test_valid_operators(self):
        for op in ["eq", "neq", "gt", "lt", "gte", "lte", "in", "contains"]:
            f = MetadataFilter(field="x", value="y", operator=op)
            assert f.operator == op

    def test_invalid_operator(self):
        with pytest.raises(ValueError, match="operator"):
            MetadataFilter(field="x", value="y", operator="LIKE")

    def test_default_operator(self):
        f = MetadataFilter(field="department", value="engineering")
        assert f.operator == "eq"


class TestSearchMode:
    def test_values(self):
        assert SearchMode.SEMANTIC.value == "semantic"
        assert SearchMode.KEYWORD.value == "keyword"
        assert SearchMode.HYBRID.value == "hybrid"
