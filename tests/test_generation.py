"""
Tests for prompt construction and evaluation metrics.
"""

import pytest

from core.evaluation import (
    evaluate_retrieval,
    evaluate_faithfulness,
    evaluate_query,
)
from generation.prompts import build_rag_prompt, build_context_block, SYSTEM_PROMPT


class TestPrompts:
    def test_rag_prompt_includes_query(self):
        prompt = build_rag_prompt("What is RAG?", "Some context here.")
        assert "What is RAG?" in prompt

    def test_rag_prompt_includes_context(self):
        prompt = build_rag_prompt("query", "Important context information.")
        assert "Important context information." in prompt

    def test_system_prompt_has_grounding(self):
        assert "ONLY" in SYSTEM_PROMPT
        assert "context" in SYSTEM_PROMPT.lower()

    def test_system_prompt_has_refusal(self):
        assert "don't have enough information" in SYSTEM_PROMPT.lower()

    def test_build_context_block(self):
        class FakeChunk:
            class metadata:
                source = "doc.txt"

            content = "Some text."

        block = build_context_block([FakeChunk()])
        assert "[Source: doc.txt]" in block
        assert "Some text." in block


class TestRetrievalEvaluation:
    def test_perfect_retrieval(self):
        scores = evaluate_retrieval(
            relevant_ids={"a", "b"},
            retrieved_ids=["a", "b", "c"],
            k=3,
        )
        assert scores.precision_at_k == pytest.approx(2 / 3, abs=0.01)
        assert scores.recall_at_k == 1.0
        assert scores.mrr == 1.0
        assert scores.hit_rate == 1.0

    def test_no_hits(self):
        scores = evaluate_retrieval(
            relevant_ids={"x", "y"},
            retrieved_ids=["a", "b", "c"],
            k=3,
        )
        assert scores.precision_at_k == 0.0
        assert scores.recall_at_k == 0.0
        assert scores.mrr == 0.0
        assert scores.hit_rate == 0.0

    def test_partial_recall(self):
        scores = evaluate_retrieval(
            relevant_ids={"a", "b", "c"},
            retrieved_ids=["a", "d", "e"],
            k=3,
        )
        assert scores.precision_at_k == pytest.approx(1 / 3, abs=0.01)
        assert scores.recall_at_k == pytest.approx(1 / 3, abs=0.01)

    def test_mrr_second_position(self):
        scores = evaluate_retrieval(
            relevant_ids={"b"},
            retrieved_ids=["a", "b", "c"],
            k=3,
        )
        assert scores.mrr == pytest.approx(0.5)

    def test_empty_relevant(self):
        scores = evaluate_retrieval(
            relevant_ids=set(),
            retrieved_ids=["a", "b"],
            k=2,
        )
        assert scores.recall_at_k == 0.0


class TestFaithfulness:
    def test_faithful_answer(self):
        context = [
            "RAG reduces hallucinations by grounding responses in retrieved documents.",
            "It uses vector databases for semantic search.",
        ]
        answer = (
            "RAG reduces hallucinations by grounding responses in retrieved documents."
        )
        score = evaluate_faithfulness(answer, context)
        assert score > 0.5

    def test_unfaithful_answer(self):
        context = ["Python is a programming language."]
        answer = "Quantum computing will revolutionize medicine by 2030."
        score = evaluate_faithfulness(answer, context)
        assert score < 0.5

    def test_empty_answer(self):
        score = evaluate_faithfulness("", ["some context"])
        assert score == 0.0

    def test_empty_context(self):
        score = evaluate_faithfulness("Some answer.", [])
        assert score <= 0.5


class TestFullEvaluation:
    def test_full_eval(self):
        result = evaluate_query(
            query="What is RAG?",
            answer="RAG enhances LLMs with retrieved context.",
            context_chunks=[
                "RAG enhances Large Language Models with retrieved context from documents."
            ],
            relevant_ids={"doc1"},
            retrieved_ids=["doc1", "doc2"],
            k=2,
        )
        assert result.retrieval.hit_rate == 1.0
        assert result.answer.faithfulness > 0
