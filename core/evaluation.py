"""
RAG evaluation metrics — measure retrieval + answer quality.
Use from tests or the /evaluate endpoint.

Retrieval metrics:
  - Precision@K, Recall@K, MRR, NDCG, Hit Rate

Answer metrics:
  - Faithfulness (heuristic — does the answer stick to context?)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


@dataclass
class RetrievalScores:
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0


@dataclass
class AnswerScores:
    faithfulness: float = 0.0
    answer_coverage: float = 0.0


@dataclass
class EvalResult:
    query: str
    retrieval: RetrievalScores
    answer: AnswerScores
    details: dict = field(default_factory=dict)


def evaluate_retrieval(
    relevant_ids: set[str],
    retrieved_ids: list[str],
    k: int = 5,
) -> RetrievalScores:
    """
    Compute standard IR metrics.

    Args:
        relevant_ids:  ground-truth set of relevant document/chunk IDs
        retrieved_ids: ordered list of IDs returned by the system
        k:             cutoff
    """
    top_k = retrieved_ids[:k]

    # Precision@K
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    precision = hits / k if k > 0 else 0.0

    # Recall@K
    recall = hits / len(relevant_ids) if relevant_ids else 0.0

    # MRR — reciprocal rank of first relevant result
    mrr = 0.0
    for i, rid in enumerate(top_k):
        if rid in relevant_ids:
            mrr = 1.0 / (i + 1)
            break

    # Hit rate — did we find at least one?
    hit_rate = 1.0 if hits > 0 else 0.0

    # NDCG@K
    dcg = sum(
        (1.0 if rid in relevant_ids else 0.0) / math.log2(i + 2)
        for i, rid in enumerate(top_k)
    )
    # ideal DCG: all relevant docs first
    ideal_rels = sorted(
        [1.0 if rid in relevant_ids else 0.0 for rid in top_k],
        reverse=True,
    )
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return RetrievalScores(
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
        ndcg_at_k=round(ndcg, 4),
        hit_rate=round(hit_rate, 4),
    )


def evaluate_faithfulness(
    answer: str,
    context_chunks: list[str],
) -> float:
    """
    Heuristic faithfulness: what fraction of answer sentences
    have meaningful overlap with the provided context?

    Production systems use LLM-as-judge; this is a free proxy.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0

    context_words = set()
    for chunk in context_chunks:
        context_words.update(chunk.lower().split())

    # Remove stop words from comparison
    stop = {
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
    }

    faithful = 0
    for sentence in sentences:
        words = set(sentence.lower().split()) - stop
        if not words:
            faithful += 1
            continue
        overlap = len(words & context_words) / len(words)
        if overlap >= 0.25:
            faithful += 1

    return round(faithful / len(sentences), 4)


def evaluate_query(
    query: str,
    answer: str,
    context_chunks: list[str],
    relevant_ids: set[str] | None = None,
    retrieved_ids: list[str] | None = None,
    k: int = 5,
) -> EvalResult:
    """Full evaluation combining retrieval and answer metrics."""

    retrieval = RetrievalScores()
    if relevant_ids and retrieved_ids:
        retrieval = evaluate_retrieval(relevant_ids, retrieved_ids, k)

    faithfulness = evaluate_faithfulness(answer, context_chunks)

    return EvalResult(
        query=query,
        retrieval=retrieval,
        answer=AnswerScores(faithfulness=faithfulness),
    )
