"""RAGAS-style evaluation metrics for RAG pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.rag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationSample:
    """A single evaluation example with ground truth."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.name}: {self.score:.3f}"


class RAGMetricsCalculator:
    """
    RAGAS-inspired metrics computed without a live LLM judge.

    For production evaluation, the faithfulness and answer_relevancy
    metrics should use an LLM judge — here they use embedding cosine
    similarity as an approximation.
    """

    def __init__(self, embedding_provider=None) -> None:
        self._ep = embedding_provider

    # ------------------------------------------------------------------ #
    #  Faithfulness                                                        #
    # ------------------------------------------------------------------ #

    async def faithfulness(self, sample: EvaluationSample) -> MetricResult:
        """
        Measures whether the answer is grounded in the provided contexts.

        Approximation: cosine similarity between answer embedding and
        mean context embedding.
        """
        if not self._ep:
            return MetricResult("faithfulness", 0.0, {"note": "No embedding provider"})

        all_texts = [sample.answer] + sample.contexts
        vectors = await self._ep.embed_batch(all_texts)

        answer_vec = np.array(vectors[0])
        context_vecs = np.array(vectors[1:])
        mean_context = context_vecs.mean(axis=0)

        score = float(_cosine(answer_vec, mean_context))
        return MetricResult("faithfulness", score, {"method": "embedding_cosine"})

    # ------------------------------------------------------------------ #
    #  Answer Relevancy                                                    #
    # ------------------------------------------------------------------ #

    async def answer_relevancy(self, sample: EvaluationSample) -> MetricResult:
        """
        Measures whether the answer is relevant to the question.

        Approximation: cosine similarity between answer and question embeddings.
        """
        if not self._ep:
            return MetricResult("answer_relevancy", 0.0, {"note": "No embedding provider"})

        vecs = await self._ep.embed_batch([sample.question, sample.answer])
        score = float(_cosine(np.array(vecs[0]), np.array(vecs[1])))
        return MetricResult("answer_relevancy", score, {"method": "embedding_cosine"})

    # ------------------------------------------------------------------ #
    #  Context Precision                                                   #
    # ------------------------------------------------------------------ #

    async def context_precision(self, sample: EvaluationSample) -> MetricResult:
        """
        Proportion of retrieved contexts that are relevant to the question.

        Uses embedding similarity with threshold 0.5 as relevance signal.
        """
        if not self._ep or not sample.contexts:
            return MetricResult("context_precision", 0.0)

        all_texts = [sample.question] + sample.contexts
        vecs = await self._ep.embed_batch(all_texts)
        q_vec = np.array(vecs[0])

        relevant = sum(
            1 for ctx_vec in vecs[1:]
            if _cosine(q_vec, np.array(ctx_vec)) >= 0.5
        )
        score = relevant / len(sample.contexts)
        return MetricResult("context_precision", score, {"relevant_contexts": relevant, "total": len(sample.contexts)})

    # ------------------------------------------------------------------ #
    #  Context Recall                                                      #
    # ------------------------------------------------------------------ #

    async def context_recall(self, sample: EvaluationSample) -> MetricResult:
        """
        How much of the ground truth is covered by the retrieved contexts.

        Requires ground_truth.  Approximation: similarity between ground_truth
        and each context, taking the max.
        """
        if not self._ep or not sample.ground_truth:
            return MetricResult("context_recall", 0.0, {"note": "No ground_truth provided"})

        all_texts = [sample.ground_truth] + sample.contexts
        vecs = await self._ep.embed_batch(all_texts)
        gt_vec = np.array(vecs[0])

        max_sim = max(
            _cosine(gt_vec, np.array(ctx_vec))
            for ctx_vec in vecs[1:]
        ) if vecs[1:] else 0.0
        return MetricResult("context_recall", float(max_sim))

    # ------------------------------------------------------------------ #
    #  Answer Similarity                                                   #
    # ------------------------------------------------------------------ #

    async def answer_similarity(self, sample: EvaluationSample) -> MetricResult:
        """Cosine similarity between generated answer and ground truth."""
        if not self._ep or not sample.ground_truth:
            return MetricResult("answer_similarity", 0.0, {"note": "No ground_truth"})

        vecs = await self._ep.embed_batch([sample.answer, sample.ground_truth])
        score = float(_cosine(np.array(vecs[0]), np.array(vecs[1])))
        return MetricResult("answer_similarity", score)

    # ------------------------------------------------------------------ #
    #  NDCG@k                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def ndcg_at_k(relevances: list[float], k: int) -> MetricResult:
        """
        Normalized Discounted Cumulative Gain at k.

        Args:
            relevances: List of relevance scores for ranked results (higher = better)
            k: Cutoff position
        """
        truncated = relevances[:k]
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(truncated))
        ideal = sorted(relevances, reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
        score = dcg / idcg if idcg > 0 else 0.0
        return MetricResult("ndcg@k", score, {"k": k, "dcg": dcg, "idcg": idcg})

    async def evaluate_all(self, sample: EvaluationSample) -> list[MetricResult]:
        """Run all applicable metrics on a sample."""
        results = await asyncio.gather(
            self.faithfulness(sample),
            self.answer_relevancy(sample),
            self.context_precision(sample),
            self.context_recall(sample),
            self.answer_similarity(sample),
        )
        return list(results)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


import asyncio  # noqa: E402
