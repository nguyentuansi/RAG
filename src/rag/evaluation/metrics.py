"""RAG evaluation metrics: faithfulness, relevancy, precision, recall, NDCG."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.rag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    ground_truth_contexts: list[str] = field(default_factory=list)


@dataclass
class MetricResult:
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    dcg = sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(ideal)
    )
    return dcg / idcg if idcg > 0 else 0.0


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality across multiple dimensions.

    Faithfulness, answer relevancy, context precision/recall use embedding
    similarity as a proxy. For production, replace with an LLM-judge call.
    """

    def __init__(self, embedding_provider=None) -> None:
        self.embedding_provider = embedding_provider

    async def faithfulness(self, sample: EvaluationSample) -> MetricResult:
        """
        Measure whether the answer is grounded in the retrieved contexts.
        Score = fraction of answer sentences that can be attributed to a context.
        """
        if not self.embedding_provider:
            return MetricResult(name="faithfulness", score=0.0, details={"error": "no_embedding_provider"})

        answer_sentences = [s.strip() for s in sample.answer.split(".") if len(s.strip()) > 10]
        if not answer_sentences:
            return MetricResult(name="faithfulness", score=1.0)

        context_text = " ".join(sample.contexts)
        answer_embs = await self.embedding_provider.embed_batch(answer_sentences)
        context_emb = await self.embedding_provider.embed_text(context_text)

        attributed = sum(
            1 for emb in answer_embs if _cosine_sim(emb, context_emb) > 0.65
        )
        score = attributed / len(answer_sentences)
        return MetricResult(
            name="faithfulness",
            score=round(score, 4),
            details={"attributed": attributed, "total_sentences": len(answer_sentences)},
        )

    async def answer_relevancy(self, sample: EvaluationSample) -> MetricResult:
        """Cosine similarity between question embedding and answer embedding."""
        if not self.embedding_provider:
            return MetricResult(name="answer_relevancy", score=0.0)

        q_emb = await self.embedding_provider.embed_text(sample.question)
        a_emb = await self.embedding_provider.embed_text(sample.answer)
        score = _cosine_sim(q_emb, a_emb)
        return MetricResult(name="answer_relevancy", score=round(score, 4))

    async def context_precision(self, sample: EvaluationSample) -> MetricResult:
        """
        Fraction of retrieved contexts that are relevant to the question.
        Relevance threshold: cosine_sim(context, question) > 0.5.
        """
        if not sample.contexts or not self.embedding_provider:
            return MetricResult(name="context_precision", score=0.0)

        q_emb = await self.embedding_provider.embed_text(sample.question)
        ctx_embs = await self.embedding_provider.embed_batch(sample.contexts)
        relevant = sum(1 for emb in ctx_embs if _cosine_sim(emb, q_emb) > 0.5)
        score = relevant / len(sample.contexts)
        return MetricResult(
            name="context_precision",
            score=round(score, 4),
            details={"relevant": relevant, "total": len(sample.contexts)},
        )

    async def context_recall(self, sample: EvaluationSample) -> MetricResult:
        """
        Fraction of ground-truth contexts covered by retrieved contexts.
        Requires sample.ground_truth_contexts to be populated.
        """
        if not sample.ground_truth_contexts or not self.embedding_provider:
            return MetricResult(name="context_recall", score=0.0, details={"skipped": "no_ground_truth"})

        gt_embs = await self.embedding_provider.embed_batch(sample.ground_truth_contexts)
        ctx_embs = await self.embedding_provider.embed_batch(sample.contexts)

        covered = 0
        for gt_emb in gt_embs:
            if any(_cosine_sim(gt_emb, c_emb) > 0.65 for c_emb in ctx_embs):
                covered += 1

        score = covered / len(gt_embs)
        return MetricResult(
            name="context_recall",
            score=round(score, 4),
            details={"covered": covered, "total_ground_truth": len(gt_embs)},
        )

    async def answer_similarity(self, sample: EvaluationSample) -> MetricResult:
        """Cosine similarity between generated answer and ground truth."""
        if not sample.ground_truth or not self.embedding_provider:
            return MetricResult(name="answer_similarity", score=0.0, details={"skipped": "no_ground_truth"})

        a_emb = await self.embedding_provider.embed_text(sample.answer)
        gt_emb = await self.embedding_provider.embed_text(sample.ground_truth)
        score = _cosine_sim(a_emb, gt_emb)
        return MetricResult(name="answer_similarity", score=round(score, 4))

    def ndcg_at_k(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> MetricResult:
        relevances = [1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids]
        score = _ndcg_at_k(relevances, k)
        return MetricResult(
            name=f"ndcg@{k}",
            score=round(score, 4),
            details={"k": k, "relevant_found": sum(relevances[:k])},
        )

    async def evaluate_all(self, sample: EvaluationSample) -> dict[str, MetricResult]:
        """Run all applicable metrics for a single sample."""
        results = {}
        for metric_fn in [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
            self.answer_similarity,
        ]:
            result = await metric_fn(sample)
            results[result.name] = result
        return results
