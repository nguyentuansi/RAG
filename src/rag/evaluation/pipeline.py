"""RAG evaluation pipeline: run metrics, compare configs, generate reports."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.rag.core.logging import get_logger
from src.rag.evaluation.metrics import EvaluationSample, MetricResult, RAGMetricsCalculator

logger = get_logger(__name__)


class EvaluationResult:
    def __init__(
        self,
        name: str,
        samples: list[EvaluationSample],
        metric_results: list[list[MetricResult]],
    ) -> None:
        self.name = name
        self.samples = samples
        self.metric_results = metric_results
        self.evaluated_at = datetime.now(timezone.utc).isoformat()

    def aggregate(self) -> dict[str, float]:
        """Return mean score per metric across all samples."""
        if not self.metric_results:
            return {}
        metric_names = [r.name for r in self.metric_results[0]]
        aggregated: dict[str, float] = {}
        for metric_name in metric_names:
            scores = [
                next((r.score for r in sample_results if r.name == metric_name), 0.0)
                for sample_results in self.metric_results
            ]
            aggregated[metric_name] = round(sum(scores) / len(scores), 4) if scores else 0.0
        return aggregated

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evaluated_at": self.evaluated_at,
            "num_samples": len(self.samples),
            "aggregate_scores": self.aggregate(),
            "per_sample": [
                {
                    "question": s.question[:120],
                    "scores": {r.name: r.score for r in results},
                }
                for s, results in zip(self.samples, self.metric_results)
            ],
        }


class EvaluationPipeline:
    """Orchestrates evaluation runs over a dataset of QA pairs."""

    def __init__(self, embedding_provider=None) -> None:
        self._calculator = RAGMetricsCalculator(embedding_provider)

    async def run_evaluation(
        self,
        samples: list[EvaluationSample],
        *,
        name: str = "evaluation",
        concurrency: int = 4,
    ) -> EvaluationResult:
        """Run all metrics on each sample with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_one(sample: EvaluationSample) -> list[MetricResult]:
            async with semaphore:
                return await self._calculator.evaluate_all(sample)

        logger.info("evaluation_started", name=name, samples=len(samples))
        all_results = await asyncio.gather(*[evaluate_one(s) for s in samples])
        logger.info("evaluation_complete", name=name)

        result = EvaluationResult(name=name, samples=samples, metric_results=list(all_results))
        logger.info("evaluation_scores", **result.aggregate())
        return result

    async def compare_configurations(
        self,
        samples: list[EvaluationSample],
        configurations: dict[str, Any],
    ) -> dict[str, EvaluationResult]:
        """
        Compare multiple RAG configurations side-by-side.

        Each key in configurations is a name; the value is a dict with a
        'pipeline' callable that takes a question and returns (answer, contexts).
        """
        results: dict[str, EvaluationResult] = {}
        for config_name, config in configurations.items():
            pipeline_fn = config.get("pipeline")
            if not pipeline_fn:
                continue
            augmented_samples = []
            for sample in samples:
                answer, contexts = await pipeline_fn(sample.question)
                augmented_samples.append(
                    EvaluationSample(
                        question=sample.question,
                        answer=answer,
                        contexts=contexts,
                        ground_truth=sample.ground_truth,
                    )
                )
            results[config_name] = await self.run_evaluation(augmented_samples, name=config_name)
        return results

    def save_results(self, result: EvaluationResult, output_path: Path) -> None:
        """Write evaluation results to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("evaluation_saved", path=str(output_path))

    def generate_report(self, result: EvaluationResult) -> str:
        """Return a markdown-formatted evaluation report."""
        scores = result.aggregate()
        lines = [
            f"# RAG Evaluation Report: {result.name}",
            f"\nEvaluated at: {result.evaluated_at}",
            f"Samples: {len(result.samples)}",
            "\n## Aggregate Scores\n",
        ]
        for metric, score in sorted(scores.items()):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"- **{metric}**: {score:.3f} `{bar}`")

        lines.append("\n## Per-Sample Results\n")
        for i, (sample, metric_results) in enumerate(
            zip(result.samples, result.metric_results), 1
        ):
            lines.append(f"### Sample {i}: {sample.question[:80]}...")
            for mr in metric_results:
                lines.append(f"  - {mr.name}: {mr.score:.3f}")

        return "\n".join(lines)
