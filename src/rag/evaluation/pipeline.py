"""RAG evaluation pipeline: dataset loading, bulk evaluation, report generation."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.rag.core.logging import get_logger
from src.rag.evaluation.metrics import EvaluationSample, MetricResult, RAGEvaluator

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    run_id: str
    timestamp: str
    sample_count: int
    metric_averages: dict[str, float]
    per_sample_results: list[dict[str, Any]]
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Evaluation Report [{self.run_id}]", f"Samples: {self.sample_count}"]
        for metric, score in sorted(self.metric_averages.items()):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {metric:<25} {bar} {score:.4f}")
        return "\n".join(lines)


class EvaluationPipeline:
    def __init__(self, evaluator: RAGEvaluator) -> None:
        self.evaluator = evaluator

    def load_dataset(self, path: str | Path) -> list[EvaluationSample]:
        """Load evaluation dataset from a JSONL file."""
        samples: list[EvaluationSample] = []
        path = Path(path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(EvaluationSample(
                    question=obj["question"],
                    answer=obj["answer"],
                    contexts=obj.get("contexts", []),
                    ground_truth=obj.get("ground_truth"),
                    ground_truth_contexts=obj.get("ground_truth_contexts", []),
                ))
        logger.info("evaluation_dataset_loaded", path=str(path), samples=len(samples))
        return samples

    async def run_evaluation(
        self,
        samples: list[EvaluationSample],
        *,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        if not run_id:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        per_sample: list[dict[str, Any]] = []
        metric_scores: dict[str, list[float]] = {}

        for i, sample in enumerate(samples):
            logger.debug("evaluating_sample", index=i, question=sample.question[:60])
            results = await self.evaluator.evaluate_all(sample)
            sample_dict: dict[str, Any] = {
                "index": i,
                "question": sample.question,
                "metrics": {k: asdict(v) for k, v in results.items()},
            }
            per_sample.append(sample_dict)
            for name, result in results.items():
                metric_scores.setdefault(name, []).append(result.score)

        averages = {
            name: round(statistics.mean(scores), 4)
            for name, scores in metric_scores.items()
        }

        report = EvaluationReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_count=len(samples),
            metric_averages=averages,
            per_sample_results=per_sample,
            config=config or {},
        )
        logger.info("evaluation_complete", run_id=run_id, samples=len(samples), metrics=averages)
        return report

    def generate_report(self, report: EvaluationReport) -> dict[str, Any]:
        return {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "sample_count": report.sample_count,
            "metric_averages": report.metric_averages,
            "config": report.config,
        }

    def save_results(self, report: EvaluationReport, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for sample_result in report.per_sample_results:
                f.write(json.dumps(sample_result) + "\n")
        logger.info("evaluation_results_saved", path=str(path))

    async def compare_configurations(
        self,
        samples: list[EvaluationSample],
        configs: list[dict[str, Any]],
    ) -> list[EvaluationReport]:
        """Run evaluation for multiple pipeline configurations and compare."""
        reports = []
        for cfg in configs:
            run_id = cfg.get("name", f"config-{len(reports)}")
            report = await self.run_evaluation(samples, run_id=run_id, config=cfg)
            reports.append(report)
        return reports
