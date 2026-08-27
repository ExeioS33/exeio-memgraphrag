"""Run a dataset through an answering function and aggregate the metrics.

The paper (arXiv:2606.00610) publishes single numbers with no run count, no
standard deviation and no confidence interval, while claiming gains as small as
+2.10 absolute points. A campaign that cannot say whether 58.9 instead of 59.25
is a regression or noise cannot defend either verdict, so this runner is built
around repetition: :func:`run_campaign` executes N identical runs and reports
mean and standard deviation per metric, and the non-regression threshold in
``golden.py`` is expressed in those standard deviations rather than in a
hand-picked constant.
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .benchmark import LatencyStats
from .datasets import EvaluationExample
from .judge import JudgeVerdict, LLMAccJudge
from .qa_metrics import exact_match, f1_score, str_acc
from .retrieval_metrics import DEFAULT_K_LIST, context_relevance, evidence_recall, recall_at_k

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """What the system under test returned for one question."""

    example_id: str
    question: str
    answer: str = ""
    retrieved_docs: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "answer": self.answer,
            "retrieved_docs": list(self.retrieved_docs),
            "latency_s": round(self.latency_s, 6),
            "error": self.error,
        }


#: A callable that answers one example. ``scripts/evaluate.py`` binds this to a
#: live ``MemGraphRAG`` instance; tests bind it to a dictionary.
AnswerFn = Callable[[EvaluationExample], Awaitable[Prediction]]


@dataclass
class RunResult:
    """One complete pass over the evaluation set."""

    metrics: dict[str, float] = field(default_factory=dict)
    predictions: list[Prediction] = field(default_factory=list)
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    latency: LatencyStats | None = None
    seconds: float = 0.0
    failures: int = 0

    def to_dict(self, include_predictions: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metrics": {key: round(value, 6) for key, value in self.metrics.items()},
            "seconds": round(self.seconds, 3),
            "failures": self.failures,
            "latency": self.latency.to_dict() if self.latency else None,
        }
        if include_predictions:
            payload["predictions"] = [pred.to_dict() for pred in self.predictions]
            payload["verdicts"] = [verdict.to_dict() for verdict in self.verdicts]
        return payload


@dataclass(frozen=True)
class MetricStat:
    """Mean and spread of one metric across the runs of a campaign."""

    mean: float
    stdev: float
    minimum: float
    maximum: float
    runs: int
    values: list[float] = field(default_factory=list)

    @property
    def ci95(self) -> float:
        """Half-width of the 95% confidence interval of the mean (normal approximation).

        Reported alongside the standard deviation because they answer different
        questions: the stdev says how noisy a single run is, the interval says how
        precisely this campaign located the mean. Meaningless for a single run,
        where it is 0.0 by construction — which is the point: one run measures no
        variance at all, it only looks like it does.
        """
        if self.runs < 2:
            return 0.0
        return 1.96 * self.stdev / math.sqrt(self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "stdev": round(self.stdev, 6),
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "runs": self.runs,
            "ci95": round(self.ci95, 6),
            "values": [round(value, 6) for value in self.values],
        }


def aggregate_runs(runs: Sequence[Mapping[str, float]]) -> dict[str, MetricStat]:
    """Mean / sample standard deviation per metric across runs.

    Uses the *sample* standard deviation (n-1). With a single run the spread is
    reported as 0.0, and callers must treat that as "unmeasured", not "stable".
    A metric missing from some runs is aggregated over the runs that have it,
    rather than being silently zero-filled.
    """
    keys: list[str] = []
    for run in runs:
        for key in run:
            if key not in keys:
                keys.append(key)
    stats: dict[str, MetricStat] = {}
    for key in keys:
        values = [float(run[key]) for run in runs if key in run]
        if not values:
            continue
        stats[key] = MetricStat(
            mean=statistics.fmean(values),
            stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
            minimum=min(values),
            maximum=max(values),
            runs=len(values),
            values=values,
        )
    return stats


@dataclass
class CampaignReport:
    """N runs of the same evaluation, plus the aggregate that makes them usable."""

    dataset: str
    examples: int
    runs: list[RunResult] = field(default_factory=list)
    metrics: dict[str, MetricStat] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_predictions: bool = True) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "examples": self.examples,
            "runs": len(self.runs),
            "metrics": {key: stat.to_dict() for key, stat in self.metrics.items()},
            "metadata": dict(self.metadata),
            "run_details": [run.to_dict(include_predictions) for run in self.runs],
        }


async def _answer_all(
    examples: Sequence[EvaluationExample],
    answer_fn: AnswerFn,
    concurrency: int,
) -> list[Prediction]:
    """Answer every example, preserving order and never letting one failure abort the run."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(index: int) -> tuple[int, Prediction]:
        example = examples[index]
        async with semaphore:
            started = time.perf_counter()
            try:
                prediction = await answer_fn(example)
            except Exception as exc:  # noqa: BLE001 - a failed query scores as wrong
                logger.warning("query failed for %s: %s", example.id, exc)
                return index, Prediction(
                    example_id=example.id,
                    question=example.question,
                    latency_s=time.perf_counter() - started,
                    error=str(exc)[:300],
                )
            if prediction.latency_s <= 0:
                prediction.latency_s = time.perf_counter() - started
            return index, prediction

    pairs = await asyncio.gather(*(_one(index) for index in range(len(examples))))
    return [prediction for _, prediction in sorted(pairs, key=lambda pair: pair[0])]


async def run_once(
    examples: Sequence[EvaluationExample],
    answer_fn: AnswerFn,
    judge: LLMAccJudge | None = None,
    k_list: Sequence[int] = DEFAULT_K_LIST,
    concurrency: int = 1,
) -> RunResult:
    """Evaluate one pass: answer every question, then score answers and retrieval.

    All four paper metrics plus the EM / F1 / Recall@k floor are computed in one
    place so that a report can never contain Str-Acc from one code path and
    Evidence Recall from another.
    """
    started = time.perf_counter()
    predictions = await _answer_all(examples, answer_fn, concurrency)

    gold_answers = [example.gold_answers for example in examples]
    gold_docs = [example.gold_docs for example in examples]
    answers = [prediction.answer for prediction in predictions]
    retrieved = [prediction.retrieved_docs for prediction in predictions]

    metrics: dict[str, float] = {}
    for result in (
        str_acc(gold_answers, answers),
        exact_match(gold_answers, answers),
        f1_score(gold_answers, answers),
        recall_at_k(gold_docs, retrieved, k_list),
        evidence_recall(gold_docs, retrieved),
        context_relevance(gold_docs, retrieved),
    ):
        metrics.update(result.pooled)

    verdicts: list[JudgeVerdict] = []
    if judge is not None:
        judged, verdicts = await judge.ajudge(
            [example.question for example in examples], gold_answers, answers
        )
        metrics.update(judged.pooled)

    return RunResult(
        metrics=metrics,
        predictions=predictions,
        verdicts=verdicts,
        latency=LatencyStats.from_samples([pred.latency_s for pred in predictions]),
        seconds=time.perf_counter() - started,
        failures=sum(1 for pred in predictions if pred.error),
    )


async def run_campaign(
    examples: Sequence[EvaluationExample],
    answer_fn: AnswerFn,
    runs: int = 1,
    judge: LLMAccJudge | None = None,
    k_list: Sequence[int] = DEFAULT_K_LIST,
    concurrency: int = 1,
    dataset: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CampaignReport:
    """Execute ``runs`` identical passes and aggregate them.

    The runs are executed one after another, not concurrently: overlapping them
    would have them contend for the same LLM endpoint and the latency figures
    would measure the campaign's own load.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    results = [
        await run_once(examples, answer_fn, judge=judge, k_list=k_list, concurrency=concurrency)
        for _ in range(runs)
    ]
    report_metadata = dict(metadata or {})
    if judge is not None:
        report_metadata.setdefault("judge_prompt_version", judge.prompt_version)
        report_metadata.setdefault("judge_model", judge.model or "")
        report_metadata.setdefault("judge_calls", judge.calls)
    return CampaignReport(
        dataset=dataset or (examples[0].dataset if examples else ""),
        examples=len(examples),
        runs=results,
        metrics=aggregate_runs([result.metrics for result in results]),
        metadata=report_metadata,
    )
