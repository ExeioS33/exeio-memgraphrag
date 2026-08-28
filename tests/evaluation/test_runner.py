"""Campaign runner: metric wiring, failure handling, and the variance figures.

Item 44 of the audit: the paper reports single numbers with no run count and no
spread while claiming +2.10 point gains. These tests pin that the harness cannot
report a mean without also reporting how noisy it was.
"""

from __future__ import annotations

import pytest

from memgraphrag.evaluation.datasets import EvaluationExample
from memgraphrag.evaluation.judge import LLMAccJudge
from memgraphrag.evaluation.runner import (
    Prediction,
    aggregate_runs,
    run_campaign,
    run_once,
)

pytestmark = pytest.mark.offline

EXAMPLES = [
    EvaluationExample(
        id="q1",
        question="Where was he born?",
        gold_answers=["Paris"],
        gold_docs=["Doc A"],
        dataset="fixture",
    ),
    EvaluationExample(
        id="q2",
        question="When did she die?",
        gold_answers=["875"],
        gold_docs=["Doc B"],
        dataset="fixture",
    ),
]


def _answers(mapping: dict[str, tuple[str, list[str]]]):
    async def answer_fn(example: EvaluationExample) -> Prediction:
        answer, docs = mapping[example.id]
        return Prediction(
            example_id=example.id,
            question=example.question,
            answer=answer,
            retrieved_docs=docs,
        )

    return answer_fn


async def test_run_once_reports_every_frozen_metric() -> None:
    """One pass emits the four paper metrics and the EM / F1 / Recall floor together."""
    result = await run_once(
        EXAMPLES,
        _answers({"q1": ("Paris", ["Doc A"]), "q2": ("874", ["Doc C"])}),
    )
    for metric in ("StrAcc", "ExactMatch", "F1", "Recall@5", "EvidenceRecall", "ContextRelevance"):
        assert metric in result.metrics
    assert result.metrics["StrAcc"] == 0.5
    assert result.metrics["EvidenceRecall"] == 0.5


async def test_a_failing_query_scores_as_wrong_instead_of_aborting_the_run() -> None:
    async def answer_fn(example: EvaluationExample) -> Prediction:
        if example.id == "q2":
            raise RuntimeError("engine exploded")
        return Prediction(example_id=example.id, question=example.question, answer="Paris")

    result = await run_once(EXAMPLES, answer_fn)
    assert result.failures == 1
    assert result.metrics["StrAcc"] == 0.5
    assert "engine exploded" in result.predictions[1].error


async def test_predictions_keep_dataset_order_under_concurrency() -> None:
    import asyncio

    async def answer_fn(example: EvaluationExample) -> Prediction:
        if example.id == "q1":
            await asyncio.sleep(0.02)
        return Prediction(example_id=example.id, question=example.question, answer="x")

    result = await run_once(EXAMPLES, answer_fn, concurrency=4)
    assert [pred.example_id for pred in result.predictions] == ["q1", "q2"]


async def test_run_once_times_every_query_for_the_latency_summary() -> None:
    result = await run_once(EXAMPLES, _answers({"q1": ("Paris", []), "q2": ("875", [])}))
    assert result.latency is not None
    assert result.latency.count == 2
    assert result.latency.p95 >= result.latency.p50


async def test_campaign_reports_a_standard_deviation_across_runs() -> None:
    """A metric that moves between runs must show up as spread, not as a new truth."""
    calls = {"n": 0}

    async def flaky(example: EvaluationExample) -> Prediction:
        calls["n"] += 1
        # q1 is answered correctly only on the first run.
        answer = "Paris" if (example.id == "q1" and calls["n"] <= 2) else "Lyon"
        return Prediction(example_id=example.id, question=example.question, answer=answer)

    report = await run_campaign(EXAMPLES, flaky, runs=2, dataset="fixture")
    stat = report.metrics["StrAcc"]
    assert stat.runs == 2
    assert stat.values == [0.5, 0.0]
    assert stat.stdev > 0
    assert stat.mean == 0.25


async def test_single_run_reports_zero_spread_and_no_confidence_interval() -> None:
    report = await run_campaign(
        EXAMPLES, _answers({"q1": ("Paris", []), "q2": ("875", [])}), runs=1
    )
    stat = report.metrics["StrAcc"]
    assert stat.runs == 1
    assert stat.stdev == 0.0
    assert stat.ci95 == 0.0


async def test_campaign_records_the_judge_prompt_version_it_used() -> None:
    """A judged score is only comparable to another produced by the same prompt."""

    async def judge_reply(prompt: str, **kwargs: object) -> str:
        return '{"verdict": "correct", "reason": "ok"}'

    judge = LLMAccJudge(complete=judge_reply, model="judge-model")
    report = await run_campaign(
        EXAMPLES, _answers({"q1": ("Paris", []), "q2": ("875", [])}), runs=1, judge=judge
    )
    assert report.metrics["LLMAcc"].mean == 1.0
    assert report.metadata["judge_prompt_version"] == judge.prompt_version
    assert report.metadata["judge_model"] == "judge-model"
    assert report.metadata["judge_calls"] == 2


async def test_zero_runs_is_rejected() -> None:
    with pytest.raises(ValueError):
        await run_campaign(EXAMPLES, _answers({}), runs=0)


def test_aggregate_uses_the_sample_standard_deviation() -> None:
    stats = aggregate_runs([{"m": 0.0}, {"m": 1.0}])
    assert stats["m"].mean == 0.5
    # Sample stdev of {0, 1} is 0.7071…, not the population 0.5.
    assert stats["m"].stdev == pytest.approx(0.70710678, rel=1e-6)
    assert stats["m"].ci95 == pytest.approx(1.96 * 0.70710678 / (2**0.5), rel=1e-6)


def test_aggregate_ignores_runs_that_lack_a_metric_instead_of_zero_filling() -> None:
    """Zero-filling a missing judged score would invent a regression out of nothing."""
    stats = aggregate_runs([{"StrAcc": 0.8, "LLMAcc": 0.9}, {"StrAcc": 0.8}])
    assert stats["LLMAcc"].runs == 1
    assert stats["LLMAcc"].mean == 0.9


def test_report_serialises_without_predictions_when_asked() -> None:
    from memgraphrag.evaluation.runner import CampaignReport, RunResult

    report = CampaignReport(
        dataset="fixture",
        examples=1,
        runs=[RunResult(metrics={"StrAcc": 1.0}, predictions=[Prediction("q1", "?")])],
        metrics=aggregate_runs([{"StrAcc": 1.0}]),
    )
    payload = report.to_dict(include_predictions=False)
    assert "predictions" not in payload["run_details"][0]
    assert payload["metrics"]["StrAcc"]["runs"] == 1
