"""Golden sets: round-tripping the file, and judging a drop against measured noise."""

from __future__ import annotations

from pathlib import Path

import pytest

from memgraphrag.evaluation.datasets import EvaluationExample
from memgraphrag.evaluation.golden import (
    GOLDEN_FORMAT,
    GoldenSet,
    GoldenSetError,
    build_golden_set,
    compare_to_golden,
    load_golden_set,
    write_golden_set,
)
from memgraphrag.evaluation.runner import CampaignReport, aggregate_runs

pytestmark = pytest.mark.offline

EXAMPLES = [
    EvaluationExample(
        id="q1",
        question="Where?",
        gold_answers=["Paris"],
        gold_docs=["Doc A"],
        dataset="fixture",
    )
]


def _report(values: list[dict[str, float]]) -> CampaignReport:
    return CampaignReport(dataset="fixture", examples=1, runs=[], metrics=aggregate_runs(values))


def _golden(mean: float, stdev: float, metric: str = "StrAcc") -> GoldenSet:
    return GoldenSet(
        name="ref",
        dataset="fixture",
        examples=list(EXAMPLES),
        metrics={metric: {"mean": mean, "stdev": stdev, "runs": 5}},
    )


def test_golden_set_round_trips_through_disk(tmp_path: Path) -> None:
    report = _report([{"StrAcc": 0.60}, {"StrAcc": 0.64}])
    golden = build_golden_set("hotpot-50", EXAMPLES, report, metadata={"mode": "ppr"})
    path = write_golden_set(tmp_path / "golden.json", golden)

    loaded = load_golden_set(path)
    assert loaded.dataset == "fixture"
    assert loaded.examples[0].gold_docs == ["Doc A"]
    assert loaded.mean("StrAcc") == pytest.approx(0.62)
    assert loaded.stdev("StrAcc") > 0
    assert loaded.metadata["mode"] == "ppr"
    assert golden.to_dict()["format"] == GOLDEN_FORMAT


def test_a_foreign_json_file_is_rejected(tmp_path: Path) -> None:
    stray = tmp_path / "notes.json"
    stray.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(GoldenSetError):
        load_golden_set(stray)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError):
        load_golden_set(tmp_path / "absent.json")


def test_a_drop_inside_the_measured_noise_is_not_a_regression() -> None:
    """59.25 -> 58.90 with sigma 0.5 is noise, and the harness must say so."""
    comparison = compare_to_golden(_golden(0.5925, 0.005), _report([{"StrAcc": 0.589}]))
    assert comparison.passed
    assert comparison.comparisons[0].regressed is False
    assert comparison.comparisons[0].z_score == pytest.approx(-0.7, abs=0.05)


def test_a_drop_beyond_the_tolerance_is_a_regression() -> None:
    comparison = compare_to_golden(
        _golden(0.5925, 0.005), _report([{"StrAcc": 0.55}]), tolerance_sigma=2.0
    )
    assert not comparison.passed
    assert [row.metric for row in comparison.regressions] == ["StrAcc"]


def test_the_noisier_of_the_two_measurements_sets_the_bar() -> None:
    """A current run with wide spread cannot be declared a regression on a tight reference."""
    noisy = _report([{"StrAcc": 0.50}, {"StrAcc": 0.68}])
    comparison = compare_to_golden(_golden(0.5925, 0.005), noisy, tolerance_sigma=2.0)
    row = comparison.comparisons[0]
    assert row.sigma > 0.005
    assert row.regressed is False


def test_without_measured_variance_the_check_falls_back_and_says_so() -> None:
    """A golden set built from one run cannot certify anything; the report admits it."""
    comparison = compare_to_golden(_golden(0.60, 0.0), _report([{"StrAcc": 0.55}]))
    row = comparison.comparisons[0]
    assert row.regressed is True
    assert "no variance" in row.sigma_basis


def test_audit_counters_regress_when_they_rise() -> None:
    """A tripled unparsed-judge rate is bad news even though the number went up."""
    reference = _golden(0.01, 0.002, metric="LLMAccUnparsedRate")
    comparison = compare_to_golden(reference, _report([{"LLMAccUnparsedRate": 0.30}]))
    assert comparison.comparisons[0].regressed is True

    improved = compare_to_golden(reference, _report([{"LLMAccUnparsedRate": 0.0}]))
    assert improved.comparisons[0].improved is True
    assert improved.passed


def test_a_metric_present_on_only_one_side_is_listed_as_missing() -> None:
    comparison = compare_to_golden(_golden(0.6, 0.01), _report([{"LLMAcc": 0.7}]))
    assert sorted(comparison.missing) == ["LLMAcc", "StrAcc"]
    assert comparison.passed


def test_a_real_improvement_is_reported_as_such() -> None:
    comparison = compare_to_golden(_golden(0.50, 0.005), _report([{"StrAcc": 0.62}]))
    assert comparison.comparisons[0].improved is True
    assert comparison.passed
