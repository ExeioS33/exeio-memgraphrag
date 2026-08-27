"""Golden sets and non-regression comparison in units of measured variance.

A golden set here is two things in one file: the frozen question set (so a later
campaign scores the *same* questions) and the reference metric distribution
(mean and standard deviation over N runs). The comparison is expressed in
standard deviations rather than in absolute points because an absolute threshold
is a guess about noise, and this harness measures that noise instead — see
``runner.run_campaign`` and ``docs/Evaluation.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .datasets import EvaluationExample
from .runner import CampaignReport, MetricStat

#: File marker, so a mistyped path fails loudly instead of comparing to garbage.
GOLDEN_FORMAT = "memgraphrag-golden-set"
GOLDEN_VERSION = 1

#: Metrics where a *rise* is the bad news. These are the audit counters that
#: qualify a score (verbose Str-Acc hits, truncated answers, unparsed judge
#: replies); a campaign whose Str-Acc held steady while its unparsed rate tripled
#: has a broken judge, and comparing it "higher is better" would hide that.
LOWER_IS_BETTER = frozenset({"StrAccVerboseHitRate", "StrAccTruncatedRate", "LLMAccUnparsedRate"})

#: Used only when the reference recorded no variance (a single run). One absolute
#: point of the 0..1 scale: deliberately crude, and reported as such.
DEFAULT_ABSOLUTE_FLOOR = 0.01


class GoldenSetError(ValueError):
    """The file is not a MemGraphRAG golden set, or is of an unknown version."""


@dataclass
class GoldenSet:
    """Reference questions plus the metric distribution they produced."""

    name: str
    dataset: str
    examples: list[EvaluationExample] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def mean(self, metric: str) -> float | None:
        entry = self.metrics.get(metric)
        return float(entry["mean"]) if entry and "mean" in entry else None

    def stdev(self, metric: str) -> float:
        entry = self.metrics.get(metric) or {}
        return float(entry.get("stdev") or 0.0)

    def runs(self, metric: str) -> int:
        entry = self.metrics.get(metric) or {}
        return int(entry.get("runs") or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": GOLDEN_FORMAT,
            "version": GOLDEN_VERSION,
            "name": self.name,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "metrics": {key: dict(value) for key, value in self.metrics.items()},
            "examples": [example.to_dict() for example in self.examples],
        }


def build_golden_set(
    name: str,
    examples: Sequence[EvaluationExample],
    report: CampaignReport | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GoldenSet:
    """Freeze a question set and, when a campaign is given, its metric distribution."""
    metrics: dict[str, dict[str, float]] = {}
    if report is not None:
        metrics = {key: _stat_to_entry(stat) for key, stat in report.metrics.items()}
    merged = dict(report.metadata) if report is not None else {}
    merged.update(dict(metadata or {}))
    return GoldenSet(
        name=name,
        dataset=(report.dataset if report is not None else "")
        or (examples[0].dataset if examples else ""),
        examples=list(examples),
        metrics=metrics,
        metadata=merged,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _stat_to_entry(stat: MetricStat) -> dict[str, float]:
    return {
        "mean": round(stat.mean, 6),
        "stdev": round(stat.stdev, 6),
        "runs": stat.runs,
        "min": round(stat.minimum, 6),
        "max": round(stat.maximum, 6),
    }


def write_golden_set(path: str | Path, golden: GoldenSet) -> Path:
    """Write a golden set as indented JSON (it is meant to be read in a diff)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(golden.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target


def load_golden_set(path: str | Path) -> GoldenSet:
    """Read a golden set, rejecting anything that is not one."""
    source = Path(path)
    if not source.exists():
        raise GoldenSetError(f"golden set not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != GOLDEN_FORMAT:
        raise GoldenSetError(f"{source} is not a {GOLDEN_FORMAT} file")
    if int(payload.get("version") or 0) > GOLDEN_VERSION:
        raise GoldenSetError(
            f"{source} is version {payload.get('version')}, this build reads {GOLDEN_VERSION}"
        )
    examples = [
        EvaluationExample(
            id=str(item.get("id") or ""),
            question=str(item.get("question") or ""),
            gold_answers=[str(answer) for answer in item.get("gold_answers") or []],
            gold_docs=[str(doc) for doc in item.get("gold_docs") or []],
            dataset=str(item.get("dataset") or payload.get("dataset") or ""),
            question_type=str(item.get("question_type") or ""),
        )
        for item in payload.get("examples") or []
    ]
    return GoldenSet(
        name=str(payload.get("name") or source.stem),
        dataset=str(payload.get("dataset") or ""),
        examples=examples,
        metrics={
            str(key): {name: float(value) for name, value in (entry or {}).items()}
            for key, entry in (payload.get("metrics") or {}).items()
        },
        metadata=dict(payload.get("metadata") or {}),
        created_at=str(payload.get("created_at") or ""),
    )


@dataclass(frozen=True)
class MetricComparison:
    """One metric, reference vs. current, with the noise it was judged against."""

    metric: str
    reference_mean: float
    current_mean: float
    delta: float
    sigma: float
    sigma_basis: str
    tolerance_sigma: float
    regressed: bool
    improved: bool

    @property
    def z_score(self) -> float:
        """Change in standard deviations; 0.0 when no variance was ever measured."""
        return self.delta / self.sigma if self.sigma > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "reference_mean": round(self.reference_mean, 6),
            "current_mean": round(self.current_mean, 6),
            "delta": round(self.delta, 6),
            "sigma": round(self.sigma, 6),
            "sigma_basis": self.sigma_basis,
            "z_score": round(self.z_score, 3),
            "tolerance_sigma": self.tolerance_sigma,
            "regressed": self.regressed,
            "improved": self.improved,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """Verdict of a golden-set comparison."""

    comparisons: list[MetricComparison] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricComparison]:
        return [row for row in self.comparisons if row.regressed]

    @property
    def passed(self) -> bool:
        return not self.regressions

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "regressions": [row.metric for row in self.regressions],
            "missing_metrics": list(self.missing),
            "comparisons": [row.to_dict() for row in self.comparisons],
        }


def compare_to_golden(
    reference: GoldenSet,
    current: CampaignReport | Mapping[str, MetricStat],
    tolerance_sigma: float = 2.0,
    absolute_floor: float = DEFAULT_ABSOLUTE_FLOOR,
    metrics: Sequence[str] | None = None,
) -> ComparisonReport:
    """Compare a campaign to a golden set, in standard deviations.

    ``sigma`` is the larger of the reference and current standard deviations: the
    noisier of the two measurements is the one that decides what "the same
    result" means. When neither side measured any variance — a golden set built
    from a single run — the check falls back to ``absolute_floor`` and says so in
    ``sigma_basis``, so nobody mistakes the resulting green for evidence.
    """
    current_metrics: Mapping[str, MetricStat] = (
        current.metrics if isinstance(current, CampaignReport) else current
    )
    names = list(metrics) if metrics else sorted(set(reference.metrics) | set(current_metrics))

    comparisons: list[MetricComparison] = []
    missing: list[str] = []
    for name in names:
        reference_mean = reference.mean(name)
        stat = current_metrics.get(name)
        if reference_mean is None or stat is None:
            missing.append(name)
            continue
        delta = stat.mean - reference_mean
        # Audit counters read the other way round: normalise to "positive = better"
        # so a single rule decides every row.
        oriented = -delta if name in LOWER_IS_BETTER else delta
        sigma = max(reference.stdev(name), stat.stdev)
        if sigma > 0:
            basis = "measured"
            regressed = oriented < -tolerance_sigma * sigma
            improved = oriented > tolerance_sigma * sigma
        else:
            basis = "absolute-floor (no variance measured)"
            regressed = oriented < -absolute_floor
            improved = oriented > absolute_floor
        comparisons.append(
            MetricComparison(
                metric=name,
                reference_mean=reference_mean,
                current_mean=stat.mean,
                delta=delta,
                sigma=sigma,
                sigma_basis=basis,
                tolerance_sigma=tolerance_sigma,
                regressed=regressed,
                improved=improved,
            )
        )
    return ComparisonReport(comparisons=comparisons, missing=missing)
