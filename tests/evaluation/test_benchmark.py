"""Latency percentiles, load driving and LLM call accounting.

Item 47 of the audit: docs/Reproduce.md promises p50/p95, RPS and cost per query
with no script behind them. These tests pin the primitives that script uses; the
timings here are of stubs, so they assert shape and accounting, never durations.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from memgraphrag.evaluation.benchmark import CallMeter, LatencyStats, percentile, run_load

pytestmark = pytest.mark.offline


def test_percentile_is_nearest_rank_and_returns_an_observed_value() -> None:
    """An interpolated p95 reports a latency no request ever experienced."""
    samples = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(samples, 0.5) == 3.0
    assert percentile(samples, 0.95) == 10.0
    assert percentile(samples, 0.0) == 1.0
    assert percentile([], 0.5) == 0.0


def test_latency_stats_summarise_a_sample_set() -> None:
    stats = LatencyStats.from_samples([0.1, 0.2, 0.3, 0.4])
    assert stats.count == 4
    assert stats.p50 == 0.2
    assert stats.minimum == 0.1 and stats.maximum == 0.4
    assert stats.to_dict()["p95_s"] == 0.4


def test_empty_sample_set_is_zeroed_not_an_error() -> None:
    assert LatencyStats.from_samples([]).count == 0


async def test_run_load_times_every_call_and_reports_throughput() -> None:
    async def call(item: int) -> int:
        await asyncio.sleep(0.001)
        return item * 2

    result = await run_load(call, [1, 2, 3], concurrency=2)
    assert len(result.latencies) == 3
    assert result.results == [2, 4, 6]
    assert result.errors == 0
    assert result.throughput > 0
    assert result.to_dict()["latency"]["count"] == 3


async def test_warmup_calls_are_executed_but_not_measured() -> None:
    """The first query pays for storage load and PPR build; that is start-up, not p95."""
    seen: list[int] = []

    async def call(item: int) -> int:
        seen.append(item)
        return item

    result = await run_load(call, [1, 2], concurrency=1, warmup=1)
    assert seen == [1, 1, 2]
    assert len(result.latencies) == 2


async def test_a_failing_call_is_counted_as_an_error_not_as_a_fast_success() -> None:
    async def call(item: int) -> int:
        if item == 2:
            raise RuntimeError("timeout")
        return item

    result = await run_load(call, [1, 2, 3], concurrency=1)
    assert result.errors == 1
    assert len(result.latencies) == 2
    # Errors still consumed wall time, so they belong in the throughput denominator.
    assert result.throughput == pytest.approx(3 / result.wall_seconds, rel=1e-3)


async def test_call_meter_counts_calls_and_tokens_through_the_wrapper() -> None:
    """Wrapping the completion func is what makes per-query LLM cost observable."""
    meter = CallMeter()

    async def complete(prompt: str, **kwargs: Any) -> str:
        return "an answer"

    metered = meter.wrap(complete)
    await metered("a question", system_prompt="be brief")
    await metered("another question")

    snapshot = meter.snapshot()
    assert snapshot["llm_calls"] == 2
    assert snapshot["prompt_tokens"] > 0
    assert snapshot["completion_tokens"] > 0
    assert snapshot["total_tokens"] == snapshot["prompt_tokens"] + snapshot["completion_tokens"]
    assert meter.per_query(2)["llm_calls_per_query"] == 1.0


async def test_metered_wrapper_returns_the_original_result() -> None:
    meter = CallMeter()

    async def complete(prompt: str, **kwargs: Any) -> str:
        return "verbatim"

    assert await meter.wrap(complete)("q") == "verbatim"


def test_token_count_falls_back_when_no_tokenizer_is_available() -> None:
    """A missing tiktoken model must degrade the estimate, not break the benchmark."""
    meter = CallMeter()
    meter._tokenizer = None  # noqa: SLF001 - simulating the ImportError path
    assert meter.count_tokens("abcdefgh") == 2
    assert meter.count_tokens("") == 0


def test_per_query_cost_is_zero_for_zero_queries() -> None:
    assert CallMeter().per_query(0)["tokens_per_query"] == 0.0
