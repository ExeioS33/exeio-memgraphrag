"""Performance instrumentation: latency percentiles, throughput, LLM call cost.

``docs/Reproduce.md`` promises RPS, p50/p95 and cost per query but ships no way
to measure any of them, and the paper's "0.061 s per retrieval" has never been
checkable in this repository. These are the primitives ``scripts/bench.py``
drives; they are deliberately free of any engine import so they can be tested
offline against a stub.

Token counts produced here are **locally estimated** with the project tokenizer,
not the provider's billed usage: the OpenAI-compatible binding in
``memgraphrag/llm/`` does not surface ``response.usage``, so a caller wanting
billed tokens must read them from the provider. The estimate is still the right
tool for comparing two engine configurations against the same endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def percentile(values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile of ``values`` (``quantile`` in 0..1).

    Nearest-rank rather than interpolated: an interpolated p95 reports a latency
    that no request actually experienced, which is the wrong number to put in a
    service objective.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if quantile <= 0:
        return ordered[0]
    rank = max(1, min(len(ordered), int(-(-quantile * len(ordered) // 1))))
    return ordered[rank - 1]


@dataclass(frozen=True)
class LatencyStats:
    """Summary of one latency sample set, in seconds."""

    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencyStats":
        if not samples:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            count=len(samples),
            mean=sum(samples) / len(samples),
            p50=percentile(samples, 0.50),
            p95=percentile(samples, 0.95),
            p99=percentile(samples, 0.99),
            minimum=min(samples),
            maximum=max(samples),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_s": round(self.mean, 6),
            "p50_s": round(self.p50, 6),
            "p95_s": round(self.p95, 6),
            "p99_s": round(self.p99, 6),
            "min_s": round(self.minimum, 6),
            "max_s": round(self.maximum, 6),
        }


@dataclass
class CallMeter:
    """Counts LLM calls and estimates their tokens by wrapping a completion func.

    Wrapping is how per-query LLM cost becomes observable without touching
    ``memgraphrag/llm/``: the engine is handed ``meter.wrap(openai_complete)``
    and every call it makes on any path — OpenIE, fact rerank, answer generation
    — lands in the same counters.
    """

    model: str = "gpt-4o-mini"
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from ..utils.tokenizer import TiktokenTokenizer

            self._tokenizer = TiktokenTokenizer(self.model)
        except Exception as exc:  # noqa: BLE001 - estimation must never break a bench
            logger.debug("tokenizer unavailable (%s); falling back to a length heuristic", exc)
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Token count, or a 4-characters-per-token estimate when tiktoken is absent."""
        if not text:
            return 0
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:  # noqa: BLE001
                pass
        return max(1, len(text) // 4)

    def record(self, prompt: str, response: str, seconds: float) -> None:
        self.calls += 1
        self.prompt_tokens += self.count_tokens(prompt)
        self.completion_tokens += self.count_tokens(response)
        self.seconds += seconds

    def wrap(self, complete: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Return ``complete`` with call/token accounting attached."""

        async def _metered(prompt: str, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = await complete(prompt, **kwargs)
            elapsed = time.perf_counter() - started
            system_prompt = str(kwargs.get("system_prompt") or "")
            self.record(f"{system_prompt}\n{prompt}", str(result), elapsed)
            return result

        return _metered

    def snapshot(self) -> dict[str, float]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "llm_seconds": round(self.seconds, 4),
        }

    def per_query(self, queries: int) -> dict[str, float]:
        """Cost normalised per query — the shape ``docs/Reproduce.md`` asks for."""
        if queries <= 0:
            return {"llm_calls_per_query": 0.0, "tokens_per_query": 0.0}
        return {
            "llm_calls_per_query": round(self.calls / queries, 4),
            "tokens_per_query": round((self.prompt_tokens + self.completion_tokens) / queries, 2),
        }


@dataclass(frozen=True)
class LoadResult:
    """Outcome of driving ``run_load``: per-call latencies plus wall-clock throughput."""

    latencies: list[float]
    wall_seconds: float
    errors: int
    results: list[Any] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        """Completed calls per second, errors included: they cost time too."""
        total = len(self.latencies) + self.errors
        return total / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency": LatencyStats.from_samples(self.latencies).to_dict(),
            "wall_seconds": round(self.wall_seconds, 4),
            "errors": self.errors,
            "throughput_rps": round(self.throughput, 4),
        }


async def run_load(
    call: Callable[[T], Awaitable[Any]],
    items: Sequence[T],
    concurrency: int = 1,
    warmup: int = 0,
) -> LoadResult:
    """Run ``call`` over ``items`` at a fixed concurrency, timing each call.

    ``warmup`` calls are executed and discarded first: the first query of a
    process pays for lazy storage loads and PPR graph construction, and letting
    that land in the sample makes p95 a measurement of start-up.
    """
    for item in items[:warmup]:
        try:
            await call(item)
        except Exception as exc:  # noqa: BLE001 - a failing warm-up is not the measurement
            logger.warning("warm-up call failed: %s", exc)

    semaphore = asyncio.Semaphore(max(1, concurrency))
    latencies: list[float] = []
    results: list[Any] = []
    errors = 0

    async def _one(item: T) -> tuple[float | None, Any]:
        async with semaphore:
            started = time.perf_counter()
            try:
                result = await call(item)
            except Exception as exc:  # noqa: BLE001
                logger.warning("load call failed: %s", exc)
                return None, None
            return time.perf_counter() - started, result

    wall_started = time.perf_counter()
    outcomes = await asyncio.gather(*(_one(item) for item in items))
    wall_seconds = time.perf_counter() - wall_started

    for elapsed, result in outcomes:
        if elapsed is None:
            errors += 1
            continue
        latencies.append(elapsed)
        results.append(result)

    return LoadResult(
        latencies=latencies, wall_seconds=wall_seconds, errors=errors, results=results
    )
