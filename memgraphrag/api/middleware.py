"""Request correlation and Prometheus metrics for the MemGraphRAG API.

Two pure-ASGI middlewares plus a hand-written Prometheus text renderer:

* :class:`RequestContextMiddleware` — accepts (or mints) ``X-Request-ID``, echoes it
  on the response and publishes it through a :mod:`contextvars` variable so log
  records can carry it. MemGraphRAG runs the whole index/retrieve pipeline on one
  asyncio loop, so ``[STAGE]`` / ``[LLM]`` lines from concurrent requests interleave;
  without a correlation id an operator cannot tell which line belongs to which call.
* :class:`MetricsMiddleware` — in-process counters/histogram keyed by the matched
  *route template*, so ``/documents/{doc_id}`` cannot explode label cardinality.
* :func:`render_prometheus` — Prometheus text exposition format, written by hand so
  the service image gains no dependency.

Scope: like :mod:`memgraphrag.api.rate_limit`, the registry lives in one process.
Behind several workers each process reports its own counters; scrape them per
instance (or move to a shared backend) before enabling ``WORKERS>1``.

Provenance: MemGraphRAG-native; no LightRAG counterpart.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from typing import Any

REQUEST_ID_HEADER = "X-Request-ID"

# A request id ends up in log lines and in a response header. Anything outside this
# alphabet is dropped so a caller cannot forge log lines (CR/LF) or split headers.
_REQUEST_ID_ALLOWED = re.compile(r"[^A-Za-z0-9._:-]")
_REQUEST_ID_MAX_LEN = 128

# Route label used when no route matched: keeps 404 scans from creating one metric
# series per probed URL.
UNMATCHED_ROUTE = "<unmatched>"

# Seconds. Chosen for an LLM-backed API: sub-second calls are cache/health traffic,
# the long tail is indexing and generation.
DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

_request_id: ContextVar[str] = ContextVar("memgraphrag_request_id", default="")


def get_request_id() -> str:
    """Return the current request's correlation id, or ``""`` outside a request."""
    return _request_id.get()


def set_request_id(value: str) -> Any:
    """Bind ``value`` as the current correlation id; returns the reset token."""
    return _request_id.set(value)


def new_request_id() -> str:
    """Mint a correlation id for a request that arrived without one."""
    return uuid.uuid4().hex


def sanitize_request_id(value: str | None) -> str:
    """Normalise a caller-supplied id, or mint one when it is unusable.

    Callers control ``X-Request-ID``; echoing it verbatim would let them inject
    newlines into logs or terminate the response header early.
    """
    cleaned = _REQUEST_ID_ALLOWED.sub("", str(value or ""))[:_REQUEST_ID_MAX_LEN]
    return cleaned or new_request_id()


class RequestIdLogFilter(logging.Filter):
    """Attach ``request_id`` to every record so formatters can print it.

    Records emitted outside a request (startup, shutdown, background drains that
    outlive their request) get ``-`` rather than raising a formatting KeyError.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class RequestContextMiddleware:
    """Pure-ASGI ``X-Request-ID`` propagation.

    Implemented as raw ASGI rather than ``BaseHTTPMiddleware`` on purpose: the latter
    runs the endpoint in a separate anyio task, so a ``ContextVar`` set in its
    ``dispatch`` is not reliably visible to the handler — which is precisely what the
    log correlation needs.
    """

    def __init__(self, app: Any, header_name: str = REQUEST_ID_HEADER) -> None:
        self.app = app
        self.header_name = header_name
        self._header_key = header_name.lower().encode("latin-1")

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        incoming = None
        for key, value in scope.get("headers") or ():
            if key.lower() == self._header_key:
                incoming = value.decode("latin-1", "replace")
                break
        request_id = sanitize_request_id(incoming)

        # Handlers that never touch contextvars can still read it off the request.
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        encoded = request_id.encode("latin-1")

        async def send_with_request_id(message: Mapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((self._header_key, encoded))
                message = {**message, "headers": headers}
            await send(message)

        token = set_request_id(request_id)
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            _request_id.reset(token)


class MetricsRegistry:
    """Counters + a latency histogram, keyed by ``(method, route, status)``.

    Small enough to keep in a dict: the label set is bounded by the number of routes
    the app declares, not by the traffic it serves.
    """

    def __init__(self, buckets: Iterable[float] = DEFAULT_LATENCY_BUCKETS) -> None:
        self.buckets: tuple[float, ...] = tuple(sorted(float(b) for b in buckets))
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = {}
        self._bucket_counts: dict[tuple[str, str], list[int]] = {}
        self._duration_sum: dict[tuple[str, str], float] = {}
        self._duration_count: dict[tuple[str, str], int] = {}
        self._in_flight = 0

    def observe(
        self, method: str, route: str, status_code: int, duration: float
    ) -> None:
        key = (method, route)
        with self._lock:
            counter_key = (method, route, str(int(status_code)))
            self._requests[counter_key] = self._requests.get(counter_key, 0) + 1

            counts = self._bucket_counts.get(key)
            if counts is None:
                counts = [0] * len(self.buckets)
                self._bucket_counts[key] = counts
            for i, bound in enumerate(self.buckets):
                if duration <= bound:
                    counts[i] += 1
            self._duration_sum[key] = self._duration_sum.get(key, 0.0) + float(duration)
            self._duration_count[key] = self._duration_count.get(key, 0) + 1

    def enter(self) -> None:
        with self._lock:
            self._in_flight += 1

    def leave(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent copy so rendering never races with live requests."""
        with self._lock:
            return {
                "buckets": self.buckets,
                "requests": dict(self._requests),
                "bucket_counts": {k: list(v) for k, v in self._bucket_counts.items()},
                "duration_sum": dict(self._duration_sum),
                "duration_count": dict(self._duration_count),
                "in_flight": self._in_flight,
            }


class MetricsMiddleware:
    """Record request counts and latencies into a :class:`MetricsRegistry`."""

    def __init__(self, app: Any, registry: MetricsRegistry | None = None) -> None:
        self.app = app
        self.registry = registry or MetricsRegistry()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method") or "GET").upper()
        status_holder = {"code": 500}

        async def send_capturing_status(message: Mapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status_holder["code"] = int(message.get("status") or 0)
            await send(message)

        started = time.perf_counter()
        self.registry.enter()
        try:
            await self.app(scope, receive, send_capturing_status)
        finally:
            self.registry.leave()
            # Starlette stores the matched APIRoute on the scope during routing, so
            # this runs after the fact and yields the template, not the raw path.
            self.registry.observe(
                method,
                route_label(scope),
                status_holder["code"],
                time.perf_counter() - started,
            )


def route_label(scope: Mapping[str, Any]) -> str:
    """Best-effort route *template* for a finished request.

    Falls back to :data:`UNMATCHED_ROUTE` rather than the raw path: an unrouted URL is
    attacker-chosen, and using it as a label would let a scan grow the registry
    without bound.
    """
    route = scope.get("route")
    path = getattr(route, "path_format", None) or getattr(route, "path", None)
    if path:
        return str(path)
    return UNMATCHED_ROUTE


def _escape_label_value(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _labels(pairs: Iterable[tuple[str, Any]]) -> str:
    body = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in pairs)
    return "{" + body + "}" if body else ""


def _format_float(value: float) -> str:
    if value == float("inf"):
        return "+Inf"
    return repr(float(value))


def render_prometheus(
    registry: MetricsRegistry,
    *,
    gauges: Mapping[str, tuple[float, str]] | None = None,
) -> str:
    """Render ``registry`` (plus extra ``gauges``) as Prometheus text format.

    ``gauges`` maps metric name → ``(value, help_text)``; used for state that lives on
    the app rather than in the registry (``pipeline_busy``, readiness).
    """
    snap = registry.snapshot()
    buckets: tuple[float, ...] = snap["buckets"]
    lines: list[str] = []

    lines.append("# HELP memgraphrag_http_requests_total Total HTTP requests served.")
    lines.append("# TYPE memgraphrag_http_requests_total counter")
    for (method, route, code), count in sorted(snap["requests"].items()):
        labels = _labels((("method", method), ("route", route), ("code", code)))
        lines.append(f"memgraphrag_http_requests_total{labels} {count}")

    lines.append(
        "# HELP memgraphrag_http_request_duration_seconds "
        "HTTP request latency in seconds."
    )
    lines.append("# TYPE memgraphrag_http_request_duration_seconds histogram")
    for key in sorted(snap["duration_count"]):
        method, route = key
        counts = snap["bucket_counts"].get(key) or [0] * len(buckets)
        total = snap["duration_count"][key]
        for bound, count in zip(buckets, counts):
            labels = _labels(
                (("method", method), ("route", route), ("le", _format_float(bound)))
            )
            lines.append(
                f"memgraphrag_http_request_duration_seconds_bucket{labels} {count}"
            )
        inf_labels = _labels(
            (("method", method), ("route", route), ("le", "+Inf"))
        )
        lines.append(
            f"memgraphrag_http_request_duration_seconds_bucket{inf_labels} {total}"
        )
        pair = _labels((("method", method), ("route", route)))
        lines.append(
            "memgraphrag_http_request_duration_seconds_sum"
            f"{pair} {snap['duration_sum'].get(key, 0.0)!r}"
        )
        lines.append(
            f"memgraphrag_http_request_duration_seconds_count{pair} {total}"
        )

    lines.append(
        "# HELP memgraphrag_http_requests_in_flight HTTP requests being served now."
    )
    lines.append("# TYPE memgraphrag_http_requests_in_flight gauge")
    lines.append(f"memgraphrag_http_requests_in_flight {snap['in_flight']}")

    for name, (value, help_text) in sorted((gauges or {}).items()):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {float(value)!r}")

    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_LATENCY_BUCKETS",
    "REQUEST_ID_HEADER",
    "UNMATCHED_ROUTE",
    "MetricsMiddleware",
    "MetricsRegistry",
    "RequestContextMiddleware",
    "RequestIdLogFilter",
    "get_request_id",
    "new_request_id",
    "render_prometheus",
    "route_label",
    "sanitize_request_id",
    "set_request_id",
]

