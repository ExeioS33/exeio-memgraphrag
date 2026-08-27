"""In-process fixed-window rate limiting for authentication endpoints.

Scope: this limiter lives in one process. It is sufficient for the supported
single-worker deployment (``WORKERS=1``); behind several workers or replicas each
process keeps its own counters, so the effective limit multiplies by the worker
count. A shared backend (Redis / Postgres) is required before enabling multi-worker.

Provenance: same intent as LightRAG ``lightrag/api/login_rate_limit.py``, reduced to
what a single-process deployment needs.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class FixedWindowRateLimiter:
    """Allow at most ``max_attempts`` per ``window_seconds`` for a given key.

    Keys are caller-defined (client IP, username, …). Buckets for keys that have gone
    quiet are dropped on access so memory stays bounded by the number of *active*
    keys rather than by every key ever seen.
    """

    def __init__(self, max_attempts: int, window_seconds: float) -> None:
        self.max_attempts = int(max_attempts)
        self.window_seconds = float(window_seconds)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def check(self, key: str) -> float | None:
        """Record an attempt. Return ``None`` if allowed, else seconds to wait."""
        if self.max_attempts <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket
            self._prune(bucket, now)
            if len(bucket) >= self.max_attempts:
                return max(0.0, self.window_seconds - (now - bucket[0]))
            bucket.append(now)
            # Opportunistic sweep so idle keys do not accumulate forever.
            if len(self._hits) > 1024:
                for k in [k for k, b in self._hits.items() if not b]:
                    del self._hits[k]
            return None

    def reset(self, key: str) -> None:
        """Forget a key's attempts (called after a successful authentication)."""
        with self._lock:
            self._hits.pop(key, None)


def client_key(request: object) -> str:
    """Best-effort client identity for rate limiting.

    Uses the socket peer only. ``X-Forwarded-For`` is deliberately ignored: it is
    attacker-controlled unless a trusted proxy rewrites it, and trusting it here
    would hand out a free limiter reset per forged header.
    """
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return str(host or "unknown")
