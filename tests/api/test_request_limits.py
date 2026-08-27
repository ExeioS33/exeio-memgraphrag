"""Tests for anti-abuse limits: login throttling, upload size cap, type allowlist."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from memgraphrag.api.rate_limit import FixedWindowRateLimiter, client_key
from memgraphrag.api.routers.documents import (
    _reject_unsupported_suffix,
    _spool_upload,
)
from memgraphrag.api.server import create_app
from tests.api.test_auth_edge_cases import _mock_rag, _test_args


@pytest.mark.offline
def test_rate_limiter_blocks_after_budget_and_reports_retry_after() -> None:
    limiter = FixedWindowRateLimiter(max_attempts=3, window_seconds=60.0)
    assert [limiter.check("1.2.3.4") for _ in range(3)] == [None, None, None]

    retry_after = limiter.check("1.2.3.4")
    assert retry_after is not None and 0 < retry_after <= 60.0

    # Buckets are per key.
    assert limiter.check("5.6.7.8") is None

    # A successful auth clears the budget.
    limiter.reset("1.2.3.4")
    assert limiter.check("1.2.3.4") is None


@pytest.mark.offline
def test_rate_limiter_disabled_when_budget_non_positive() -> None:
    limiter = FixedWindowRateLimiter(max_attempts=0, window_seconds=60.0)
    assert all(limiter.check("k") is None for _ in range(50))


@pytest.mark.offline
def test_client_key_ignores_forwarded_headers() -> None:
    class _Req:
        def __init__(self) -> None:
            self.client = type("C", (), {"host": "10.0.0.1"})()
            self.headers = {"X-Forwarded-For": "1.1.1.1"}

    # Trusting X-Forwarded-For would hand out a free reset per forged header.
    assert client_key(_Req()) == "10.0.0.1"
    assert client_key(type("R", (), {"client": None})()) == "unknown"


@pytest.mark.offline
def test_login_is_throttled() -> None:
    app = create_app(
        _test_args(
            auth_accounts="admin:pw123",
            token_secret="unit-test-secret",
            login_max_attempts=3,
            login_window_seconds=60.0,
        ),
        testing=True,
        rag=_mock_rag(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        for _ in range(3):
            resp = client.post("/login", data={"username": "admin", "password": "no"})
            assert resp.status_code == 401

        throttled = client.post("/login", data={"username": "admin", "password": "no"})
        assert throttled.status_code == 429
        assert int(throttled.headers["Retry-After"]) >= 1


@pytest.mark.offline
def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        _reject_unsupported_suffix("payload.exe")
    assert exc.value.status_code == 415

    # A suffix every build can parse must pass.
    _reject_unsupported_suffix("notes.txt")


@pytest.mark.offline
async def test_spool_upload_enforces_cap_and_cleans_up(tmp_path: Path) -> None:
    class _Upload:
        def __init__(self, payload: bytes) -> None:
            self._buf = payload

        async def read(self, size: int = -1) -> bytes:
            if size is None or size < 0:
                chunk, self._buf = self._buf, b""
                return chunk
            chunk, self._buf = self._buf[:size], self._buf[size:]
            return chunk

    ok_dest = tmp_path / "small.txt"
    written = await _spool_upload(_Upload(b"x" * 1000), ok_dest, max_bytes=10_000)
    assert written == 1000
    assert ok_dest.read_bytes() == b"x" * 1000

    big_dest = tmp_path / "big.txt"
    with pytest.raises(HTTPException) as exc:
        await _spool_upload(_Upload(b"x" * 5_000_000), big_dest, max_bytes=1024)
    assert exc.value.status_code == 413
    # A rejected upload must not leave a partial file in the input directory.
    assert not big_dest.exists()
