"""Offline tests for Langfuse helpers (no network)."""

from __future__ import annotations

import os

import pytest

from memgraphrag.observability.langfuse_trace import (
    flush_langfuse,
    get_langfuse_client,
    is_langfuse_enabled,
    observation,
    reset_langfuse_client_for_tests,
    truncate_docs,
    update_observation,
)


@pytest.fixture(autouse=True)
def _clear_langfuse_env(monkeypatch):
    reset_langfuse_client_for_tests()
    for key in (
        "LANGFUSE_ENABLE_TRACE",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    reset_langfuse_client_for_tests()


@pytest.mark.offline
def test_disabled_without_flag():
    assert is_langfuse_enabled() is False
    assert get_langfuse_client() is None
    with observation("noop") as span:
        assert span is None
    flush_langfuse()  # no-op


@pytest.mark.offline
def test_disabled_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLE_TRACE", "true")
    assert is_langfuse_enabled() is False


@pytest.mark.offline
def test_enabled_with_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLE_TRACE", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert is_langfuse_enabled() is True


@pytest.mark.offline
def test_truncate_docs():
    docs = ["a" * 1000, "b", "c", "d", "e", "f"]
    out = truncate_docs(docs, max_docs=3, max_chars=10)
    assert len(out) == 4  # 3 + overflow marker
    assert out[0].endswith("...")
    assert "more" in out[-1]


@pytest.mark.offline
def test_update_none_span_is_safe():
    update_observation(None, output={"ok": True})
