"""Offline tests for outbound TLS helpers."""

from __future__ import annotations

import os
import ssl
from pathlib import Path

import pytest

from memgraphrag.utils.http_ssl import (
    build_ssl_context,
    describe_ssl_verify,
    find_configured_ca_path,
    reset_ssl_verify_cache,
    ssl_verify,
)


@pytest.fixture(autouse=True)
def _clear_ssl_cache(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "SSL_VERIFY",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "MEMGRAPHRAG_SSL_CERT_FILE",
        "MEMGRAPHRAG_CORP_CA_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    reset_ssl_verify_cache()
    yield
    reset_ssl_verify_cache()


@pytest.mark.offline
def test_ssl_verify_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSL_VERIFY", "false")
    reset_ssl_verify_cache()
    assert ssl_verify() is False


@pytest.mark.offline
def test_ssl_verify_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ignore local certs/corporate-ca.crt so this asserts the no-CA path.
    monkeypatch.setattr(
        "memgraphrag.utils.http_ssl.find_configured_ca_path", lambda: None
    )
    reset_ssl_verify_cache()
    assert ssl_verify() is True


@pytest.mark.offline
def test_ssl_context_with_custom_ca(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Minimal self-signed-looking PEM is enough for load_verify_locations;
    # use certifi's first cert by copying a slice of the bundle.
    import certifi

    bundle = Path(certifi.where()).read_text(encoding="utf-8")
    # first PEM block
    start = bundle.index("-----BEGIN CERTIFICATE-----")
    end = bundle.index("-----END CERTIFICATE-----", start) + len(
        "-----END CERTIFICATE-----"
    )
    ca = tmp_path / "corp.pem"
    ca.write_text(bundle[start:end] + "\n", encoding="utf-8")
    monkeypatch.setenv("MEMGRAPHRAG_SSL_CERT_FILE", str(ca))
    reset_ssl_verify_cache()
    verify = ssl_verify()
    assert isinstance(verify, ssl.SSLContext)
    assert find_configured_ca_path() == str(ca.resolve())
    info = describe_ssl_verify()
    assert info["kind"] == "sslcontext"
    assert info["ca_path"] == str(ca.resolve())
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        assert verify.verify_flags & ssl.VERIFY_X509_STRICT == 0


@pytest.mark.offline
def test_build_ssl_context_relaxes_strict() -> None:
    ctx = build_ssl_context(None)
    assert isinstance(ctx, ssl.SSLContext)
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        assert ctx.verify_flags & ssl.VERIFY_X509_STRICT == 0
