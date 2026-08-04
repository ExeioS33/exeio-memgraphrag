"""TLS verify helpers for outbound HTTPS (OpenAI client, Docling, clients).

Corporate TLS inspection (e.g. Fortinet) requires a custom CA. httpx/openai use
certifi by default and ignore ``SSL_CERT_FILE``, so callers must pass
``verify=ssl_verify()``.

On OpenSSL 3 / Python 3.13, some inspection CAs trigger
``certificate verify failed: Missing Authority Key Identifier`` when
``VERIFY_X509_STRICT`` is enabled. When a custom CA is configured we therefore
build an ``ssl.SSLContext`` that:

1. Trusts certifi *plus* the custom CA (merged), and
2. Clears ``VERIFY_X509_STRICT`` so MITM chains without AKI still verify.

Env:
  SSL_VERIFY — ``true``/``false`` (default true). When false, verification is off.
  SSL_CERT_FILE / REQUESTS_CA_BUNDLE / MEMGRAPHRAG_SSL_CERT_FILE /
  MEMGRAPHRAG_CORP_CA_FILE / CURL_CA_BUNDLE — PEM file used as extra trust.
  Relative paths are resolved against the process cwd, then the package repo
  ``certs/`` directory when present.
"""

from __future__ import annotations

import logging
import os
import ssl
from functools import lru_cache
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

VerifyType = Union[bool, str, ssl.SSLContext]

_ENV_CA_KEYS = (
    "MEMGRAPHRAG_SSL_CERT_FILE",
    "MEMGRAPHRAG_CORP_CA_FILE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def _truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_ca_path(raw: str) -> str | None:
    """Return an existing CA path, trying cwd-relative and repo certs/ fallbacks."""
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        # repo_root/certs/<name> when running from the package checkout
        here = Path(__file__).resolve()
        repo_root = here.parents[2]  # .../memgraphrag/memgraphrag/utils → repo
        candidates.append(repo_root / raw)
        candidates.append(repo_root / "certs" / path.name)
    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand.resolve())
        except OSError:
            continue
    return None


def find_configured_ca_path() -> str | None:
    """Return the first configured CA path that exists on disk, else None."""
    for key in _ENV_CA_KEYS:
        raw = (os.getenv(key) or "").strip()
        if not raw:
            continue
        resolved = _resolve_ca_path(raw)
        if resolved:
            return resolved
        logger.warning("%s=%s does not exist; ignoring", key, raw)

    # Convention used by Compose / local labs
    for candidate in (
        Path("certs/corporate-ca.crt"),
        Path(__file__).resolve().parents[2] / "certs" / "corporate-ca.crt",
        Path("/app/certs/corporate-ca.crt"),
    ):
        try:
            if candidate.is_file():
                return str(candidate.resolve())
        except OSError:
            continue
    return None


def build_ssl_context(ca_path: str | None = None) -> ssl.SSLContext:
    """Build an httpx-compatible SSLContext with optional extra CA + OpenSSL3 relax.

    Always loads the system/certifi defaults, then appends ``ca_path`` when given.
    """
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover — certifi nearly always present
        ctx = ssl.create_default_context()

    if ca_path:
        ctx.load_verify_locations(cafile=ca_path)

    # OpenSSL 3 / CPython 3.13: Fortinet (and similar) inspection CAs often omit
    # Authority Key Identifier on intermediates. Strict mode then rejects them.
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT

    return ctx


@lru_cache(maxsize=1)
def ssl_verify() -> VerifyType:
    """Return httpx/openai ``verify`` value: True, False, path, or SSLContext."""
    if not _truthy(os.getenv("SSL_VERIFY"), default=True):
        logger.warning("SSL_VERIFY=false — outbound TLS certificate checks disabled")
        return False

    ca_path = find_configured_ca_path()
    if ca_path:
        logger.info("Using custom TLS CA bundle: %s", ca_path)
        return build_ssl_context(ca_path)

    return True


def reset_ssl_verify_cache() -> None:
    """Test helper / UI helper to clear the cached verify decision."""
    ssl_verify.cache_clear()


def describe_ssl_verify() -> dict[str, Any]:
    """Non-secret summary for CLI/UI diagnostics."""
    verify = ssl_verify()
    if verify is False:
        kind = "disabled"
    elif isinstance(verify, ssl.SSLContext):
        kind = "sslcontext"
    elif isinstance(verify, str):
        kind = "ca_path"
    else:
        kind = "default"
    return {
        "kind": kind,
        "ca_path": find_configured_ca_path(),
        "ssl_verify_env": os.getenv("SSL_VERIFY"),
        "env_keys_set": [k for k in _ENV_CA_KEYS if (os.getenv(k) or "").strip()],
    }
