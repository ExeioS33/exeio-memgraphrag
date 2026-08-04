"""TLS verify helpers for outbound HTTPS (OpenAI client, Docling, etc.).

Corporate TLS inspection (e.g. Fortinet) requires a custom CA. httpx/openai use
certifi by default and ignore ``SSL_CERT_FILE``, so callers must pass
``verify=ssl_verify()``.

Env:
  SSL_VERIFY — ``true``/``false`` (default true). When false, verification is off.
  SSL_CERT_FILE / REQUESTS_CA_BUNDLE / MEMGRAPHRAG_SSL_CERT_FILE — PEM file or
  directory used as the verify trust store when set.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def _truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def ssl_verify() -> Any:
    """Return httpx/openai ``verify`` value: True, False, or a CA path string."""
    if not _truthy(os.getenv("SSL_VERIFY"), default=True):
        logger.warning("SSL_VERIFY=false — outbound TLS certificate checks disabled")
        return False

    for key in (
        "MEMGRAPHRAG_SSL_CERT_FILE",
        "MEMGRAPHRAG_CORP_CA_FILE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        path = (os.getenv(key) or "").strip()
        if path:
            if os.path.exists(path):
                logger.info("Using custom TLS CA bundle from %s=%s", key, path)
                return path
            logger.warning("%s=%s does not exist; falling back to default verify", key, path)

    return True


def reset_ssl_verify_cache() -> None:
    """Test helper to clear the cached verify decision."""
    ssl_verify.cache_clear()
