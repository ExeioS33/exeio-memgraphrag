"""Langfuse tracing helpers for MemGraphRAG retrieval.

Enabled when ``LANGFUSE_ENABLE_TRACE`` is truthy and both
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are set. Host/base URL
uses ``LANGFUSE_BASE_URL`` or legacy ``LANGFUSE_HOST``.

All helpers no-op when disabled or when the ``langfuse`` package is missing,
so retrieval stays functional without the optional dependency.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from memgraphrag.utils.env import get_env_value

logger = logging.getLogger(__name__)

_CLIENT: Any = None
_INIT_ATTEMPTED = False


def is_langfuse_enabled() -> bool:
    """Return True when tracing is opted in and API keys are present."""
    if not get_env_value("LANGFUSE_ENABLE_TRACE", False, bool):
        return False
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public_key and secret_key)


def get_langfuse_client() -> Any | None:
    """Lazy-init a shared Langfuse client, or ``None`` when disabled/unavailable."""
    global _CLIENT, _INIT_ATTEMPTED
    if not is_langfuse_enabled():
        return None
    if _CLIENT is not None:
        return _CLIENT
    if _INIT_ATTEMPTED:
        return None
    _INIT_ATTEMPTED = True
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning(
            "LANGFUSE_ENABLE_TRACE is set but langfuse is not installed; "
            "install memgraphrag[api] / `pip install langfuse`"
        )
        return None

    kwargs: dict[str, Any] = {
        "public_key": os.environ["LANGFUSE_PUBLIC_KEY"].strip(),
        "secret_key": os.environ["LANGFUSE_SECRET_KEY"].strip(),
    }
    base_url = (os.getenv("LANGFUSE_BASE_URL") or "").strip() or (
        os.getenv("LANGFUSE_HOST") or ""
    ).strip()
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")

    try:
        _CLIENT = Langfuse(**kwargs)
        logger.info(
            "Langfuse tracing enabled (host=%s)",
            base_url or "default",
        )
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse client: %s", exc)
        _CLIENT = None
    return _CLIENT


def reset_langfuse_client_for_tests() -> None:
    """Clear cached client (unit tests only)."""
    global _CLIENT, _INIT_ATTEMPTED
    _CLIENT = None
    _INIT_ATTEMPTED = False


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> Iterator[Any | None]:
    """Create a Langfuse observation, or yield ``None`` when tracing is off.

    Prefer ``as_type=\"retriever\"`` for retrieval stages and
    ``as_type=\"generation\"`` for LLM calls.
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    kwargs: dict[str, Any] = {"name": name, "as_type": as_type}
    if input is not None:
        kwargs["input"] = input
    if metadata:
        kwargs["metadata"] = metadata
    if model is not None:
        kwargs["model"] = model

    try:
        cm = client.start_as_current_observation(**kwargs)
    except TypeError:
        # Older SDKs used start_as_current_span / start_as_current_generation.
        if as_type == "generation" and hasattr(client, "start_as_current_generation"):
            cm = client.start_as_current_generation(
                name=name, input=input, metadata=metadata, model=model
            )
        elif hasattr(client, "start_as_current_span"):
            cm = client.start_as_current_span(name=name, input=input, metadata=metadata)
        else:
            logger.debug("Langfuse client missing observation APIs; skipping %s", name)
            yield None
            return

    # Do NOT wrap `with cm as span: yield span` in a try/except that yields again.
    # When the caller's body raises, contextlib throws into the generator at the
    # yield; catching it and yielding a second time makes Python raise
    # "RuntimeError: generator didn't stop after throw()", which replaced every real
    # engine exception (timeouts, 429s) with an opaque error whenever tracing was on.
    # Enter and exit are guarded separately so only Langfuse's own failures are
    # swallowed; the caller's exception always propagates with its original type.
    try:
        span = cm.__enter__()
    except Exception as exc:
        logger.debug("Langfuse observation %s failed open: %s", name, exc)
        yield None
        return

    try:
        yield span
    except BaseException:
        try:
            cm.__exit__(*sys.exc_info())
        except Exception as exc:
            logger.debug("Langfuse observation %s failed to close: %s", name, exc)
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception as exc:
            logger.debug("Langfuse observation %s failed to close: %s", name, exc)


def update_observation(
    span: Any | None,
    *,
    output: Any | None = None,
    metadata: dict[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
) -> None:
    """Best-effort update of an active observation."""
    if span is None:
        return
    payload: dict[str, Any] = {}
    if output is not None:
        payload["output"] = output
    if metadata:
        payload["metadata"] = metadata
    if level is not None:
        payload["level"] = level
    if status_message is not None:
        payload["status_message"] = status_message
    if not payload:
        return
    try:
        span.update(**payload)
    except Exception as exc:
        logger.debug("Langfuse span.update failed: %s", exc)


def flush_langfuse() -> None:
    """Flush buffered events (call after a query request)."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.debug("Langfuse flush failed: %s", exc)


def truncate_docs(docs: list[str], *, max_docs: int = 5, max_chars: int = 400) -> list[str]:
    """Compact passage snippets for Langfuse payloads (avoid huge traces)."""
    out: list[str] = []
    for doc in docs[:max_docs]:
        text = doc if isinstance(doc, str) else str(doc)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        out.append(text)
    if len(docs) > max_docs:
        out.append(f"...(+{len(docs) - max_docs} more)")
    return out
