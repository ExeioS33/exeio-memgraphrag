"""Structured Main/Step logging helpers for MemGraphRAG server operators.

Convention (grep-friendly)::

    [MAIN] ingest.process | pending=3
    [STEP] ingest.process.parse | doc_id=doc-abc engine=legacy
    [DONE] ingest.process | ok processed=2 failed=0
    [FAIL] ingest.process | doc_id=doc-abc error=...

Never log secrets, full prompts, or huge document bodies. Prefer ids,
counts, lengths, and truncated text via :func:`truncate`.
"""

from __future__ import annotations

import logging
from typing import Any

DEFAULT_TRUNCATE = 160


def truncate(value: Any, limit: int = DEFAULT_TRUNCATE) -> str:
    """Return a single-line preview truncated to ``limit`` characters."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _format_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    if not parts:
        return ""
    return " | " + " ".join(parts)


def main_step(logger: logging.Logger, name: str, **fields: Any) -> None:
    """Log the start of a major flow (``[MAIN]``)."""
    logger.info("[MAIN] %s%s", name, _format_fields(fields))


def sub_step(logger: logging.Logger, name: str, **fields: Any) -> None:
    """Log a sub-step inside a main flow (``[STEP]``)."""
    logger.info("[STEP] %s%s", name, _format_fields(fields))


def done_step(logger: logging.Logger, name: str, **fields: Any) -> None:
    """Log successful completion of a main flow (``[DONE]``)."""
    logger.info("[DONE] %s%s", name, _format_fields(fields))


def fail_step(
    logger: logging.Logger,
    name: str,
    *,
    exc: BaseException | None = None,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Log failure of a main/sub step (``[FAIL]``).

    Pass ``exc_info=True`` (or set ``exc``) to attach a traceback via
    ``logger.exception`` / ``logger.error``.
    """
    if exc is not None and "error" not in fields:
        fields["error"] = truncate(exc, 200)
    msg = "[FAIL] %s%s" % (name, _format_fields(fields))
    if exc_info:
        logger.exception(msg)
    else:
        logger.warning(msg)


__all__ = [
    "DEFAULT_TRUNCATE",
    "done_step",
    "fail_step",
    "main_step",
    "sub_step",
    "truncate",
]
