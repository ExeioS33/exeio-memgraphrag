"""Framework-aligned logging for MemGraphRAG server operators.

Engine phases follow the research pipeline
([XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG))::

    [INDEX] Memory-based Indexing Graph Construction | chunks=16
    [STAGE] Performing OpenIE | cached=10 extract=6
    [LLM] agent=openie.ner action=complete model=… prompt_chars=420
    [STAGE] Extracting schema | batches=3
    [STAGE] Graph construction completed! | passages=99 facts=2115
    [RETRIEVE] Memory-guided Online Retrieval | mode=ppr query=…
    [STAGE] Preparing for fast retrieval.
    [STAGE] Retrieving | linking_top_k=50
    [EMBED] context=query_to_fact n=1
    [LLM] agent=qa.reading action=complete …
    [DONE] …
    [FAIL] …

HTTP / file-pipeline plumbing may still use ``[MAIN]`` / ``[STEP]`` for
request boundaries (upload, parse, chunk). Prefer ``phase`` / ``stage`` /
``llm_call`` / ``embed_call`` inside the MemGraphRAG engine.

Never log secrets, full prompts, or huge document bodies. Prefer ids,
counts, lengths, and truncated text via :func:`truncate`.
"""

from __future__ import annotations

import logging
from typing import Any

DEFAULT_TRUNCATE = 160

# Paper / framework phase titles (operator-facing banners).
INDEX_PHASE = "Memory-based Indexing Graph Construction"
RETRIEVE_PHASE = "Memory-guided Online Retrieval"


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
    """Log the start of an API / pipeline flow (``[MAIN]``)."""
    logger.info("[MAIN] %s%s", name, _format_fields(fields))


def sub_step(logger: logging.Logger, name: str, **fields: Any) -> None:
    """Log a sub-step inside an API / pipeline flow (``[STEP]``)."""
    logger.info("[STEP] %s%s", name, _format_fields(fields))


def phase(
    logger: logging.Logger,
    title: str,
    *,
    kind: str | None = None,
    **fields: Any,
) -> None:
    """Log a framework phase banner (``[INDEX]`` / ``[RETRIEVE]``).

    ``kind`` defaults from ``title`` when it matches :data:`INDEX_PHASE` or
    :data:`RETRIEVE_PHASE`.
    """
    if kind is None:
        if title == INDEX_PHASE or "Indexing" in title:
            kind = "INDEX"
        elif title == RETRIEVE_PHASE or "Retrieval" in title:
            kind = "RETRIEVE"
        else:
            kind = "PHASE"
    logger.info("[%s] %s%s", kind, title, _format_fields(fields))


def stage(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log a framework stage line (``[STAGE]``), matching upstream wording."""
    logger.info("[STAGE] %s%s", message, _format_fields(fields))


def llm_call(
    logger: logging.Logger,
    *,
    agent: str,
    action: str = "complete",
    **fields: Any,
) -> None:
    """Log an agent / LLM invocation (``[LLM]``) without prompt bodies."""
    logger.info(
        "[LLM] agent=%s action=%s%s",
        agent,
        action,
        _format_fields(fields),
    )


def embed_call(logger: logging.Logger, *, context: str, **fields: Any) -> None:
    """Log an embedding invocation (``[EMBED]``)."""
    logger.info("[EMBED] context=%s%s", context, _format_fields(fields))


def done_step(logger: logging.Logger, name: str, **fields: Any) -> None:
    """Log successful completion (``[DONE]``)."""
    logger.info("[DONE] %s%s", name, _format_fields(fields))


def fail_step(
    logger: logging.Logger,
    name: str,
    *,
    exc: BaseException | None = None,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Log failure of a main/sub/stage step (``[FAIL]``).

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
    "INDEX_PHASE",
    "RETRIEVE_PHASE",
    "done_step",
    "embed_call",
    "fail_step",
    "llm_call",
    "main_step",
    "phase",
    "stage",
    "sub_step",
    "truncate",
]
