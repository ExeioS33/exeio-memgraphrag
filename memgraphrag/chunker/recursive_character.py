"""Recursive character chunking — the MemGraphRAG ``R`` strategy.

Adapted from LightRAG ``lightrag/chunker/recursive_character.py``.
Uses ``langchain-text-splitters`` when available; otherwise a simple
recursive separator fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]

DEFAULT_R_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]


class Tokenizer(Protocol):
    def encode(self, content: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


def _simple_recursive_split(
    text: str,
    *,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
    length_fn,
) -> list[str]:
    """Minimal recursive splitter used when langchain is unavailable."""
    if not text:
        return []
    if length_fn(text) <= chunk_size:
        return [text] if text.strip() else []

    seps = list(separators) if separators else [""]
    separator = seps[-1]
    next_seps: list[str] = []
    for i, candidate in enumerate(seps):
        if candidate == "" or candidate in text:
            separator = candidate
            next_seps = seps[i + 1 :]
            break

    if separator:
        parts = text.split(separator)
    else:
        # Character-level fallback
        parts = list(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = length_fn(separator) if separator else 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        joined = separator.join(current) if separator else "".join(current)
        body = joined.strip()
        if body:
            chunks.append(body)
        current = []
        current_len = 0

    for part in parts:
        if not part and separator:
            continue
        part_len = length_fn(part)
        if part_len > chunk_size and next_seps:
            flush()
            chunks.extend(
                _simple_recursive_split(
                    part,
                    separators=next_seps,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_fn=length_fn,
                )
            )
            continue
        extra = sep_len if current else 0
        if current and current_len + extra + part_len > chunk_size:
            flush()
            # Soft overlap: keep trailing fragment when overlap > 0
            if chunk_overlap > 0 and chunks:
                # Overlap is approximate (token-based); skip complex rebuild.
                pass
        current.append(part)
        current_len += extra + part_len
    flush()
    return chunks


def chunking_by_recursive_character(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    chunk_overlap_token_size: int = 100,
    separators: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recursive character splitter — the ``R`` chunking strategy."""
    if not content or not content.strip():
        return []

    def length_function(text: str) -> int:
        return len(tokenizer.encode(text))

    seps = list(separators) if separators is not None else list(DEFAULT_R_SEPARATORS)
    chunk_size = max(int(chunk_token_size), 1)
    chunk_overlap = max(int(chunk_overlap_token_size), 0)
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size - 1, 0)

    if _LANGCHAIN_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=seps,
            strip_whitespace=True,
        )
        pieces = splitter.split_text(content)
    else:
        logger.info(
            "[recursive_character] langchain-text-splitters not installed; "
            "using simple recursive fallback"
        )
        pieces = _simple_recursive_split(
            content,
            separators=seps,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_fn=length_function,
        )

    results: list[dict[str, Any]] = []
    for piece in pieces:
        body = piece.strip()
        if not body:
            continue
        results.append(
            {
                "tokens": length_function(body),
                "content": body,
                "chunk_order_index": len(results),
            }
        )

    if not results:
        body = content.strip()
        if body:
            results.append(
                {
                    "tokens": length_function(body),
                    "content": body,
                    "chunk_order_index": 0,
                }
            )
    return results
