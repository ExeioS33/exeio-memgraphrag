"""Paragraph-semantic chunking — the MemGraphRAG ``P`` strategy.

Adapted from LightRAG ``lightrag/chunker/paragraph_semantic.py``.
Reads ``blocks.jsonl`` when present; otherwise falls back to the ``R`` chunker.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    def encode(self, content: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


def _load_blocks_from_jsonl(blocks_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(blocks_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "content":
                rows.append(obj)
    return rows


def _merge_blocks_by_token_size(
    tokenizer: Tokenizer,
    rows: list[dict[str, Any]],
    chunk_token_size: int,
) -> list[dict[str, Any]]:
    """Greedy merge of content blocks up to ``chunk_token_size`` tokens."""
    target = max(int(chunk_token_size), 1)
    results: list[dict[str, Any]] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_heading: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal buf, buf_tokens, buf_heading
        if not buf:
            return
        body = "\n\n".join(buf).strip()
        if body:
            item: dict[str, Any] = {
                "tokens": len(tokenizer.encode(body)),
                "content": body,
                "chunk_order_index": len(results),
            }
            if buf_heading is not None:
                item["heading"] = buf_heading
            results.append(item)
        buf = []
        buf_tokens = 0
        buf_heading = None

    for row in rows:
        text = str(row.get("content") or "").strip()
        if not text:
            continue
        n = len(tokenizer.encode(text))
        # Oversized single blocks must be window-split (Docling/P path).
        if n > target:
            flush()
            from memgraphrag.chunker.token_size import chunking_by_token_size

            for piece in chunking_by_token_size(
                tokenizer,
                text,
                chunk_token_size=target,
                chunk_overlap_token_size=min(100, max(target // 10, 0)),
            ):
                piece["chunk_order_index"] = len(results)
                piece["heading"] = {
                    "level": int(row.get("level") or 0),
                    "heading": str(row.get("heading") or ""),
                    "parent_headings": list(row.get("parent_headings") or []),
                }
                results.append(piece)
            continue
        if buf and buf_tokens + n > target:
            flush()
        if not buf:
            buf_heading = {
                "level": int(row.get("level") or 0),
                "heading": str(row.get("heading") or ""),
                "parent_headings": list(row.get("parent_headings") or []),
            }
        buf.append(text)
        buf_tokens += n
        if n >= target:
            flush()
    flush()
    return results


def chunking_by_paragraph_semantic(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 2000,
    *,
    blocks_path: str | None = None,
    chunk_overlap_token_size: int = 100,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Paragraph-semantic chunker (``P``). Falls back to ``R`` without sidecar."""
    rows: list[dict[str, Any]] = []
    fallback_reason: str | None = None

    if not blocks_path:
        fallback_reason = "blocks_path is empty"
    else:
        try:
            rows = _load_blocks_from_jsonl(blocks_path)
        except OSError as exc:
            fallback_reason = f"cannot read blocks.jsonl at {blocks_path}: {exc}"
        else:
            if not rows:
                fallback_reason = f"blocks.jsonl at {blocks_path} contains no content rows"

    if fallback_reason is not None:
        logger.warning(
            "[paragraph_semantic] %s (doc_id=%s); falling back to recursive-character",
            fallback_reason,
            doc_id or "unknown",
        )
        from memgraphrag.chunker.recursive_character import (
            chunking_by_recursive_character,
        )

        overlap = max(int(chunk_overlap_token_size), 0)
        target = max(int(chunk_token_size), 1)
        if overlap >= target:
            overlap = max(target - 1, 0)
        return chunking_by_recursive_character(
            tokenizer,
            content,
            target,
            chunk_overlap_token_size=overlap,
        )

    return _merge_blocks_by_token_size(tokenizer, rows, chunk_token_size)
