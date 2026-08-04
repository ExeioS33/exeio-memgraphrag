"""Fixed-size token-window chunking — the MemGraphRAG ``F`` strategy.

Adapted from LightRAG ``lightrag/chunker/token_size.py``.
"""

from __future__ import annotations

from typing import Any, Protocol


class Tokenizer(Protocol):
    def encode(self, content: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


def _window_step(chunk_token_size: int, chunk_overlap_token_size: int) -> int:
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError(
            f"chunk_overlap_token_size ({chunk_overlap_token_size}) must be < "
            f"chunk_token_size ({chunk_token_size})"
        )
    return chunk_token_size - chunk_overlap_token_size


def _make_chunk(*, content: str, tokens: int, order: int) -> dict[str, Any]:
    return {
        "tokens": tokens,
        "content": content.strip(),
        "chunk_order_index": order,
    }


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """Fixed-token window chunker (``F`` strategy)."""
    if not content or not content.strip():
        return []

    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []

    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks: list[tuple[int, str]] = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    raise ValueError(
                        f"Chunk exceeds token limit: {len(_tokens)} > {chunk_token_size}"
                    )
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    step = _window_step(chunk_token_size, chunk_overlap_token_size)
                    for start in range(0, len(_tokens), step):
                        end = min(start + chunk_token_size, len(_tokens))
                        chunk_content = tokenizer.decode(_tokens[start:end])
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(_make_chunk(content=chunk, tokens=_len, order=index))
    else:
        step = _window_step(chunk_token_size, chunk_overlap_token_size)
        for index, start in enumerate(range(0, len(tokens), step)):
            end = min(start + chunk_token_size, len(tokens))
            chunk_content = tokenizer.decode(tokens[start:end])
            results.append(
                _make_chunk(
                    content=chunk_content,
                    tokens=min(chunk_token_size, len(tokens) - start),
                    order=index,
                )
            )
    return results


def chunking_by_fixed_token(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    chunk_overlap_token_size: int = 100,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
) -> list[dict[str, Any]]:
    """File-chunker contract for the ``F`` strategy."""
    return chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        chunk_overlap_token_size=chunk_overlap_token_size,
        chunk_token_size=chunk_token_size,
    )
