"""MemGraphRAG chunking strategies (F / R / P).

Adapted from LightRAG ``lightrag/chunker/__init__.py``.
"""

from __future__ import annotations

from typing import Any, Callable

from memgraphrag.chunker.paragraph_semantic import chunking_by_paragraph_semantic
from memgraphrag.chunker.recursive_character import chunking_by_recursive_character
from memgraphrag.chunker.token_size import (
    chunking_by_fixed_token,
    chunking_by_token_size,
)

ChunkerFn = Callable[..., list[dict[str, Any]]]

_CHUNKERS: dict[str, ChunkerFn] = {
    "F": chunking_by_fixed_token,
    "R": chunking_by_recursive_character,
    "P": chunking_by_paragraph_semantic,
}


def get_chunker(strategy: str) -> ChunkerFn:
    """Return the chunker callable for strategy ``F``, ``R``, or ``P``."""
    key = (strategy or "F").strip().upper()
    try:
        return _CHUNKERS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r}; expected one of F, R, P"
        ) from exc


__all__ = [
    "chunking_by_fixed_token",
    "chunking_by_paragraph_semantic",
    "chunking_by_recursive_character",
    "chunking_by_token_size",
    "get_chunker",
]
