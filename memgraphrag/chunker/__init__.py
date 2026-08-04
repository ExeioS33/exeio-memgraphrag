"""MemGraphRAG chunking strategies (F / R / P).

Adapted from LightRAG ``lightrag/chunker/__init__.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from memgraphrag.chunker.paragraph_semantic import chunking_by_paragraph_semantic
from memgraphrag.chunker.recursive_character import chunking_by_recursive_character
from memgraphrag.chunker.token_size import (
    chunking_by_fixed_token,
    chunking_by_token_size,
)
from memgraphrag.utils.step_log import done_step, main_step, sub_step

logger = logging.getLogger(__name__)

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
        fn = _CHUNKERS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r}; expected one of F, R, P"
        ) from exc

    def _logged(
        tokenizer: Any,
        content: str,
        chunk_token_size: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        main_step(
            logger,
            "chunk.run",
            strategy=key,
            content_chars=len(content or ""),
            chunk_token_size=chunk_token_size,
            overlap=kwargs.get("chunk_overlap_token_size"),
        )
        if key == "P":
            sub_step(
                logger,
                "chunk.run.blocks",
                has_blocks_path=bool(kwargs.get("blocks_path")),
                doc_id=kwargs.get("doc_id") or "-",
            )
        chunks = fn(tokenizer, content, chunk_token_size, **kwargs)
        done_step(logger, "chunk.run", strategy=key, chunks=len(chunks))
        return chunks

    return _logged


__all__ = [
    "chunking_by_fixed_token",
    "chunking_by_paragraph_semantic",
    "chunking_by_recursive_character",
    "chunking_by_token_size",
    "get_chunker",
]
