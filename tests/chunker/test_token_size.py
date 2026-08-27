"""Tests for the fixed-token (F) chunker."""

from __future__ import annotations

import pytest

from memgraphrag.chunker.token_size import chunking_by_token_size
from memgraphrag.utils.tokenizer import TiktokenTokenizer

pytestmark = pytest.mark.offline


def test_token_size_overlapping_chunks() -> None:
    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    # Build content long enough to need multiple overlapping windows.
    content = " ".join(f"word{i}" for i in range(500))

    chunk_token_size = 50
    chunk_overlap_token_size = 10
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
    )

    assert len(chunks) >= 2
    assert all(c["tokens"] <= chunk_token_size for c in chunks)
    assert all(c["content"] for c in chunks)
    assert [c["chunk_order_index"] for c in chunks] == list(range(len(chunks)))

    # Verify sliding-window overlap on the raw token stream (before strip).
    all_tokens = tokenizer.encode(content)
    step = chunk_token_size - chunk_overlap_token_size
    expected_windows = list(range(0, len(all_tokens), step))
    assert len(chunks) == len(expected_windows)

    first = all_tokens[0:chunk_token_size]
    second = all_tokens[step : step + chunk_token_size]
    assert first[-chunk_overlap_token_size:] == second[:chunk_overlap_token_size]
    # Decoded chunk contents should reconstruct prefixes of the source.
    assert chunks[0]["content"] in content
    assert chunks[1]["content"] in content
