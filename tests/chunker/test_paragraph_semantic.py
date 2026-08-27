"""Tests for paragraph-semantic (P) chunker fallback."""

from __future__ import annotations

import pytest

from memgraphrag.chunker.paragraph_semantic import chunking_by_paragraph_semantic
from memgraphrag.chunker.recursive_character import chunking_by_recursive_character
from memgraphrag.utils.tokenizer import TiktokenTokenizer

pytestmark = pytest.mark.offline


def test_paragraph_semantic_falls_back_to_r_without_sidecar() -> None:
    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    content = (
        "First paragraph about Alice.\n\n"
        "Second paragraph about Bob.\n\n"
        "Third paragraph about Carol and Dave together."
    )

    p_chunks = chunking_by_paragraph_semantic(
        tokenizer,
        content,
        chunk_token_size=40,
        blocks_path=None,
        chunk_overlap_token_size=5,
    )
    r_chunks = chunking_by_recursive_character(
        tokenizer,
        content,
        40,
        chunk_overlap_token_size=5,
    )

    assert p_chunks
    assert [c["content"] for c in p_chunks] == [c["content"] for c in r_chunks]
