"""Tests for the legacy parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from memgraphrag.parser.base import ParseContext
from memgraphrag.parser.legacy.parser import LegacyParser
from memgraphrag.parser.registry import get_parser

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_legacy_parser_txt(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Hello MemGraphRAG.\nSecond line.\n", encoding="utf-8")

    parser = LegacyParser()
    result = await parser.parse(ParseContext(doc_id="doc-1", file_path=str(path), content_data={}))

    assert result.parse_format == "raw"
    assert result.parse_engine == "legacy"
    assert "Hello MemGraphRAG" in result.content
    assert result.blocks_path == ""


@pytest.mark.asyncio
async def test_get_parser_legacy() -> None:
    parser = get_parser("legacy")
    assert parser.engine_name == "legacy"


def test_get_parser_docling_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    from memgraphrag.parser.registry import ParserUnavailableError

    with pytest.raises(ParserUnavailableError):
        get_parser("docling")
