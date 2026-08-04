"""Unified parser contract for MemGraphRAG engines.

Adapted from LightRAG ``lightrag/parser/base.py`` — MemGraphRAG-native naming
and a leaner ``ParseContext`` without LightRAG pipeline coupling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParseContext:
    """Inputs handed to :meth:`BaseParser.parse`."""

    doc_id: str
    file_path: str
    content_data: dict[str, Any] = field(default_factory=dict)
    rag: Any = None
    parsed_dir: Path | None = None

    def source_path(self) -> Path:
        """Resolve the on-disk source file for this document."""
        source = self.content_data.get("source_file") or self.file_path
        return Path(source)

    def resolve_parsed_dir(self) -> Path:
        """Return ``parsed_dir``, or derive ``<parent>/__parsed__/<stem>.parsed``."""
        if self.parsed_dir is not None:
            return Path(self.parsed_dir)
        source = self.source_path()
        base = source.stem or self.doc_id
        return source.parent / "__parsed__" / f"{base}.parsed"


@dataclass
class ParseResult:
    """Structured parser output."""

    doc_id: str
    file_path: str
    parse_format: str
    content: str
    blocks_path: str = ""
    parse_engine: str | None = None
    parse_warnings: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "parse_format": self.parse_format,
            "content": self.content,
            "blocks_path": self.blocks_path,
        }
        if self.parse_engine is not None:
            out["parse_engine"] = self.parse_engine
        if self.parse_warnings:
            out["parse_warnings"] = self.parse_warnings
        return out


class BaseParser(ABC):
    """Abstract base for every parser engine."""

    engine_name: str

    @abstractmethod
    async def parse(self, ctx: ParseContext) -> ParseResult:
        """Parse one document and return its :class:`ParseResult`."""
        ...
