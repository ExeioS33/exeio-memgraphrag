"""Legacy engine adapter: in-process plain-text extraction (RAW output).

Adapted from LightRAG ``lightrag/parser/legacy/parser.py``.
"""

from __future__ import annotations

import asyncio
import os

from memgraphrag.parser.base import BaseParser, ParseContext, ParseResult
from memgraphrag.parser.registry import (
    PARSER_ENGINE_LEGACY,
    ParserSpec,
    register_parser,
    suffix_capabilities,
)

FULL_DOCS_FORMAT_RAW = "raw"


class LegacyParser(BaseParser):
    """Extract plain text in-process and return a ``raw`` document."""

    engine_name = PARSER_ENGINE_LEGACY

    async def parse(self, ctx: ParseContext) -> ParseResult:
        from memgraphrag.parser.legacy.extractors import (
            LegacyExtractionError,
            extract_text,
        )

        source = ctx.source_path()
        if not source.is_file():
            # Allow content already supplied in content_data
            inline = ctx.content_data.get("content")
            if isinstance(inline, str) and inline.strip():
                return ParseResult(
                    doc_id=ctx.doc_id,
                    file_path=ctx.file_path,
                    parse_format=FULL_DOCS_FORMAT_RAW,
                    content=inline,
                    blocks_path="",
                    parse_engine=self.engine_name,
                )
            raise FileNotFoundError(f"legacy source file not found: {source}")

        suffix = source.suffix.lower().lstrip(".")
        if suffix and suffix not in suffix_capabilities(self.engine_name):
            raise ValueError(
                f"legacy parser does not support .{suffix}: "
                f"doc_id={ctx.doc_id} file={ctx.file_path}"
            )

        file_bytes = await asyncio.to_thread(source.read_bytes)
        pdf_password = os.getenv("PDF_DECRYPT_PASSWORD") or None
        text = await asyncio.to_thread(
            extract_text, file_bytes, suffix or "txt", pdf_password=pdf_password
        )
        if not text.strip():
            raise LegacyExtractionError(
                f"extracted no usable text from {ctx.file_path} (doc_id={ctx.doc_id})"
            )

        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_RAW,
            content=text,
            blocks_path="",
            parse_engine=self.engine_name,
        )


# Ensure registry entry is present when this module is imported.
_LEGACY_SUFFIXES = suffix_capabilities(PARSER_ENGINE_LEGACY) or frozenset(
    {"txt", "md", "pdf", "docx", "pptx", "xlsx"}
)
register_parser(
    ParserSpec(
        engine_name=PARSER_ENGINE_LEGACY,
        impl="memgraphrag.parser.legacy.parser:LegacyParser",
        suffixes=_LEGACY_SUFFIXES,
    )
)
