"""Legacy engine adapter: in-process plain-text extraction (RAW output).

Adapted from LightRAG ``lightrag/parser/legacy/parser.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os

from memgraphrag.parser.base import BaseParser, ParseContext, ParseResult
from memgraphrag.parser.registry import (
    PARSER_ENGINE_LEGACY,
    ParserSpec,
    register_parser,
    suffix_capabilities,
)
from memgraphrag.utils.step_log import done_step, fail_step, main_step, sub_step

logger = logging.getLogger(__name__)

FULL_DOCS_FORMAT_RAW = "raw"


class LegacyParser(BaseParser):
    """Extract plain text in-process and return a ``raw`` document."""

    engine_name = PARSER_ENGINE_LEGACY

    async def parse(self, ctx: ParseContext) -> ParseResult:
        from memgraphrag.parser.legacy.extractors import (
            LegacyExtractionError,
            extract_text,
        )

        main_step(
            logger,
            "parse.legacy",
            doc_id=ctx.doc_id,
            file=os.path.basename(ctx.file_path) or ctx.file_path,
        )
        source = ctx.source_path()
        if not source.is_file():
            # Allow content already supplied in content_data
            inline = ctx.content_data.get("content")
            if isinstance(inline, str) and inline.strip():
                sub_step(
                    logger,
                    "parse.legacy.inline",
                    doc_id=ctx.doc_id,
                    chars=len(inline),
                )
                done_step(
                    logger,
                    "parse.legacy",
                    doc_id=ctx.doc_id,
                    source="inline",
                    chars=len(inline),
                )
                return ParseResult(
                    doc_id=ctx.doc_id,
                    file_path=ctx.file_path,
                    parse_format=FULL_DOCS_FORMAT_RAW,
                    content=inline,
                    blocks_path="",
                    parse_engine=self.engine_name,
                )
            fail_step(
                logger,
                "parse.legacy",
                doc_id=ctx.doc_id,
                error="source_not_found",
            )
            raise FileNotFoundError(f"legacy source file not found: {source}")

        suffix = source.suffix.lower().lstrip(".")
        file_bytes = await asyncio.to_thread(source.read_bytes)
        sub_step(
            logger,
            "parse.legacy.read",
            doc_id=ctx.doc_id,
            suffix=suffix or "-",
            bytes=len(file_bytes),
        )
        # Defense in depth: arXiv-style names (``2605.18490v1``) may lack ``.pdf``
        # even though the payload is a real PDF. Prefer magic over a bogus suffix.
        if file_bytes[:4] == b"%PDF" or file_bytes[:5] == b"%PDF-":
            suffix = "pdf"
            sub_step(logger, "parse.legacy.magic", detected="pdf")
        elif suffix and suffix not in suffix_capabilities(self.engine_name):
            fail_step(
                logger,
                "parse.legacy",
                doc_id=ctx.doc_id,
                error=f"unsupported_suffix.{suffix}",
            )
            raise ValueError(
                f"legacy parser does not support .{suffix}: "
                f"doc_id={ctx.doc_id} file={ctx.file_path}"
            )

        pdf_password = os.getenv("PDF_DECRYPT_PASSWORD") or None
        sub_step(
            logger,
            "parse.legacy.extract_text",
            suffix=suffix or "txt",
            bytes=len(file_bytes),
        )
        text = await asyncio.to_thread(
            extract_text, file_bytes, suffix or "txt", pdf_password=pdf_password
        )
        if not text.strip():
            fail_step(
                logger,
                "parse.legacy",
                doc_id=ctx.doc_id,
                error="empty_extract",
            )
            raise LegacyExtractionError(
                f"extracted no usable text from {ctx.file_path} (doc_id={ctx.doc_id})"
            )

        done_step(
            logger,
            "parse.legacy",
            doc_id=ctx.doc_id,
            chars=len(text),
            suffix=suffix or "txt",
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
