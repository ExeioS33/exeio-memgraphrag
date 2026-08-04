"""Docling parser integration for MemGraphRAG.

Registers ``DoclingParser`` in the registry. The engine is only usable when
``DOCLING_ENDPOINT`` is configured — :func:`get_parser` raises
:class:`ParserUnavailableError` otherwise.
"""

from __future__ import annotations

import os

from memgraphrag.parser.registry import (
    PARSER_ENGINE_DOCLING,
    ParserSpec,
    register_parser,
)

_DOCLING_SUFFIXES = frozenset(
    {
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "md",
        "html",
        "xhtml",
        "png",
        "jpg",
        "jpeg",
        "tiff",
        "webp",
        "bmp",
    }
)


def _docling_endpoint_configured() -> bool:
    return bool(os.getenv("DOCLING_ENDPOINT", "").strip())


# Always register so capability queries / get_parser gating work; the
# endpoint_configured gate keeps it unavailable until DOCLING_ENDPOINT is set.
register_parser(
    ParserSpec(
        engine_name=PARSER_ENGINE_DOCLING,
        impl="memgraphrag.parser.external.docling.parser:DoclingParser",
        suffixes=_DOCLING_SUFFIXES,
        endpoint_configured=_docling_endpoint_configured,
        endpoint_requirement=lambda: "DOCLING_ENDPOINT",
        extra_suffixes_env="DOCLING_ADDITIONAL_SUFFIXES",
    )
)

# Eager-import the class only when the endpoint is configured so cold starts
# without Docling stay cheap.
if os.getenv("DOCLING_ENDPOINT", "").strip():
    from memgraphrag.parser.external.docling.parser import DoclingParser  # noqa: F401

__all__ = ["PARSER_ENGINE_DOCLING"]
