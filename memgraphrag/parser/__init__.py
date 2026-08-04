"""Document parsing layer for MemGraphRAG.

Importing this package registers the legacy parser. Docling registers itself
when ``DOCLING_ENDPOINT`` is configured (see ``parser.external.docling``).
"""

from memgraphrag.parser.base import BaseParser, ParseContext, ParseResult
from memgraphrag.parser.registry import (
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_LEGACY,
    ParserUnavailableError,
    available_engine_suffixes,
    get_parser,
    register_parser,
    supported_parser_engines,
)
from memgraphrag.parser.routing import (
    ParserDirectives,
    resolve_file_parser_engine,
    resolve_parser_directives,
)

# Side-effect: register LegacyParser
from memgraphrag.parser import legacy as _legacy  # noqa: F401
from memgraphrag.parser.external import docling as _docling  # noqa: F401

__all__ = [
    "BaseParser",
    "ParseContext",
    "ParseResult",
    "PARSER_ENGINE_DOCLING",
    "PARSER_ENGINE_LEGACY",
    "ParserDirectives",
    "ParserUnavailableError",
    "available_engine_suffixes",
    "get_parser",
    "register_parser",
    "resolve_file_parser_engine",
    "resolve_parser_directives",
    "supported_parser_engines",
]
