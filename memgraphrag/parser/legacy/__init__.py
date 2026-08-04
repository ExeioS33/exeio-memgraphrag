"""Legacy parser engine: simple in-process text extraction (no sidecar)."""

from memgraphrag.parser.legacy.extractors import (
    LegacyExtractionError,
    extract_text,
)
from memgraphrag.parser.legacy.parser import LegacyParser

__all__ = ["LegacyExtractionError", "LegacyParser", "extract_text"]
