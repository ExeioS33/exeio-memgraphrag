"""Parser engine routing from ``MEMGRAPHRAG_PARSER`` and filename hints.

Adapted from LightRAG ``lightrag/parser/routing.py`` — simplified to engine
+ process-options resolution without the full chunk-param DSL.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memgraphrag.parser.registry import (
    PARSER_ENGINE_LEGACY,
    engine_endpoint_configured,
    engine_endpoint_requirement,
    suffix_capabilities,
    supported_parser_engines,
)

# Trailing parser-hint: ``name.[engine].ext`` or ``name.[engine-options].ext``
_PARSER_HINT_RE = re.compile(r"\.\[([^\]]*)\](\.[^.]+)$")

PROCESS_OPTION_CHUNK_FIXED = "F"
PROCESS_OPTION_CHUNK_RECURSIVE = "R"
PROCESS_OPTION_CHUNK_PARAGRAPH = "P"
PROCESS_OPTION_CHUNK_CHARS = frozenset(
    {
        PROCESS_OPTION_CHUNK_FIXED,
        PROCESS_OPTION_CHUNK_RECURSIVE,
        PROCESS_OPTION_CHUNK_PARAGRAPH,
    }
)
SUPPORTED_PROCESS_OPTIONS = frozenset({"i", "t", "e", "!"} | PROCESS_OPTION_CHUNK_CHARS)


class ParserRoutingConfigError(ValueError):
    """Raised when ``MEMGRAPHRAG_PARSER`` contains an invalid routing rule."""


class FilenameParserHintError(ValueError):
    """Raised when a filename parser hint is invalid for ingestion."""


def normalize_parser_engine(engine: Any) -> str:
    """Normalize engine hints such as ``docling-ocr`` → ``docling``."""
    text = str(engine or "").strip()
    paren = text.find("(")
    if paren != -1:
        text = text[:paren]
    return text.split("-", 1)[0].strip().lower()


def sanitize_process_options(options: Any) -> str:
    if not options:
        return ""
    return "".join(ch for ch in str(options) if ch in SUPPORTED_PROCESS_OPTIONS)


def parse_chunking_strategy(process_options: Any) -> str:
    """Return the selected chunker char (F/R/P); default ``F``."""
    raw = sanitize_process_options(process_options)
    for ch in raw:
        if ch in PROCESS_OPTION_CHUNK_CHARS:
            return ch
    return PROCESS_OPTION_CHUNK_FIXED


def parser_suffix(file_path: str | Path) -> str:
    return Path(file_path).suffix.lower().lstrip(".")


def parser_engine_supports_suffix(engine: str, suffix: str) -> bool:
    return suffix.lower().lstrip(".") in suffix_capabilities(engine)


def parser_rules_from_env() -> str:
    return os.getenv("MEMGRAPHRAG_PARSER", "").strip()


def split_engine_and_options(bracket_inner: str) -> tuple[str | None, str]:
    """Decompose a bracket-hint inner string into ``(engine, options)``."""
    inner = (bracket_inner or "").strip()
    if not inner:
        return None, ""
    if inner.startswith("-"):
        return None, inner[1:].strip()
    if "-" in inner:
        head, _, tail = inner.partition("-")
        engine_candidate = normalize_parser_engine(head)
        if engine_candidate in supported_parser_engines():
            return engine_candidate, tail.strip()
        return None, ""
    engine_candidate = normalize_parser_engine(inner)
    if engine_candidate in supported_parser_engines():
        return engine_candidate, ""
    return None, ""


def _filename_hint_match(file_path: str | Path) -> tuple[str | None, str] | None:
    basename = Path(file_path).name
    m = _PARSER_HINT_RE.search(basename)
    if not m:
        return None
    engine, options = split_engine_and_options(m.group(1))
    if engine in supported_parser_engines():
        return engine, sanitize_process_options(options)
    if engine is None and options:
        return None, sanitize_process_options(options)
    return None


def filename_parser_directives(file_path: str | Path) -> tuple[str | None, str]:
    found = _filename_hint_match(file_path)
    if not found:
        return None, ""
    return found


def _engine_is_usable(
    engine: str,
    suffix: str,
    *,
    require_external_endpoint: bool,
) -> bool:
    if engine not in supported_parser_engines():
        return False
    if not parser_engine_supports_suffix(engine, suffix):
        return False
    if require_external_endpoint and not engine_endpoint_configured(engine):
        return False
    return True


def _validate_filename_hint(
    file_path: str | Path,
    *,
    require_external_endpoint: bool,
) -> None:
    basename = Path(file_path).name
    m = _PARSER_HINT_RE.search(basename)
    if not m:
        return
    inner = m.group(1).strip()
    if not inner:
        raise FilenameParserHintError(f"Invalid filename parser hint in {basename!r}: empty hint")
    engine, options = split_engine_and_options(inner)
    if engine is None and not options and not inner.startswith("-"):
        raise FilenameParserHintError(
            f"Invalid filename parser hint in {basename!r}: unsupported engine {inner!r}"
        )
    if engine and engine not in supported_parser_engines():
        supported = ", ".join(sorted(supported_parser_engines()))
        raise FilenameParserHintError(
            f"Invalid filename parser hint in {basename!r}: "
            f"unsupported engine {engine!r}; supported: {supported}"
        )
    if engine:
        suffix = parser_suffix(file_path)
        if not parser_engine_supports_suffix(engine, suffix):
            raise FilenameParserHintError(
                f"Invalid filename parser hint in {basename!r}: "
                f"engine {engine!r} does not support suffix {suffix!r}"
            )
        req = engine_endpoint_requirement(engine)
        if require_external_endpoint and req and not engine_endpoint_configured(engine):
            raise FilenameParserHintError(
                f"Invalid filename parser hint in {basename!r}: requires {req} to be configured"
            )


def _matching_rule_directives(
    file_path: str | Path,
    *,
    parser_rules: str | None,
    require_external_endpoint: bool,
) -> tuple[str | None, str]:
    suffix = parser_suffix(file_path)
    rules = parser_rules_from_env() if parser_rules is None else parser_rules.strip()
    if not rules:
        return None, ""
    for item in re.split(r"[;,]", rules):
        item = item.strip()
        if not item or ":" not in item:
            continue
        pattern, engine_hint = item.split(":", 1)
        pattern = pattern.strip().lower()
        engine_hint = engine_hint.strip()
        head, _, options_str = engine_hint.partition("-")
        engine = normalize_parser_engine(head)
        if not fnmatch.fnmatch(suffix, pattern):
            continue
        if _engine_is_usable(engine, suffix, require_external_endpoint=require_external_endpoint):
            return engine, sanitize_process_options(options_str)
    return None, ""


@dataclass(frozen=True)
class ParserDirectives:
    """Fully resolved per-file parser directives."""

    engine: str
    process_options: str = ""


def resolve_parser_directives(
    file_path: str | Path,
    *,
    parser_rules: str | None = None,
    require_external_endpoint: bool = True,
) -> ParserDirectives:
    """Resolve engine and process options for a file.

    Order: filename ``[hint]`` → ``MEMGRAPHRAG_PARSER`` rules → ``legacy``.
    """
    suffix = parser_suffix(file_path)
    _validate_filename_hint(file_path, require_external_endpoint=require_external_endpoint)

    hinted_engine, hinted_options = filename_parser_directives(file_path)
    if hinted_engine and not _engine_is_usable(
        hinted_engine, suffix, require_external_endpoint=require_external_endpoint
    ):
        hinted_engine = None

    rule_engine, rule_options = _matching_rule_directives(
        file_path,
        parser_rules=parser_rules,
        require_external_endpoint=require_external_endpoint,
    )

    engine = hinted_engine or rule_engine or PARSER_ENGINE_LEGACY
    options_str = hinted_options or rule_options
    return ParserDirectives(
        engine=engine,
        process_options=sanitize_process_options(options_str),
    )


def resolve_file_parser_engine(
    file_path: str | Path,
    *,
    parser_rules: str | None = None,
    require_external_endpoint: bool = True,
) -> str:
    return resolve_parser_directives(
        file_path,
        parser_rules=parser_rules,
        require_external_endpoint=require_external_endpoint,
    ).engine
