"""Central registry for MemGraphRAG parser engines.

Adapted from LightRAG ``lightrag/parser/registry.py`` — leaner table with
``legacy`` and ``docling`` only. Capability metadata stays import-cheap;
implementations load lazily via :func:`get_parser`.
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from memgraphrag.parser.base import BaseParser

PARSER_ENGINE_LEGACY = "legacy"
PARSER_ENGINE_DOCLING = "docling"

_SUFFIX_TOKEN = re.compile(r"^[a-z0-9]+$")


def _env_endpoint_configured(env_name: str) -> Callable[[], bool]:
    return lambda: bool(os.getenv(env_name, "").strip())


def _parse_env_suffixes(env_name: str) -> frozenset[str]:
    valid: set[str] = set()
    for raw in os.getenv(env_name, "").split(","):
        token = raw.strip().lower().lstrip(".")
        if token and _SUFFIX_TOKEN.match(token):
            valid.add(token)
    return frozenset(valid)


@dataclass(frozen=True)
class ParserSpec:
    """Lightweight metadata for one parser engine."""

    engine_name: str
    impl: str
    suffixes: frozenset[str]
    user_selectable: bool = True
    endpoint_configured: Callable[[], bool] = field(default=lambda: True)
    endpoint_requirement: Callable[[], str | None] = field(default=lambda: None)
    extra_suffixes_env: str | None = None

    def effective_suffixes(self) -> frozenset[str]:
        if self.extra_suffixes_env:
            return self.suffixes | _parse_env_suffixes(self.extra_suffixes_env)
        return self.suffixes


_LEGACY_SUFFIXES = frozenset(
    {
        "txt",
        "md",
        "mdx",
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "csv",
        "json",
        "xml",
        "yaml",
        "yml",
        "html",
        "htm",
        "log",
        "py",
        "js",
        "ts",
        "java",
        "go",
        "rb",
        "php",
        "c",
        "h",
        "cpp",
        "hpp",
        "sh",
        "sql",
    }
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

_REGISTRY: dict[str, ParserSpec] = {
    PARSER_ENGINE_LEGACY: ParserSpec(
        engine_name=PARSER_ENGINE_LEGACY,
        impl="memgraphrag.parser.legacy.parser:LegacyParser",
        suffixes=_LEGACY_SUFFIXES,
    ),
    PARSER_ENGINE_DOCLING: ParserSpec(
        engine_name=PARSER_ENGINE_DOCLING,
        impl="memgraphrag.parser.external.docling.parser:DoclingParser",
        suffixes=_DOCLING_SUFFIXES,
        endpoint_configured=_env_endpoint_configured("DOCLING_ENDPOINT"),
        endpoint_requirement=lambda: "DOCLING_ENDPOINT",
        extra_suffixes_env="DOCLING_ADDITIONAL_SUFFIXES",
    ),
}

_INSTANCE_CACHE: dict[tuple[str, str], "BaseParser"] = {}


class ParserUnavailableError(RuntimeError):
    """Raised when a parser engine cannot be used (missing endpoint, etc.)."""


def register_parser(spec: ParserSpec) -> None:
    """Register (or override) a parser engine spec."""
    _REGISTRY[spec.engine_name] = spec
    # Drop cached instance if impl changed.
    stale = [k for k in _INSTANCE_CACHE if k[0] == spec.engine_name and k[1] != spec.impl]
    for key in stale:
        _INSTANCE_CACHE.pop(key, None)


def parser_specs_snapshot() -> dict[str, ParserSpec]:
    return dict(_REGISTRY)


def get_parser(engine: str, *, specs: dict[str, ParserSpec] | None = None):
    """Return a cached parser instance for ``engine``.

    Raises:
        KeyError: Unknown engine name.
        ParserUnavailableError: Engine requires an endpoint that is not configured.
    """
    table = specs if specs is not None else _REGISTRY
    spec = table.get(engine)
    if spec is None:
        raise KeyError(f"Unknown parser engine: {engine!r}")
    if not spec.endpoint_configured():
        req = spec.endpoint_requirement() or "endpoint"
        raise ParserUnavailableError(
            f"Parser engine {engine!r} is unavailable: {req} is not configured"
        )
    cache_key = (engine, spec.impl)
    inst = _INSTANCE_CACHE.get(cache_key)
    if inst is None:
        module_path, _, cls_name = spec.impl.partition(":")
        cls = getattr(importlib.import_module(module_path), cls_name)
        inst = cls()
        _INSTANCE_CACHE[cache_key] = inst
    return inst


def supported_parser_engines(
    specs: dict[str, ParserSpec] | None = None,
) -> frozenset[str]:
    table = specs if specs is not None else _REGISTRY
    return frozenset(name for name, spec in table.items() if spec.user_selectable)


def available_engine_suffixes(
    specs: dict[str, ParserSpec] | None = None,
) -> frozenset[str]:
    """Suffixes parseable by a currently usable (endpoint-gated) engine."""
    table = specs if specs is not None else _REGISTRY
    out: set[str] = set()
    for spec in table.values():
        if spec.user_selectable and spec.endpoint_configured():
            out |= spec.effective_suffixes()
    return frozenset(out)


def suffix_capabilities(engine: str, specs: dict[str, ParserSpec] | None = None) -> frozenset[str]:
    table = specs if specs is not None else _REGISTRY
    spec = table.get(engine)
    return spec.effective_suffixes() if spec is not None else frozenset()


def engine_endpoint_configured(engine: str, specs: dict[str, ParserSpec] | None = None) -> bool:
    table = specs if specs is not None else _REGISTRY
    spec = table.get(engine)
    return spec.endpoint_configured() if spec is not None else True


def engine_endpoint_requirement(
    engine: str, specs: dict[str, ParserSpec] | None = None
) -> str | None:
    table = specs if specs is not None else _REGISTRY
    spec = table.get(engine)
    return spec.endpoint_requirement() if spec is not None else None
