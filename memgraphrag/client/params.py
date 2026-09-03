"""Query parameter registry, presets, and default optimization grids.

Single source of truth for CLI options, Streamlit sliders, and the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

ParamKind = Literal["choice", "int", "float", "bool", "str"]


@dataclass(frozen=True)
class ParamSpec:
    """Descriptor for one tunable query parameter."""

    name: str
    kind: ParamKind
    emoji: str
    help: str
    default: Any
    choices: Optional[tuple[Any, ...]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    # Included in the default optimization sweep grid when True.
    sweepable: bool = True
    # Default discrete values for the optimization grid.
    grid: tuple[Any, ...] = field(default_factory=tuple)


QUERY_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        name="mode",
        kind="choice",
        emoji="🎛️",
        help=(
            "Retrieval mode: ppr (graph), naive (dense), context, bypass (LLM only), "
            "agent (the model searches for itself — one extra LLM round trip or more)"
        ),
        default="ppr",
        choices=("ppr", "naive", "context", "bypass", "agent"),
        grid=("ppr", "naive"),
    ),
    ParamSpec(
        name="top_k",
        kind="int",
        emoji="🔢",
        help="Number of passages returned after ranking",
        default=5,
        min=1,
        max=50,
        step=1,
        grid=(3, 5, 10),
    ),
    ParamSpec(
        name="linking_top_k",
        kind="int",
        emoji="🔗",
        help="How many seed nodes to link from the query embedding",
        default=5,
        min=1,
        max=50,
        step=1,
        grid=(3, 5, 10),
    ),
    ParamSpec(
        name="passage_node_weight",
        kind="float",
        emoji="⚖️",
        help="Relative weight of passage nodes in the PPR seed vector",
        default=0.05,
        min=0.0,
        max=1.0,
        step=0.01,
        grid=(0.01, 0.05, 0.1),
    ),
    ParamSpec(
        name="damping",
        kind="float",
        emoji="🌊",
        help="PPR damping factor (teleport probability = 1 - damping)",
        default=0.5,
        min=0.05,
        max=0.95,
        step=0.05,
        grid=(0.3, 0.5, 0.7),
    ),
    ParamSpec(
        name="fact_similarity_threshold",
        kind="float",
        emoji="📏",
        help="Minimum cosine similarity when matching query facts",
        default=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
        grid=(0.3, 0.5, 0.7),
    ),
    ParamSpec(
        name="skip_fact_rerank",
        kind="bool",
        emoji="⏩",
        help="Skip fact-level reranking after retrieval",
        default=False,
        grid=(False, True),
    ),
    ParamSpec(
        name="schema_top_k",
        kind="int",
        emoji="🧬",
        help="How many ontology schemas to link from the query embedding",
        default=5,
        min=0,
        max=50,
        step=1,
        grid=(0, 3, 5),
    ),
    ParamSpec(
        name="schema_node_weight",
        kind="float",
        emoji="🧩",
        help="Relative weight of schema-expanded seeds in the PPR seed vector",
        default=0.1,
        min=0.0,
        max=1.0,
        step=0.01,
        grid=(0.05, 0.1, 0.2),
    ),
    ParamSpec(
        name="user_prompt",
        kind="str",
        emoji="📝",
        help="Optional extra instruction appended to the system prompt",
        default=None,
        sweepable=False,
    ),
)

PARAM_BY_NAME: dict[str, ParamSpec] = {p.name: p for p in QUERY_PARAMS}

# Playful presets for the Streamlit UI / CLI.
PRESETS: dict[str, dict[str, Any]] = {
    "🎯 Precise": {
        "mode": "ppr",
        "top_k": 3,
        "linking_top_k": 3,
        "passage_node_weight": 0.05,
        "damping": 0.7,
        "fact_similarity_threshold": 0.7,
        "skip_fact_rerank": False,
        "schema_top_k": 3,
        "schema_node_weight": 0.1,
    },
    "⚖️ Balanced": {
        "mode": "ppr",
        "top_k": 5,
        "linking_top_k": 5,
        "passage_node_weight": 0.05,
        "damping": 0.5,
        "fact_similarity_threshold": 0.5,
        "skip_fact_rerank": False,
        "schema_top_k": 5,
        "schema_node_weight": 0.1,
    },
    "🌊 Broad": {
        "mode": "ppr",
        "top_k": 10,
        "linking_top_k": 10,
        "passage_node_weight": 0.1,
        "damping": 0.3,
        "fact_similarity_threshold": 0.3,
        "skip_fact_rerank": True,
        "schema_top_k": 10,
        "schema_node_weight": 0.15,
    },
    "🧲 Dense-only": {
        "mode": "naive",
        "top_k": 5,
        "linking_top_k": 5,
        "passage_node_weight": 0.05,
        "damping": 0.5,
        "fact_similarity_threshold": 0.5,
        "skip_fact_rerank": True,
        "schema_top_k": 0,
        "schema_node_weight": 0.0,
    },
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv", ".json", ".html", ".htm"}
)


def default_sweep_grid() -> dict[str, list[Any]]:
    """Return the default discrete grid used by the optimizer."""
    return {p.name: list(p.grid) for p in QUERY_PARAMS if p.sweepable and p.grid}


def defaults() -> dict[str, Any]:
    """Return server-side default query params (None omitted for optional str)."""
    out: dict[str, Any] = {}
    for p in QUERY_PARAMS:
        if p.default is not None:
            out[p.name] = p.default
    return out


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown / empty values so they are not sent to the API."""
    known = set(PARAM_BY_NAME)
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key not in known:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        out[key] = value
    return out
