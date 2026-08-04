"""PPR engine factory for MemGraphRAG."""

from __future__ import annotations

from typing import Any

from memgraphrag.ppr.base import PPREngine
from memgraphrag.ppr.igraph_engine import IgraphPPREngine
from memgraphrag.ppr.neo4j_gds_engine import Neo4jGDSPPREngine


def get_ppr_engine(name: str = "igraph", **kwargs: Any) -> PPREngine:
    """Construct a PPR engine by name (``igraph`` | ``neo4j_gds``)."""
    key = (name or "igraph").strip().lower()
    if key in {"igraph", "ig"}:
        return IgraphPPREngine(**kwargs)
    if key in {"neo4j_gds", "neo4j", "gds"}:
        return Neo4jGDSPPREngine(**kwargs)
    raise ValueError(f"Unknown PPR engine: {name!r} (expected 'igraph' or 'neo4j_gds')")


__all__ = [
    "PPREngine",
    "IgraphPPREngine",
    "Neo4jGDSPPREngine",
    "get_ppr_engine",
]
