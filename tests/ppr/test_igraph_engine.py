"""Synthetic-graph tests for IgraphPPREngine."""

from __future__ import annotations

import pytest

from memgraphrag.ppr.igraph_engine import IgraphPPREngine


@pytest.mark.offline
def test_igraph_ppr_prefers_seeded_passage():
    # entity-a -- passage-1
    # entity-b -- passage-2
    # entity-a -- entity-b
    edges = [
        ("entity-a", "chunk-1"),
        ("entity-b", "chunk-2"),
        ("entity-a", "entity-b"),
    ]
    engine = IgraphPPREngine(
        edges=edges,
        edge_weights=[1.0, 1.0, 0.5],
        passage_ids={"chunk-1", "chunk-2"},
        directed=False,
    )
    scores = engine.run({"entity-a": 1.0}, damping=0.5)
    assert set(scores.keys()) == {"chunk-1", "chunk-2"}
    assert scores["chunk-1"] >= scores["chunk-2"]


@pytest.mark.offline
def test_igraph_ppr_empty_seeds_uniform_fallback():
    edges = [("entity-a", "chunk-1"), ("entity-a", "chunk-2")]
    engine = IgraphPPREngine(edges=edges, passage_ids={"chunk-1", "chunk-2"})
    scores = engine.run({}, damping=0.85)
    assert len(scores) == 2
    assert all(v >= 0 for v in scores.values())


@pytest.mark.offline
def test_get_ppr_engine_factory():
    from memgraphrag.ppr import get_ppr_engine

    eng = get_ppr_engine(
        "igraph",
        edges=[("a", "chunk-x")],
        passage_ids={"chunk-x"},
    )
    scores = eng.run({"a": 1.0})
    assert "chunk-x" in scores
