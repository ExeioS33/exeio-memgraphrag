"""Unit tests for RetrievalStateManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from memgraphrag.exceptions import NotReadyError
from memgraphrag.ppr.igraph_engine import IgraphPPREngine
from memgraphrag.retrieval import RetrievalStateManager
from memgraphrag.storage import shared as shared_storage
from memgraphrag.storage.igraph_impl import IgraphStorage


@pytest.fixture
async def graph_storage(tmp_path: Path):
    store = IgraphStorage(
        workspace="test_ws",
        namespace="memory",
        global_config={"working_dir": str(tmp_path), "is_directed_graph": False},
    )
    await store.initialize()
    await store.upsert_node(
        "entity-a",
        {"id": "entity-a", "label": "Entity", "content": "alice"},
    )
    await store.upsert_node(
        "chunk-1",
        {"id": "chunk-1", "label": "Passage", "content": "Alice works here."},
    )
    await store.upsert_edge(
        "entity-a",
        "chunk-1",
        {"type": "PASSAGE_ENTITY", "weight": 1.0},
    )
    yield store
    await store.finalize()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_not_ready_raises(graph_storage) -> None:
    mgr = RetrievalStateManager(graph=graph_storage, workspace="test_ws")
    assert mgr.is_ready is False
    with pytest.raises(NotReadyError):
        mgr.require_ready()
    with pytest.raises(NotReadyError):
        mgr.get_ppr_or_raise()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_warm_up_hydrates_ppr(graph_storage) -> None:
    mgr = RetrievalStateManager(
        graph=graph_storage,
        workspace="test_ws",
        ppr_engine_name="igraph",
    )
    await mgr.warm_up()
    assert mgr.is_ready is True
    engine = mgr.get_ppr_or_raise()
    assert isinstance(engine, IgraphPPREngine)
    scores = engine.run({"entity-a": 1.0}, damping=0.5)
    assert "chunk-1" in scores
    assert scores["chunk-1"] > 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_full_reload(graph_storage) -> None:
    mgr = RetrievalStateManager(graph=graph_storage, workspace="test_ws")
    await mgr.warm_up()
    v1 = mgr.version
    await graph_storage.upsert_node(
        "chunk-2",
        {"id": "chunk-2", "label": "Passage", "content": "More text."},
    )
    await graph_storage.upsert_edge(
        "entity-a",
        "chunk-2",
        {"type": "PASSAGE_ENTITY", "weight": 1.0},
    )
    await mgr.full_reload()
    assert mgr.is_ready is True
    assert mgr.version >= v1
    scores = mgr.get_ppr_or_raise().run({"entity-a": 1.0})
    assert "chunk-2" in scores


@pytest.mark.offline
@pytest.mark.asyncio
async def test_refresh_incremental(graph_storage) -> None:
    mgr = RetrievalStateManager(graph=graph_storage, workspace="test_ws")
    await mgr.warm_up()
    await mgr.refresh_incremental(
        nodes=[
            {"id": "entity-b", "label": "Entity", "content": "bob"},
            {"id": "chunk-2", "label": "Passage", "content": "Bob passage"},
        ],
        edges=[
            {"source": "entity-b", "target": "chunk-2", "weight": 1.0},
            {"source": "entity-a", "target": "entity-b", "weight": 0.5},
        ],
    )
    assert mgr.is_ready is True
    assert isinstance(mgr.ppr, IgraphPPREngine)
    assert "chunk-2" in mgr.ppr.passage_ids
    scores = mgr.ppr.run({"entity-b": 1.0}, damping=0.5)
    assert scores["chunk-2"] >= scores.get("chunk-1", 0.0)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cross_worker_signal(graph_storage) -> None:
    ws = "signal_ws"
    mgr = RetrievalStateManager(graph=graph_storage, workspace=ws)
    await mgr.warm_up()

    # Simulate another worker marking refresh needed
    await shared_storage.set_refresh_flag(ws)
    reloaded = await mgr.consume_cross_worker_signal()
    assert reloaded is True
    assert mgr.is_ready is True

    # Flag already consumed
    reloaded_again = await mgr.consume_cross_worker_signal()
    assert reloaded_again is False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_signal_refresh_helper(graph_storage) -> None:
    ws = "signal_helper_ws"
    mgr = RetrievalStateManager(graph=graph_storage, workspace=ws)
    await mgr.signal_refresh()
    assert await shared_storage.consume_refresh_flag(ws) is True
