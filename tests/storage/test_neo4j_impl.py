"""Unit tests for Neo4JStorage (mocked driver or import-only)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.offline

REQUIRED_METHODS = (
    "initialize",
    "finalize",
    "has_node",
    "has_edge",
    "upsert_node",
    "upsert_edge",
    "get_node",
    "get_edge",
    "get_all_nodes",
    "get_all_edges",
    "clear",
    "node_degree",
)


def test_neo4j_storage_imports_and_has_required_methods() -> None:
    from memgraphrag.storage.neo4j_impl import Neo4JStorage
    from memgraphrag.base import BaseGraphStorage

    assert issubclass(Neo4JStorage, BaseGraphStorage)
    for name in REQUIRED_METHODS:
        assert hasattr(Neo4JStorage, name), f"missing method: {name}"
        assert callable(getattr(Neo4JStorage, name))


def test_neo4j_storage_registered_in_factory() -> None:
    from memgraphrag.storage.factory import get_storage_class

    cls = get_storage_class("Neo4JStorage")
    assert cls.__name__ == "Neo4JStorage"


def test_normalize_helpers() -> None:
    from memgraphrag.storage.neo4j_impl import (
        _normalize_edge_type,
        _normalize_node_label,
    )

    assert _normalize_node_label({"label": "Passage"}) == "Passage"
    assert _normalize_node_label({"layer": "entity"}) == "Entity"
    assert _normalize_node_label({"node_type": "Type"}) == "Type"
    assert _normalize_edge_type({"type": "PASSAGE_ENTITY"}) == "PASSAGE_ENTITY"
    assert _normalize_edge_type({"type": "entity-similarity"}) == "ENTITY_SIMILARITY"


@pytest.mark.asyncio
async def test_neo4j_missing_package_raises_on_initialize() -> None:
    from memgraphrag.storage import neo4j_impl

    storage = neo4j_impl.Neo4JStorage(
        workspace="test",
        namespace="graph",
        global_config={},
    )
    with (
        patch.object(neo4j_impl, "_NEO4J_AVAILABLE", False),
        patch.object(neo4j_impl, "AsyncGraphDatabase", None),
    ):
        with pytest.raises(ImportError, match="neo4j package"):
            await storage.initialize()


@pytest.mark.asyncio
async def test_neo4j_methods_with_mocked_driver() -> None:
    """Exercise CRUD paths against a mocked AsyncDriver."""
    from memgraphrag.storage.neo4j_impl import Neo4JStorage

    storage = Neo4JStorage(
        workspace="ws1",
        namespace="memory_graph",
        global_config={},
    )

    # Build a fake session / result chain
    record_exists = {"exists": True}
    record_degree = {"degree": 2}
    node_props = {
        "entity_id": "entity-a",
        "node_id": "entity-a",
        "node_type": "Entity",
        "content": "alice",
        "workspace": "ws1",
    }

    class FakeResult:
        def __init__(self, records: list[Any] | None = None, single_rec: Any = None):
            self._records = list(records or [])
            self._single = single_rec
            self._idx = 0

        async def single(self):
            return self._single

        async def consume(self):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx >= len(self._records):
                raise StopAsyncIteration
            rec = self._records[self._idx]
            self._idx += 1
            return rec

    class FakeSession:
        def __init__(self):
            self.runs: list[tuple[str, dict]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def run(self, query: str, **params):
            self.runs.append((query, params))
            q = query.upper()
            if "COUNT(N) > 0" in q or "COUNT(R) > 0" in q or "AS EXISTS" in q:
                return FakeResult(single_rec=record_exists)
            if "AS DEGREE" in q:
                return FakeResult(single_rec=record_degree)
            if "RETURN N, LABELS" in q and "LIMIT 1" in q:
                return FakeResult(single_rec={"n": node_props, "labels": ["ws1", "Entity"]})
            if "RETURN TYPE(R)" in q and "LIMIT 1" in q:
                return FakeResult(
                    single_rec={
                        "rel_type": "PASSAGE_ENTITY",
                        "props": {"weight": 1.0, "workspace": "ws1"},
                    }
                )
            if "RETURN N, LABELS" in q:
                return FakeResult(records=[{"n": node_props, "labels": ["ws1", "Entity"]}])
            if "RETURN A.ENTITY_ID AS SOURCE" in q:
                return FakeResult(
                    records=[
                        {
                            "source": "entity-a",
                            "target": "chunk-1",
                            "rel_type": "PASSAGE_ENTITY",
                            "props": {"weight": 1.0},
                        }
                    ]
                )
            return FakeResult(single_rec={"ok": 1})

    fake_session = FakeSession()
    fake_driver = MagicMock()
    fake_driver.session = MagicMock(return_value=fake_session)
    fake_driver.close = AsyncMock()

    mock_adb = MagicMock()
    mock_adb.driver = MagicMock(return_value=fake_driver)

    with (
        patch("memgraphrag.storage.neo4j_impl.AsyncGraphDatabase", mock_adb),
        patch("memgraphrag.storage.neo4j_impl._NEO4J_AVAILABLE", True),
        patch.dict(
            "os.environ",
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USERNAME": "neo4j",
                "NEO4J_PASSWORD": "test",
                "NEO4J_DATABASE": "neo4j",
            },
            clear=False,
        ),
    ):
        await storage.initialize()
        assert storage._driver is fake_driver

        assert await storage.has_node("entity-a") is True
        assert await storage.has_edge("entity-a", "chunk-1") is True

        await storage.upsert_node(
            "entity-a",
            {"label": "Entity", "content": "alice"},
        )
        await storage.upsert_edge(
            "entity-a",
            "chunk-1",
            {"type": "PASSAGE_ENTITY", "weight": 1.0},
        )

        node = await storage.get_node("entity-a")
        assert node is not None
        assert node["id"] == "entity-a"
        assert node["label"] == "Entity"

        edge = await storage.get_edge("entity-a", "chunk-1")
        assert edge is not None
        assert edge["type"] == "PASSAGE_ENTITY"
        assert edge["weight"] == 1.0

        nodes = await storage.get_all_nodes()
        assert len(nodes) == 1
        edges = await storage.get_all_edges()
        assert len(edges) == 1
        assert await storage.node_degree("entity-a") == 2

        await storage.clear()
        await storage.finalize()
        fake_driver.close.assert_awaited()
