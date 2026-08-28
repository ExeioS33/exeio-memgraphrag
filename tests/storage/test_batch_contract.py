"""The deferred-write contract belongs to every storage kind, not just graphs.

``batch()`` started life on ``BaseGraphStorage`` only, then ``JsonKVStorage`` and
``NanoVectorDBStorage`` grew their own copies. A caller wrapping an ingestion loop
in ``async with storage.batch():`` therefore worked or raised ``AttributeError``
depending on which backend the deployment had selected. The default now sits on
``StorageNameSpace``, so these tests assert the method is reachable from every ABC
and that backends with per-statement durability inherit a no-op instead of failing.
"""

from __future__ import annotations

from typing import Any

import pytest

from memgraphrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocStatusStorage,
    StorageNameSpace,
)

pytestmark = pytest.mark.offline

STORAGE_ABCS = [BaseKVStorage, BaseVectorStorage, BaseGraphStorage, DocStatusStorage]


class _StubKV(BaseKVStorage):
    """Minimal concrete KV backend: no persistence, hence no batch() override."""

    async def initialize(self) -> None: ...

    async def finalize(self) -> None: ...

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return []

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return set(keys)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None: ...

    async def delete(self, ids: list[str]) -> None: ...


@pytest.mark.parametrize("abc", STORAGE_ABCS, ids=lambda c: c.__name__)
def test_every_storage_abc_exposes_batch(abc: type) -> None:
    assert hasattr(abc, "batch"), f"{abc.__name__} has no batch(); callers cannot be generic"


@pytest.mark.asyncio
async def test_inherited_batch_is_a_usable_no_op() -> None:
    storage = _StubKV(workspace="", namespace="chunks", global_config={})
    async with storage.batch() as handle:
        await storage.upsert({"a": {"content": "x"}})
    assert handle is storage


def test_database_backends_inherit_the_default_rather_than_reimplementing_it() -> None:
    # PostgreSQL commits per statement, so it must NOT override batch(); if it ever
    # does, the override has to be reviewed against the shared-pool refcounting.
    pg = pytest.importorskip("memgraphrag.storage.postgres_impl")
    assert pg.PGKVStorage.batch is StorageNameSpace.batch
    assert pg.PGVectorStorage.batch is StorageNameSpace.batch


def test_file_backends_override_the_default() -> None:
    from memgraphrag.storage.igraph_impl import IgraphStorage
    from memgraphrag.storage.json_kv_impl import JsonKVStorage
    from memgraphrag.storage.nano_vector_db_impl import NanoVectorDBStorage

    for backend in (IgraphStorage, JsonKVStorage, NanoVectorDBStorage):
        assert backend.batch is not StorageNameSpace.batch, (
            f"{backend.__name__} rewrites its whole state per write and must defer"
        )
