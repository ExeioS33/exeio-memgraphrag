"""Unit tests for NanoVectorDBStorage deferred persistence."""

from __future__ import annotations

import os
from typing import Any

import pytest

from memgraphrag.storage.nano_vector_db_impl import NanoVectorDBStorage

pytestmark = pytest.mark.offline

DIM = 4


def _storage(tmp_path: Any, namespace: str = "chunks") -> NanoVectorDBStorage:
    return NanoVectorDBStorage(
        workspace="",
        namespace=namespace,
        global_config={"working_dir": str(tmp_path), "embedding_dim": DIM},
    )


def _count_saves(storage: NanoVectorDBStorage) -> list[int]:
    """Wrap the client's save() so tests can count real persistence calls."""
    saves: list[int] = []
    real_save = storage._client.save

    def counting_save(*args: Any, **kwargs: Any) -> Any:
        saves.append(1)
        return real_save(*args, **kwargs)

    storage._client.save = counting_save  # type: ignore[method-assign]
    return saves


def _vector(seed: float) -> list[float]:
    return [seed, 0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_batch_saves_once_instead_of_per_upsert(tmp_path: Any) -> None:
    storage = _storage(tmp_path)
    await storage.initialize()
    saves = _count_saves(storage)

    async with storage.batch():
        for i in range(5):
            await storage.upsert({f"chunk-{i}": {"content": str(i), "embedding": _vector(i)}})
        assert saves == [], "a batch must not serialise the store per upsert"

    assert len(saves) == 1
    assert os.path.exists(tmp_path / "vdb_chunks.json")


@pytest.mark.asyncio
async def test_batch_defers_deletes_too(tmp_path: Any) -> None:
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert(
        {
            "a": {"content": "a", "embedding": _vector(1.0)},
            "b": {"content": "b", "embedding": _vector(2.0)},
        }
    )
    saves = _count_saves(storage)

    async with storage.batch():
        await storage.delete(["a"])
        await storage.delete(["b"])
        assert saves == []

    assert len(saves) == 1


@pytest.mark.asyncio
async def test_upsert_outside_a_batch_persists_immediately(tmp_path: Any) -> None:
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert({"a": {"content": "a", "embedding": _vector(1.0)}})

    reloaded = _storage(tmp_path)
    await reloaded.initialize()
    hits = await reloaded.query(_vector(1.0), top_k=1)
    assert hits and hits[0]["id"] == "a"


@pytest.mark.asyncio
async def test_finalize_flushes_pending_batch_writes(tmp_path: Any) -> None:
    storage = _storage(tmp_path)
    await storage.initialize()
    batch = storage.batch()
    await batch.__aenter__()
    await storage.upsert({"a": {"content": "a", "embedding": _vector(1.0)}})
    # finalize() must persist even if the batch block never exits cleanly.
    await storage.finalize()

    reloaded = _storage(tmp_path)
    await reloaded.initialize()
    hits = await reloaded.query(_vector(1.0), top_k=1)
    assert hits and hits[0]["id"] == "a"
    await batch.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_finalize_without_writes_does_not_save(tmp_path: Any) -> None:
    storage = _storage(tmp_path)
    await storage.initialize()
    saves = _count_saves(storage)

    await storage.finalize()

    assert saves == []
