"""Unit tests for JsonKVStorage: corrupt-file handling and deferred flush."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import pytest

from memgraphrag.storage.json_kv_impl import CorruptKVFileError, JsonKVStorage

pytestmark = pytest.mark.offline


def _storage(tmp_path: Any, namespace: str = "doc_status") -> JsonKVStorage:
    return JsonKVStorage(
        workspace="",
        namespace=namespace,
        global_config={"working_dir": str(tmp_path)},
    )


@pytest.mark.asyncio
async def test_initialize_on_missing_file_starts_empty(tmp_path: Any) -> None:
    storage = _storage(tmp_path)
    await storage.initialize()
    assert await storage.get_all() == {}


@pytest.mark.asyncio
async def test_corrupt_file_fails_initialize_and_is_preserved(tmp_path: Any) -> None:
    path = tmp_path / "doc_status.json"
    path.write_text('{"doc-1": {"status": "PROCESSED"', encoding="utf-8")

    storage = _storage(tmp_path)
    with pytest.raises(CorruptKVFileError):
        await storage.initialize()

    # The damaged index must survive: an upsert over an empty dict would erase it.
    assert path.read_text(encoding="utf-8").startswith('{"doc-1"')


@pytest.mark.asyncio
async def test_non_object_json_fails_initialize(tmp_path: Any) -> None:
    path = tmp_path / "doc_status.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    storage = _storage(tmp_path)
    with pytest.raises(CorruptKVFileError):
        await storage.initialize()


@pytest.mark.asyncio
async def test_quarantine_mode_moves_the_file_aside(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMGRAPHRAG_KV_QUARANTINE_CORRUPT", "true")
    path = tmp_path / "doc_status.json"
    original = '{"doc-1": {"status": "PROCESSED"'
    path.write_text(original, encoding="utf-8")

    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert({"doc-2": {"status": "PENDING"}})
    await storage.finalize()

    quarantined = glob.glob(str(tmp_path / "doc_status.json.corrupt-*"))
    assert len(quarantined) == 1
    assert Path(quarantined[0]).read_text(encoding="utf-8") == original
    assert json.loads(path.read_text(encoding="utf-8"))["doc-2"]["status"] == "PENDING"


@pytest.mark.asyncio
async def test_batch_defers_the_flush_to_the_outermost_exit(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memgraphrag.storage import json_kv_impl

    writes: list[int] = []
    real_write = json_kv_impl._write_json

    def counting_write(file_path: str, data: dict[str, Any]) -> None:
        writes.append(len(data))
        real_write(file_path, data)

    monkeypatch.setattr(json_kv_impl, "_write_json", counting_write)

    storage = _storage(tmp_path, namespace="chunks")
    await storage.initialize()
    async with storage.batch():
        for i in range(5):
            await storage.upsert({f"chunk-{i}": {"content": str(i)}})
        assert writes == [], "a batch must not rewrite the file per upsert"

    assert writes == [5]
    assert len(json.loads(Path(tmp_path / "chunks.json").read_text(encoding="utf-8"))) == 5


@pytest.mark.asyncio
async def test_nested_batch_flushes_once(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from memgraphrag.storage import json_kv_impl

    writes: list[int] = []
    real_write = json_kv_impl._write_json

    def counting_write(file_path: str, data: dict[str, Any]) -> None:
        writes.append(len(data))
        real_write(file_path, data)

    monkeypatch.setattr(json_kv_impl, "_write_json", counting_write)

    storage = _storage(tmp_path, namespace="chunks")
    await storage.initialize()
    async with storage.batch():
        await storage.upsert({"a": {"v": 1}})
        async with storage.batch():
            await storage.upsert({"b": {"v": 2}})
        assert writes == [], "the inner batch must not flush"

    assert writes == [2]


@pytest.mark.asyncio
async def test_upsert_outside_a_batch_still_persists(tmp_path: Any) -> None:
    storage = _storage(tmp_path, namespace="chunks")
    await storage.initialize()
    await storage.upsert({"a": {"v": 1}})

    assert os.path.exists(tmp_path / "chunks.json")

    reloaded = _storage(tmp_path, namespace="chunks")
    await reloaded.initialize()
    assert (await reloaded.get_by_id("a"))["v"] == 1
