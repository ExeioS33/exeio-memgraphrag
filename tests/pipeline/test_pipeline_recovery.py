"""Tests for crash recovery and the bounded content preview in doc-status."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memgraphrag.base import DocStatus
from memgraphrag.namespace import NameSpace
from memgraphrag.pipeline import (
    CONTENT_SUMMARY_LIMIT,
    enqueue_document,
    process_pending,
    reset_interrupted_documents,
)
from memgraphrag.storage.json_doc_status_impl import JsonDocStatusStorage


async def _storage(tmp_path: Path) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        workspace="",
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path)},
    )
    await storage.initialize()
    return storage


@pytest.mark.offline
async def test_reset_interrupted_documents_requeues_stuck_docs(tmp_path: Path) -> None:
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc-parsing": {
                "status": DocStatus.PARSING.value,
                "file_path": "/inputs/a.txt",
                "metadata": {"memory_sub_stage": None},
            },
            "doc-processing": {
                "status": DocStatus.PROCESSING.value,
                "file_path": "/inputs/b.txt",
                "chunk_ids": ["chunk-half-written"],
                "metadata": {"memory_sub_stage": "conflict_detection"},
            },
            "doc-processed": {
                "status": DocStatus.PROCESSED.value,
                "file_path": "/inputs/c.txt",
                "chunk_ids": ["chunk-good"],
            },
            "doc-failed": {
                "status": DocStatus.FAILED.value,
                "file_path": "/inputs/d.txt",
            },
        }
    )

    recovered = await reset_interrupted_documents(storage)

    # process_pending only looks at PENDING, so before this sweep an OOM-kill left
    # these two wedged in PARSING/PROCESSING for the lifetime of the storage.
    assert sorted(recovered) == ["doc-parsing", "doc-processing"]
    pending = await storage.get_docs_by_statuses([DocStatus.PENDING])
    assert set(pending) == {"doc-parsing", "doc-processing"}
    resumed = await storage.get_by_id("doc-processing")
    assert resumed["metadata"]["memory_sub_stage"] is None
    assert resumed["metadata"]["recovered_from"] == DocStatus.PROCESSING.value
    # A torn write may have recorded chunks that were never installed.
    assert "chunk_ids" not in resumed
    # Terminal states are left alone.
    assert (await storage.get_by_id("doc-processed"))["status"] == (DocStatus.PROCESSED.value)
    assert (await storage.get_by_id("doc-failed"))["status"] == DocStatus.FAILED.value

    await storage.finalize()


@pytest.mark.offline
async def test_reset_interrupted_documents_is_a_noop_without_stuck_docs(
    tmp_path: Path,
) -> None:
    storage = await _storage(tmp_path)
    await storage.upsert({"doc-ok": {"status": DocStatus.PROCESSED.value, "file_path": "/x.txt"}})

    assert await reset_interrupted_documents(storage) == []

    await storage.finalize()


@pytest.mark.offline
async def test_enqueue_keeps_only_a_bounded_preview_of_the_body(
    tmp_path: Path,
) -> None:
    storage = await _storage(tmp_path)
    body = "lorem ipsum " * 500

    record = await enqueue_document(
        doc_id="doc-preview",
        file_path=str(tmp_path / "preview.txt"),
        doc_status_storage=storage,
        content=body,
        parse_engine="legacy",
    )

    # The full body used to be stored here, so listing 10k documents serialized the
    # whole corpus into one response.
    assert "content" not in record
    assert record["content_length"] == len(body)
    assert len(record["content_summary"]) <= CONTENT_SUMMARY_LIMIT
    assert (await storage.get_by_id("doc-preview"))["content_summary"] == (
        record["content_summary"]
    )

    await storage.finalize()


@pytest.mark.offline
async def test_recovered_document_is_reprocessed_end_to_end(tmp_path: Path) -> None:
    storage = await _storage(tmp_path)
    source = tmp_path / "stuck.txt"
    source.write_text("a document a previous worker died on", encoding="utf-8")
    await storage.upsert(
        {
            "doc-stuck": {
                "status": DocStatus.PROCESSING.value,
                "file_path": str(source),
                "parse_engine": "legacy",
                "chunk_options": {},
                "metadata": {"memory_sub_stage": "openie"},
            }
        }
    )
    engine = MagicMock()
    engine.aindex_with_memory = AsyncMock()

    # Without the sweep this run has nothing to do: the document is not PENDING.
    assert (await process_pending(engine, storage))["processed"] == 0

    await reset_interrupted_documents(storage)
    summary = await process_pending(engine, storage)

    assert summary["processed"] == 1
    assert (await storage.get_by_id("doc-stuck"))["status"] == DocStatus.PROCESSED.value
    engine.aindex_with_memory.assert_awaited_once()

    await storage.finalize()


@pytest.mark.offline
async def test_drain_background_tasks_waits_then_cancels_stragglers() -> None:
    pytest.importorskip("fastapi")
    from memgraphrag.api.server import drain_background_tasks

    finished: list[str] = []

    async def quick() -> None:
        await asyncio.sleep(0)
        finished.append("quick")

    async def endless() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            finished.append("cancelled")
            raise

    app = SimpleNamespace(
        state=SimpleNamespace(
            background_tasks={
                asyncio.ensure_future(quick()),
                asyncio.ensure_future(endless()),
            }
        )
    )

    # A SIGTERM used to kill indexing mid-write because nothing awaited these.
    cancelled = await drain_background_tasks(app, timeout=0.2)

    assert cancelled == 1
    assert "quick" in finished
    assert "cancelled" in finished


@pytest.mark.offline
async def test_track_background_task_registers_and_clears() -> None:
    pytest.importorskip("fastapi")
    from memgraphrag.api.routers.documents import track_background_task

    app = SimpleNamespace(state=SimpleNamespace(background_tasks=set()))

    async def work() -> None:
        await asyncio.sleep(0)

    task = track_background_task(app, work())
    # Registered while running, so the shutdown drain can find it.
    assert task in app.state.background_tasks
    await task
    await asyncio.sleep(0)
    # ...and unregistered once done, so the set cannot grow without bound.
    assert task not in app.state.background_tasks
