"""Tests for pipeline enqueue → PENDING."""

from __future__ import annotations

from pathlib import Path

import pytest

from memgraphrag.base import DocStatus
from memgraphrag.namespace import NameSpace
from memgraphrag.pipeline import enqueue_document
from memgraphrag.storage.json_doc_status_impl import JsonDocStatusStorage


@pytest.mark.asyncio
async def test_pipeline_enqueue_sets_pending(tmp_path: Path) -> None:
    storage = JsonDocStatusStorage(
        workspace="",
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path)},
    )
    await storage.initialize()

    src = tmp_path / "note.txt"
    src.write_text("hello pipeline", encoding="utf-8")

    record = await enqueue_document(
        doc_id="doc-enqueue-1",
        file_path=str(src),
        doc_status_storage=storage,
        parse_engine="legacy",
        chunk_options={"chunk_token_size": 100},
    )

    assert record["status"] == DocStatus.PENDING.value

    loaded = await storage.get_by_id("doc-enqueue-1")
    assert loaded is not None
    assert loaded["status"] == DocStatus.PENDING.value
    assert loaded["file_path"] == str(src)
    assert loaded["parse_engine"] == "legacy"
    assert loaded["chunk_options"]["chunk_token_size"] == 100

    pending = await storage.get_docs_by_statuses([DocStatus.PENDING])
    assert "doc-enqueue-1" in pending

    await storage.finalize()
