"""Tests for document admin: chunk tracking, delete, clear, lock, drop."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from memgraphrag.base import DocStatus
from memgraphrag.pipeline import _assign_chunk_ids
from memgraphrag.utils.hashing import compute_mdhash_id

pytestmark = pytest.mark.offline


def test_assign_chunk_ids_are_content_hashes():
    chunks = [
        {"content": "Hello world", "chunk_order_index": 0},
        {"content": "Hello world", "chunk_order_index": 1},
        {"content": "Different", "chunk_order_index": 2},
    ]
    prepared = _assign_chunk_ids(chunks)
    assert len(prepared) == 3
    expected = compute_mdhash_id("Hello world", prefix="chunk-")
    assert prepared[0]["idx"] == expected
    assert prepared[1]["idx"] == expected
    assert prepared[2]["idx"] == compute_mdhash_id("Different", prefix="chunk-")
    assert all(c["idx"].startswith("chunk-") for c in prepared)


async def _fake_embed(texts, **kwargs):
    return np.zeros((len(texts), 8), dtype=np.float32)


async def _fake_llm(user: str, system_prompt: str = "", **kwargs: Any) -> str:
    # Minimal ontology / conflict JSON so ingest can complete without network.
    if "ontology" in (system_prompt or "").lower() or "ontology" in user.lower():
        return '{"ontology_triples": []}'
    if "conflict" in (system_prompt or "").lower() or "conflict" in user.lower():
        return '{"has_conflict": false, "conflicts": [], "summary": "none"}'
    return '{"named_entities": ["Alice"], "triples": [["Alice", "lives_in", "Paris"]]}'


def _make_rag(tmp_path, *, monkeypatch=None):
    from memgraphrag.core import MemGraphRAG

    if monkeypatch is not None:
        monkeypatch.setenv("CONFLICT_ENABLED", "false")
        monkeypatch.setenv("ONTOLOGY_MIN_FREQUENCY", "1")

    return MemGraphRAG(
        working_dir=str(tmp_path / "wd"),
        llm_model_func=_fake_llm,
        embedding_func=_fake_embed,
        embedding_dim=8,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="IgraphStorage",
        doc_status_storage="JsonDocStatusStorage",
    )


@pytest.mark.asyncio
async def test_json_kv_and_nano_drop(tmp_path):
    from memgraphrag.storage.json_kv_impl import JsonKVStorage
    from memgraphrag.storage.nano_vector_db_impl import NanoVectorDBStorage

    common = dict(
        workspace="",
        global_config={"working_dir": str(tmp_path), "embedding_dim": 8},
        embedding_func=None,
    )
    kv = JsonKVStorage(namespace="kv_test", **common)
    await kv.initialize()
    await kv.upsert({"a": {"x": 1}, "b": {"y": 2}})
    assert len(await kv.get_all()) == 2
    await kv.drop()
    assert await kv.get_all() == {}

    vdb = NanoVectorDBStorage(namespace="vec_test", **common)
    await vdb.initialize()
    await vdb.upsert(
        {
            "v1": {"content": "hi", "embedding": [0.0] * 8},
            "v2": {"content": "bye", "embedding": [0.1] * 8},
        }
    )
    await vdb.drop()
    # Fresh client should have no hits for previous ids
    hits = await vdb.query([0.0] * 8, top_k=10)
    assert hits == [] or all(h.get("id") not in {"v1", "v2"} for h in hits)


@pytest.mark.asyncio
async def test_delete_pending_and_not_found(tmp_path, monkeypatch):
    rag = _make_rag(tmp_path, monkeypatch=monkeypatch)
    await rag.initialize_storages()
    await rag.doc_status.upsert(
        {
            "doc-pending": {
                "status": DocStatus.PENDING.value,
                "file_path": "inline:doc-pending",
                "chunk_ids": [],
            }
        }
    )
    out = await rag.adelete_by_doc_ids(["doc-pending", "doc-missing"])
    assert out["results"]["doc-pending"]["status"] == "deleted"
    assert out["results"]["doc-missing"]["status"] == "not_found"
    assert await rag.doc_status.get_by_id("doc-pending") is None
    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_corpus_accumulate_and_delete_rebuild(tmp_path, monkeypatch):
    from memgraphrag.openie.openai_openie import OpenIE

    rag = _make_rag(tmp_path, monkeypatch=monkeypatch)
    await rag.initialize_storages()

    # Bypass LLM OpenIE with deterministic docs keyed by chunk ids
    async def fake_batch(docs):
        out = []
        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                idx = str(doc.get("idx") or i)
                passage = str(doc.get("content") or doc.get("passage") or "")
            else:
                idx, passage = str(i), str(doc)
            # Distinct triples per passage so memory sizes differ
            entity = "Alice" if "A" in passage else "Bob"
            city = "Paris" if "A" in passage else "Lyon"
            out.append(
                {
                    "idx": idx,
                    "passage": passage,
                    "extracted_entities": [entity, city],
                    "extracted_triples": [[entity, "lives_in", city]],
                }
            )
        return out

    rag.openie = OpenIE(_fake_llm, max_concurrency=2)
    rag.openie.batch_openie = fake_batch  # type: ignore[method-assign]

    chunk_a = {"idx": "", "content": "Doc A: Alice lives in Paris."}
    chunk_b = {"idx": "", "content": "Doc B: Bob lives in Lyon."}
    prepared_a = rag._normalize_chunks([chunk_a])
    prepared_b = rag._normalize_chunks([chunk_b])
    id_a, id_b = prepared_a[0]["idx"], prepared_b[0]["idx"]

    await rag.ainsert(prepared_a, run_conflicts=False)
    await rag.doc_status.upsert(
        {
            "doc-a": {
                "status": DocStatus.PROCESSED.value,
                "file_path": "inline:doc-a",
                "chunk_ids": [id_a],
                "chunk_count": 1,
            }
        }
    )
    assert len(rag.memory.passage_layer) == 1

    await rag.ainsert(prepared_b, run_conflicts=False)
    await rag.doc_status.upsert(
        {
            "doc-b": {
                "status": DocStatus.PROCESSED.value,
                "file_path": "inline:doc-b",
                "chunk_ids": [id_b],
                "chunk_count": 1,
            }
        }
    )
    assert len(rag.memory.passage_layer) == 2
    assert id_a in rag._passage_ids and id_b in rag._passage_ids

    # Shared chunk refcount: create doc-c pointing at same chunk as doc-a
    await rag.doc_status.upsert(
        {
            "doc-c": {
                "status": DocStatus.PROCESSED.value,
                "file_path": "inline:doc-c",
                "chunk_ids": [id_a],
                "chunk_count": 1,
            }
        }
    )
    del_a = await rag.adelete_by_doc_ids(["doc-a"])
    assert del_a["results"]["doc-a"]["status"] == "deleted"
    assert id_a not in del_a["chunks_dropped"]  # still referenced by doc-c
    assert await rag.openie_kv.get_by_id(id_a) is not None
    assert len(rag.memory.passage_layer) == 2  # a(shared)+b

    del_c = await rag.adelete_by_doc_ids(["doc-c"])
    assert id_a in del_c["chunks_dropped"]
    assert await rag.openie_kv.get_by_id(id_a) is None
    assert len(rag.memory.passage_layer) == 1
    assert rag._passage_ids == [id_b]

    # Delete last doc → empty corpus
    await rag.adelete_by_doc_ids(["doc-b"])
    assert rag.memory is None
    assert await rag.memory_kv.get_by_id("memory") is None
    nodes = await rag.graph.get_all_nodes()
    assert nodes == []
    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_aclear_all_drops_storages(tmp_path, monkeypatch):
    rag = _make_rag(tmp_path, monkeypatch=monkeypatch)
    await rag.initialize_storages()

    async def fake_batch(docs):
        out = []
        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                idx = str(doc.get("idx") or i)
                passage = str(doc.get("content") or "")
            else:
                idx, passage = str(i), str(doc)
            out.append(
                {
                    "idx": idx,
                    "passage": passage,
                    "extracted_entities": ["X"],
                    "extracted_triples": [["X", "rel", "Y"]],
                }
            )
        return out

    rag.openie.batch_openie = fake_batch  # type: ignore[method-assign]
    prepared = rag._normalize_chunks([{"content": "Clear me please."}])
    await rag.ainsert(prepared, run_conflicts=False)
    await rag.doc_status.upsert(
        {
            "doc-z": {
                "status": DocStatus.PROCESSED.value,
                "chunk_ids": [prepared[0]["idx"]],
            }
        }
    )
    result = await rag.aclear_all(delete_files=False)
    assert result["status"] == "ok"
    assert await rag.doc_status.get_all() == {}
    assert await rag.openie_kv.get_all() == {}
    assert await rag.memory_kv.get_all() == {}
    assert rag.memory is None
    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_pipeline_lock_returns_409(tmp_path):
    from httpx import ASGITransport, AsyncClient

    from memgraphrag.api.server import create_app

    rag = _make_rag(tmp_path)
    await rag.initialize_storages()
    app = create_app(testing=True, rag=rag)
    # Simulate busy pipeline
    app.state.pipeline_busy = True
    await app.state.pipeline_lock.acquire()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/documents/doc-x")
        assert resp.status_code == 409
        resp2 = await client.delete("/documents/", params={"confirm": "true"})
        assert resp2.status_code == 409
        resp3 = await client.delete("/documents/", params={"confirm": "false"})
        # Still busy — 409 takes precedence only after confirm check?
        # confirm=false returns 400 before lock
        assert resp3.status_code == 400

    app.state.pipeline_lock.release()
    app.state.pipeline_busy = False
    await rag.finalize_storages()
