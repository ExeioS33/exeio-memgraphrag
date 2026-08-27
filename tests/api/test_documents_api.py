"""Tests for document identity, listing limits, graph exposure and observability.

Covers the ingestion paths (`/documents/upload`, `/documents/text`, `/documents/scan`)
agreeing on one content-addressed doc id, the listing no longer shipping document
bodies, `/graphs` returning a connected sub-graph without passage text, and the
`X-Request-ID` / `/metrics` plumbing added in `memgraphrag.api.middleware`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgraphrag.api.routers.documents import inline_source_name
from memgraphrag.api.server import create_app
from memgraphrag.base import DocStatus
from memgraphrag.namespace import NameSpace
from memgraphrag.storage.json_doc_status_impl import JsonDocStatusStorage
from memgraphrag.utils.hashing import compute_mdhash_id
from tests.api.test_auth_edge_cases import _test_args


async def _doc_status(tmp_path: Path) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        workspace="",
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path / "rag")},
    )
    await storage.initialize()
    return storage


def _rag_with_doc_status(storage: JsonDocStatusStorage) -> MagicMock:
    rag = MagicMock()
    rag.working_dir = "/tmp/memgraphrag-test"
    rag.workspace = ""
    rag.doc_status = storage
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    return rag


def _app(tmp_path: Path, storage: JsonDocStatusStorage, **args: object):
    app = create_app(
        _test_args(input_dir=str(tmp_path / "inputs"), **args),
        testing=True,
        rag=_rag_with_doc_status(storage),
    )
    # The drain worker would otherwise run the real parse→chunk→index pipeline; these
    # tests are about what ingestion *records*, not about what it indexes.
    app.state.pipeline_busy = True
    return app


@pytest.mark.offline
async def test_upload_scan_and_text_agree_on_one_content_hash_id(tmp_path: Path) -> None:
    storage = await _doc_status(tmp_path)
    app = _app(tmp_path, storage)
    body = "the same document, three doors"

    with TestClient(app) as client:
        uploaded = client.post(
            "/documents/upload",
            files={"file": ("note.txt", body.encode("utf-8"), "text/plain")},
        ).json()
        texted = client.post("/documents/text", json={"text": body}).json()

        # Drop the upload's record so /scan sees the file as new rather than known.
        await storage.delete([uploaded["doc_id"]])
        scanned = client.post("/documents/scan").json()

    expected = compute_mdhash_id(body, prefix="doc-")
    # Before the fix these were md5("note.txt:30"), md5(body) and md5("<path>:30") —
    # three ids for one document.
    assert uploaded["doc_id"] == expected
    assert texted["doc_id"] == expected
    assert scanned["enqueued"] >= 1
    assert await storage.get_by_id(expected) is not None

    await storage.finalize()


@pytest.mark.offline
async def test_same_name_and_size_revisions_get_distinct_ids(tmp_path: Path) -> None:
    storage = await _doc_status(tmp_path)
    app = _app(tmp_path, storage)

    with TestClient(app) as client:
        first = client.post(
            "/documents/upload",
            files={"file": ("report.txt", b"revision one..", "text/plain")},
        ).json()
        second = client.post(
            "/documents/upload",
            files={"file": ("report.txt", b"revision two..", "text/plain")},
        ).json()

    # Same name, same byte count: hashing name+size collapsed both onto one doc_id, so
    # the second ingest overwrote the first record and orphaned its chunks.
    assert first["doc_id"] != second["doc_id"]
    assert Path(first["path"]).read_bytes() == b"revision one.."
    assert Path(second["path"]).read_bytes() == b"revision two.."
    assert first["path"] != second["path"]
    assert await storage.get_by_id(first["doc_id"]) is not None
    assert await storage.get_by_id(second["doc_id"]) is not None

    await storage.finalize()


@pytest.mark.offline
async def test_reuploading_indexed_content_is_refused_and_keeps_chunk_ids(
    tmp_path: Path,
) -> None:
    storage = await _doc_status(tmp_path)
    app = _app(tmp_path, storage)

    with TestClient(app) as client:
        first = client.post(
            "/documents/upload",
            files={"file": ("dup.txt", b"identical bytes", "text/plain")},
        ).json()
        doc_id = first["doc_id"]
        record = await storage.get_by_id(doc_id)
        record["status"] = DocStatus.PROCESSED.value
        record["chunk_ids"] = ["chunk-aaa", "chunk-bbb"]
        await storage.upsert({doc_id: record})

        again = client.post(
            "/documents/upload",
            files={"file": ("dup.txt", b"identical bytes", "text/plain")},
        )

    assert again.status_code == 200
    assert again.json()["status"] == "duplicate"
    after = await storage.get_by_id(doc_id)
    # Re-enqueuing would have rewritten the record as PENDING with no chunk_ids,
    # leaving those two chunks unreachable in openie_kv forever.
    assert after["status"] == DocStatus.PROCESSED.value
    assert after["chunk_ids"] == ["chunk-aaa", "chunk-bbb"]

    await storage.finalize()


@pytest.mark.offline
async def test_scan_skips_documents_that_are_already_indexed(tmp_path: Path) -> None:
    storage = await _doc_status(tmp_path)
    app = _app(tmp_path, storage)
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "known.txt").write_text("already indexed", encoding="utf-8")
    doc_id = compute_mdhash_id("already indexed", prefix="doc-")
    await storage.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSED.value,
                "file_path": str(inputs / "known.txt"),
                "chunk_ids": ["chunk-keep"],
                "created_at": 1,
                "updated_at": 1,
            }
        }
    )

    with TestClient(app) as client:
        result = client.post("/documents/scan").json()

    assert result["skipped"] == 1
    assert result["enqueued"] == 0
    assert (await storage.get_by_id(doc_id))["chunk_ids"] == ["chunk-keep"]

    await storage.finalize()


@pytest.mark.offline
async def test_text_ingest_stores_the_body_on_disk_not_in_doc_status(
    tmp_path: Path,
) -> None:
    storage = await _doc_status(tmp_path)
    app = _app(tmp_path, storage)
    body = "a body that must not live inside the status record"

    with TestClient(app) as client:
        doc_id = client.post("/documents/text", json={"text": body}).json()["doc_id"]

    record = await storage.get_by_id(doc_id)
    assert "content" not in record
    assert record["content_length"] == len(body)
    assert record["content_summary"]
    spooled = tmp_path / "inputs" / "__inline__" / inline_source_name(doc_id)
    assert spooled.read_text(encoding="utf-8") == body

    await storage.finalize()


@pytest.mark.offline
def test_inline_source_name_cannot_escape_the_input_directory() -> None:
    # doc_id comes straight from the request body.
    name = inline_source_name("../../etc/passwd")
    assert "/" not in name and ".." not in name
    assert Path(name).name == name
    # Distinct ids that sanitise to the same stem must not share a file.
    assert inline_source_name("a/b") != inline_source_name("a:b")


@pytest.mark.offline
async def test_list_documents_paginates_and_never_returns_a_body(
    tmp_path: Path,
) -> None:
    storage = await _doc_status(tmp_path)
    await storage.upsert(
        {
            f"doc-{i:03d}": {
                "status": DocStatus.PROCESSED.value,
                "file_path": f"/inputs/{i}.txt",
                # A record written by an earlier version, body and all.
                "content": "x" * 5000,
                "created_at": i,
                "updated_at": i,
            }
            for i in range(25)
        }
    )
    app = _app(tmp_path, storage)

    with TestClient(app) as client:
        page = client.get("/documents/", params={"limit": 10, "offset": 0}).json()
        tail = client.get("/documents/", params={"limit": 10, "offset": 20}).json()
        filtered = client.get("/documents/", params={"status": "pending"}).json()
        bad = client.get("/documents/", params={"status": "nonsense"})

    assert page["total"] == 25
    assert len(page["statuses"]) == 10
    assert page["next_offset"] == 10
    assert len(tail["statuses"]) == 5
    assert tail["next_offset"] is None
    # Unpaginated, this response carried 25 x 5 kB of document text.
    for record in page["statuses"].values():
        assert "content" not in record
        assert record["content_length"] == 5000
    assert filtered["statuses"] == {}
    assert bad.status_code == 400

    await storage.finalize()


@pytest.mark.offline
async def test_graph_export_keeps_edges_connected_and_hides_passage_text(
    tmp_path: Path,
) -> None:
    storage = await _doc_status(tmp_path)
    rag = _rag_with_doc_status(storage)
    rag.graph = MagicMock()
    rag.graph.get_all_nodes = AsyncMock(
        return_value=[
            {"id": f"n{i}", "label": "Passage", "props": {"content": "secret " * 50},
             "content": "secret " * 50}
            for i in range(10)
        ]
    )
    rag.graph.get_all_edges = AsyncMock(
        return_value=[
            {"source": "n0", "target": "n1", "type": "rel"},
            # Endpoint outside the first page: keeping this edge produced a dangling
            # endpoint in every visualisation client.
            {"source": "n1", "target": "n9", "type": "rel"},
        ]
    )
    app = create_app(
        _test_args(input_dir=str(tmp_path / "inputs")), testing=True, rag=rag
    )

    with TestClient(app) as client:
        data = client.get("/graphs", params={"limit": 3}).json()

    node_ids = {n["id"] for n in data["nodes"]}
    assert len(node_ids) == 3
    for edge in data["edges"]:
        assert edge["source"] in node_ids and edge["target"] in node_ids
    assert data["total_nodes"] == 10
    assert data["truncated"] is True
    for node in data["nodes"]:
        assert "content" not in node
        assert "content" not in node["props"]
        assert node["content_length"] == len("secret " * 50)

    await storage.finalize()


@pytest.mark.offline
async def test_health_hides_paths_and_reports_honest_readiness(tmp_path: Path) -> None:
    storage = await _doc_status(tmp_path)
    app = _app(tmp_path, storage)

    with TestClient(app) as client:
        healthy = client.get("/health").json()
        assert healthy["ready"] is True

        # A failed prepare_retrieval used to leave /health saying "healthy" with no
        # hint that every query was going to fail.
        app.state.retrieval_ready = False
        app.state.retrieval_error = "embedding endpoint unreachable"
        degraded = client.get("/health")
        not_ready = client.get("/health/ready")

    assert degraded.status_code == 200, "liveness must survive a retrieval failure"
    assert degraded.json()["ready"] is False
    assert degraded.json()["retrieval_status"] == "error"
    # /health is whitelisted, so these were server filesystem layout for anonymous
    # callers.
    assert "working_dir" not in healthy
    assert "workspace" not in healthy
    assert not_ready.status_code == 503
    assert not_ready.json()["ready"] is False

    await storage.finalize()


@pytest.mark.offline
async def test_request_id_is_echoed_generated_and_sanitized(tmp_path: Path) -> None:
    storage = await _doc_status(tmp_path)
    app = _app(tmp_path, storage)

    with TestClient(app) as client:
        echoed = client.get("/health", headers={"X-Request-ID": "trace-42"})
        generated = client.get("/health")
        dirty = client.get("/health", headers={"X-Request-ID": 'ab cd"ef'})

    assert echoed.headers["X-Request-ID"] == "trace-42"
    assert generated.headers["X-Request-ID"]
    assert generated.headers["X-Request-ID"] != echoed.headers["X-Request-ID"]
    # Caller-controlled: anything that could forge a log line or split a header goes.
    assert dirty.headers["X-Request-ID"] == "abcdef"

    await storage.finalize()


@pytest.mark.offline
async def test_metrics_are_authenticated_and_labelled_by_route(tmp_path: Path) -> None:
    storage = await _doc_status(tmp_path)
    app = _app(tmp_path, storage, key="metrics-key", whitelist_paths="/health")

    with TestClient(app) as client:
        client.get("/health")
        client.get("/documents/doc-missing", headers={"X-API-Key": "metrics-key"})
        anonymous = client.get("/metrics")
        authorised = client.get("/metrics", headers={"X-API-Key": "metrics-key"})

    assert anonymous.status_code == 403
    assert authorised.status_code == 200
    body = authorised.text
    assert 'memgraphrag_http_requests_total{method="GET",route="/health",code="200"}' in body
    # Route *template*, not the raw path: one series per route, not per document id.
    assert 'route="/documents/{doc_id}"' in body
    assert "memgraphrag_http_request_duration_seconds_bucket" in body
    assert "memgraphrag_pipeline_busy" in body

    await storage.finalize()
