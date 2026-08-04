"""Offline unit tests for MemGraphRAGClient (httpx.MockTransport)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from memgraphrag.client.http import MemGraphRAGClient


def _json_response(request: httpx.Request, payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
        request=request,
    )


@pytest.mark.offline
def test_health_and_api_key_header() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["api_key"] = request.headers.get("x-api-key")
        return _json_response(
            request,
            {
                "status": "healthy",
                "core_version": "0.1.0",
                "api_version": "0.1.0",
                "pipeline_busy": False,
            },
        )

    transport = httpx.MockTransport(handler)
    with MemGraphRAGClient(
        base_url="http://test", api_key="secret-key", transport=transport
    ) as client:
        data = client.health()

    assert data["status"] == "healthy"
    assert seen["path"] == "/health"
    assert seen["api_key"] == "secret-key"


@pytest.mark.offline
def test_query_passes_only_need_context() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode("utf-8")))
        return _json_response(request, {"answer": "ok", "docs": [], "doc_scores": []})

    transport = httpx.MockTransport(handler)
    with MemGraphRAGClient(base_url="http://test", transport=transport) as client:
        client.query("hi", mode="bypass", only_need_context=False, top_k=3)

    assert bodies[0]["query"] == "hi"
    assert bodies[0]["mode"] == "bypass"
    assert bodies[0]["top_k"] == 3
    assert bodies[0]["only_need_context"] is False


@pytest.mark.offline
def test_query_data_and_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/query/data":
            return _json_response(
                request,
                {"data": {"docs": ["a", "b"], "doc_scores": [0.9, 0.4]}},
            )
        if request.url.path == "/query/stream":
            sse = (
                'data: {"response": "Hel"}\n\n'
                'data: {"response": "lo"}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse.encode("utf-8"),
                request=request,
            )
        return _json_response(request, {"error": "unexpected"}, status=404)

    transport = httpx.MockTransport(handler)
    with MemGraphRAGClient(base_url="http://test", transport=transport) as client:
        data = client.query_data("q", top_k=2)
        chunks = list(client.query_stream("q"))

    assert data["data"]["docs"] == ["a", "b"]
    assert chunks == ['{"response": "Hel"}', '{"response": "lo"}', "[DONE]"]


@pytest.mark.offline
def test_documents_upload_list_scan_clear(tmp_path) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/documents/upload":
            return _json_response(request, {"doc_id": "d1", "filename": "note.txt"})
        if request.url.path == "/documents/":
            if request.method == "GET":
                return _json_response(
                    request, {"statuses": {"d1": {"status": "processed", "file_path": "note.txt"}}}
                )
            if request.method == "DELETE":
                return _json_response(request, {"status": "not_implemented"})
        if request.url.path == "/documents/text":
            return _json_response(request, {"doc_id": "t1"})
        if request.url.path == "/documents/scan":
            return _json_response(request, {"files_found": 2, "input_dir": "/inputs"})
        return _json_response(request, {"error": "unexpected"}, status=404)

    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")

    transport = httpx.MockTransport(handler)
    with MemGraphRAGClient(base_url="http://test", transport=transport) as client:
        up = client.upload_file(note)
        listed = client.list_documents()
        texted = client.insert_text("inline")
        scanned = client.scan_input_dir()
        cleared = client.clear_documents()
        by_bytes = client.upload_bytes(b"x", "x.md")

    assert up["doc_id"] == "d1"
    assert listed["statuses"]["d1"]["status"] == "processed"
    assert texted["doc_id"] == "t1"
    assert scanned["files_found"] == 2
    assert cleared["status"] == "not_implemented"
    assert by_bytes["doc_id"] == "d1"
    assert ("POST", "/documents/upload") in calls
    assert ("GET", "/documents/") in calls
    assert ("DELETE", "/documents/") in calls


@pytest.mark.offline
def test_graph_endpoints() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graph/label/list":
            return _json_response(request, {"labels": ["Passage", "Fact"]})
        if request.url.path == "/graphs":
            assert request.url.params.get("limit") == "10"
            assert request.url.params.get("label") == "Passage"
            return _json_response(
                request,
                {
                    "nodes": [{"id": "n1", "label": "Passage"}],
                    "edges": [{"source": "n1", "target": "n2"}],
                },
            )
        return _json_response(request, {"error": "unexpected"}, status=404)

    transport = httpx.MockTransport(handler)
    with MemGraphRAGClient(base_url="http://test", transport=transport) as client:
        labels = client.list_labels()
        graph = client.explore_graph(label="Passage", limit=10)

    assert labels["labels"] == ["Passage", "Fact"]
    assert len(graph["nodes"]) == 1
    assert len(graph["edges"]) == 1


@pytest.mark.offline
def test_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, {"detail": "nope"}, status=401)

    transport = httpx.MockTransport(handler)
    with MemGraphRAGClient(base_url="http://test", transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            client.health()
    assert "401" in str(exc.value)


@pytest.mark.offline
def test_upload_url_rejects_bad_schemes() -> None:
    with MemGraphRAGClient(base_url="http://test", verify=False) as client:
        with pytest.raises(ValueError, match="empty"):
            client.upload_url("  ")
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client.upload_url("ftp://example.com/a.pdf")


@pytest.mark.offline
def test_upload_url_wraps_ssl_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from memgraphrag.client import http as http_mod
    from memgraphrag.client.http import ClientSSLError

    class _BoomClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.verify = kwargs.get("verify")

        def __enter__(self) -> "_BoomClient":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def get(self, url: str) -> Any:
            raise httpx.ConnectError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                "unable to get local issuer certificate (_ssl.c:1028)"
            )

    # Build the API client first (real httpx), then swap Client for URL download only.
    client = MemGraphRAGClient(
        base_url="http://test",
        transport=httpx.MockTransport(lambda r: _json_response(r, {})),
        verify=True,
    )
    try:
        monkeypatch.setattr(http_mod.httpx, "Client", _BoomClient)
        with pytest.raises(ClientSSLError, match="MEMGRAPHRAG_SSL_CERT_FILE"):
            client.upload_url("https://example.com/paper.pdf")
    finally:
        client.close()
