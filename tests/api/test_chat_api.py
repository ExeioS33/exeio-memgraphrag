"""Chat thread and message routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgraphrag.api.config import namespace_from_dict
from memgraphrag.api.server import create_app

pytestmark = pytest.mark.offline


def _mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    return rag


def _client(**overrides) -> TestClient:
    args = namespace_from_dict(overrides)
    app = create_app(args, testing=True, rag=_mock_rag())
    return TestClient(app)


def test_thread_crud_round_trip() -> None:
    with _client() as client:
        created = client.post("/chat/threads", json={"title": "Budget"})
        assert created.status_code == 200
        thread_id = created.json()["id"]

        listed = client.get("/chat/threads")
        assert listed.status_code == 200
        assert [t["id"] for t in listed.json()["threads"]] == [thread_id]
        assert listed.json()["total"] == 1

        client.post(
            f"/chat/threads/{thread_id}/messages",
            json={"role": "user", "content": "Quel est le budget ?"},
        )
        client.post(
            f"/chat/threads/{thread_id}/messages",
            json={
                "role": "assistant",
                "content": "42 euros.",
                "references": [{"reference_id": "1", "file_path": "/a.pdf", "content": None}],
            },
        )

        fetched = client.get(f"/chat/threads/{thread_id}").json()
        assert [m["role"] for m in fetched["messages"]] == ["user", "assistant"]
        assert fetched["messages"][1]["references"][0]["file_path"] == "/a.pdf"

        renamed = client.patch(f"/chat/threads/{thread_id}", json={"title": "Autre"})
        assert renamed.json()["title"] == "Autre"

        assert client.delete(f"/chat/threads/{thread_id}").status_code == 200
        assert client.get(f"/chat/threads/{thread_id}").status_code == 404


def test_patch_only_touches_the_fields_it_was_given() -> None:
    """PATCH {"title": ...} must not blank the thread's model."""
    with _client() as client:
        thread_id = client.post("/chat/threads", json={"model": "gpt-4o-mini"}).json()["id"]
        patched = client.patch(f"/chat/threads/{thread_id}", json={"title": "Renommé"})
        assert patched.json()["title"] == "Renommé"
        assert patched.json()["model"] == "gpt-4o-mini"


def test_unknown_thread_is_404() -> None:
    with _client() as client:
        assert client.get("/chat/threads/nope").status_code == 404
        assert client.delete("/chat/threads/nope").status_code == 404
        assert client.patch("/chat/threads/nope", json={"title": "x"}).status_code == 404
        assert (
            client.post(
                "/chat/threads/nope/messages", json={"role": "user", "content": "hi"}
            ).status_code
            == 404
        )


def test_invalid_role_is_rejected_by_validation() -> None:
    with _client() as client:
        thread_id = client.post("/chat/threads", json={}).json()["id"]
        response = client.post(
            f"/chat/threads/{thread_id}/messages",
            json={"role": "system", "content": "nope"},
        )
        assert response.status_code == 422


def test_routes_answer_503_without_an_application_database() -> None:
    """No APP_DATABASE_URL means no chat store — an explicit 503, not a silent
    degradation to some other backend."""
    args = namespace_from_dict({})
    app = create_app(args, testing=True, rag=_mock_rag())
    app.state.chat_store = None
    with TestClient(app) as client:
        response = client.get("/chat/threads")
        assert response.status_code == 503
        assert "APP_DATABASE_URL" in response.json()["detail"]
        # The rest of the API is unaffected.
        assert client.get("/health").status_code == 200
