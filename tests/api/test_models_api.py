"""Per-request model selection and the client parameter registry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgraphrag.api.config import namespace_from_dict
from memgraphrag.api.routers.query import models_for
from memgraphrag.api.server import create_app
from memgraphrag.utils.misc import QuerySolution

pytestmark = pytest.mark.offline


def _mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    rag.top_k = 10
    rag.arag_qa = AsyncMock(
        return_value=QuerySolution(question="q", docs=[], answer="ok", sources=[])
    )
    return rag


def _client(**overrides) -> TestClient:
    app = create_app(namespace_from_dict(overrides), testing=True, rag=_mock_rag())
    return TestClient(app)


def test_default_model_is_always_offered() -> None:
    """Without LLM_MODELS the running model must still be selectable — otherwise the
    picker shows the one choice guaranteed to work and 400s on it."""
    args = namespace_from_dict({"llm_model": "gpt-4o-mini", "llm_models": ""})
    assert models_for(args) == ["gpt-4o-mini"]


def test_allow_list_is_prefixed_by_the_default() -> None:
    args = namespace_from_dict({"llm_model": "a", "llm_models": "b, c"})
    assert models_for(args) == ["a", "b", "c"]
    args = namespace_from_dict({"llm_model": "b", "llm_models": "b,c"})
    assert models_for(args) == ["b", "c"]


def test_models_endpoint_reports_default_and_list() -> None:
    with _client(llm_model="gpt-4o-mini", llm_models="gpt-4o") as client:
        body = client.get("/models").json()
    assert body["default"] == "gpt-4o-mini"
    assert body["models"] == ["gpt-4o-mini", "gpt-4o"]


def test_query_accepts_an_allowed_model() -> None:
    with _client(llm_model="gpt-4o-mini", llm_models="gpt-4o") as client:
        response = client.post("/query", json={"query": "q", "model": "gpt-4o"})
    assert response.status_code == 200


def test_query_rejects_an_unlisted_model() -> None:
    with _client(llm_model="gpt-4o-mini", llm_models="") as client:
        response = client.post("/query", json={"query": "q", "model": "claude-opus-5"})
    assert response.status_code == 400
    assert "claude-opus-5" in response.json()["detail"]


def test_query_without_a_model_is_unaffected() -> None:
    with _client() as client:
        assert client.post("/query", json={"query": "q"}).status_code == 200


def test_query_params_registry_is_served() -> None:
    with _client() as client:
        body = client.get("/query/params").json()
    names = {spec["name"] for spec in body["params"]}
    # Registry comes from memgraphrag/client/params.py; these are its anchors.
    assert {"mode", "top_k", "damping"} <= names
    assert body["presets"], "presets drive the UI's quick-settings chips"
    assert ".pdf" in body["supported_extensions"]
