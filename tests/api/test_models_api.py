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


def test_models_endpoint_reports_default_providers_and_locked_embedding() -> None:
    with _client(
        llm_model="gpt-4o-mini",
        llm_models="gpt-4o",
        embedding_model="intfloat/multilingual-e5-large-instruct",
        embedding_dim=1024,
    ) as client:
        body = client.get("/models").json()

    assert body["default"] == {"provider": "default", "model": "gpt-4o-mini"}
    assert body["models"] == ["gpt-4o-mini", "gpt-4o"]

    ids = [p["id"] for p in body["providers"]]
    assert {"default", "together", "ollama", "openai", "vllm"} <= set(ids)

    # Ollama needs no credential, so it is always usable; the others depend on env.
    ollama = next(p for p in body["providers"] if p["id"] == "ollama")
    assert ollama["available"] is True
    assert ollama["base_url"] == "http://localhost:11434/v1"

    # The embedding model is reported but never selectable — the corpus is indexed
    # with it, so offering to change it would promise something that breaks answers.
    assert body["embedding"]["locked"] is True
    assert body["embedding"]["dim"] == 1024
    assert body["embedding"]["reason"]


def test_unknown_provider_is_rejected() -> None:
    with _client() as client:
        response = client.post("/query", json={"query": "q", "provider": "anthropic"})
    assert response.status_code == 400
    assert "anthropic" in response.json()["detail"]


def test_provider_without_a_credential_is_rejected_by_name(monkeypatch) -> None:
    """A missing key is an operator problem and must say so, rather than being
    forwarded and surfacing as an opaque 401 from someone else's API."""
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    with _client() as client:
        response = client.post("/query", json={"query": "q", "provider": "together"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "together" in detail and "TOGETHER_API_KEY" in detail


def test_provider_allow_list_comes_from_its_own_env(monkeypatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")
    monkeypatch.setenv("TOGETHER_MODELS", "openai/gpt-oss-20b")
    with _client() as client:
        ok = client.post(
            "/query", json={"query": "q", "provider": "together", "model": "openai/gpt-oss-20b"}
        )
        bad = client.post("/query", json={"query": "q", "provider": "together", "model": "gpt-4o"})
    assert ok.status_code == 200
    assert bad.status_code == 400
    assert "gpt-4o" in bad.json()["detail"]


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
