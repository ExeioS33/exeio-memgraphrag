"""Edge-case tests for auth, query-before-ready, and empty corpus."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgraphrag.api.server import create_app
from memgraphrag.retrieval import RetrievalStateManager


def _mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.working_dir = "/tmp/memgraphrag-test"
    rag.workspace = ""
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    rag.arag_qa = AsyncMock(side_effect=Exception("not ready"))
    rag.aquery = AsyncMock(side_effect=Exception("not ready"))
    rag.aretrieve = AsyncMock(side_effect=Exception("not ready"))
    return rag


def _test_args(**kwargs) -> SimpleNamespace:
    base = dict(
        host="127.0.0.1",
        port=9621,
        workers=1,
        working_dir="/tmp/memgraphrag-test",
        input_dir="/tmp/memgraphrag-inputs",
        workspace="",
        key=None,
        log_level="INFO",
        ssl=False,
        ssl_certfile=None,
        ssl_keyfile=None,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="IgraphStorage",
        doc_status_storage="JsonDocStatusStorage",
        llm_binding="openai",
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
        top_k=10,
        linking_top_k=50,
        passage_node_weight=0.05,
        damping=0.5,
        fact_similarity_threshold=0.6,
        skip_fact_rerank=True,
        ppr_engine="igraph",
        max_async_llm=4,
        auth_accounts="",
        token_secret=None,
        cors_origins="*",
        whitelist_paths="/health,/docs,/openapi.json,/api/*",
        ollama_model_name="memgraphrag",
        ollama_model_tag="latest",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.offline
def test_health_with_api_key_still_whitelisted() -> None:
    app = create_app(
        _test_args(key="secret-key", whitelist_paths="/health"),
        testing=True,
        rag=_mock_rag(),
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


@pytest.mark.offline
def test_query_requires_api_key_when_configured() -> None:
    rag = _mock_rag()
    rag.arag_qa = AsyncMock(return_value=[{"answer": "ok", "docs": []}])
    rag.aquery = AsyncMock(return_value=[{"answer": "ok", "docs": []}])
    rag.aretrieve = AsyncMock(return_value=[])
    app = create_app(
        _test_args(key="secret-key", whitelist_paths="/health"),
        testing=True,
        rag=rag,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        denied = client.post("/query", json={"query": "hi"})
        assert denied.status_code in (401, 403)
        ok = client.post(
            "/query",
            json={"query": "hi"},
            headers={"X-API-Key": "secret-key"},
        )
        assert ok.status_code != 401


@pytest.mark.offline
def test_retrieval_state_not_ready_by_default() -> None:
    mgr = RetrievalStateManager(rag=None, graph=None)
    assert mgr.is_ready is False
    with pytest.raises(Exception):
        mgr.require_ready()


@pytest.mark.offline
def test_ollama_api_not_whitelisted_by_default() -> None:
    """/api/* must not be reachable without a credential.

    The Ollama emulation router is mounted on /api and its chat/generate routes call
    the billed LLM (the /bypass prefix skips retrieval entirely). It used to sit in
    the default WHITELIST_PATHS, and the whitelist short-circuits before any token or
    key check — so the most expensive surface was open even with auth configured.
    """
    from memgraphrag.api.config import parse_args

    defaults = parse_args([])
    assert "/api/*" not in (defaults.whitelist_paths or "")

    app = create_app(
        _test_args(key="secret-key", whitelist_paths=defaults.whitelist_paths),
        testing=True,
        rag=_mock_rag(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        denied = client.post(
            "/api/chat",
            json={"model": "memgraphrag", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert denied.status_code in (401, 403)


@pytest.mark.offline
def test_create_app_honours_auth_accounts_argument() -> None:
    """AuthHandler must be built from `args`, not from the import-time global_args.

    It used to be a module-level singleton, so create_app(args) ran with no accounts
    and /login fell into its "authentication is disabled" branch, returning 200 and a
    valid guest token for any password.
    """
    app = create_app(
        _test_args(auth_accounts="admin:pw123", token_secret="unit-test-secret"),
        testing=True,
        rag=_mock_rag(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        bad = client.post("/login", data={"username": "admin", "password": "wrong"})
        assert bad.status_code == 401

        good = client.post("/login", data={"username": "admin", "password": "pw123"})
        assert good.status_code == 200
        assert good.json()["auth_mode"] == "enabled"


@pytest.mark.offline
def test_wildcard_cors_disables_credentials() -> None:
    """allow_origins=['*'] with allow_credentials=True makes Starlette reflect the
    caller's Origin, which is a CSRF primitive against every mutating endpoint."""
    from fastapi.middleware.cors import CORSMiddleware

    app = create_app(_test_args(cors_origins="*"), testing=True, rag=_mock_rag())
    cors = [m for m in app.user_middleware if m.cls is CORSMiddleware]
    assert len(cors) == 1
    assert cors[0].kwargs["allow_credentials"] is False

    app_explicit = create_app(
        _test_args(cors_origins="https://app.exeio.test"),
        testing=True,
        rag=_mock_rag(),
    )
    cors_explicit = [m for m in app_explicit.user_middleware if m.cls is CORSMiddleware]
    assert cors_explicit[0].kwargs["allow_credentials"] is True
