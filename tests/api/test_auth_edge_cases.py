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
