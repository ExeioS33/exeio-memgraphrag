"""Health endpoint smoke test for MemGraphRAG API."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgraphrag.api.server import create_app


def _mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.working_dir = "/tmp/memgraphrag-test"
    rag.workspace = ""
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
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
def test_health_endpoint() -> None:
    app = create_app(_test_args(), testing=True, rag=_mock_rag())
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "api_version" in data
    assert "core_version" in data
