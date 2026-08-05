"""API wiring tests for structured query output."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgraphrag.api.server import create_app
from memgraphrag.utils.misc import QuerySolution


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


def _mock_rag(sol: QuerySolution) -> MagicMock:
    rag = MagicMock()
    rag.working_dir = "/tmp/memgraphrag-test"
    rag.workspace = ""
    rag.top_k = 10
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    rag.arag_qa = AsyncMock(return_value=sol)
    rag.aquery = AsyncMock(return_value=sol)
    rag.aretrieve = AsyncMock(return_value=[sol])
    return rag


@pytest.mark.offline
def test_query_returns_structured_fields() -> None:
    sol = QuerySolution(
        question="What is PPR?",
        docs=["Personalized PageRank ranks passages."],
        doc_scores=[0.91],
        answer="Personalized PageRank over the memory graph [1] (paper.pdf).",
        thought="Passage 1 (paper.pdf) mentions PPR.",
        citations=[1],
        confidence="high",
        structured=True,
        sources=["paper.pdf"],
    )
    app = create_app(_test_args(), testing=True, rag=_mock_rag(sol))
    with TestClient(app) as client:
        resp = client.post(
            "/query",
            json={"query": "What is PPR?", "mode": "ppr", "top_k": 5},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Personalized PageRank over the memory graph [1] (paper.pdf)."
    assert body["response"] == body["answer"]
    assert body["thought"] == "Passage 1 (paper.pdf) mentions PPR."
    assert body["citations"] == [1]
    assert body["confidence"] == "high"
    assert body["structured"] is True
    assert body["sources"] == ["paper.pdf"]
    assert body["references"] == [
        {"reference_id": "1", "file_path": "paper.pdf", "content": None}
    ]
    assert body["docs"]


@pytest.mark.offline
def test_query_passes_structured_output_false() -> None:
    sol = QuerySolution(question="q", docs=[], answer="ok")
    rag = _mock_rag(sol)
    app = create_app(_test_args(), testing=True, rag=rag)
    with TestClient(app) as client:
        resp = client.post(
            "/query",
            json={
                "query": "hello world",
                "mode": "bypass",
                "structured_output": False,
            },
        )
    assert resp.status_code == 200
    assert rag.arag_qa.await_count == 1
    _args, kwargs = rag.arag_qa.await_args
    param = kwargs.get("param") or (_args[1] if len(_args) > 1 else None)
    assert param is not None
    assert param.structured_output is False
