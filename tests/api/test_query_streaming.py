"""POST /query/stream now emits one frame per token.

The wire format is unchanged — `references`, then `response` frames, then [DONE] —
so a client that concatenates `response` values keeps working; it just gets many
small frames instead of one big one.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgraphrag.api.config import namespace_from_dict
from memgraphrag.api.server import create_app
from memgraphrag.utils.misc import QuerySolution

pytestmark = pytest.mark.offline

REFS = [{"reference_id": "1", "file_path": "/corpus/a.pdf", "content": None}]


def _base_rag() -> MagicMock:
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    rag.top_k = 10
    return rag


def _streaming_rag(tokens: list[str]) -> MagicMock:
    rag = _base_rag()

    async def astream_qa(_query: str, param: Any = None) -> AsyncIterator[dict[str, Any]]:
        assert param is not None
        yield {"references": REFS}
        for token in tokens:
            yield {"token": token}
        yield {"done": True, "answer": "".join(tokens)}

    rag.astream_qa = astream_qa
    return rag


def _frames(body: str) -> list[Any]:
    """Parse an SSE body into its decoded payloads."""
    out: list[Any] = []
    for block in body.split("\n\n"):
        line = block.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        out.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return out


def _client(rag: MagicMock) -> TestClient:
    app = create_app(namespace_from_dict({}), testing=True, rag=rag)
    return TestClient(app)


def test_tokens_arrive_as_separate_frames() -> None:
    with _client(_streaming_rag(["Le ", "budget ", "est ", "de 42."])) as client:
        response = client.post("/query/stream", json={"query": "budget ?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _frames(response.text)

    assert frames[0] == {"references": REFS}, "references must precede the first token"
    assert frames[-1] == "[DONE]"
    tokens = [f["response"] for f in frames[1:-1] if isinstance(f, dict) and "response" in f]
    assert tokens == ["Le ", "budget ", "est ", "de 42."]
    assert "".join(tokens) == "Le budget est de 42."


def test_falls_back_to_the_buffered_answer_without_astream_qa() -> None:
    """An engine that cannot stream must still answer, not fail after the 200."""

    class _NoStreamRag:
        top_k = 10
        initialize_storages = AsyncMock()
        finalize_storages = AsyncMock()
        prepare_retrieval = AsyncMock()

        async def arag_qa(self, query: str, param: Any = None) -> QuerySolution:
            assert param is not None
            sol = QuerySolution(question=query, docs=[], answer="réponse complète", sources=[])
            sol.ensure_references()
            return sol

    app = create_app(namespace_from_dict({}), testing=True, rag=_NoStreamRag())
    with TestClient(app) as client:
        response = client.post("/query/stream", json={"query": "q"})

    frames = _frames(response.text)
    assert frames[-1] == "[DONE]"
    tokens = [f["response"] for f in frames if isinstance(f, dict) and "response" in f]
    assert "".join(tokens) == "réponse complète"


def test_a_mocked_engine_does_not_silently_stream_nothing() -> None:
    """MagicMock answers True to hasattr(__aiter__), so the generator check must be
    inspect.isasyncgen — otherwise this yields an empty stream instead of falling
    back, and the failure is invisible."""
    rag = _base_rag()
    rag.arag_qa = AsyncMock(
        return_value=QuerySolution(question="q", docs=[], answer="secours", sources=[])
    )
    with _client(rag) as client:
        response = client.post("/query/stream", json={"query": "q"})

    tokens = [
        f["response"] for f in _frames(response.text) if isinstance(f, dict) and "response" in f
    ]
    assert "".join(tokens) == "secours"


def test_engine_failure_is_reported_in_band() -> None:
    """The 200 and headers are already committed, so an error can only be a frame."""
    rag = _base_rag()
    rag.arag_qa = AsyncMock(side_effect=RuntimeError("moteur indisponible"))
    with _client(rag) as client:
        response = client.post("/query/stream", json={"query": "q"})

    assert response.status_code == 200
    errors = [f for f in _frames(response.text) if isinstance(f, dict) and "error" in f]
    assert errors and "moteur indisponible" in errors[0]["error"]
