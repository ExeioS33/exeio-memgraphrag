"""Tests for structured query payload normalization used by CLI / Streamlit."""

from __future__ import annotations

import pytest

from memgraphrag.client.result import (
    merge_stream_event,
    normalize_query_payload,
    source_label,
)


@pytest.mark.offline
def test_normalize_query_payload_structured() -> None:
    raw = {
        "question": "What is PPR?",
        "answer": "Personalized PageRank [1] (paper.pdf).",
        "thought": "Passage 1 explains PPR.",
        "citations": [1],
        "confidence": "high",
        "structured": True,
        "sources": ["paper.pdf"],
        "references": [
            {"reference_id": "1", "file_path": "paper.pdf", "content": None}
        ],
        "docs": ["PPR ranks passages."],
        "doc_scores": [0.91],
    }
    out = normalize_query_payload(raw)
    assert out["answer"].startswith("Personalized PageRank")
    assert out["response"] == out["answer"]
    assert out["citations"] == [1]
    assert out["sources"] == ["paper.pdf"]
    assert out["references"][0]["file_path"] == "paper.pdf"
    assert out["structured"] is True


@pytest.mark.offline
def test_normalize_builds_references_from_sources() -> None:
    out = normalize_query_payload(
        {"answer": "ok", "sources": ["a.pdf", "b.md"], "docs": ["x", "y"]}
    )
    assert out["references"] == [
        {"reference_id": "1", "file_path": "a.pdf", "content": None},
        {"reference_id": "2", "file_path": "b.md", "content": None},
    ]


@pytest.mark.offline
def test_merge_stream_event_keeps_structured_fields() -> None:
    acc, text = merge_stream_event(
        {},
        {
            "response": "Hello",
            "answer": "Hello from paper.pdf [1].",
            "thought": "Cited passage 1.",
            "citations": [1],
            "confidence": "medium",
            "structured": True,
            "sources": ["paper.pdf"],
            "references": [
                {"reference_id": "1", "file_path": "paper.pdf", "content": None}
            ],
        },
    )
    assert "Hello" in text
    assert acc["answer"] == "Hello from paper.pdf [1]."
    assert acc["references"][0]["file_path"] == "paper.pdf"
    assert acc["citations"] == [1]


@pytest.mark.offline
def test_source_label() -> None:
    assert source_label({"reference_id": "2", "file_path": "x.pdf"}, 9) == "[2] x.pdf"
    assert source_label({}, 3) == "[3] unknown"


@pytest.mark.offline
def test_query_passes_structured_output_flag() -> None:
    import json
    from typing import Any

    import httpx

    from memgraphrag.client.http import MemGraphRAGClient

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "answer": "ok",
                    "structured": True,
                    "sources": ["a.pdf"],
                    "references": [
                        {
                            "reference_id": "1",
                            "file_path": "a.pdf",
                            "content": None,
                        }
                    ],
                    "docs": [],
                    "doc_scores": [],
                }
            ).encode("utf-8"),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with MemGraphRAGClient(base_url="http://test", transport=transport) as client:
        out = client.query("hi", mode="ppr", structured_output=True, top_k=3)

    assert bodies[0]["structured_output"] is True
    assert bodies[0]["top_k"] == 3
    assert out["references"][0]["file_path"] == "a.pdf"
