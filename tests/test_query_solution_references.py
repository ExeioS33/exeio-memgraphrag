"""LightRAG-style query references."""

from __future__ import annotations

import pytest

from memgraphrag.api.routers.query import _solution_payload
from memgraphrag.utils.misc import QuerySolution

pytestmark = pytest.mark.offline


def test_ensure_references_unique_by_file_path() -> None:
    sol = QuerySolution(
        question="q",
        docs=["a", "b", "c"],
        sources=["paper.pdf", "guide.md", "paper.pdf"],
    )
    refs = sol.ensure_references()
    assert refs == [
        {"reference_id": "1", "file_path": "paper.pdf", "content": None},
        {"reference_id": "2", "file_path": "guide.md", "content": None},
    ]


def test_solution_payload_lightrag_shape() -> None:
    sol = QuerySolution(
        question="q",
        docs=["passage"],
        answer="Thought: x\nAnswer: hello",
        sources=["doc.pdf"],
    )
    payload = _solution_payload(sol)
    assert set(payload.keys()) == {"response", "references"}
    assert payload["response"] == "Thought: x\nAnswer: hello"
    assert payload["references"][0]["file_path"] == "doc.pdf"
