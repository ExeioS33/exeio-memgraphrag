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


@pytest.mark.offline
def test_to_dict_returns_every_doc_by_default() -> None:
    """Serialisation must not silently drop retrieved passages.

    `to_dict` hard-coded `docs[:5]`, so a request with top_k=20 was answered with 20
    passages by the engine and 5 by the API, with no total and no truncation flag.
    """
    sol = QuerySolution(
        question="q",
        docs=[f"passage {i}" for i in range(8)],
        doc_scores=[1.0 - i / 10 for i in range(8)],
        sources=[f"doc{i}.pdf" for i in range(8)],
    )

    full = sol.to_dict()
    assert len(full["docs"]) == 8
    assert len(full["doc_scores"]) == 8
    assert len(full["sources"]) == 8
    assert full["total_docs"] == 8

    # Callers that want a bound must ask for one, and still learn the real total.
    capped = sol.to_dict(max_docs=3)
    assert len(capped["docs"]) == 3
    assert len(capped["doc_scores"]) == 3
    assert capped["total_docs"] == 8
