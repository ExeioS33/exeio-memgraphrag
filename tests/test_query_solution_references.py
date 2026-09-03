"""LightRAG-style query references."""

from __future__ import annotations

import pytest

from memgraphrag.api.routers.query import _solution_payload
from memgraphrag.utils.misc import QuerySolution

pytestmark = pytest.mark.offline


def test_one_reference_per_passage_numbered_like_the_fences() -> None:
    """The numbering has to line up with what the model was told to cite.

    `fence_passages` labels passages [1..n] in the order of `docs`, and the QA
    system prompt asks for those numbers. References used to be collapsed per
    document, so three passages from two files produced two references and an
    answer citing [3] pointed at nothing. One entry per passage is what makes a
    citation resolvable — and what lets a click open the exact chunk.
    """
    sol = QuerySolution(
        question="q",
        docs=["a", "b", "c"],
        sources=["paper.pdf", "guide.md", "paper.pdf"],
        passage_ids=["chunk-1", "chunk-2", "chunk-3"],
        source_paths=["/corpus/paper.pdf", "/corpus/sub/guide.md", "/corpus/paper.pdf"],
    )
    refs = sol.ensure_references()
    assert [r["reference_id"] for r in refs] == ["1", "2", "3"]
    assert [r["file_path"] for r in refs] == ["paper.pdf", "guide.md", "paper.pdf"]
    assert [r["chunk_id"] for r in refs] == ["chunk-1", "chunk-2", "chunk-3"]
    assert refs[1]["source_path"] == "/corpus/sub/guide.md"


def test_reference_numbering_can_continue_a_multi_hop_turn() -> None:
    """A second retrieval in one turn must not restart the numbering at 1.

    Two hops each renumbering from 1 put two different `[1]` markers in the same
    answer, and the list under it no longer matched the prose.
    """
    second_hop = QuerySolution(
        question="q",
        docs=["d", "e"],
        sources=["other.pdf", "other.pdf"],
        passage_ids=["chunk-9", "chunk-10"],
    )
    refs = second_hop.ensure_references(start=4)
    assert [r["reference_id"] for r in refs] == ["4", "5"]


def test_missing_provenance_degrades_to_unknown_without_dropping_the_slot() -> None:
    """A passage with no doc-status record still needs its number.

    Skipping it would shift every later reference by one and misattribute them.
    """
    sol = QuerySolution(question="q", docs=["a", "b"], sources=["", "guide.md"])
    refs = sol.ensure_references()
    assert [r["reference_id"] for r in refs] == ["1", "2"]
    assert refs[0]["file_path"] == "unknown"
    assert refs[0]["chunk_id"] is None


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
