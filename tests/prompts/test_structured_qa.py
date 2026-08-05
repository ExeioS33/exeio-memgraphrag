"""Unit tests for structured RAG QA prompts and parsing."""

from __future__ import annotations

from memgraphrag.prompts.templates import (
    RAG_QA_STRUCTURED_SYSTEM,
    parse_structured_qa,
    render_rag_qa_structured,
)


def test_render_rag_qa_structured_numbers_passages() -> None:
    system, user = render_rag_qa_structured(
        "What is MemGraphRAG?",
        ["First passage about memory.", "Second passage about PPR."],
        sources=["paper.pdf", "guide.md"],
    )
    assert system == RAG_QA_STRUCTURED_SYSTEM
    assert "[Passage 1 | Source: paper.pdf]" in user
    assert "[Passage 2 | Source: guide.md]" in user
    assert "First passage about memory." in user
    assert "What is MemGraphRAG?" in user
    assert "Always cite Source filenames" in user
    assert "MUST reference document sources" in system
    assert "Benchmarks" in system
    assert "Markdown tables" in system
    assert "domain-adapted" in user


def test_parse_structured_qa_valid_json() -> None:
    raw = """{
      "thought": "Passage 1 defines the system.",
      "answer": "A memory-based GraphRAG engine [1] (paper.pdf).",
      "citations": [1, 2],
      "sources": [{"passage": 1, "file_path": "paper.pdf"}],
      "confidence": "High"
    }"""
    parsed = parse_structured_qa(raw)
    assert parsed["structured"] is True
    assert parsed["answer"] == "A memory-based GraphRAG engine [1] (paper.pdf)."
    assert parsed["thought"] == "Passage 1 defines the system."
    assert parsed["citations"] == [1, 2]
    assert parsed["sources"] == [{"passage": 1, "file_path": "paper.pdf"}]
    assert parsed["confidence"] == "high"


def test_parse_structured_qa_fenced_json() -> None:
    raw = """```json
{"thought": "ok", "answer": "yes", "citations": ["1"], "confidence": "medium"}
```"""
    parsed = parse_structured_qa(raw)
    assert parsed["structured"] is True
    assert parsed["answer"] == "yes"
    assert parsed["citations"] == [1]
    assert parsed["confidence"] == "medium"


def test_parse_structured_qa_legacy_thought_answer() -> None:
    raw = "Thought: because of X.\nAnswer: final answer"
    parsed = parse_structured_qa(raw)
    assert parsed["structured"] is False
    assert parsed["answer"] == "final answer"
    assert parsed["thought"] == "because of X."
    assert parsed["citations"] == []


def test_parse_structured_qa_plain_text_fallback() -> None:
    parsed = parse_structured_qa("just a plain answer")
    assert parsed["structured"] is False
    assert parsed["answer"] == "just a plain answer"
    assert parsed["confidence"] is None
