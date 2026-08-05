"""Normalize MemGraphRAG ``/query`` payloads for CLI and Streamlit clients."""

from __future__ import annotations

from typing import Any


def _as_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    out: list[int] = []
    for item in values:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def normalize_query_payload(data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a ``/query`` (or final stream event) body into a common shape."""
    data = data if isinstance(data, dict) else {}
    answer = data.get("answer") or data.get("response") or ""
    citations = data.get("citations") or []
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    references = data.get("references") or []
    if not isinstance(references, list):
        references = []
    # Build references from sources when the server omitted them.
    if not references and sources:
        references = [
            {
                "reference_id": str(i),
                "file_path": str(src or "unknown"),
                "content": None,
            }
            for i, src in enumerate(sources, start=1)
        ]
    docs = data.get("docs") or []
    if not isinstance(docs, list):
        docs = []
    scores = data.get("doc_scores") or []
    if not isinstance(scores, list):
        scores = []
    return {
        "question": data.get("question"),
        "answer": str(answer) if answer is not None else "",
        "response": str(answer) if answer is not None else "",
        "thought": data.get("thought"),
        "citations": _as_int_list(citations),
        "confidence": data.get("confidence"),
        "structured": bool(data.get("structured")),
        "sources": [str(s) for s in sources],
        "references": [r for r in references if isinstance(r, dict)],
        "docs": [str(d) for d in docs],
        "doc_scores": scores,
    }


def merge_stream_event(
    acc: dict[str, Any], obj: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Merge one SSE JSON object into an accumulator.

    Returns ``(updated_acc, display_text)`` where ``display_text`` is the
    answer/response fragment suitable for progressive rendering.
    """
    merged = dict(acc)
    for key in (
        "question",
        "answer",
        "response",
        "thought",
        "citations",
        "confidence",
        "structured",
        "sources",
        "references",
        "docs",
        "doc_scores",
    ):
        if key in obj and obj[key] is not None:
            merged[key] = obj[key]
    # Prefer answer; fall back to response. If both present and response is a
    # short token delta appended to answer, keep the longer answer field.
    answer = merged.get("answer") or merged.get("response") or ""
    display = str(obj.get("answer") or obj.get("response") or obj.get("error") or "")
    if not merged.get("answer") and answer:
        merged["answer"] = answer
    return normalize_query_payload(merged), display


def source_label(ref: dict[str, Any], index: int) -> str:
    """Human-readable label for a reference row."""
    path = str(ref.get("file_path") or ref.get("source") or "unknown")
    rid = str(ref.get("reference_id") or index)
    return f"[{rid}] {path}"
