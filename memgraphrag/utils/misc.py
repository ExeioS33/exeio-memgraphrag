"""
Miscellaneous helpers for MemGraphRAG.

Adapted from MemGraphRAG/code/src/utils/misc_utils.py (QuerySolution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class QuerySolution:
    """Container for a retrieval / QA result."""

    question: str
    docs: List[str]
    doc_scores: Optional[ArrayLike] = None
    answer: Optional[str] = None
    sources: Optional[List[str]] = None
    """Document source label per retrieved passage (basename / file_path)."""
    passage_ids: Optional[List[str]] = None
    """Chunk ids aligned with ``docs``."""
    source_paths: Optional[List[str]] = None
    """Full document paths aligned with ``docs`` (``sources`` holds basenames)."""
    references: Optional[List[dict[str, Any]]] = None
    """API-facing source references (``reference_id``, ``file_path``, …)."""
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None

    def build_references(self, *, start: int = 1) -> list[dict[str, Any]]:
        """Rebuild ``references`` from the aligned lists, numbered from ``start``.

        Unlike :meth:`ensure_references` this never returns a cached list, which is
        the whole point: by the time a multi-hop caller sees a solution, the engine
        has already numbered it from 1, so asking politely for an offset got the
        original list back and the second hop restarted at ``[1]``. The bug is only
        visible with a model that actually re-calls the tool — most do not.
        """
        self.references = None
        return self.ensure_references(start=start)

    def ensure_references(self, *, start: int = 1) -> list[dict[str, Any]]:
        """One reference per retrieved passage, numbered like the prompt's fences.

        The numbering is the whole point. ``fence_passages`` labels passages
        ``[1..n]`` in the order of ``docs`` and the QA system prompt tells the model
        to cite those numbers, so a reference list collapsed per document — which is
        what this used to build — left ``[7]`` in an answer pointing at nothing
        whenever ten passages came from three files. Keeping one entry per passage
        makes a citation resolvable, and carrying ``chunk_id`` lets the UI open the
        exact passage rather than the top of a PDF.

        ``start`` offsets the numbering for a multi-hop turn, where a second
        retrieval must continue the first one's sequence instead of restarting at 1.
        It is ignored when references are already built — use
        :meth:`build_references` when the offset has to win.
        """
        if self.references is not None:
            return list(self.references)
        sources = list(self.sources or [])
        chunk_ids = list(self.passage_ids or [])
        paths = list(self.source_paths or [])
        refs: list[dict[str, Any]] = []
        for index in range(len(self.docs or [])):
            label = str(sources[index] if index < len(sources) else "").strip() or "unknown"
            chunk_id = (
                str(chunk_ids[index]) if index < len(chunk_ids) and chunk_ids[index] else None
            )
            path = str(paths[index]) if index < len(paths) and paths[index] else None
            refs.append(
                {
                    "reference_id": str(start + index),
                    "file_path": label,
                    # The full path when doc-status knew one; `file_path` stays the
                    # basename because that is what every existing client renders.
                    "source_path": path,
                    "chunk_id": chunk_id,
                    "content": None,
                }
            )
        self.references = refs
        return refs

    def to_dict(self, *, max_docs: int | None = None) -> dict[str, Any]:
        """Serialise the solution.

        ``max_docs`` bounds the passage lists; ``None`` (the default) returns every
        retrieved passage. The bound used to be hard-coded at 5, so a request with
        ``top_k=20`` was answered with 20 passages by the engine and 5 by the API,
        with no ``total`` and no indication that anything had been dropped.
        """
        limit = slice(None) if max_docs is None else slice(0, max_docs)
        doc_scores = None
        if self.doc_scores is not None:
            scores = self.doc_scores
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            doc_scores = [round(float(v), 4) for v in list(scores)[limit]]
        refs = self.ensure_references()
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[limit],
            "doc_scores": doc_scores,
            "gold_docs": self.gold_docs,
            "sources": list(self.sources or [])[limit],
            "references": refs,
            "total_docs": len(self.docs),
        }
