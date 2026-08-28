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
    references: Optional[List[dict[str, Any]]] = None
    """API-facing source references (``reference_id``, ``file_path``, …)."""
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None

    def ensure_references(self) -> list[dict[str, Any]]:
        """Build unique ``references`` from retrieved passage sources (LightRAG-style)."""
        if self.references is not None:
            return list(self.references)
        seen: dict[str, str] = {}
        refs: list[dict[str, Any]] = []
        for src in list(self.sources or []):
            label = str(src or "").strip() or "unknown"
            if label in seen:
                continue
            rid = str(len(refs) + 1)
            seen[label] = rid
            refs.append(
                {
                    "reference_id": rid,
                    "file_path": label,
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
