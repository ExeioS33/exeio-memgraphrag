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
    thought: Optional[str] = None
    citations: Optional[List[int]] = None
    confidence: Optional[str] = None
    structured: bool = False
    sources: Optional[List[str]] = None
    """Document source label per retrieved passage (basename / file_path)."""
    passage_ids: Optional[List[str]] = None
    """Chunk ids aligned with ``docs``."""
    references: Optional[List[dict[str, Any]]] = None
    """API-facing source references (``reference_id``, ``file_path``, …)."""
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None

    def ensure_references(self) -> list[dict[str, Any]]:
        """Build ``references`` from retrieved passage sources (always)."""
        if self.references is not None:
            return list(self.references)
        refs: list[dict[str, Any]] = []
        sources = list(self.sources or [])
        for i, _doc in enumerate(self.docs or [], start=1):
            src = ""
            if i - 1 < len(sources):
                src = str(sources[i - 1] or "").strip()
            refs.append(
                {
                    "reference_id": str(i),
                    "file_path": src or "unknown",
                    "content": None,
                }
            )
        self.references = refs
        return refs

    def to_dict(self) -> dict[str, Any]:
        doc_scores = None
        if self.doc_scores is not None:
            scores = self.doc_scores
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            doc_scores = [round(float(v), 4) for v in list(scores)[:5]]
        refs = self.ensure_references()
        return {
            "question": self.question,
            "answer": self.answer,
            "thought": self.thought,
            "citations": list(self.citations or []),
            "confidence": self.confidence,
            "structured": bool(self.structured),
            "sources": list(self.sources or [])[:5],
            "references": refs[:5],
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": doc_scores,
            "gold_docs": self.gold_docs,
        }
