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
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None

    def to_dict(self) -> dict[str, Any]:
        doc_scores = None
        if self.doc_scores is not None:
            scores = self.doc_scores
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            doc_scores = [round(float(v), 4) for v in list(scores)[:5]]
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": doc_scores,
            "gold_docs": self.gold_docs,
        }
