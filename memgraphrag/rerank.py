"""Fact filtering / reranking helpers for MemGraphRAG retrieval.

Provenance: simplified from MemGraphRAG ``code/src/rerank.py`` (DSPyFilter).
POC focuses on similarity threshold filtering; LLM filter is a stub that
falls back to threshold behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class FactFilter:
    """Filter candidate facts by similarity score (and optional LLM stub)."""

    def __init__(self, default_threshold: float = 0.6) -> None:
        self.default_threshold = float(default_threshold)

    @staticmethod
    def threshold_filter(
        scores: Sequence[float] | np.ndarray,
        threshold: float,
    ) -> list[int]:
        """Return indices whose score is ``>= threshold`` (stable ascending order)."""
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim == 0:
            return [0] if float(arr) >= threshold else []
        return [int(i) for i, s in enumerate(arr.tolist()) if float(s) >= threshold]

    def llm_filter(
        self,
        query: str,
        candidate_facts: Sequence[Any],
        candidate_indices: Sequence[int],
        scores: Sequence[float] | np.ndarray | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> list[int]:
        """Stub LLM fact filter — falls back to threshold filtering.

        When LLM reranking is not wired, keep facts above ``threshold``.
        If scores are missing, return all candidate indices unchanged.
        """
        thr = self.default_threshold if threshold is None else float(threshold)
        logger.debug(
            "llm_filter stub for query=%r n=%d; falling back to threshold=%.3f",
            query[:80],
            len(candidate_facts),
            thr,
        )
        if scores is None:
            return list(candidate_indices)
        kept_local = self.threshold_filter(scores, thr)
        # Map local score indices back to candidate_indices when aligned 1:1
        if len(scores) == len(candidate_indices):
            return [int(candidate_indices[i]) for i in kept_local]
        return kept_local
