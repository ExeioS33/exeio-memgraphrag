"""Fact filtering / reranking helpers for MemGraphRAG retrieval.

Provenance: simplified from MemGraphRAG ``code/src/rerank.py`` (DSPyFilter), whose
role is to drop retrieved triples that are topically similar to the query but do not
actually help answer it — something a cosine threshold cannot do.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from memgraphrag.prompts.templates import render_fact_rerank
from memgraphrag.utils.json_llm import extract_json_object

logger = logging.getLogger(__name__)

#: Upper bound on triples submitted to the reranker. The candidates arrive sorted by
#: similarity, so the tail is the least promising part anyway, and an unbounded prompt
#: is how the conflict path used to blow through the context window.
MAX_RERANK_CANDIDATES = 50


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
        """Threshold fallback used when no LLM is available.

        Kept for callers without an event loop; :meth:`allm_filter` is the real
        reranker.
        """
        thr = self.default_threshold if threshold is None else float(threshold)
        if scores is None:
            return list(candidate_indices)
        kept_local = self.threshold_filter(scores, thr)
        # Map local score indices back to candidate_indices when aligned 1:1
        if len(scores) == len(candidate_indices):
            return [int(candidate_indices[i]) for i in kept_local]
        return kept_local

    async def allm_filter(
        self,
        query: str,
        candidate_facts: Sequence[Any],
        candidate_indices: Sequence[int],
        scores: Sequence[float] | np.ndarray | None = None,
        threshold: float | None = None,
        llm_model_func: Any | None = None,
        max_candidates: int = MAX_RERANK_CANDIDATES,
        **kwargs: Any,
    ) -> list[int]:
        """Ask the LLM which candidate facts actually help answer ``query``.

        This used to be a stub that logged and then called ``threshold_filter`` with
        the same threshold, so both branches of the caller produced identical results
        and ``SKIP_FACT_RERANK`` was a lever with no effect. It now performs a real
        selection, and falls back to the threshold — loudly — when no LLM is wired or
        the call fails, so a degraded run is never mistaken for a reranked one.
        """
        thr = self.default_threshold if threshold is None else float(threshold)
        fallback = self.llm_filter(
            query, candidate_facts, candidate_indices, scores=scores, threshold=thr
        )
        if llm_model_func is None or not candidate_facts:
            logger.debug("Fact rerank unavailable (no LLM); using threshold %.3f", thr)
            return fallback

        head = list(candidate_indices)[:max_candidates]
        facts = [candidate_facts[i] for i in range(min(len(candidate_facts), len(head)))]
        system, user = render_fact_rerank(query, facts)
        try:
            raw = await llm_model_func(
                user,
                system_prompt=system,
                agent="retrieve.fact_rerank",
                llm_action="complete",
            )
        except Exception as exc:
            logger.warning(
                "Fact rerank LLM call failed (%s); falling back to threshold %.3f",
                exc,
                thr,
            )
            return fallback

        data = extract_json_object(str(raw))
        picked = data.get("relevant_facts")
        if not isinstance(picked, list):
            logger.warning(
                "Fact rerank returned no usable selection; falling back to threshold"
            )
            return fallback

        kept: list[int] = []
        for value in picked:
            try:
                pos = int(value)
            except (TypeError, ValueError):
                continue
            # The prompt numbers facts from 1.
            if 1 <= pos <= len(head):
                kept.append(int(head[pos - 1]))
        if not kept:
            # An empty selection is a legitimate answer ("nothing here helps"), but so
            # is a malformed one. Prefer recall and say which happened.
            logger.info("Fact rerank kept nothing; falling back to threshold %.3f", thr)
            return fallback
        return sorted(set(kept))
