"""Tests for FactFilter threshold and LLM stub fallback."""

from __future__ import annotations

import numpy as np
import pytest

from memgraphrag.rerank import FactFilter


@pytest.mark.offline
def test_threshold_filter_basic():
    ff = FactFilter(default_threshold=0.5)
    scores = [0.1, 0.5, 0.9, 0.49]
    assert FactFilter.threshold_filter(scores, 0.5) == [1, 2]


@pytest.mark.offline
def test_threshold_filter_numpy():
    scores = np.array([0.0, 0.6, 0.59, 1.0])
    assert FactFilter.threshold_filter(scores, 0.6) == [1, 3]


@pytest.mark.offline
def test_threshold_filter_empty():
    assert FactFilter.threshold_filter([], 0.5) == []


@pytest.mark.offline
def test_llm_filter_falls_back_to_threshold():
    ff = FactFilter(default_threshold=0.7)
    facts = [("a", "r", "b"), ("c", "r", "d"), ("e", "r", "f")]
    indices = [10, 20, 30]
    scores = [0.2, 0.8, 0.75]
    kept = ff.llm_filter("q", facts, indices, scores=scores, threshold=0.7)
    assert kept == [20, 30]


@pytest.mark.offline
def test_llm_filter_without_scores_returns_all():
    ff = FactFilter()
    kept = ff.llm_filter("q", ["f1", "f2"], [0, 1], scores=None)
    assert kept == [0, 1]
