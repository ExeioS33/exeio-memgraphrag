"""Recall@k, Evidence Recall and Context Relevance over document identities."""

from __future__ import annotations

import pytest

from memgraphrag.evaluation.retrieval_metrics import (
    context_relevance,
    evidence_recall,
    recall_at_k,
)

pytestmark = pytest.mark.offline

GOLD = [["Teutberga", "Lothair II"]]


def test_recall_at_k_respects_the_cut_off() -> None:
    retrieved = [["Noise", "Teutberga", "Lothair II"]]
    result = recall_at_k(GOLD, retrieved, k_list=[1, 2, 5])
    assert result.pooled["Recall@1"] == 0.0
    assert result.pooled["Recall@2"] == 0.5
    assert result.pooled["Recall@5"] == 1.0


def test_document_identity_survives_case_and_punctuation() -> None:
    """A retriever returning "lothair ii." found the gold document; scoring says so."""
    result = recall_at_k([["Lothair II"]], [["lothair ii."]], k_list=[1])
    assert result.pooled["Recall@1"] == 1.0


def test_duplicate_passages_do_not_consume_the_top_k_budget_twice() -> None:
    """Two chunks of one document are one document; ranks are deduplicated first."""
    retrieved = [["Teutberga", "Teutberga", "Lothair II"]]
    assert recall_at_k(GOLD, retrieved, k_list=[2]).pooled["Recall@2"] == 1.0


def test_evidence_recall_scores_the_whole_context_not_a_fixed_k() -> None:
    retrieved = [["A", "B", "C", "D", "E", "F", "Teutberga", "Lothair II"]]
    assert evidence_recall(GOLD, retrieved).pooled["EvidenceRecall"] == 1.0
    assert recall_at_k(GOLD, retrieved, k_list=[5]).pooled["Recall@5"] == 0.0


def test_context_relevance_punishes_retrieving_everything() -> None:
    """The cheap way to win Evidence Recall must cost Context Relevance."""
    focused = [["Teutberga", "Lothair II"]]
    padded = [["Teutberga", "Lothair II", "Noise 1", "Noise 2", "Noise 3", "Noise 4"]]
    assert context_relevance(GOLD, focused).pooled["ContextRelevance"] == 1.0
    assert evidence_recall(GOLD, padded).pooled["EvidenceRecall"] == 1.0
    assert context_relevance(GOLD, padded).pooled["ContextRelevance"] == pytest.approx(1 / 3)


def test_empty_retrieval_is_a_failure_not_perfect_precision() -> None:
    assert context_relevance(GOLD, [[]]).pooled["ContextRelevance"] == 0.0
    assert evidence_recall(GOLD, [[]]).pooled["EvidenceRecall"] == 0.0


def test_unlabelled_question_scores_zero_rather_than_raising_the_mean() -> None:
    result = evidence_recall([[], ["Teutberga"]], [["Teutberga"], ["Teutberga"]])
    assert result.pooled["EvidenceRecall"] == 0.5


def test_misaligned_inputs_raise() -> None:
    with pytest.raises(ValueError):
        recall_at_k(GOLD, [["a"], ["b"]])
