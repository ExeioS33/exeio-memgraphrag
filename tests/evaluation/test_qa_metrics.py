"""Str-Acc / EM / F1 pin the definitions frozen in docs/Evaluation.md."""

from __future__ import annotations

import pytest

from memgraphrag.evaluation.qa_metrics import (
    STR_ACC_WINDOW_TOKENS,
    contains_token_span,
    exact_match,
    f1_score,
    str_acc,
)

pytestmark = pytest.mark.offline


def test_str_acc_accepts_a_gold_answer_embedded_in_prose() -> None:
    """The paper's metric is inclusion, not equality: a sentence answer still counts."""
    result = str_acc(
        [["Chris Evans"]], ["The actor is **Chris Evans**, known for Captain America."]
    )
    assert result.pooled["StrAcc"] == 1.0


def test_str_acc_rejects_a_gold_answer_that_is_only_a_substring_of_a_word() -> None:
    """A raw substring test scores these as hits; token containment must not.

    "art" inside "started" and "US" inside "USSR" are exactly the accidental hits
    that inflate an inclusion metric on short gold answers.
    """
    assert str_acc([["art"]], ["The war started in 1941."]).pooled["StrAcc"] == 0.0
    assert str_acc([["US"]], ["The USSR collapsed in 1991."]).pooled["StrAcc"] == 0.0


def test_str_acc_requires_the_gold_tokens_to_be_contiguous() -> None:
    """Bag-of-tokens inclusion would accept a shuffled answer; the frozen rule does not."""
    assert str_acc([["New York"]], ["York is new to the list."]).pooled["StrAcc"] == 0.0
    assert str_acc([["New York"]], ["He lives in New York."]).pooled["StrAcc"] == 1.0


def test_str_acc_ignores_gold_matches_beyond_the_window() -> None:
    """Inclusion is monotone in length: without a bound, padding buys the point."""
    padding = " ".join(["filler"] * (STR_ACC_WINDOW_TOKENS + 10))
    assert str_acc([["Paris"]], [f"{padding} Paris"]).pooled["StrAcc"] == 0.0
    assert str_acc([["Paris"]], [f"Paris {padding}"]).pooled["StrAcc"] == 1.0


def test_str_acc_reports_verbosity_and_truncation_rates() -> None:
    """A Str-Acc figure without these two counters cannot be audited."""
    long_answer = "Paris " + " ".join(["and more context"] * 80)
    result = str_acc([["Paris"], ["Lyon"]], [long_answer, "Lyon"])
    assert result.pooled["StrAcc"] == 1.0
    # Both answers are right, but only one of them buried the answer in 240 tokens.
    assert result.pooled["StrAccVerboseHitRate"] == 0.5
    assert result.pooled["StrAccTruncatedRate"] == 0.5


def test_str_acc_uses_the_best_of_several_gold_answers() -> None:
    result = str_acc([["June 1982", "1982"]], ["He signed in 1982."])
    assert result.pooled["StrAcc"] == 1.0


def test_empty_gold_answer_never_scores_a_free_point() -> None:
    assert not contains_token_span("", "anything at all", STR_ACC_WINDOW_TOKENS)
    assert str_acc([[""]], [""]).pooled["StrAcc"] == 0.0


def test_exact_match_is_whole_answer_equality_after_normalization() -> None:
    assert exact_match([["Paris"]], ["the **Paris**"]).pooled["ExactMatch"] == 1.0
    assert exact_match([["Paris"]], ["The answer is Paris."]).pooled["ExactMatch"] == 0.0


def test_f1_is_token_overlap_and_scores_nothing_for_an_empty_answer() -> None:
    result = f1_score([["Chris Evans"], ["Paris"]], ["Chris Evans", ""])
    assert result.per_example[0]["F1"] == 1.0
    assert result.per_example[1]["F1"] == 0.0


def test_misaligned_inputs_raise_instead_of_zipping_short() -> None:
    with pytest.raises(ValueError):
        exact_match([["a"], ["b"]], ["a"])
