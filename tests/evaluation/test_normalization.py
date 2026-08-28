"""Normalization is the whole metric: every QA score is a comparison of its output."""

from __future__ import annotations

import pytest

from memgraphrag.evaluation.normalization import (
    normalize_answer,
    normalize_doc_key,
    normalize_tokens,
    strip_markdown,
)

pytestmark = pytest.mark.offline


def test_markdown_decoration_does_not_change_the_answer() -> None:
    """A bolded correct answer must normalize to the same tokens as a bare one."""
    assert normalize_answer("**Chris Evans**") == normalize_answer("Chris Evans")
    assert normalize_answer("- `Paris`") == "paris"
    assert normalize_answer("## The Answer") == "answer"


def test_citation_markers_are_dropped() -> None:
    """The QA prompt asks for citations; they must not count as answer tokens."""
    assert normalize_answer("Paris [1]") == "paris"
    assert normalize_answer("Paris 【2】") == "paris"


def test_typographic_variants_fold_onto_ascii() -> None:
    """An en dash in a date range is the same answer as a hyphen."""
    assert normalize_answer("1941–1945") == normalize_answer("1941-1945")
    assert normalize_answer("O’Brien") == normalize_answer("O'Brien")


def test_punctuation_becomes_a_separator_not_a_deletion() -> None:
    """Deleting punctuation (the MRQA recipe) welds tokens: "U.S.-based" -> "usbased"."""
    assert normalize_tokens("U.S.-based") == ["u", "s", "based"]


def test_articles_are_dropped_from_answers_but_kept_in_doc_keys() -> None:
    """Two Wikipedia titles differing only by "The" are two different documents."""
    assert normalize_answer("the newcomers") == "newcomers"
    assert normalize_doc_key("The Newcomers") != normalize_doc_key("Newcomers")


def test_doc_key_ignores_case_and_trailing_punctuation() -> None:
    assert normalize_doc_key("Lothair II.") == normalize_doc_key("lothair ii")


def test_strip_markdown_keeps_link_text() -> None:
    assert strip_markdown("[Chris Evans](https://example.org)").strip() == "Chris Evans"


def test_empty_input_is_empty_not_an_error() -> None:
    assert normalize_answer("") == ""
    assert normalize_tokens(None) == []  # type: ignore[arg-type]
