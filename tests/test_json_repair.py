"""Malformed LLM JSON is repaired instead of silently becoming an empty result."""

from __future__ import annotations

import pytest

from memgraphrag.utils.json_llm import extract_json_object

pytestmark = pytest.mark.offline


def test_trailing_comma_and_truncated_tail_are_repaired():
    raw = '{"named_entities": ["Plateforme Agréée", "DGFiP",], "triples": [["A", "r", "B"]'
    data = extract_json_object(raw)
    assert data["named_entities"] == ["Plateforme Agréée", "DGFiP"]
    assert data["triples"] == [["A", "r", "B"]]


def test_unescaped_inner_quote_is_repaired():
    raw = '{"ontology_triples": [{"triple": ["Cas "9"", "concerne", "Distributeur"], "ontology": ["Cas d\'usage", "concerne", "Acteur"]}]}'
    data = extract_json_object(raw)
    assert data.get("ontology_triples"), data
    assert data["ontology_triples"][0]["ontology"][2] == "Acteur"


def test_hopeless_text_still_returns_empty():
    assert extract_json_object("Je ne peux pas répondre à cette question.") == {}
    assert extract_json_object("") == {}
