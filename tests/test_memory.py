"""Unit tests for ThreeLayerMemory."""

from __future__ import annotations

from pathlib import Path

import pytest

from memgraphrag.memory import PassageNode, ThreeLayerMemory

pytestmark = pytest.mark.offline


def test_empty_memory() -> None:
    memory = ThreeLayerMemory()
    assert memory.schema_layer == []
    assert memory.fact_layer == []
    assert memory.passage_layer == []
    assert memory.to_dict()["stats"] == {
        "num_schemas": 0,
        "num_facts": 0,
        "num_passages": 0,
    }


def test_build_from_raw_openie_results() -> None:
    data = {
        "docs": [
            {
                "idx": "chunk-1",
                "passage": "Alice works at Acme.",
                "extracted_triples": [
                    ["Alice", "works_at", "Acme"],
                    {"processed_triple": ["Alice", "lives_in", "Paris"]},
                ],
            },
            {
                "idx": "chunk-2",
                "passage": "Bob knows Alice.",
                "extracted_triples": [["Bob", "knows", "Alice"]],
            },
        ]
    }
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(data)

    assert len(memory.schema_layer) == 0
    assert len(memory.passage_layer) == 2
    assert len(memory.fact_layer) == 3
    assert all(f.schema_idx == -1 for f in memory.fact_layer)
    assert memory.passage_layer[0].chunk_id == "chunk-1"
    assert set(f.content for f in memory.fact_layer) == {
        ("Alice", "works_at", "Acme"),
        ("Alice", "lives_in", "Paris"),
        ("Bob", "knows", "Alice"),
    }


def test_to_dict_from_dict_roundtrip() -> None:
    data = {
        "docs": [
            {
                "idx": "c1",
                "passage": "X relates to Y.",
                "extracted_triples": [["X", "relates_to", "Y"]],
            }
        ]
    }
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(data)
    restored = ThreeLayerMemory.from_dict(memory.to_dict())

    assert len(restored.fact_layer) == len(memory.fact_layer)
    assert len(restored.passage_layer) == len(memory.passage_layer)
    assert restored.fact_layer[0].content == memory.fact_layer[0].content
    assert restored.passage_layer[0].chunk_id == memory.passage_layer[0].chunk_id
    assert restored.passage_layer[0].modality == "text"
    assert restored._fact_to_idx[("X", "relates_to", "Y")] == 0
    assert restored._chunk_id_to_idx["c1"] == 0


def test_save_load_tmp_path(tmp_path: Path) -> None:
    data = {
        "docs": [
            {
                "idx": "p1",
                "passage": "Cats chase mice.",
                "extracted_triples": [["Cats", "chase", "mice"]],
            }
        ]
    }
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(data)

    path = tmp_path / "memory.json"
    memory.save(str(path))
    loaded = ThreeLayerMemory.load(str(path))

    assert len(loaded.fact_layer) == 1
    assert loaded.fact_layer[0].content == ("Cats", "chase", "mice")
    assert loaded.passage_layer[0].content == "Cats chase mice."
    assert loaded.passage_layer[0].modality == "text"


def test_duplicate_triples_merge_passage_indices() -> None:
    shared = ["Entity", "related_to", "Other"]
    data = {
        "docs": [
            {
                "idx": "a",
                "passage": "Passage A.",
                "extracted_triples": [shared, shared],
            },
            {
                "idx": "b",
                "passage": "Passage B.",
                "extracted_triples": [list(shared)],
            },
        ]
    }
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(data)

    assert len(memory.fact_layer) == 1
    fact = memory.fact_layer[0]
    assert fact.passage_indices == [0, 1]
    assert fact.frequency == 2
    assert 0 in memory.passage_layer[0].fact_indices
    assert 0 in memory.passage_layer[1].fact_indices


def test_passages_without_triples_are_still_indexed() -> None:
    """A passage with no usable triple must still enter the passage layer.

    These chunks used to be dropped entirely, which meant no PassageNode, no entry in
    chunks_vdb, and therefore a chunk unreachable even by the dense fallback: a cover
    page or a table of figures vanished from the corpus with no error. An orphan
    passage carrying no fact still answers dense queries.
    """
    data = {
        "docs": [
            {"idx": "empty-list", "passage": "No triples.", "extracted_triples": []},
            {"idx": "missing", "passage": "Also none.", "extracted_triples": None},
            {
                "idx": "invalid",
                "passage": "Bad entries only.",
                "extracted_triples": [None, {"foo": "bar"}, ["too", "short"]],
            },
        ]
    }
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(data)

    assert [p.chunk_id for p in memory.passage_layer] == [
        "empty-list",
        "missing",
        "invalid",
    ]
    # No triple was usable, so no fact — but the text remains retrievable.
    assert memory.fact_layer == []
    assert all(not p.fact_indices for p in memory.passage_layer)


def test_modality_field_default() -> None:
    node = PassageNode(idx=0, chunk_id="c", content="hello")
    assert node.modality == "text"

    data = {
        "docs": [
            {
                "idx": "m1",
                "passage": "Text passage.",
                "extracted_triples": [["A", "r", "B"]],
            }
        ]
    }
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(data)
    assert memory.passage_layer[0].modality == "text"
    assert memory.to_dict()["passage_layer"][0]["modality"] == "text"

    payload = memory.to_dict()
    del payload["passage_layer"][0]["modality"]
    restored = ThreeLayerMemory.from_dict(payload)
    assert restored.passage_layer[0].modality == "text"
