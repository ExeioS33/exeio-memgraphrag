"""Edge cases for empty corpus and duplicate chunk IDs."""

from __future__ import annotations

from memgraphrag.memory import ThreeLayerMemory


def test_empty_openie_docs_build_empty_memory():
    mem = ThreeLayerMemory()
    mem.build_from_raw_openie_results({"docs": []})
    assert len(mem.schema_layer) == 0
    assert len(mem.fact_layer) == 0
    assert len(mem.passage_layer) == 0


def test_malformed_triples_are_skipped():
    mem = ThreeLayerMemory()
    mem.build_from_raw_openie_results(
        {
            "docs": [
                {
                    "idx": "c1",
                    "passage": "hello",
                    "extracted_triples": [
                        None,
                        {"bad": True},
                        ["only", "two"],
                        ["a", "rel", "b"],
                    ],
                }
            ]
        }
    )
    assert len(mem.fact_layer) == 1
    assert mem.fact_layer[0].content == ("a", "rel", "b")


def test_duplicate_chunk_id_reuses_passage():
    mem = ThreeLayerMemory()
    mem.build_from_raw_openie_results(
        {
            "docs": [
                {
                    "idx": "c1",
                    "passage": "hello",
                    "extracted_triples": [["a", "r", "b"]],
                },
                {
                    "idx": "c1",
                    "passage": "hello again",
                    "extracted_triples": [["a", "r", "c"]],
                },
            ]
        }
    )
    assert len(mem.passage_layer) == 1
    assert len(mem.fact_layer) == 2
