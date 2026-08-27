"""Offline tests for ontology / conflict / memory mutation framework layers."""

from __future__ import annotations

from typing import Any

import pytest

from memgraphrag.memory import ThreeLayerMemory
from memgraphrag.utils.json_llm import extract_json_object

pytestmark = pytest.mark.offline


def _raw_docs() -> dict[str, Any]:
    return {
        "docs": [
            {
                "idx": "c1",
                "passage": "Alice was born in Paris. Bob was born in Lyon.",
                "extracted_triples": [
                    ["Alice", "born_in", "Paris"],
                    ["Bob", "born_in", "Lyon"],
                ],
            },
            {
                "idx": "c2",
                "passage": "Alice lives in Berlin.",
                "extracted_triples": [
                    ["Alice", "born_in", "Berlin"],
                    ["Alice", "lives_in", "Berlin"],
                ],
            },
            {
                "idx": "c3",
                "passage": "Carol works at Acme.",
                "extracted_triples": [["Carol", "works_at", "Acme"]],
            },
        ]
    }


def test_extract_json_object_fenced_and_raw():
    assert extract_json_object('```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json_object('prefix {"b": 2} suffix')["b"] == 2
    assert extract_json_object("not json") == {}


def test_link_fact_to_schema_and_filter():
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(_raw_docs())
    assert all(f.schema_idx == -1 for f in memory.fact_layer)

    # Link three facts to same ontology, one to another, leave one untyped
    ont_born = ("Person", "born_in", "City")
    ont_live = ("Person", "lives_in", "City")
    for fact in memory.fact_layer:
        if fact.content[1] == "born_in":
            memory.link_fact_to_schema(fact.idx, ont_born)
        elif fact.content[1] == "lives_in":
            memory.link_fact_to_schema(fact.idx, ont_live)

    memory.recompute_schema_frequencies()
    assert len(memory.schema_layer) == 2
    born = next(s for s in memory.schema_layer if s.content == ont_born)
    assert born.frequency == 3  # Alice-Paris, Bob-Lyon, Alice-Berlin

    # Filter min_frequency=2 drops lives_in (freq 1) and keeps born_in
    stats = memory.filter_schemas_by_frequency(2)
    assert stats["kept"] == 1
    assert stats["dropped"] == 1
    assert len(memory.schema_layer) == 1
    assert memory.schema_layer[0].content == ont_born
    assert memory.schema_layer[0].idx == 0
    for fact in memory.fact_layer:
        if fact.content[1] == "born_in":
            assert fact.schema_idx == 0
        else:
            assert fact.schema_idx == -1


def test_remove_and_replace_fact_index_consistency():
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(_raw_docs())
    memory.link_fact_to_schema(0, ("Person", "born_in", "City"))
    memory.link_fact_to_schema(1, ("Person", "born_in", "City"))

    n_before = len(memory.fact_layer)
    assert memory.remove_fact(0) is True
    assert len(memory.fact_layer) == n_before - 1
    # All fact.idx sequential
    assert [f.idx for f in memory.fact_layer] == list(range(len(memory.fact_layer)))
    # Schema fact_indices remapped
    schema = memory.schema_layer[0]
    assert all(0 <= i < len(memory.fact_layer) for i in schema.fact_indices)
    for p in memory.passage_layer:
        assert all(0 <= i < len(memory.fact_layer) for i in p.fact_indices)

    # Replace merges into existing triple
    target = memory.fact_layer[0].content
    other_idx = 1 if len(memory.fact_layer) > 1 else 0
    if other_idx != 0:
        passages_before = set(memory.fact_layer[0].passage_indices)
        surviving = memory.replace_fact(other_idx, target)
        # replace_fact returns the index of the surviving fact, and the replaced
        # fact's passages must have been merged into it rather than dropped.
        assert surviving == 0
        assert target in {f.content for f in memory.fact_layer}
        assert passages_before <= set(memory.fact_layer[surviving].passage_indices)


def test_conflict_candidate_groups():
    from memgraphrag.core import MemGraphRAG

    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(_raw_docs())
    rag = MemGraphRAG(working_dir="/tmp/mgr-test-framework", llm_model_func=None)
    groups = rag._conflict_candidate_groups(memory, max_groups=50)
    # Alice born_in Paris/Berlin share (head, relation)
    assert any(len(g) >= 2 for g in groups)
    flat = {i for g in groups for i in g}
    assert flat  # non-empty


@pytest.mark.asyncio
async def test_extract_schema_parses_and_links(tmp_path):
    from memgraphrag.core import MemGraphRAG

    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(
        {
            "docs": [
                {
                    "idx": "c1",
                    "passage": "Alice was born in Paris.",
                    "extracted_triples": [["Alice", "born_in", "Paris"]],
                }
            ]
        }
    )

    async def fake_llm(user: str, system_prompt: str = "", **kwargs: Any) -> str:
        return """{
          "ontology_triples": [
            {
              "triple": ["Alice", "born_in", "Paris"],
              "ontology": ["Person", "born_in", "City"]
            }
          ]
        }"""

    rag = MemGraphRAG(
        working_dir=str(tmp_path / "wd"),
        llm_model_func=fake_llm,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="IgraphStorage",
        doc_status_storage="JsonDocStatusStorage",
    )
    await rag.initialize_storages()
    # Seed openie_kv empty so cache miss
    out = await rag.extract_schema(memory)
    assert len(out.schema_layer) == 1
    assert out.schema_layer[0].content == ("Person", "born_in", "City")
    assert out.fact_layer[0].schema_idx == 0
    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_filter_ontology_env_and_resolve(tmp_path, monkeypatch):
    from memgraphrag.core import MemGraphRAG

    monkeypatch.setenv("ONTOLOGY_MIN_FREQUENCY", "2")
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(_raw_docs())
    for fact in memory.fact_layer:
        if fact.content[1] == "born_in":
            memory.link_fact_to_schema(fact.idx, ("Person", "born_in", "City"))
        elif fact.content[1] == "lives_in":
            memory.link_fact_to_schema(fact.idx, ("Person", "lives_in", "City"))

    rag = MemGraphRAG(working_dir=str(tmp_path / "wd2"), llm_model_func=None)
    memory = await rag.filter_ontology(memory)
    assert len(memory.schema_layer) == 1

    # Resolve discarded
    async def resolve_llm(user: str, system_prompt: str = "", **kwargs: Any) -> str:
        return """{
          "resolved_triples": [
            {
              "original_triple": ["Alice", "born_in", "Berlin"],
              "resolution": "discarded",
              "resolved_triple": null,
              "conflict_type": "mutual",
              "reason": "conflicts with Paris birthplace"
            }
          ],
          "unresolved_conflicts": [],
          "summary": "ok"
        }"""

    rag.llm_model_func = resolve_llm
    before = len(memory.fact_layer)
    memory, resolution = await rag.resolve_conflicts(
        memory,
        {
            "has_conflict": True,
            "conflicts": [
                {
                    "triple1": ["Alice", "born_in", "Paris"],
                    "triple2": ["Alice", "born_in", "Berlin"],
                    "is_hard_conflict": True,
                }
            ],
        },
    )
    assert resolution["summary"]["discarded"] == 1
    assert len(memory.fact_layer) == before - 1
    assert ("Alice", "born_in", "Berlin") not in {f.content for f in memory.fact_layer}


@pytest.mark.asyncio
async def test_install_graph_has_schema_nodes(tmp_path):
    import numpy as np

    from memgraphrag.core import MemGraphRAG

    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(
        {
            "docs": [
                {
                    "idx": "c1",
                    "passage": "Alice was born in Paris.",
                    "extracted_triples": [["Alice", "born_in", "Paris"]],
                }
            ]
        }
    )
    memory.link_fact_to_schema(0, ("Person", "born_in", "City"))
    # Normalize chunk ids like embed path does
    for p in memory.passage_layer:
        from memgraphrag.utils.hashing import compute_mdhash_id

        p.chunk_id = compute_mdhash_id(p.content, prefix="chunk-")

    async def fake_embed(texts, **kwargs):
        return np.zeros((len(texts), 8), dtype=np.float32)

    rag = MemGraphRAG(
        working_dir=str(tmp_path / "wd3"),
        embedding_func=fake_embed,
        embedding_dim=8,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="IgraphStorage",
        doc_status_storage="JsonDocStatusStorage",
    )
    await rag.initialize_storages()
    await rag._install_memory_graph(memory)
    nodes = await rag.graph.get_all_nodes()
    labels = {n.get("label") or n.get("node_type") for n in nodes}
    assert "Schema" in labels or any((n.get("layer") == "schema") for n in nodes)
    edges = await rag.graph.get_all_edges()
    types = {(e.get("type") or e.get("edge_type") or "").upper() for e in edges}
    assert "FACT_SCHEMA" in types or "TYPE_RELATION" in types
    await rag.finalize_storages()
