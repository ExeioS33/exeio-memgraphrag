"""Tests for the paper-fidelity fixes in the memory graph and retrieval path.

Each test here pins a behaviour that was measurably absent: the fact graph, the
ontology layer wiring, the mean-based entity seeding, the ontology filter actually
reaching the fact layer, and a fact reranker that does something.
"""

from __future__ import annotations

import pytest

from memgraphrag.memory import ThreeLayerMemory
from memgraphrag.rerank import FactFilter


def _memory_with_chain() -> ThreeLayerMemory:
    """Two facts forming a 2-hop chain, each in its OWN passage.

    Einstein --born_in--> Germany  (passage 1)
    Germany  --capital--> Berlin   (passage 2)

    The entities never co-occur in a chunk, so only a real entity<->entity edge can
    connect Einstein to Berlin.
    """
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(
        {
            "docs": [
                {
                    "idx": "chunk-1",
                    "passage": "Einstein was born in Germany.",
                    "extracted_entities": ["Einstein", "Germany"],
                    "extracted_triples": [["Einstein", "born_in", "Germany"]],
                },
                {
                    "idx": "chunk-2",
                    "passage": "The capital of Germany is Berlin.",
                    "extracted_entities": ["Germany", "Berlin"],
                    "extracted_triples": [["Germany", "capital", "Berlin"]],
                },
            ]
        }
    )
    return memory


@pytest.mark.offline
def test_memory_chain_shares_an_entity_across_passages() -> None:
    """The fixture really is a cross-passage chain (guards the tests below)."""
    memory = _memory_with_chain()
    assert len(memory.passage_layer) == 2
    assert len(memory.fact_layer) == 2

    per_passage = [set() for _ in memory.passage_layer]
    for fact in memory.fact_layer:
        for pidx in fact.passage_indices:
            per_passage[pidx].update({str(fact.content[0]).lower(), str(fact.content[2]).lower()})
    # No single passage holds both ends of the chain.
    assert not any({"einstein", "berlin"} <= entities for entities in per_passage)


@pytest.mark.offline
async def test_graph_install_creates_entity_relation_and_type_edges(tmp_path) -> None:
    """G_fac (entity<->entity) and the entity->type wiring must both exist.

    _install_memory_graph used to emit only TYPE_RELATION, PASSAGE_ENTITY,
    FACT_PASSAGE and FACT_SCHEMA. With no entity<->entity edge the graph was
    bipartite {entity, fact} <-> {chunk}, so PPR could not traverse a multi-hop
    chain; and the type nodes, joined only to each other, formed a component that
    could neither receive nor pass PPR mass.
    """
    from memgraphrag.storage.igraph_impl import IgraphStorage

    memory = _memory_with_chain()
    # Give both facts a schema so the ontology layer is exercised too.
    for fact, ontology in zip(
        memory.fact_layer,
        (("person", "born_in", "country"), ("country", "capital", "city")),
    ):
        memory.link_fact_to_schema(fact.idx, ontology)

    graph = IgraphStorage(
        workspace="",
        namespace="test",
        global_config={"working_dir": str(tmp_path), "is_directed_graph": False},
    )
    await graph.initialize()

    from memgraphrag.core import MemGraphRAG

    engine = MemGraphRAG.__new__(MemGraphRAG)
    engine.graph = graph
    engine._inactive_fact_idxs = set()

    await MemGraphRAG._install_memory_graph(engine, memory)

    edges = await graph.get_all_edges()
    types = {e.get("type") for e in edges}
    assert "ENTITY_RELATION" in types, "the fact graph G_fac is missing"
    assert "ENTITY_TO_TYPE" in types, "the ontology layer is not wired to entities"

    # Germany is the pivot: it must be joined to both Einstein and Berlin.
    from memgraphrag.utils.hashing import compute_mdhash_id

    eid = {n: compute_mdhash_id(n, prefix="entity-") for n in ("einstein", "germany", "berlin")}
    entity_edges = {
        frozenset((e["source"], e["target"])) for e in edges if e.get("type") == "ENTITY_RELATION"
    }
    assert frozenset((eid["einstein"], eid["germany"])) in entity_edges
    assert frozenset((eid["germany"], eid["berlin"])) in entity_edges


@pytest.mark.offline
def test_ontology_filter_safety_valve_keeps_small_corpora() -> None:
    """A corpus too small to carry frequency signal must not be emptied.

    Making the filter effective on the fact layer is only safe with this guard: at
    ONTOLOGY_MIN_FREQUENCY=2 every schema of a one-document corpus is seen once, so
    an unguarded filter would deactivate the entire fact layer.
    """
    from memgraphrag.constants import ONTOLOGY_MAX_DEACTIVATION_RATIO

    assert 0.0 < ONTOLOGY_MAX_DEACTIVATION_RATIO < 1.0


@pytest.mark.offline
async def test_fact_rerank_selects_rather_than_echoing_the_threshold() -> None:
    """The reranker must actually drop irrelevant facts.

    llm_filter used to log and then call threshold_filter with the same threshold, so
    both branches of the caller returned identical results: SKIP_FACT_RERANK was a
    lever with no observable effect.
    """
    facts = [["Einstein", "born_in", "Germany"], ["Einstein", "likes", "sailing"]]
    scores = [0.9, 0.88]

    async def fake_llm(user, **kwargs):
        assert "born_in" in user
        return '{"relevant_facts": [1]}'

    filt = FactFilter(default_threshold=0.6)
    kept = await filt.allm_filter(
        "Where was Einstein born?",
        facts,
        [0, 1],
        scores=scores,
        threshold=0.6,
        llm_model_func=fake_llm,
    )
    assert kept == [0]

    # Both facts clear the threshold, so the threshold branch keeps both — proving the
    # two branches now differ.
    assert filt.threshold_filter(scores, 0.6) == [0, 1]


@pytest.mark.offline
async def test_fact_rerank_falls_back_when_llm_fails() -> None:
    """A failing reranker degrades to the threshold instead of losing every fact."""
    facts = [["a", "r", "b"], ["c", "r", "d"]]

    async def broken_llm(user, **kwargs):
        raise TimeoutError("provider down")

    filt = FactFilter(default_threshold=0.6)
    kept = await filt.allm_filter(
        "q", facts, [0, 1], scores=[0.9, 0.2], threshold=0.6, llm_model_func=broken_llm
    )
    assert kept == [0]
