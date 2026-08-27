"""Deterministic end-to-end test of the whole engine, with no network.

Before this, no test exercised the pipeline: `_retrieve_one`, `_run_ppr`,
`detect_conflicts` and `process_pending` appeared in `tests/` only as AsyncMock
replacements, so every stage was verified in isolation and never chained. This test
runs the real chain — OpenIE, memory build, schema extraction, ontology filter,
conflict detection and resolution, graph install, PPR retrieval, answer generation —
against a scripted LLM and a hash-based embedding, so it is fast and reproducible.

The corpus encodes a deliberate multi-hop chain whose links never share a passage,
plus a genuine contradiction, so retrieval and conflict handling are both observable.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from memgraphrag.base import QueryParam
from memgraphrag.core import MemGraphRAG

pytestmark = pytest.mark.offline

EMBED_DIM = 64

# Each fact lives in its own passage, so only an entity<->entity edge can chain them.
CORPUS: list[tuple[str, str]] = [
    ("chunk-1", "Ada Lovelace was born in London."),
    ("chunk-2", "London is the capital of England."),
    ("chunk-3", "England is part of the United Kingdom."),
    ("chunk-4", "Charles Babbage designed the Analytical Engine."),
    ("chunk-5", "Ada Lovelace wrote notes on the Analytical Engine."),
    # Deliberate contradiction with chunk-1: same head and relation, different tail.
    # Without it the conflict stage has no candidate group and never runs.
    ("chunk-6", "Ada Lovelace was born in Paris."),
]

TRIPLES: dict[str, list[list[str]]] = {
    "Ada Lovelace was born in London.": [["Ada Lovelace", "born in", "London"]],
    "London is the capital of England.": [["London", "capital of", "England"]],
    "England is part of the United Kingdom.": [["England", "part of", "United Kingdom"]],
    "Charles Babbage designed the Analytical Engine.": [
        ["Charles Babbage", "designed", "Analytical Engine"]
    ],
    "Ada Lovelace wrote notes on the Analytical Engine.": [
        ["Ada Lovelace", "wrote notes on", "Analytical Engine"]
    ],
    "Ada Lovelace was born in Paris.": [["Ada Lovelace", "born in", "Paris"]],
}

TYPES = {
    "Ada Lovelace": "person",
    "Charles Babbage": "person",
    "London": "city",
    "England": "country",
    "United Kingdom": "country",
    "Analytical Engine": "machine",
    "Paris": "city",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def deterministic_embed(texts, **_: object) -> np.ndarray:
    """Hash-based bag-of-words embedding, L2-normalised.

    Real cosine geometry (shared words raise similarity) without a network call, so
    retrieval ranking is exercised rather than stubbed.
    """
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float64)
    for row, text in enumerate(texts):
        for token in _tokens(str(text)):
            digest = hashlib.md5(token.encode()).digest()
            out[row, digest[0] % EMBED_DIM] += 1.0
        norm = np.linalg.norm(out[row])
        if norm:
            out[row] /= norm
        else:
            out[row, 0] = 1.0
    return out


class ScriptedLLM:
    """Answers by agent role; records every call so the chain can be asserted on."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, prompt: str, **kwargs: object) -> str:
        agent = str(kwargs.get("agent") or "")
        self.calls.append(agent)

        if agent == "openie.ner":
            passage = self._passage(prompt)
            names = sorted({e for t in TRIPLES.get(passage, []) for e in (t[0], t[2])})
            return json.dumps({"named_entities": names})

        if agent == "openie.triple":
            passage = self._passage(prompt)
            return json.dumps({"triples": TRIPLES.get(passage, [])})

        if agent == "schema.extract":
            rows = []
            for triple in re.findall(r"\[([^\[\]]+)\]", prompt):
                parts = [p.strip().strip("'\"") for p in triple.split("','")]
                if len(parts) != 3:
                    parts = [p.strip().strip("'\" ") for p in triple.split(",")]
                if len(parts) != 3:
                    continue
                head, rel, tail = parts
                rows.append(
                    {
                        "triple": [head, rel, tail],
                        "ontology": [
                            TYPES.get(head, "thing"),
                            rel,
                            TYPES.get(tail, "thing"),
                        ],
                    }
                )
            return json.dumps({"ontology_triples": rows})

        if agent == "conflict.detect":
            # Report the planted London/Paris contradiction, with the confidence the
            # engine now requires before it will act on one.
            if "Paris" in prompt and "London" in prompt:
                return json.dumps(
                    {
                        "has_conflict": True,
                        "conflicts": [
                            {
                                "triple1": ["Ada Lovelace", "born in", "London"],
                                "triple2": ["Ada Lovelace", "born in", "Paris"],
                                "conflict_type": "mutual",
                                "is_hard_conflict": True,
                                "needs_resolution": True,
                                "confidence": 0.95,
                                "conflict_reason": "birthplace is single-valued",
                            }
                        ],
                    }
                )
            return json.dumps({"has_conflict": False, "conflicts": []})

        if agent == "conflict.resolve":
            return json.dumps(
                {
                    "resolved_triples": [
                        {
                            "original_triple": ["Ada Lovelace", "born in", "London"],
                            "conflict_type": "mutual",
                            "resolution": "kept",
                            "reason": "the London passage is the reliable source",
                        },
                        {
                            "original_triple": ["Ada Lovelace", "born in", "Paris"],
                            "conflict_type": "mutual",
                            "resolution": "discarded",
                            "resolved_triple": None,
                            "reason": "contradicted by the London passage",
                        },
                    ]
                }
            )

        if agent == "retrieve.fact_rerank":
            return json.dumps({"relevant_facts": [1, 2, 3]})

        # qa.reading and anything else: answer from the passages actually supplied.
        return f"Thought: reading {prompt.count('<<<PASSAGE')} passages.\nAnswer: OK [1]"

    @staticmethod
    def _passage(prompt: str) -> str:
        for _, text in CORPUS:
            if text in prompt:
                return text
        return ""


async def _build_engine(tmp_path) -> tuple[MemGraphRAG, ScriptedLLM]:
    llm = ScriptedLLM()

    async def embedding_func(texts, **kwargs):
        return deterministic_embed(texts, **kwargs)

    rag = MemGraphRAG(
        working_dir=str(tmp_path / "storage"),
        llm_model_func=llm,
        embedding_func=embedding_func,
        embedding_dim=EMBED_DIM,
        max_async_llm=2,
    )
    await rag.initialize_storages()
    return rag, llm


async def test_full_pipeline_indexes_retrieves_and_answers(tmp_path) -> None:
    rag, llm = await _build_engine(tmp_path)

    chunks = [{"idx": cid, "content": text} for cid, text in CORPUS]
    await rag.ainsert(chunks)

    # Every stage of PROCESSING actually ran.
    assert "openie.ner" in llm.calls
    assert "openie.triple" in llm.calls
    assert "schema.extract" in llm.calls
    assert "conflict.detect" in llm.calls

    assert "conflict.resolve" in llm.calls, "a hard conflict must reach resolution"

    memory = rag.memory
    assert len(memory.passage_layer) == len(CORPUS)
    assert memory.schema_layer, "ontology extraction produced no schema"

    # The arbitration actually mutated the memory: the discarded triple is gone.
    triples = {tuple(f.content) for f in memory.fact_layer}
    assert ("Ada Lovelace", "born in", "London") in triples
    assert ("Ada Lovelace", "born in", "Paris") not in triples, (
        "conflict resolution did not discard the losing triple"
    )

    edges = await rag.graph.get_all_edges()
    edge_types = {e.get("type") for e in edges}
    assert "ENTITY_RELATION" in edge_types, "fact graph missing"
    assert "PASSAGE_ENTITY" in edge_types
    assert "FACT_PASSAGE" in edge_types

    await rag.prepare_retrieval()
    assert rag.ready_to_retrieve

    sol = await rag.aquery("Where was Ada Lovelace born?", param=QueryParam(mode="ppr", top_k=3))
    assert not isinstance(sol, str)
    assert sol.docs, "PPR returned no passage"
    assert any("London" in d for d in sol.docs), (
        f"the supporting passage was not retrieved: {sol.docs}"
    )
    assert sol.answer
    assert sol.references, "references must be populated for the API layer"

    assert "qa.reading" in llm.calls


async def test_ppr_traverses_a_chain_that_shares_no_passage(tmp_path) -> None:
    """The multi-hop property the fact graph exists for.

    "Ada Lovelace" and "United Kingdom" never co-occur in a passage; they are joined
    only through London -> England. Without entity<->entity edges the graph is
    bipartite and no walk connects them, so this is the test that distinguishes
    MemGraphRAG from a dense retriever with passage expansion.
    """
    rag, _ = await _build_engine(tmp_path)
    await rag.ainsert([{"idx": cid, "content": text} for cid, text in CORPUS])
    await rag.prepare_retrieval()

    for pid, content in rag._passage_id_to_content.items():
        if "Ada Lovelace was born" in content:
            start = pid
            break
    else:  # pragma: no cover - fixture guard
        pytest.fail("seed passage missing")

    scores = await rag._run_ppr({start: 1.0}, damping=0.5)
    reachable = {rag._passage_id_to_content.get(pid, "") for pid, s in scores.items() if s > 0}
    assert any("United Kingdom" in text for text in reachable), (
        "three-hop passage unreachable — entity<->entity edges are not connecting"
    )


async def test_empty_corpus_query_does_not_crash(tmp_path) -> None:
    """Query-before-ingest must degrade, not raise."""
    rag, _ = await _build_engine(tmp_path)
    await rag.prepare_retrieval()
    sol = await rag.aquery("anything?", param=QueryParam(mode="ppr", top_k=3))
    assert sol is not None
