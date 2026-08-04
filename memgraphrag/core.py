"""Slim MemGraphRAG core engine for indexing and retrieval.

Provenance: industrialized, async adaptation of
``MemGraphRAG/code/src/MemGraphRAG.py`` (``index_with_memory``, ``retrieve``,
``rag_qa``, ``prepare_retrieval_objects``, ``run_ppr``) using LightRAG-style
pluggable storage (``memgraphrag.storage.factory.get_storage_class``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

import numpy as np

from memgraphrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocStatusStorage,
    QueryParam,
)
from memgraphrag.constants import (
    DAMPING,
    EMBEDDING_DIM,
    FACT_SIMILARITY_THRESHOLD,
    LINKING_TOP_K,
    MAX_ASYNC_LLM,
    PASSAGE_NODE_WEIGHT,
    PPR_ENGINE,
    SKIP_FACT_RERANK,
    TOP_K,
    WORKING_DIR,
)
from memgraphrag.exceptions import NotReadyError, PipelineError
from memgraphrag.memory import ThreeLayerMemory
from memgraphrag.namespace import NameSpace
from memgraphrag.openie.openai_openie import OpenIE
from memgraphrag.prompts.templates import (
    CONFLICT_DETECTION_SYSTEM,
    CONFLICT_DETECTION_USER_TEMPLATE,
    CONFLICT_RESOLUTION_SYSTEM,
    CONFLICT_RESOLUTION_USER_TEMPLATE,
    ONTOLOGY_EXTRACTION_SYSTEM,
    ONTOLOGY_EXTRACTION_USER_TEMPLATE,
    get_query_instruction,
    render_rag_qa,
)
from memgraphrag.ppr import get_ppr_engine
from memgraphrag.ppr.base import PPREngine
from memgraphrag.ppr.igraph_engine import IgraphPPREngine
from memgraphrag.rerank import FactFilter
from memgraphrag.storage import verify_storage_implementation
from memgraphrag.storage.factory import get_storage_class
from memgraphrag.utils.env import get_env_value
from memgraphrag.utils.hashing import compute_mdhash_id
from memgraphrag.utils.misc import QuerySolution

logger = logging.getLogger(__name__)

LLMFunc = Callable[..., Awaitable[str]]
EmbedFunc = Callable[..., Awaitable[np.ndarray]]


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _triple_str(triple: Sequence[str]) -> str:
    return str(tuple(str(x) for x in triple))


def _run_sync(coro: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already in an event loop — run in a fresh loop in a thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class MemGraphRAG:
    """Memory-based GraphRAG engine (POC-oriented slim core)."""

    def __init__(
        self,
        working_dir: str = WORKING_DIR,
        workspace: str = "",
        kv_storage: str = "JsonKVStorage",
        vector_storage: str = "NanoVectorDBStorage",
        graph_storage: str = "IgraphStorage",
        doc_status_storage: str = "JsonDocStatusStorage",
        llm_model_func: LLMFunc | None = None,
        embedding_func: EmbedFunc | None = None,
        embedding_dim: int | None = None,
        ppr_engine: str = PPR_ENGINE,
        top_k: int = TOP_K,
        linking_top_k: int = LINKING_TOP_K,
        passage_node_weight: float = PASSAGE_NODE_WEIGHT,
        damping: float = DAMPING,
        fact_similarity_threshold: float = FACT_SIMILARITY_THRESHOLD,
        skip_fact_rerank: bool = SKIP_FACT_RERANK,
        max_async_llm: int = MAX_ASYNC_LLM,
        **kwargs: Any,
    ) -> None:
        self.working_dir = working_dir
        self.workspace = workspace or ""
        os.makedirs(self.working_dir, exist_ok=True)

        self.kv_storage_name = kv_storage
        self.vector_storage_name = vector_storage
        self.graph_storage_name = graph_storage
        self.doc_status_storage_name = doc_status_storage

        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self.embedding_dim = embedding_dim or get_env_value(
            "EMBEDDING_DIM", EMBEDDING_DIM, int
        )
        self.ppr_engine_name = ppr_engine or get_env_value("PPR_ENGINE", PPR_ENGINE, str)

        self.top_k = top_k
        self.linking_top_k = linking_top_k
        self.passage_node_weight = passage_node_weight
        self.damping = damping
        self.fact_similarity_threshold = fact_similarity_threshold
        self.skip_fact_rerank = skip_fact_rerank
        self.max_async_llm = max_async_llm

        self.global_config: dict[str, Any] = {
            "working_dir": self.working_dir,
            "workspace": self.workspace,
            "embedding_dim": self.embedding_dim,
            "top_k": self.top_k,
            "linking_top_k": self.linking_top_k,
            "passage_node_weight": self.passage_node_weight,
            "damping": self.damping,
            "fact_similarity_threshold": self.fact_similarity_threshold,
            "skip_fact_rerank": self.skip_fact_rerank,
            "ppr_engine": self.ppr_engine_name,
            **kwargs,
        }

        verify_storage_implementation("KV_STORAGE", self.kv_storage_name)
        verify_storage_implementation("VECTOR_STORAGE", self.vector_storage_name)
        verify_storage_implementation("GRAPH_STORAGE", self.graph_storage_name)
        verify_storage_implementation("DOC_STATUS_STORAGE", self.doc_status_storage_name)

        kv_cls = get_storage_class(self.kv_storage_name)
        vec_cls = get_storage_class(self.vector_storage_name)
        graph_cls = get_storage_class(self.graph_storage_name)
        doc_cls = get_storage_class(self.doc_status_storage_name)

        common = dict(
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
        )

        self.memory_kv: BaseKVStorage = kv_cls(namespace=NameSpace.KV_MEMORY, **common)
        self.openie_kv: BaseKVStorage = kv_cls(namespace=NameSpace.KV_OPENIE, **common)
        self.chunks_kv: BaseKVStorage = kv_cls(namespace=NameSpace.KV_TEXT_CHUNKS, **common)

        self.chunks_vdb: BaseVectorStorage = vec_cls(
            namespace=NameSpace.VECTOR_CHUNKS, **common
        )
        self.entities_vdb: BaseVectorStorage = vec_cls(
            namespace=NameSpace.VECTOR_ENTITIES, **common
        )
        self.facts_vdb: BaseVectorStorage = vec_cls(
            namespace=NameSpace.VECTOR_FACTS, **common
        )

        self.graph: BaseGraphStorage = graph_cls(
            namespace=NameSpace.GRAPH_MEMORY, **common
        )
        self.doc_status: DocStatusStorage = doc_cls(
            namespace=NameSpace.DOC_STATUS, **common
        )

        self._storages: list[Any] = [
            self.memory_kv,
            self.openie_kv,
            self.chunks_kv,
            self.chunks_vdb,
            self.entities_vdb,
            self.facts_vdb,
            self.graph,
            self.doc_status,
        ]

        self.openie = (
            OpenIE(self.llm_model_func, max_concurrency=self.max_async_llm)
            if self.llm_model_func
            else None
        )
        self.fact_filter = FactFilter(default_threshold=self.fact_similarity_threshold)
        self.memory: ThreeLayerMemory | None = None
        self.ready_to_retrieve = False
        self._ppr: PPREngine | None = None
        self._passage_ids: list[str] = []
        self._entity_ids: list[str] = []
        self._fact_ids: list[str] = []
        self._passage_id_to_content: dict[str, str] = {}
        self._fact_id_to_triple: dict[str, tuple[str, str, str]] = {}
        self._entity_to_passages: dict[str, set[str]] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Storage lifecycle
    # ------------------------------------------------------------------

    async def initialize_storages(self) -> None:
        for store in self._storages:
            await store.initialize()
        self._initialized = True
        logger.info("MemGraphRAG storages initialized under %s", self.working_dir)

    async def finalize_storages(self) -> None:
        for store in self._storages:
            await store.finalize()
        self._initialized = False
        logger.info("MemGraphRAG storages finalized")

    # ------------------------------------------------------------------
    # Schema / conflict stages (lightweight)
    # ------------------------------------------------------------------

    async def extract_schema(self, memory: ThreeLayerMemory) -> ThreeLayerMemory:
        """Lightweight ontology extraction; skips gracefully on LLM failure."""
        if not self.llm_model_func or not memory.fact_layer:
            logger.info("extract_schema: skipped (no LLM or empty facts)")
            return memory

        # Sample a few facts for POC schema tagging
        sample = memory.fact_layer[: min(8, len(memory.fact_layer))]
        triples = [list(f.content) for f in sample]
        passage = memory.passage_layer[0].content if memory.passage_layer else ""
        user = ONTOLOGY_EXTRACTION_USER_TEMPLATE.substitute(
            passage=passage, triples=str(triples)
        )
        try:
            raw = await self.llm_model_func(user, system_prompt=ONTOLOGY_EXTRACTION_SYSTEM)
            logger.debug("extract_schema LLM response length=%d", len(str(raw)))
        except Exception as exc:
            logger.warning("extract_schema failed, continuing without schema: %s", exc)
        return memory

    async def filter_ontology(self, memory: ThreeLayerMemory) -> ThreeLayerMemory:
        """Ontology filter stub — returns memory unchanged for POC."""
        logger.info(
            "filter_ontology: no-op (%d schemas, %d facts)",
            len(memory.schema_layer),
            len(memory.fact_layer),
        )
        return memory

    async def detect_conflicts(self, memory: ThreeLayerMemory) -> dict[str, Any]:
        """Lightweight conflict detection stub."""
        result: dict[str, Any] = {
            "has_conflict": False,
            "conflicts": [],
            "summary": {"hard_conflicts": 0},
        }
        if not self.llm_model_func or len(memory.fact_layer) < 2:
            logger.info("detect_conflicts: skipped")
            return result

        target = list(memory.fact_layer[0].content)
        related = [list(f.content) for f in memory.fact_layer[1:6]]
        user = CONFLICT_DETECTION_USER_TEMPLATE.substitute(
            target_triple=str(target), related_triples=str(related)
        )
        try:
            raw = await self.llm_model_func(user, system_prompt=CONFLICT_DETECTION_SYSTEM)
            logger.debug("detect_conflicts response length=%d", len(str(raw)))
        except Exception as exc:
            logger.warning("detect_conflicts failed: %s", exc)
        return result

    async def resolve_conflicts(
        self, memory: ThreeLayerMemory, conflicts: Mapping[str, Any]
    ) -> tuple[ThreeLayerMemory, dict[str, Any]]:
        """Conflict resolution stub — returns memory unchanged."""
        resolution = {"summary": {"resolved": 0}, "resolved_triples": []}
        if not conflicts.get("has_conflict"):
            logger.info("resolve_conflicts: no hard conflicts")
            return memory, resolution
        if not self.llm_model_func:
            return memory, resolution
        user = CONFLICT_RESOLUTION_USER_TEMPLATE.substitute(
            conflicting_triples_with_sources=str(conflicts.get("conflicts", []))
        )
        try:
            raw = await self.llm_model_func(
                user, system_prompt=CONFLICT_RESOLUTION_SYSTEM
            )
            logger.debug("resolve_conflicts response length=%d", len(str(raw)))
        except Exception as exc:
            logger.warning("resolve_conflicts failed: %s", exc)
        return memory, resolution

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def ainsert(self, chunks: Sequence[str] | Sequence[dict[str, str]]) -> dict[str, Any]:
        """Index already-chunked texts: OpenIE → memory → vectors → graph."""
        if not self._initialized:
            await self.initialize_storages()
        if not chunks:
            raise ValueError("ainsert requires at least one chunk")
        if self.llm_model_func is None or self.embedding_func is None:
            raise PipelineError("ainsert requires llm_model_func and embedding_func")

        assert self.openie is not None
        openie_docs = await self.openie.batch_openie(chunks)
        await self.openie_kv.upsert(
            {d["idx"]: d for d in openie_docs}
        )

        memory = ThreeLayerMemory()
        memory.build_from_raw_openie_results({"docs": openie_docs})
        memory = await self.extract_schema(memory)
        memory = await self.filter_ontology(memory)
        conflicts = await self.detect_conflicts(memory)
        memory, resolution = await self.resolve_conflicts(memory, conflicts)

        await self._embed_and_store_memory(memory, openie_docs)
        await self._install_memory_graph(memory)

        await self.memory_kv.upsert({"memory": memory.to_dict()})
        self.memory = memory
        self.ready_to_retrieve = False

        stats = {
            "num_passages": len(memory.passage_layer),
            "num_facts": len(memory.fact_layer),
            "num_schemas": len(memory.schema_layer),
            "conflict_summary": conflicts.get("summary", {}),
            "resolution_summary": resolution.get("summary", {}),
        }
        logger.info("ainsert complete: %s", stats)
        return {"memory": memory, "stats": stats, "openie_docs": openie_docs}

    def insert(self, chunks: Sequence[str] | Sequence[dict[str, str]]) -> dict[str, Any]:
        """Sync wrapper around :meth:`ainsert`."""
        return _run_sync(self.ainsert(chunks))

    async def _embed_and_store_memory(
        self, memory: ThreeLayerMemory, openie_docs: list[dict[str, Any]]
    ) -> None:
        # Passages
        passage_texts = [p.content for p in memory.passage_layer]
        passage_ids = [
            compute_mdhash_id(p.chunk_id or p.content, prefix="chunk-")
            if not str(p.chunk_id).startswith("chunk-")
            else str(p.chunk_id)
            for p in memory.passage_layer
        ]
        # Prefer stable hash of content when chunk_id is numeric index
        passage_ids = []
        for p in memory.passage_layer:
            pid = compute_mdhash_id(p.content, prefix="chunk-")
            passage_ids.append(pid)
            p.chunk_id = pid  # normalize for graph install

        if passage_texts:
            emb = await self.embedding_func(passage_texts)
            emb = np.asarray(emb)
            chunk_payload = {
                passage_ids[i]: {
                    "content": passage_texts[i],
                    "embedding": emb[i].tolist(),
                }
                for i in range(len(passage_ids))
            }
            await self.chunks_vdb.upsert(chunk_payload)
            await self.chunks_kv.upsert(
                {
                    passage_ids[i]: {"content": passage_texts[i], "idx": i}
                    for i in range(len(passage_ids))
                }
            )

        # Facts
        fact_texts = ["\t".join(f.content) for f in memory.fact_layer]
        fact_ids = [
            compute_mdhash_id(_triple_str(f.content), prefix="fact-")
            for f in memory.fact_layer
        ]
        if fact_texts:
            emb = await self.embedding_func(fact_texts)
            emb = np.asarray(emb)
            fact_payload = {
                fact_ids[i]: {
                    "content": _triple_str(memory.fact_layer[i].content),
                    "embedding": emb[i].tolist(),
                    "triple": list(memory.fact_layer[i].content),
                }
                for i in range(len(fact_ids))
            }
            await self.facts_vdb.upsert(fact_payload)

        # Entities from OpenIE + fact endpoints
        entities: list[str] = []
        seen: set[str] = set()
        for doc in openie_docs:
            for ent in doc.get("extracted_entities") or []:
                e = str(ent).strip().lower()
                if e and e not in seen:
                    seen.add(e)
                    entities.append(e)
        for fact in memory.fact_layer:
            for ent in (fact.content[0], fact.content[2]):
                e = str(ent).strip().lower()
                if e and e not in seen:
                    seen.add(e)
                    entities.append(e)

        entity_ids = [compute_mdhash_id(e, prefix="entity-") for e in entities]
        if entities:
            emb = await self.embedding_func(entities)
            emb = np.asarray(emb)
            ent_payload = {
                entity_ids[i]: {
                    "content": entities[i],
                    "embedding": emb[i].tolist(),
                }
                for i in range(len(entities))
            }
            await self.entities_vdb.upsert(ent_payload)

        self._passage_ids = passage_ids
        self._fact_ids = fact_ids
        self._entity_ids = entity_ids
        self._passage_id_to_content = dict(zip(passage_ids, passage_texts))
        self._fact_id_to_triple = {
            fact_ids[i]: tuple(memory.fact_layer[i].content)
            for i in range(len(fact_ids))
        }

    async def _install_memory_graph(self, memory: ThreeLayerMemory) -> None:
        """Write entity / passage nodes and simple co-occurrence edges."""
        await self.graph.clear()
        entity_to_passages: dict[str, set[str]] = {}

        for passage in memory.passage_layer:
            pid = passage.chunk_id
            await self.graph.upsert_node(
                pid,
                {
                    "id": pid,
                    "label": "Passage",
                    "layer": "passage",
                    "content": passage.content,
                },
            )

        for fact in memory.fact_layer:
            h, _r, t = fact.content
            for ent in (h, t):
                eid = compute_mdhash_id(str(ent).strip().lower(), prefix="entity-")
                if not await self.graph.has_node(eid):
                    await self.graph.upsert_node(
                        eid,
                        {
                            "id": eid,
                            "label": "Entity",
                            "layer": "entity",
                            "content": str(ent).strip().lower(),
                        },
                    )
                for pidx in fact.passage_indices:
                    passage = memory.get_passage_by_idx(pidx)
                    if passage is None:
                        continue
                    pid = passage.chunk_id
                    await self.graph.upsert_edge(
                        eid,
                        pid,
                        {"type": "PASSAGE_ENTITY", "weight": 1.0},
                    )
                    entity_to_passages.setdefault(eid, set()).add(pid)

            # Optional fact node
            fid = compute_mdhash_id(_triple_str(fact.content), prefix="fact-")
            if not await self.graph.has_node(fid):
                await self.graph.upsert_node(
                    fid,
                    {
                        "id": fid,
                        "label": "Fact",
                        "layer": "fact",
                        "content": _triple_str(fact.content),
                    },
                )

        self._entity_to_passages = entity_to_passages

    # ------------------------------------------------------------------
    # Retrieval prep
    # ------------------------------------------------------------------

    async def prepare_retrieval(self) -> None:
        """Load memory + graph adjacency needed for PPR retrieval."""
        if not self._initialized:
            await self.initialize_storages()

        if self.memory is None:
            stored = await self.memory_kv.get_by_id("memory")
            if stored:
                self.memory = ThreeLayerMemory.from_dict(stored)

        self._passage_id_to_content = {}
        self._passage_ids = []
        self._fact_ids = []
        self._fact_id_to_triple = {}
        self._entity_to_passages = {}

        if self.memory is not None:
            for p in self.memory.passage_layer:
                pid = p.chunk_id or compute_mdhash_id(p.content, prefix="chunk-")
                self._passage_ids.append(pid)
                self._passage_id_to_content[pid] = p.content
            for f in self.memory.fact_layer:
                fid = compute_mdhash_id(_triple_str(f.content), prefix="fact-")
                self._fact_ids.append(fid)
                self._fact_id_to_triple[fid] = tuple(f.content)
                for ent in (f.content[0], f.content[2]):
                    eid = compute_mdhash_id(str(ent).strip().lower(), prefix="entity-")
                    for pidx in f.passage_indices:
                        passage = self.memory.get_passage_by_idx(pidx)
                        if passage is None:
                            continue
                        pid = passage.chunk_id
                        self._entity_to_passages.setdefault(eid, set()).add(pid)

        # Build igraph PPR engine from graph storage edges when possible
        edges: list[tuple[str, str]] = []
        weights: list[float] = []
        try:
            raw_edges = await self.graph.get_all_edges()
            for e in raw_edges:
                src = e.get("source") or e.get("src") or e.get("source_node_id")
                tgt = e.get("target") or e.get("tgt") or e.get("target_node_id")
                if src and tgt:
                    edges.append((str(src), str(tgt)))
                    weights.append(float(e.get("weight", 1.0)))
        except Exception as exc:
            logger.warning("prepare_retrieval: could not load graph edges: %s", exc)

        try:
            self._ppr = get_ppr_engine(
                self.ppr_engine_name,
                edges=edges,
                edge_weights=weights,
                passage_ids=self._passage_ids,
            )
        except Exception as exc:
            logger.warning("PPR engine init failed (%s); using empty igraph", exc)
            self._ppr = IgraphPPREngine(
                edges=edges, edge_weights=weights, passage_ids=self._passage_ids
            )

        self.ready_to_retrieve = True
        logger.info(
            "prepare_retrieval ready: passages=%d facts=%d edges=%d",
            len(self._passage_ids),
            len(self._fact_ids),
            len(edges),
        )

    # ------------------------------------------------------------------
    # Retrieve / query
    # ------------------------------------------------------------------

    async def aretrieve(
        self,
        queries: str | Sequence[str],
        param: QueryParam | None = None,
    ) -> list[QuerySolution]:
        """Retrieve passages for one or more queries."""
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)
        if not query_list:
            return []

        param = param or QueryParam()
        if not self.ready_to_retrieve:
            await self.prepare_retrieval()
        if self.embedding_func is None:
            raise PipelineError("aretrieve requires embedding_func")

        results: list[QuerySolution] = []
        for query in query_list:
            results.append(await self._retrieve_one(query, param))
        return results

    def retrieve(
        self,
        queries: str | Sequence[str],
        param: QueryParam | None = None,
    ) -> list[QuerySolution]:
        return _run_sync(self.aretrieve(queries, param=param))

    async def _retrieve_one(self, query: str, param: QueryParam) -> QuerySolution:
        # Embed query for facts
        q_fact = await self.embedding_func(
            [query],
            context="query",
            instruction=get_query_instruction("query_to_fact"),
        )
        q_fact_vec = np.asarray(q_fact[0], dtype=np.float64).tolist()

        fact_hits = await self.facts_vdb.query(q_fact_vec, top_k=param.linking_top_k)
        scores = [float(h.get("score", h.get("distance", 0.0))) for h in fact_hits]
        # nano-vectordb often returns similarity; if distance-like, invert heuristically
        if scores and max(scores) <= 1.0 and min(scores) >= 0:
            sim_scores = scores
        else:
            sim_scores = [1.0 / (1.0 + abs(s)) for s in scores]

        if param.skip_fact_rerank:
            kept = self.fact_filter.threshold_filter(
                sim_scores, param.fact_similarity_threshold
            )
        else:
            kept = self.fact_filter.llm_filter(
                query,
                [h.get("content") for h in fact_hits],
                list(range(len(fact_hits))),
                scores=sim_scores,
                threshold=param.fact_similarity_threshold,
            )

        kept_hits = [fact_hits[i] for i in kept if i < len(fact_hits)]

        if not kept_hits:
            return await self._dense_passage_retrieve(query, param)

        # Seed PPR from entities in filtered facts
        seed_weights: dict[str, float] = {}
        for hit, score in zip(kept_hits, [sim_scores[i] for i in kept if i < len(sim_scores)]):
            triple = hit.get("triple")
            if not triple:
                content = hit.get("content", "")
                try:
                    triple = eval(content) if isinstance(content, str) else content
                except Exception:
                    triple = None
            if not (isinstance(triple, (list, tuple)) and len(triple) == 3):
                continue
            for ent in (triple[0], triple[2]):
                eid = compute_mdhash_id(str(ent).strip().lower(), prefix="entity-")
                seed_weights[eid] = seed_weights.get(eid, 0.0) + float(score)
                # also seed linked passages lightly
                for pid in self._entity_to_passages.get(eid, set()):
                    seed_weights[pid] = (
                        seed_weights.get(pid, 0.0)
                        + float(score) * param.passage_node_weight
                    )

        # Blend dense passage seeds
        q_pass = await self.embedding_func(
            [query],
            context="query",
            instruction=get_query_instruction("query_to_passage"),
        )
        q_pass_vec = np.asarray(q_pass[0], dtype=np.float64).tolist()
        passage_hits = await self.chunks_vdb.query(q_pass_vec, top_k=param.top_k)
        for hit in passage_hits:
            pid = hit.get("id") or hit.get("__id__")
            if not pid:
                # Try match by content
                content = hit.get("content", "")
                for cand_id, cand_text in self._passage_id_to_content.items():
                    if cand_text == content:
                        pid = cand_id
                        break
            if not pid:
                continue
            score = float(hit.get("score", hit.get("distance", 0.0)))
            if score > 1.0 or score < 0:
                score = 1.0 / (1.0 + abs(score))
            seed_weights[str(pid)] = (
                seed_weights.get(str(pid), 0.0) + score * param.passage_node_weight
            )

        passage_scores = await self._run_ppr(seed_weights, damping=param.damping)
        if not passage_scores:
            return await self._dense_passage_retrieve(query, param)

        ranked = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[: param.top_k]
        docs = [
            self._passage_id_to_content.get(pid, "")
            for pid, _ in top
            if self._passage_id_to_content.get(pid)
        ]
        doc_scores = [float(s) for _, s in top[: len(docs)]]
        return QuerySolution(question=query, docs=docs, doc_scores=doc_scores)

    async def _dense_passage_retrieve(
        self, query: str, param: QueryParam
    ) -> QuerySolution:
        q_pass = await self.embedding_func(
            [query],
            context="query",
            instruction=get_query_instruction("query_to_passage"),
        )
        q_pass_vec = np.asarray(q_pass[0], dtype=np.float64).tolist()
        hits = await self.chunks_vdb.query(q_pass_vec, top_k=param.top_k)
        docs: list[str] = []
        scores: list[float] = []
        for hit in hits:
            content = hit.get("content", "")
            if content:
                docs.append(content)
                scores.append(float(hit.get("score", hit.get("distance", 0.0))))
        return QuerySolution(question=query, docs=docs, doc_scores=scores)

    async def _run_ppr(
        self, seed_weights: dict[str, float], damping: float
    ) -> dict[str, float]:
        if self._ppr is None:
            # Simple fallback: aggregate entity→passage seeds
            logger.warning("No PPR engine; using seed score fallback")
            out: dict[str, float] = {}
            for nid, w in seed_weights.items():
                if nid.startswith(("chunk-", "passage-", "doc-")):
                    out[nid] = out.get(nid, 0.0) + float(w)
                else:
                    for pid in self._entity_to_passages.get(nid, set()):
                        out[pid] = out.get(pid, 0.0) + float(w)
            return out

        if asyncio.iscoroutinefunction(getattr(self._ppr, "run", None)):
            return await self._ppr.run(seed_weights, damping=damping)  # type: ignore
        return self._ppr.run(seed_weights, damping=damping)

    async def aquery(
        self,
        query: str,
        param: QueryParam | None = None,
    ) -> Union[str, QuerySolution]:
        """Query with modes: ppr / naive / context / bypass."""
        param = param or QueryParam()
        mode = param.mode

        if mode == "bypass":
            if not self.llm_model_func:
                raise PipelineError("bypass mode requires llm_model_func")
            answer = await self.llm_model_func(query, system_prompt=param.user_prompt)
            return QuerySolution(question=query, docs=[], answer=str(answer))

        if mode == "naive":
            sol = await self._dense_passage_retrieve(query, param)
        else:
            # ppr or context — both retrieve via PPR path
            sols = await self.aretrieve(query, param=param)
            sol = sols[0]

        if mode == "context" or param.only_need_context:
            return sol

        if not self.llm_model_func:
            return sol

        system, user = render_rag_qa(query, sol.docs)
        if param.user_prompt:
            user = f"{user}\n\n{param.user_prompt}"
        history = param.conversation_history or None
        answer = await self.llm_model_func(
            user, system_prompt=system, history_messages=history
        )
        sol.answer = str(answer)
        return sol

    def query(
        self, query: str, param: QueryParam | None = None
    ) -> Union[str, QuerySolution]:
        return _run_sync(self.aquery(query, param=param))
