"""Slim MemGraphRAG core engine for indexing and retrieval.

Provenance: industrialized, async adaptation of
``MemGraphRAG/code/src/MemGraphRAG.py`` (``index_with_memory``, ``retrieve``,
``rag_qa``, ``prepare_retrieval_objects``, ``run_ppr``) using LightRAG-style
pluggable storage (``memgraphrag.storage.factory.get_storage_class``).
"""

from __future__ import annotations

import ast
import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
import math
import os
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Sequence, Union

import numpy as np

from memgraphrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocStatus,
    DocStatusStorage,
    QueryParam,
)
from memgraphrag.constants import (
    CONFLICT_ENABLED,
    CONFLICT_MAX_GROUPS,
    CONFLICT_MIN_CONFIDENCE,
    DAMPING,
    EMBEDDING_DIM,
    FACT_SIMILARITY_THRESHOLD,
    LINKING_TOP_K,
    MAX_ASYNC_LLM,
    ONTOLOGY_BATCH_SIZE,
    OPENIE_CHECKPOINT_SIZE,
    ONTOLOGY_MAX_DEACTIVATION_RATIO,
    ONTOLOGY_MIN_FREQUENCY,
    PASSAGE_NODE_WEIGHT,
    PPR_ENGINE,
    SKIP_FACT_RERANK,
    TOP_K,
    WORKING_DIR,
)
from memgraphrag.exceptions import PipelineError
from memgraphrag.memory import ThreeLayerMemory
from memgraphrag.namespace import NameSpace
from memgraphrag.observability.langfuse_trace import (
    flush_langfuse,
    observation,
    truncate_docs,
    update_observation,
)
from memgraphrag.openie.openai_openie import OpenIE
from memgraphrag.ppr import get_ppr_engine
from memgraphrag.ppr.base import PPREngine
from memgraphrag.ppr.igraph_engine import IgraphPPREngine
from memgraphrag.prompts.templates import (
    CONFLICT_DETECTION_SYSTEM,
    CONFLICT_DETECTION_USER_TEMPLATE,
    CONFLICT_RESOLUTION_SYSTEM,
    CONFLICT_RESOLUTION_USER_TEMPLATE,
    ONTOLOGY_EXTRACTION_SYSTEM,
    with_language,
    ONTOLOGY_EXTRACTION_USER_TEMPLATE,
    get_query_instruction,
    render_rag_qa,
)
from memgraphrag.rerank import FactFilter
from memgraphrag.storage import verify_storage_implementation
from memgraphrag.storage.factory import get_storage_class
from memgraphrag.utils.env import get_env_value
from memgraphrag.utils.canonical import canonical_key, canonical_triple
from memgraphrag.utils.hashing import compute_mdhash_id
from memgraphrag.utils.json_llm import extract_json_object
from memgraphrag.utils.misc import QuerySolution
from memgraphrag.utils.step_log import (
    INDEX_PHASE,
    RETRIEVE_PHASE,
    done_step,
    fail_step,
    main_step,
    phase,
    stage,
    sub_step,
    truncate,
)

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


def _information_density(
    entity_to_passages: Mapping[str, set], passage_ids: Sequence[str]
) -> dict[str, float]:
    """Per-passage information density, the sigma(...) factor of the paper's Eq. 19.

    ``P_init(p) = Sim(q, d_p) * alpha * sigma( sum_{e in E_p} IDF(e) / log(|E_p|+1) )``

    Only ``Sim(q, d_p) * alpha`` was implemented; the density factor was missing
    entirely, so every retrieved passage was weighted the same regardless of how
    discriminating its entities are. IDF is computed over the passage collection
    (each passage is a document), which is the only collection the engine has.

    The sigmoid is centred on the mean raw density of the corpus so the factor
    actually spreads over (0, 1): an uncentred sigmoid on unbounded positive values
    saturates at 1 and silently reduces to the constant it replaced.
    """
    passage_entities: dict[str, set] = {pid: set() for pid in passage_ids}
    for eid, pids in (entity_to_passages or {}).items():
        for pid in pids or ():
            bucket = passage_entities.get(str(pid))
            if bucket is not None:
                bucket.add(eid)

    total = len(passage_ids) or 1
    doc_freq: dict[str, int] = {}
    for entities in passage_entities.values():
        for eid in entities:
            doc_freq[eid] = doc_freq.get(eid, 0) + 1

    raw: dict[str, float] = {}
    for pid, entities in passage_entities.items():
        if not entities:
            raw[pid] = 0.0
            continue
        idf_sum = sum(math.log(total / (1.0 + doc_freq.get(eid, 0))) + 1.0 for eid in entities)
        raw[pid] = idf_sum / math.log(len(entities) + 1.0)

    values = [v for v in raw.values() if v]
    centre = (sum(values) / len(values)) if values else 0.0
    spread = max(1e-6, (max(values) - min(values)) / 4.0) if len(values) > 1 else 1.0
    return {pid: 1.0 / (1.0 + math.exp(-(value - centre) / spread)) for pid, value in raw.items()}


def _hit_score(hit: Mapping[str, Any]) -> float:
    """Read the cosine similarity of a vector-store hit.

    Single reader for the BaseVectorStorage score contract: ``score`` is authoritative,
    ``distance`` is the deprecated alias carrying the same value.
    """
    raw = hit.get("score")
    if raw is None:
        raw = hit.get("distance", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _is_cached_openie(rec: Any) -> bool:
    """Whether a stored OpenIE record counts as a cache hit.

    A record that extracted successfully but found nothing is still a hit: treating
    "empty" as "missing" re-ran two LLM calls for that chunk on every single insert,
    forever. Records written before the ``failed`` flag existed are recognised by
    their content.
    """
    if not isinstance(rec, dict):
        return False
    if rec.get("failed"):
        return False
    if "failed" in rec:
        return True
    return bool(rec.get("extracted_triples") or rec.get("extracted_entities"))


def _corpus_entity_surfaces(memory: Any, openie_docs: Sequence[dict] | None) -> list[str]:
    """Canonical, order-stable list of normalized entity surfaces for a corpus.

    Single source of truth for the entity set. ``_embed_and_store_memory`` and
    ``prepare_retrieval`` must derive the *same* list: the indexing path diffs it
    against the stored ids to decide what to embed and what to delete, so any
    divergence either re-embeds the whole corpus or deletes live entities. The two
    used to disagree — indexing included OpenIE ``extracted_entities`` while the
    retrieval path derived entities from fact endpoints only.
    """
    entities: list[str] = []
    seen: set[str] = set()
    for doc in openie_docs or []:
        for ent in (doc or {}).get("extracted_entities") or []:
            e = canonical_key(ent)
            if e and e not in seen:
                seen.add(e)
                entities.append(e)
    for fact in getattr(memory, "fact_layer", []) or []:
        for ent in (fact.content[0], fact.content[2]):
            e = canonical_key(ent)
            if e and e not in seen:
                seen.add(e)
                entities.append(e)
    return entities


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
        self.embedding_dim = embedding_dim or get_env_value("EMBEDDING_DIM", EMBEDDING_DIM, int)
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

        self.chunks_vdb: BaseVectorStorage = vec_cls(namespace=NameSpace.VECTOR_CHUNKS, **common)
        self.entities_vdb: BaseVectorStorage = vec_cls(
            namespace=NameSpace.VECTOR_ENTITIES, **common
        )
        self.facts_vdb: BaseVectorStorage = vec_cls(namespace=NameSpace.VECTOR_FACTS, **common)
        self.schemas_vdb: BaseVectorStorage = vec_cls(namespace=NameSpace.VECTOR_SCHEMAS, **common)

        self.graph: BaseGraphStorage = graph_cls(namespace=NameSpace.GRAPH_MEMORY, **common)
        self.doc_status: DocStatusStorage = doc_cls(namespace=NameSpace.DOC_STATUS, **common)

        self._storages: list[Any] = [
            self.memory_kv,
            self.openie_kv,
            self.chunks_kv,
            self.chunks_vdb,
            self.entities_vdb,
            self.facts_vdb,
            self.schemas_vdb,
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
        self._inactive_fact_idxs: set[int] = set()
        self._passage_density: dict[str, float] = {}
        # Guards the corpus rewrite (embed + graph install) against readers.
        self._corpus_lock = asyncio.Lock()
        self._fact_ids: list[str] = []
        self._schema_ids: list[str] = []
        self._passage_id_to_content: dict[str, str] = {}
        self._fact_id_to_triple: dict[str, tuple[str, str, str]] = {}
        self._schema_id_to_idx: dict[str, int] = {}
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
    # Schema / conflict stages
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_triple_key(triple: Sequence[str]) -> tuple[str, str, str]:
        return (
            str(triple[0]).strip(),
            str(triple[1]).strip(),
            str(triple[2]).strip(),
        )

    @staticmethod
    def _triple_lookup_key(triple: Sequence[str]) -> tuple[str, str, str]:
        """Matching key for a triple: accent- and case-insensitive.

        `.lower()` alone left `Réforme` and `reforme` as different facts, so a French
        corpus grew one entity, one vector and one graph node per spelling.
        """
        return canonical_triple(MemGraphRAG._normalize_triple_key(triple))

    def _parse_ontology_triples(
        self, raw: str
    ) -> list[tuple[tuple[str, str, str], tuple[str, str, str]]]:
        """Return list of (fact_triple, ontology_triple) from LLM JSON."""
        data = extract_json_object(raw)
        items = data.get("ontology_triples") or []
        if not isinstance(items, list):
            return []
        out: list[tuple[tuple[str, str, str], tuple[str, str, str]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            triple = item.get("triple")
            ontology = item.get("ontology")
            if not (
                isinstance(triple, (list, tuple))
                and len(triple) == 3
                and isinstance(ontology, (list, tuple))
                and len(ontology) == 3
            ):
                continue
            out.append(
                (
                    self._normalize_triple_key(triple),
                    self._normalize_triple_key(ontology),
                )
            )
        return out

    async def _apply_cached_ontology(self, memory: ThreeLayerMemory) -> int:
        """Apply ``extracted_triple_ontology`` from openie_kv when present."""
        linked = 0
        try:
            cached = await self.openie_kv.get_all()
        except Exception:
            cached = {}
        if not cached:
            return 0

        fact_by_key = {self._triple_lookup_key(f.content): f.idx for f in memory.fact_layer}
        for _doc_id, doc in cached.items():
            if not isinstance(doc, dict):
                continue
            ont_map = doc.get("extracted_triple_ontology") or {}
            if not isinstance(ont_map, dict) or not ont_map:
                continue
            for triple_key, ontology in ont_map.items():
                try:
                    triple_tuple = ast.literal_eval(str(triple_key))
                    if not (isinstance(triple_tuple, (list, tuple)) and len(triple_tuple) == 3):
                        continue
                except Exception:
                    continue
                if not (isinstance(ontology, (list, tuple)) and len(ontology) == 3):
                    continue
                fact_idx = fact_by_key.get(self._triple_lookup_key(triple_tuple))
                if fact_idx is None:
                    continue
                memory.link_fact_to_schema(fact_idx, self._normalize_triple_key(ontology))
                linked += 1
        return linked

    async def _persist_ontology_to_openie(self, memory: ThreeLayerMemory) -> None:
        """Write per-doc ontology maps back into openie_kv for cache reuse."""
        try:
            cached = await self.openie_kv.get_all()
        except Exception:
            return
        if not cached:
            return

        by_chunk: dict[str, dict[str, list[str]]] = {}
        by_content: dict[str, dict[str, list[str]]] = {}
        for fact in memory.fact_layer:
            if fact.schema_idx < 0:
                continue
            schema = memory.get_schema_by_idx(fact.schema_idx)
            if schema is None:
                continue
            key = str(tuple(fact.content))
            for pidx in fact.passage_indices:
                passage = memory.get_passage_by_idx(pidx)
                if passage is None:
                    continue
                by_chunk.setdefault(str(passage.chunk_id), {})[key] = list(schema.content)
                by_content.setdefault(passage.content, {})[key] = list(schema.content)

        updates: dict[str, Any] = {}
        for doc_id, doc in cached.items():
            if not isinstance(doc, dict):
                continue
            chunk_key = str(doc.get("idx", doc_id))
            ont_map = by_chunk.get(chunk_key) or by_content.get(doc.get("passage") or "")
            if ont_map:
                updated = dict(doc)
                updated["extracted_triple_ontology"] = ont_map
                updates[str(doc_id)] = updated
        if updates:
            await self.openie_kv.upsert(updates)

    async def extract_schema(self, memory: ThreeLayerMemory) -> ThreeLayerMemory:
        """Extract ontology schemas for all facts (batched, cache-aware)."""
        if not memory.fact_layer:
            stage(logger, "Extracting schema", skipped=True, reason="empty_facts")
            return memory

        cached_links = await self._apply_cached_ontology(memory)
        unlinked = [f for f in memory.fact_layer if f.schema_idx < 0]
        if cached_links and not unlinked:
            memory.recompute_schema_frequencies()
            stage(
                logger,
                "Extracting schema",
                source="cache",
                linked=cached_links,
                schemas=len(memory.schema_layer),
            )
            return memory

        if not self.llm_model_func:
            stage(
                logger,
                "Extracting schema",
                skipped=True,
                reason="no_llm",
                facts=len(memory.fact_layer),
                cached_links=cached_links,
            )
            return memory

        batch_size = max(1, get_env_value("ONTOLOGY_BATCH_SIZE", ONTOLOGY_BATCH_SIZE, int))
        # Group unlinked facts by passage for contextual typing.
        by_passage: dict[int, list[int]] = {}
        for fact in unlinked:
            if fact.passage_indices:
                for pidx in fact.passage_indices:
                    by_passage.setdefault(pidx, []).append(fact.idx)
            else:
                by_passage.setdefault(-1, []).append(fact.idx)

        batches: list[tuple[str, list[int]]] = []
        for pidx, fact_idxs in by_passage.items():
            passage = (
                memory.get_passage_by_idx(pidx).content
                if pidx >= 0 and memory.get_passage_by_idx(pidx)
                else ""
            )
            for i in range(0, len(fact_idxs), batch_size):
                batches.append((passage, fact_idxs[i : i + batch_size]))

        stage(
            logger,
            "Extracting schema",
            batches=len(batches),
            unlinked=len(unlinked),
            cached_links=cached_links,
            batch_size=batch_size,
            agent="schema.extract",
        )

        sem = asyncio.Semaphore(max(1, self.max_async_llm))
        fact_by_key = {self._triple_lookup_key(f.content): f.idx for f in memory.fact_layer}
        linked = 0
        failed_batches = 0

        async def _run_batch(passage: str, fact_idxs: list[int]) -> int:
            nonlocal failed_batches
            triples = [list(memory.fact_layer[i].content) for i in fact_idxs]
            user = ONTOLOGY_EXTRACTION_USER_TEMPLATE.substitute(
                passage=passage, triples=str(triples)
            )
            pairs: list = []
            raw = ""
            # A batch whose answer cannot be parsed is asked once more before its
            # facts are left untyped: on the RFE corpus 2 % of batches came back as
            # malformed JSON on the first call, almost none on the second.
            for _attempt in range(2):
                try:
                    async with sem:
                        raw = await self.llm_model_func(
                            user,
                            system_prompt=with_language(ONTOLOGY_EXTRACTION_SYSTEM),
                            agent="schema.extract",
                            llm_action="complete",
                        )
                except Exception as exc:
                    fail_step(logger, "schema.extract_batch", exc=exc)
                    failed_batches += 1
                    return 0
                pairs = self._parse_ontology_triples(str(raw))
                if pairs:
                    break
            if not pairs:
                failed_batches += 1
                fail_step(
                    logger,
                    "schema.extract_batch",
                    reason="empty_or_unparsed",
                    response_chars=len(str(raw)),
                )
                return 0
            count = 0
            for triple, ontology in pairs:
                fact_idx = fact_by_key.get(self._triple_lookup_key(triple))
                if fact_idx is None:
                    continue
                memory.link_fact_to_schema(fact_idx, ontology)
                count += 1
            return count

        results = await asyncio.gather(*[_run_batch(p, idxs) for p, idxs in batches])
        linked = sum(results)
        memory.recompute_schema_frequencies()
        try:
            await self._persist_ontology_to_openie(memory)
        except Exception as exc:
            fail_step(logger, "schema.cache_write", exc=exc)

        stage(
            logger,
            "Extracting schema done",
            linked=linked,
            schemas=len(memory.schema_layer),
            failed_batches=failed_batches,
            facts_untyped=sum(1 for f in memory.fact_layer if f.schema_idx < 0),
        )
        return memory

    async def filter_ontology(self, memory: ThreeLayerMemory) -> ThreeLayerMemory:
        """Frequency-based ontology filter with schema reindexing.

        Two defects are addressed here. First, the filter used to prune only the
        schema layer: facts whose schema was dropped kept living in ``fact_layer``,
        so they stayed in ``facts_vdb`` and in the graph and remained fully
        retrievable. ``ONTOLOGY_MIN_FREQUENCY`` therefore moved schema seeds and
        FACT_SCHEMA edges but never affected factual recall — the very thing the
        paper's ablation measures. We now track which facts the filter deactivates
        and exclude them from the vector store and from the graph.

        Second, making the filter real exposes a trap: with an absolute threshold on
        a small corpus, nearly every schema is seen once, so the filter would
        deactivate almost every fact and silently empty the index. A corpus that
        small simply carries no frequency signal, so we skip filtering rather than
        destroy it, and say so in the log.
        """
        min_freq = get_env_value("ONTOLOGY_MIN_FREQUENCY", ONTOLOGY_MIN_FREQUENCY, int)
        max_deactivation = get_env_value(
            "ONTOLOGY_MAX_DEACTIVATION_RATIO", ONTOLOGY_MAX_DEACTIVATION_RATIO, float
        )
        before = len(memory.schema_layer)
        self._inactive_fact_idxs = set()

        if min_freq <= 0:
            stage(logger, "Ontology filtering", before=before, min_frequency=min_freq, noop=True)
            return memory

        # Dry run: how much of the fact layer would this threshold silence?
        typed_before = {f.idx for f in memory.fact_layer if f.schema_idx >= 0}
        doomed = {
            idx
            for schema in memory.schema_layer
            if (schema.frequency or 0) < min_freq
            for idx in (schema.fact_indices or [])
        }
        ratio = (len(doomed) / len(typed_before)) if typed_before else 0.0

        if typed_before and ratio > max_deactivation:
            stage(
                logger,
                "Ontology filtering",
                before=before,
                min_frequency=min_freq,
                skipped="corpus too small for frequency signal",
                would_deactivate_facts=len(doomed),
                typed_facts=len(typed_before),
                ratio=round(ratio, 3),
                max_ratio=max_deactivation,
            )
            return memory

        stats = memory.filter_schemas_by_frequency(min_freq)
        typed_after = {f.idx for f in memory.fact_layer if f.schema_idx >= 0}
        self._inactive_fact_idxs = typed_before - typed_after
        stage(
            logger,
            "Ontology filtering",
            before=before,
            kept=stats["kept"],
            dropped=stats["dropped"],
            min_frequency=min_freq,
            deactivated_facts=len(self._inactive_fact_idxs),
        )
        return memory

    def _active_facts(self, memory: ThreeLayerMemory) -> list[Any]:
        """Facts that survived ontology filtering (see ``filter_ontology``)."""
        inactive = getattr(self, "_inactive_fact_idxs", None) or set()
        if not inactive:
            return list(memory.fact_layer)
        return [f for f in memory.fact_layer if f.idx not in inactive]

    def _conflict_candidate_groups(
        self, memory: ThreeLayerMemory, max_groups: int
    ) -> list[list[int]]:
        """Group facts that share (head, relation) or (relation, tail)."""
        hr: dict[tuple[str, str], list[int]] = {}
        rt: dict[tuple[str, str], list[int]] = {}
        for fact in memory.fact_layer:
            h, r, t = self._triple_lookup_key(fact.content)
            hr.setdefault((h, r), []).append(fact.idx)
            rt.setdefault((r, t), []).append(fact.idx)

        seen: set[frozenset[int]] = set()
        candidates: list[list[int]] = []
        for bucket in list(hr.values()) + list(rt.values()):
            uniq = sorted(set(bucket))
            if len(uniq) < 2:
                continue
            key = frozenset(uniq)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(uniq)

        # The cap used to `break` out of the loop, so the surviving groups were
        # whichever came first in an order derived from the MD5 sort of chunk ids —
        # arbitrary, and reshuffled on every ingest, meaning a newly added document's
        # conflicts could simply never be examined. Rank instead: the biggest buckets
        # are the most ambiguous, so they earn the budget, and the ordering is stable.
        candidates.sort(key=lambda g: (-len(g), g))
        self._conflict_groups_truncated = max(0, len(candidates) - max_groups)
        return candidates[:max_groups]

    async def detect_conflicts(self, memory: ThreeLayerMemory) -> dict[str, Any]:
        """Detect hard conflicts among fact groups via LLM."""
        result: dict[str, Any] = {
            "has_conflict": False,
            "conflicts": [],
            "summary": {"hard_conflicts": 0, "groups_checked": 0},
        }
        enabled = get_env_value("CONFLICT_ENABLED", CONFLICT_ENABLED, bool)
        if not enabled:
            stage(logger, "Detecting conflicts", skipped=True, reason="disabled")
            return result
        if not self.llm_model_func or len(memory.fact_layer) < 2:
            stage(
                logger,
                "Detecting conflicts",
                skipped=True,
                facts=len(memory.fact_layer),
            )
            return result

        max_groups = max(1, get_env_value("CONFLICT_MAX_GROUPS", CONFLICT_MAX_GROUPS, int))
        groups = self._conflict_candidate_groups(memory, max_groups)
        truncated = getattr(self, "_conflict_groups_truncated", 0)
        stage(
            logger,
            "Detecting conflicts",
            groups=len(groups),
            max_groups=max_groups,
            # Silent truncation reads as "no conflicts" to an operator. Surface it.
            groups_skipped_over_budget=truncated,
            agent="conflict.detect",
        )
        if truncated:
            logger.warning(
                "Conflict detection budget exhausted: %d candidate groups were not "
                "examined (CONFLICT_MAX_GROUPS=%d). Raise the budget or expect "
                "undetected conflicts.",
                truncated,
                max_groups,
            )
        if not groups:
            return result

        sem = asyncio.Semaphore(max(1, self.max_async_llm))
        min_confidence = get_env_value("CONFLICT_MIN_CONFIDENCE", CONFLICT_MIN_CONFIDENCE, float)
        hard: list[dict[str, Any]] = []

        async def _check_group(idxs: list[int]) -> list[dict[str, Any]]:
            target_idx = idxs[0]
            related_idxs = idxs[1:]
            target = list(memory.fact_layer[target_idx].content)
            related = [list(memory.fact_layer[i].content) for i in related_idxs]
            user = CONFLICT_DETECTION_USER_TEMPLATE.substitute(
                target_triple=str(target), related_triples=str(related)
            )
            try:
                async with sem:
                    raw = await self.llm_model_func(
                        user,
                        system_prompt=CONFLICT_DETECTION_SYSTEM,
                        agent="conflict.detect",
                        llm_action="complete",
                    )
            except Exception as exc:
                fail_step(logger, "conflict.detect_group", exc=exc)
                return []
            data = extract_json_object(str(raw))
            conflicts = data.get("conflicts") or []
            if not isinstance(conflicts, list):
                return []
            found: list[dict[str, Any]] = []
            for item in conflicts:
                if not isinstance(item, dict):
                    continue
                if not item.get("is_hard_conflict"):
                    continue
                # Guard rails the research code has and this port had lost. Without
                # them a single hallucinated `is_hard_conflict: true` is enough to
                # destroy a correct fact, since resolution may discard triples.
                if str(item.get("conflict_type") or "").strip().lower() in (
                    "duplicate",
                    "none",
                    "uncertain",
                ):
                    continue
                try:
                    confidence = float(item.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                if confidence < min_confidence:
                    continue
                # Attach fact indices when triples match
                t1 = item.get("triple1")
                t2 = item.get("triple2")
                idx1 = idx2 = None
                if isinstance(t1, (list, tuple)) and len(t1) == 3:
                    key = self._triple_lookup_key(t1)
                    for i in idxs:
                        if self._triple_lookup_key(memory.fact_layer[i].content) == key:
                            idx1 = i
                            break
                if isinstance(t2, (list, tuple)) and len(t2) == 3:
                    key = self._triple_lookup_key(t2)
                    for i in idxs:
                        if self._triple_lookup_key(memory.fact_layer[i].content) == key:
                            idx2 = i
                            break
                found.append(
                    {
                        **item,
                        "fact_idx1": idx1,
                        "fact_idx2": idx2,
                        "group_indices": idxs,
                    }
                )
            return found

        group_results = await asyncio.gather(*[_check_group(g) for g in groups])
        for items in group_results:
            hard.extend(items)

        result["conflicts"] = hard
        result["has_conflict"] = bool(hard)
        result["summary"] = {
            "hard_conflicts": len(hard),
            "groups_checked": len(groups),
        }
        stage(
            logger,
            "Detecting conflicts done",
            hard_conflicts=len(hard),
            groups_checked=len(groups),
        )
        return result

    async def resolve_conflicts(
        self, memory: ThreeLayerMemory, conflicts: Mapping[str, Any]
    ) -> tuple[ThreeLayerMemory, dict[str, Any]]:
        """Resolve hard conflicts using passage evidence; mutate memory."""
        resolution: dict[str, Any] = {
            "summary": {"resolved": 0, "discarded": 0, "modified": 0, "kept": 0},
            "resolved_triples": [],
        }
        hard = list(conflicts.get("conflicts") or [])
        if not conflicts.get("has_conflict") or not hard:
            stage(logger, "Resolving conflicts", skipped=True, hard_conflicts=0)
            return memory, resolution
        if not self.llm_model_func:
            return memory, resolution

        # Build evidence bundles keyed by conflict pairs / groups
        def _bundle_for(item: Mapping[str, Any]) -> str:
            t1 = item.get("triple1")
            t2 = item.get("triple2")
            idx1 = item.get("fact_idx1")
            idx2 = item.get("fact_idx2")
            sources: list[str] = []
            for fidx in (idx1, idx2):
                if not isinstance(fidx, int) or not (0 <= fidx < len(memory.fact_layer)):
                    continue
                fact = memory.fact_layer[fidx]
                for pidx in fact.passage_indices:
                    passage = memory.get_passage_by_idx(pidx)
                    if passage is None:
                        continue
                    preview = passage.content[:800]
                    sources.append(f"fact={list(fact.content)} fact_idx={fidx} passage={preview}")
            return (
                f"conflict_type={item.get('conflict_type')}\n"
                f"triple1={t1}\ntriple2={t2}\n"
                f"reason={item.get('conflict_reason')}\n"
                f"sources:\n" + "\n---\n".join(sources)
            )

        bundles = [_bundle_for(item) for item in hard]

        # Batch by an explicit character budget. Every hard conflict, with up to 800
        # characters of evidence per source passage, used to be concatenated into ONE
        # unbounded prompt: past the context window the call raised, the except below
        # returned an all-zero summary, and the document was still marked PROCESSED —
        # "resolution failed" was indistinguishable from "no conflict found".
        budget = max(2000, get_env_value("CONFLICT_RESOLUTION_CHAR_BUDGET", 24000, int))
        batches: list[list[int]] = []
        current: list[int] = []
        current_chars = 0
        for i, bundle in enumerate(bundles):
            size = len(bundle)
            if current and current_chars + size > budget:
                batches.append(current)
                current, current_chars = [], 0
            current.append(i)
            current_chars += size
        if current:
            batches.append(current)

        stage(
            logger,
            "Resolving conflicts",
            conflicts=len(hard),
            batches=len(batches),
            char_budget=budget,
            agent="conflict.resolve",
        )

        resolved_items: list[Any] = []
        failed_batches = 0
        for batch in batches:
            user = CONFLICT_RESOLUTION_USER_TEMPLATE.substitute(
                conflicting_triples_with_sources="\n\n====\n\n".join(bundles[i] for i in batch)
            )
            try:
                raw = await self.llm_model_func(
                    user,
                    system_prompt=CONFLICT_RESOLUTION_SYSTEM,
                    agent="conflict.resolve",
                    llm_action="complete",
                )
            except Exception as exc:
                # One failed batch no longer discards the others.
                failed_batches += 1
                fail_step(logger, "conflict.resolve", exc=exc, batch_size=len(batch))
                continue
            data = extract_json_object(str(raw))
            items = data.get("resolved_triples") or []
            if not isinstance(items, list):
                failed_batches += 1
                fail_step(
                    logger,
                    "conflict.resolve",
                    reason="unparsed",
                    response_chars=len(str(raw)),
                )
                continue
            resolved_items.extend(items)

        # Make partial failure visible: an operator must be able to tell a clean run
        # from one where resolutions were lost.
        resolution["summary"]["failed_batches"] = failed_batches
        resolution["summary"]["batches"] = len(batches)
        if failed_batches:
            logger.warning(
                "Conflict resolution: %d/%d batches failed; %d conflicts may remain "
                "unresolved in the memory graph.",
                failed_batches,
                len(batches),
                len(hard),
            )

        def _find_fact(triple: Sequence[str]) -> int | None:
            key = self._triple_lookup_key(triple)
            for f in memory.fact_layer:
                if self._triple_lookup_key(f.content) == key:
                    return f.idx
            return None

        kept = discarded = modified = 0
        # Apply modifications first (content-stable lookup), then discards.
        for item in resolved_items:
            if not isinstance(item, dict):
                continue
            action = str(item.get("resolution") or "").lower()
            original = item.get("original_triple")
            resolved_t = item.get("resolved_triple")
            if action != "modified":
                continue
            if not (
                isinstance(original, (list, tuple))
                and len(original) == 3
                and isinstance(resolved_t, (list, tuple))
                and len(resolved_t) == 3
            ):
                continue
            fact_idx = _find_fact(original)
            if fact_idx is None:
                continue
            memory.replace_fact(fact_idx, self._normalize_triple_key(resolved_t))
            modified += 1

        for item in resolved_items:
            if not isinstance(item, dict):
                continue
            action = str(item.get("resolution") or "").lower()
            original = item.get("original_triple")
            if action == "kept":
                if isinstance(original, (list, tuple)) and len(original) == 3:
                    if _find_fact(original) is not None:
                        kept += 1
                continue
            if action != "discarded":
                continue
            if not (isinstance(original, (list, tuple)) and len(original) == 3):
                continue
            fact_idx = _find_fact(original)
            if fact_idx is None:
                continue
            memory.remove_fact(fact_idx)
            discarded += 1

        resolution["resolved_triples"] = resolved_items
        resolution["summary"] = {
            "resolved": kept + discarded + modified,
            "discarded": discarded,
            "modified": modified,
            "kept": kept,
        }
        stage(
            logger,
            "Resolving conflicts done",
            **resolution["summary"],
            facts=len(memory.fact_layer),
        )
        return memory, resolution

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _normalize_chunks(
        self, chunks: Sequence[str] | Sequence[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Normalize inputs to ``{idx: chunk-…, content: …}`` dicts."""
        prepared: list[dict[str, str]] = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                content = chunk.strip()
                if not content:
                    continue
                prepared.append(
                    {
                        "idx": compute_mdhash_id(content, prefix="chunk-"),
                        "content": content,
                    }
                )
                continue
            if isinstance(chunk, dict):
                content = str(chunk.get("content") or chunk.get("passage") or "").strip()
                if not content:
                    continue
                idx = str(chunk.get("idx") or chunk.get("id") or "")
                if not idx.startswith("chunk-"):
                    idx = compute_mdhash_id(content, prefix="chunk-")
                prepared.append({"idx": idx, "content": content})
                continue
            content = str(chunk).strip()
            if content:
                prepared.append(
                    {
                        "idx": compute_mdhash_id(content, prefix="chunk-"),
                        "content": content,
                    }
                )
        return prepared

    async def _doc_status_all(self) -> dict[str, dict[str, Any]]:
        """Return all doc-status records (fallback when get_all is missing)."""
        try:
            return await self.doc_status.get_all()
        except NotImplementedError:
            docs: dict[str, dict[str, Any]] = {}
            for status in DocStatus:
                try:
                    part = await self.doc_status.get_docs_by_statuses([status])
                    docs.update(part)
                except Exception:
                    pass
            return docs

    async def _load_corpus_openie_docs(
        self,
        *,
        extra_chunk_ids: Sequence[str] | None = None,
        exclude_doc_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Load cached OpenIE docs for the whole corpus (+ the chunks being inserted).

        The corpus is the union of two sources, because the engine is used two ways:

        * through the API pipeline, which records PROCESSED documents with their
          ``chunk_ids`` in doc-status;
        * embedded, calling ``ainsert`` directly (the path documented in
          ``docs/ProgramingWithCore.md``), which never writes doc-status at all.

        Deriving the corpus from doc-status alone made the embedded path destructive:
        the second ``ainsert`` saw only its own batch as the corpus, so
        ``_embed_and_store_memory`` computed every previously indexed passage as an
        orphan and deleted it. Chunks known to the OpenIE cache but claimed by no
        doc-status record are therefore kept.
        """
        chunk_ids: set[str] = set(str(c) for c in (extra_chunk_ids or []) if c)
        all_docs = await self._doc_status_all()
        claimed: set[str] = set()
        for doc_id, record in all_docs.items():
            for cid in record.get("chunk_ids") or []:
                if cid:
                    claimed.add(str(cid))
            if exclude_doc_ids and doc_id in exclude_doc_ids:
                continue
            if str(record.get("status") or "") != DocStatus.PROCESSED.value:
                continue
            for cid in record.get("chunk_ids") or []:
                if cid:
                    chunk_ids.add(str(cid))

        try:
            cached = await self.openie_kv.get_all()
        except Exception:
            cached = {}

        if not chunk_ids and not claimed:
            # No document tracking at all (embedded use): the cache *is* the corpus.
            return [dict(v) for v in cached.values() if isinstance(v, dict)]

        # Chunks nobody claims belong to embedded inserts; dropping them would delete
        # their vectors on the next insert.
        for cid in cached:
            if str(cid) not in claimed:
                chunk_ids.add(str(cid))

        id_list = sorted(chunk_ids)
        records = await self.openie_kv.get_by_ids(id_list)
        docs: list[dict[str, Any]] = []
        for cid, rec in zip(id_list, records):
            if not isinstance(rec, dict):
                continue
            doc = dict(rec)
            doc.setdefault("idx", cid)
            docs.append(doc)
        return docs

    async def _previous_vector_ids(self) -> dict[str, set[str]]:
        """Ids currently reflected in memory / in-memory caches."""
        if self._passage_ids or self._fact_ids or self._entity_ids or self._schema_ids:
            return {
                "passages": set(self._passage_ids),
                "facts": set(self._fact_ids),
                "entities": set(self._entity_ids),
                "schemas": set(self._schema_ids),
            }
        mem = await self.memory_kv.get_by_id("memory")
        if not isinstance(mem, dict):
            return {
                "passages": set(),
                "facts": set(),
                "entities": set(),
                "schemas": set(),
            }
        passages = mem.get("passage_layer") or mem.get("passages") or []
        facts = mem.get("fact_layer") or mem.get("facts") or []
        schemas = mem.get("schema_layer") or mem.get("schemas") or []
        passage_ids: set[str] = set()
        for p in passages:
            if not isinstance(p, dict):
                continue
            content = str(p.get("content") or "")
            cid = str(p.get("chunk_id") or "")
            if cid.startswith("chunk-"):
                passage_ids.add(cid)
            elif content:
                passage_ids.add(compute_mdhash_id(content, prefix="chunk-"))
        fact_ids = {
            compute_mdhash_id(_triple_str(f.get("content") or ()), prefix="fact-")
            for f in facts
            if isinstance(f, dict) and f.get("content")
        }
        schema_ids = {
            compute_mdhash_id(_triple_str(s.get("content") or ()), prefix="schema-")
            for s in schemas
            if isinstance(s, dict) and s.get("content")
        }
        entity_ids: set[str] = set()
        for f in facts:
            if not isinstance(f, dict):
                continue
            content = f.get("content") or ()
            if isinstance(content, (list, tuple)) and len(content) == 3:
                for ent in (content[0], content[2]):
                    e = canonical_key(ent)
                    if e:
                        entity_ids.add(compute_mdhash_id(e, prefix="entity-"))
        return {
            "passages": passage_ids,
            "facts": fact_ids,
            "entities": entity_ids,
            "schemas": schema_ids,
        }

    @asynccontextmanager
    async def _storage_batch(self):
        """Hold every store in batch mode for the duration of a corpus rewrite.

        The file-backed backends persist their whole state on each write, so a caller
        that upserts record by record pays a full rewrite per record. Measured today
        this changes nothing — ``_embed_and_store_memory`` already issues one bulk
        ``upsert`` per store, so the flush count is 9 with or without the batch (the
        quadratic cost that mattered was ``IgraphStorage`` writing the GraphML per node
        and per edge, fixed in the backend itself). It is wired here so the guarantee
        holds by construction: any future per-record loop inside this block collapses
        to one flush instead of silently reintroducing the problem.

        ``batch()`` is a no-op on the ABCs, so PostgreSQL passes straight through.
        ``Neo4JStorage`` overrides it for a different reason than the file backends:
        not to avoid rewrites but to amortise round trips — it buffers the writes and
        flushes them with UNWIND, which took the graph install from ~5 to ~3 800
        writes per second on the RFE corpus.
        """
        stores = [
            self.chunks_vdb,
            self.facts_vdb,
            self.entities_vdb,
            self.schemas_vdb,
            self.chunks_kv,
            self.memory_kv,
            self.openie_kv,
            self.graph,
        ]
        async with AsyncExitStack() as stack:
            for store in stores:
                batch = getattr(store, "batch", None)
                if batch is None:
                    continue
                await stack.enter_async_context(batch())
            yield

    async def _rebuild_memory_from_openie(
        self,
        openie_docs: list[dict[str, Any]],
        *,
        run_conflicts: bool = True,
    ) -> tuple[ThreeLayerMemory, dict[str, Any], dict[str, Any]]:
        """Build three-layer memory from OpenIE docs (optionally skip conflicts)."""
        memory = ThreeLayerMemory()
        stage(
            logger,
            "Building three-layer memory",
            openie_docs=len(openie_docs),
            run_conflicts=run_conflicts,
        )
        memory.build_from_raw_openie_results({"docs": openie_docs})
        stage(
            logger,
            "Built memory structure",
            passages=len(memory.passage_layer),
            facts=len(memory.fact_layer),
            schemas=len(memory.schema_layer),
        )
        memory = await self.extract_schema(memory)
        memory = await self.filter_ontology(memory)
        if run_conflicts:
            conflicts = await self.detect_conflicts(memory)
            memory, resolution = await self.resolve_conflicts(memory, conflicts)
        else:
            conflicts = {"summary": {"skipped": True}}
            resolution = {"summary": {"skipped": True}}
            stage(logger, "Detecting conflicts", skipped=True, reason="delete_rebuild")
        return memory, conflicts, resolution

    async def _extract_openie_checkpointed(self, missing_chunks: list[dict[str, Any]]) -> None:
        """Run OpenIE in sub-batches and persist each one before starting the next.

        The cache used to be written once, after every chunk of the corpus had been
        extracted. On a 1 716-chunk corpus that is ~3 400 billed LLM calls with no
        durable result until the very last one returns: a kill, a crash or a provider
        outage at minute 35 threw away minute 0 to 35. Each sub-batch is now upserted
        as soon as it completes, so a relaunch finds it through `_is_cached_openie`
        and re-bills only the sub-batch that was in flight.

        Failures are still never cached (a provider outage must not become a
        permanent "this chunk has no facts"), but a failed sub-batch no longer
        discards the successful chunks of the same sub-batch: they are written
        first, and the error is raised after the whole corpus has been attempted.
        """
        assert self.openie is not None
        size = max(1, get_env_value("OPENIE_CHECKPOINT_SIZE", OPENIE_CHECKPOINT_SIZE, int))
        total = len(missing_chunks)
        done = 0
        failed_idx: list[str] = []
        for start in range(0, total, size):
            sub = missing_chunks[start : start + size]
            new_docs = await self.openie.batch_openie(sub)
            ok_docs = [d for d in new_docs if not d.get("failed")]
            failed_docs = [d for d in new_docs if d.get("failed")]
            if ok_docs:
                await self.openie_kv.upsert({d["idx"]: d for d in ok_docs})
            done += len(sub)
            failed_idx.extend(str(d["idx"]) for d in failed_docs)
            stage(
                logger,
                "OpenIE checkpoint",
                done=done,
                total=total,
                cached=len(ok_docs),
                failed=len(failed_docs),
            )
        if failed_idx:
            # One transient provider error among thousands of calls must not abort
            # a corpus-sized run after the fact: give the failed chunks a second,
            # sequential-sized pass before giving up. Only a chunk that fails twice
            # in a row stops the run — and even then everything else is cached.
            by_idx = {c["idx"]: c for c in missing_chunks}
            retry_docs = await self.openie.batch_openie([by_idx[i] for i in failed_idx])
            ok_docs = [d for d in retry_docs if not d.get("failed")]
            still_failed = [d for d in retry_docs if d.get("failed")]
            if ok_docs:
                await self.openie_kv.upsert({d["idx"]: d for d in ok_docs})
            stage(
                logger,
                "OpenIE retry",
                retried=len(retry_docs),
                recovered=len(ok_docs),
                failed=len(still_failed),
            )
            if still_failed:
                raise PipelineError(
                    f"OpenIE failed for {len(still_failed)}/{total} chunks after a retry "
                    f"({still_failed[0].get('error', '')}). Successful chunks are cached; "
                    f"the failed ones are left unindexed rather than stored as fact-free "
                    f"and silently unreachable."
                )

    async def ainsert(
        self,
        chunks: Sequence[str] | Sequence[dict[str, str]],
        *,
        run_conflicts: bool = True,
    ) -> dict[str, Any]:
        """Index already-chunked texts into the accumulating corpus memory."""
        phase(logger, INDEX_PHASE, chunks=len(chunks))
        stage(logger, "Indexing Documents", chunks=len(chunks))
        if not self._initialized:
            await self.initialize_storages()
        prepared = self._normalize_chunks(chunks)
        if not prepared:
            raise ValueError("ainsert requires at least one chunk")
        if self.llm_model_func is None or self.embedding_func is None:
            raise PipelineError("ainsert requires llm_model_func and embedding_func")

        assert self.openie is not None
        batch_ids = [c["idx"] for c in prepared]
        existing = await self.openie_kv.get_by_ids(batch_ids)
        missing_chunks = [
            prepared[i] for i, rec in enumerate(existing) if not _is_cached_openie(rec)
        ]
        stage(
            logger,
            "Performing OpenIE",
            chunks=len(prepared),
            cached=len(prepared) - len(missing_chunks),
            extract=len(missing_chunks),
        )
        if missing_chunks:
            await self._extract_openie_checkpointed(missing_chunks)
        stage(logger, "OpenIE cache ready", docs=len(batch_ids))

        openie_docs = await self._load_corpus_openie_docs(extra_chunk_ids=batch_ids)
        memory, conflicts, resolution = await self._rebuild_memory_from_openie(
            openie_docs, run_conflicts=run_conflicts
        )

        stage(
            logger,
            "Encoding Entities",
            passages=len(memory.passage_layer),
            facts=len(memory.fact_layer),
            schemas=len(memory.schema_layer),
            corpus_openie=len(openie_docs),
        )
        # Mark the retrieval state stale BEFORE touching the stores, and hold the
        # corpus lock across the whole rewrite. Readiness used to be cleared only
        # AFTER the rebuild, while _install_memory_graph starts with graph.clear():
        # a query landing in that window still saw ready_to_retrieve=True, skipped
        # prepare_retrieval, and answered HTTP 200 from a half-empty graph. Worse,
        # _embed_and_store_memory rewrote the schema-id maps before `self.memory` was
        # swapped, so a concurrent query resolved a NEW schema id against the OLD
        # memory object and seeded PPR on unrelated entities.
        self.ready_to_retrieve = False
        async with self._corpus_lock, self._storage_batch():
            await self._embed_and_store_memory(memory, openie_docs)
            stage(logger, "Constructing Graph")
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
        stage(
            logger,
            "Graph construction completed!",
            passages=stats["num_passages"],
            facts=stats["num_facts"],
            schemas=stats["num_schemas"],
        )
        done_step(
            logger,
            INDEX_PHASE,
            passages=stats["num_passages"],
            facts=stats["num_facts"],
            schemas=stats["num_schemas"],
        )
        return {"memory": memory, "stats": stats, "openie_docs": openie_docs}

    def insert(self, chunks: Sequence[str] | Sequence[dict[str, str]]) -> dict[str, Any]:
        """Sync wrapper around :meth:`ainsert`."""
        return _run_sync(self.ainsert(chunks))

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        emb = await self.embedding_func(texts)
        return np.asarray(emb)

    async def _embed_and_store_memory(
        self, memory: ThreeLayerMemory, openie_docs: list[dict[str, Any]]
    ) -> None:
        """Embed / store memory layers, deleting orphans and skipping known ids."""
        old_ids = await self._previous_vector_ids()

        # Passages — prefer existing chunk- ids; else hash content
        passage_texts: list[str] = []
        passage_ids: list[str] = []
        for p in memory.passage_layer:
            content = p.content
            cid = str(p.chunk_id or "")
            if not cid.startswith("chunk-"):
                cid = compute_mdhash_id(content, prefix="chunk-")
            p.chunk_id = cid
            passage_ids.append(cid)
            passage_texts.append(content)

        new_passage = set(passage_ids)
        orphan_passages = sorted(old_ids["passages"] - new_passage)
        add_passages = [i for i, pid in enumerate(passage_ids) if pid not in old_ids["passages"]]
        if orphan_passages:
            await self.chunks_vdb.delete(orphan_passages)
            await self.chunks_kv.delete(orphan_passages)
        if add_passages:
            texts = [passage_texts[i] for i in add_passages]
            emb = await self._embed_batch(texts)
            chunk_payload = {
                passage_ids[add_passages[j]]: {
                    "content": texts[j],
                    "embedding": emb[j].tolist(),
                }
                for j in range(len(add_passages))
            }
            await self.chunks_vdb.upsert(chunk_payload)
            await self.chunks_kv.upsert(
                {
                    passage_ids[add_passages[j]]: {
                        "content": texts[j],
                        "idx": add_passages[j],
                    }
                    for j in range(len(add_passages))
                }
            )

        # Facts — only those the ontology filter left active. Deactivated facts used
        # to stay indexed and fully retrievable, which is why ONTOLOGY_MIN_FREQUENCY
        # never changed an answer.
        active_facts = self._active_facts(memory)
        fact_ids = [compute_mdhash_id(_triple_str(f.content), prefix="fact-") for f in active_facts]
        new_facts = set(fact_ids)
        orphan_facts = sorted(old_ids["facts"] - new_facts)
        add_facts = [i for i, fid in enumerate(fact_ids) if fid not in old_ids["facts"]]
        if orphan_facts:
            await self.facts_vdb.delete(orphan_facts)
        if add_facts:
            texts = ["\t".join(active_facts[i].content) for i in add_facts]
            emb = await self._embed_batch(texts)
            fact_payload = {
                fact_ids[add_facts[j]]: {
                    "content": _triple_str(active_facts[add_facts[j]].content),
                    "embedding": emb[j].tolist(),
                    "triple": list(active_facts[add_facts[j]].content),
                }
                for j in range(len(add_facts))
            }
            await self.facts_vdb.upsert(fact_payload)

        # Entities from OpenIE + fact endpoints
        entities = _corpus_entity_surfaces(memory, openie_docs)
        entity_ids = [compute_mdhash_id(e, prefix="entity-") for e in entities]
        new_entities = set(entity_ids)
        orphan_entities = sorted(old_ids["entities"] - new_entities)
        add_entities = [i for i, eid in enumerate(entity_ids) if eid not in old_ids["entities"]]
        if orphan_entities:
            await self.entities_vdb.delete(orphan_entities)
        if add_entities:
            texts = [entities[i] for i in add_entities]
            emb = await self._embed_batch(texts)
            ent_payload = {
                entity_ids[add_entities[j]]: {
                    "content": texts[j],
                    "embedding": emb[j].tolist(),
                }
                for j in range(len(add_entities))
            }
            await self.entities_vdb.upsert(ent_payload)

        # Schemas
        schema_ids = [
            compute_mdhash_id(_triple_str(s.content), prefix="schema-") for s in memory.schema_layer
        ]
        new_schemas = set(schema_ids)
        orphan_schemas = sorted(old_ids["schemas"] - new_schemas)
        add_schemas = [i for i, sid in enumerate(schema_ids) if sid not in old_ids["schemas"]]
        if orphan_schemas:
            await self.schemas_vdb.delete(orphan_schemas)
        if add_schemas:
            texts = ["\t".join(memory.schema_layer[i].content) for i in add_schemas]
            emb = await self._embed_batch(texts)
            schema_payload = {
                schema_ids[add_schemas[j]]: {
                    "content": _triple_str(memory.schema_layer[add_schemas[j]].content),
                    "embedding": emb[j].tolist(),
                    "triple": list(memory.schema_layer[add_schemas[j]].content),
                    "schema_idx": memory.schema_layer[add_schemas[j]].idx,
                }
                for j in range(len(add_schemas))
            }
            await self.schemas_vdb.upsert(schema_payload)

        stage(
            logger,
            "Encoding Facts",
            add_passages=len(add_passages),
            del_passages=len(orphan_passages),
            add_facts=len(add_facts),
            del_facts=len(orphan_facts),
            add_entities=len(add_entities),
            del_entities=len(orphan_entities),
            add_schemas=len(add_schemas),
            del_schemas=len(orphan_schemas),
        )

        self._passage_ids = passage_ids
        self._fact_ids = fact_ids
        self._entity_ids = entity_ids
        self._schema_ids = schema_ids
        self._schema_id_to_idx = {
            schema_ids[i]: memory.schema_layer[i].idx for i in range(len(schema_ids))
        }
        self._passage_id_to_content = dict(zip(passage_ids, passage_texts))
        self._fact_id_to_triple = {
            fact_ids[i]: tuple(memory.fact_layer[i].content) for i in range(len(fact_ids))
        }

    async def _install_memory_graph(self, memory: ThreeLayerMemory) -> None:
        """Write schema / entity / fact / passage nodes and typed edges."""
        # One persistence step for the whole install instead of one per element.
        async with self.graph.batch():
            await self._install_memory_graph_inner(memory)

    async def _install_memory_graph_inner(self, memory: ThreeLayerMemory) -> None:
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

        # Ontology schema nodes + type-level TYPE_RELATION view
        for schema in memory.schema_layer:
            sid = compute_mdhash_id(_triple_str(schema.content), prefix="schema-")
            await self.graph.upsert_node(
                sid,
                {
                    "id": sid,
                    "label": "Schema",
                    "layer": "schema",
                    "content": _triple_str(schema.content),
                    "frequency": schema.frequency,
                },
            )
            h_type, rel, t_type = schema.content
            hid = compute_mdhash_id(canonical_key(h_type), prefix="type-")
            tid = compute_mdhash_id(canonical_key(t_type), prefix="type-")
            for type_id, type_name in ((hid, h_type), (tid, t_type)):
                if not await self.graph.has_node(type_id):
                    await self.graph.upsert_node(
                        type_id,
                        {
                            "id": type_id,
                            "label": "Type",
                            "layer": "type",
                            "content": str(type_name).strip(),
                        },
                    )
            await self.graph.upsert_edge(
                hid,
                tid,
                {
                    "type": "TYPE_RELATION",
                    "relation": str(rel),
                    "weight": float(schema.frequency or 1),
                },
            )

        # Accumulate before writing. Two reasons: the entity<->entity weight is the
        # number of facts joining the pair (so it is only known after the whole
        # sweep), and the entity->passage edge used to be re-upserted once per
        # (fact x endpoint x passage), rewriting the same edge many times over.
        entity_relations: dict[tuple[str, str], dict[str, Any]] = {}
        entity_passage_edges: set[tuple[str, str]] = set()
        entity_types: dict[str, set[str]] = {}
        type_members: dict[str, set[str]] = {}

        for fact in self._active_facts(memory):
            h, rel, t = fact.content
            fid = compute_mdhash_id(_triple_str(fact.content), prefix="fact-")
            await self.graph.upsert_node(
                fid,
                {
                    "id": fid,
                    "label": "Fact",
                    "layer": "fact",
                    "content": _triple_str(fact.content),
                    "schema_idx": fact.schema_idx,
                },
            )

            endpoint_ids: list[str] = []
            for ent in (h, t):
                eid = compute_mdhash_id(canonical_key(ent), prefix="entity-")
                endpoint_ids.append(eid)
                if not await self.graph.has_node(eid):
                    await self.graph.upsert_node(
                        eid,
                        {
                            "id": eid,
                            "label": "Entity",
                            "layer": "entity",
                            "content": canonical_key(ent),
                        },
                    )
                for pidx in fact.passage_indices:
                    passage = memory.get_passage_by_idx(pidx)
                    if passage is None:
                        continue
                    pid = passage.chunk_id
                    entity_passage_edges.add((eid, pid))
                    entity_to_passages.setdefault(eid, set()).add(pid)

            # G_fac: the instantiated fact edge between the two entities. This is the
            # paper's "primary reasoning substrate" and it was simply missing — the
            # graph was bipartite {entity, fact} <-> {chunk}, so a multi-hop chain
            # like Einstein -born_in-> Germany -capital-> Berlin had no path unless
            # the three entities happened to share a chunk, and PPR degenerated into
            # two-hop passage co-occurrence.
            head_id, tail_id = endpoint_ids[0], endpoint_ids[1]
            if head_id != tail_id:
                key = (head_id, tail_id) if head_id < tail_id else (tail_id, head_id)
                entry = entity_relations.setdefault(key, {"weight": 0.0, "relations": []})
                entry["weight"] += 1.0
                if len(entry["relations"]) < 8:
                    entry["relations"].append(str(rel))

            for pidx in fact.passage_indices:
                passage = memory.get_passage_by_idx(pidx)
                if passage is None:
                    continue
                await self.graph.upsert_edge(
                    fid,
                    passage.chunk_id,
                    {"type": "FACT_PASSAGE", "weight": 1.0},
                )

            if fact.schema_idx >= 0:
                schema = memory.get_schema_by_idx(fact.schema_idx)
                if schema is not None:
                    sid = compute_mdhash_id(_triple_str(schema.content), prefix="schema-")
                    await self.graph.upsert_edge(
                        fid,
                        sid,
                        {"type": "FACT_SCHEMA", "weight": 1.0},
                    )
                    # A fact's schema types its endpoints: (h, r, t) instantiates
                    # (h_type, r, t_type), so h is an h_type and t is a t_type.
                    h_type, _srel, t_type = schema.content
                    for ent_id, type_name in (
                        (head_id, h_type),
                        (tail_id, t_type),
                    ):
                        tid = compute_mdhash_id(canonical_key(type_name), prefix="type-")
                        entity_types.setdefault(ent_id, set()).add(tid)
                        type_members.setdefault(tid, set()).add(ent_id)

        for (src, tgt), entry in entity_relations.items():
            await self.graph.upsert_edge(
                src,
                tgt,
                {
                    "type": "ENTITY_RELATION",
                    "weight": float(entry["weight"]),
                    "relations": entry["relations"],
                },
            )

        for eid, pid in entity_passage_edges:
            await self.graph.upsert_edge(
                eid,
                pid,
                {"type": "PASSAGE_ENTITY", "weight": 1.0},
            )

        # Wire the ontology layer into the graph. Type nodes existed but the only
        # edges touching them joined two type nodes, leaving the whole layer as an
        # isolated component that could neither receive nor pass PPR mass. The type
        # is meant to be a transit hub between co-typed entities, weighted
        # 1/|entities of the type| so a generic type does not flood the walk.
        for tid, members in type_members.items():
            if not members:
                continue
            weight = 1.0 / float(len(members))
            for eid in members:
                await self.graph.upsert_edge(
                    eid,
                    tid,
                    {"type": "ENTITY_TO_TYPE", "weight": weight},
                )

        self._entity_to_passages = entity_to_passages

    # ------------------------------------------------------------------
    # Document admin (delete / clear)
    # ------------------------------------------------------------------

    async def _chunk_refcount(self, exclude_doc_ids: set[str] | None = None) -> dict[str, int]:
        """Count how many *live* docs reference each chunk id.

        Only PROCESSED documents count. Counting every status kept a FAILED or stale
        PENDING record propping up chunks it never successfully contributed, so
        deleting the document that really owned them left the chunks behind as
        permanent orphans — and, symmetrically, made a refcount of 1 look like 2 and
        skipped a legitimate cleanup.
        """
        exclude = exclude_doc_ids or set()
        counts: dict[str, int] = {}
        for doc_id, record in (await self._doc_status_all()).items():
            if doc_id in exclude:
                continue
            if str(record.get("status") or "") != DocStatus.PROCESSED.value:
                continue
            for cid in record.get("chunk_ids") or []:
                key = str(cid)
                if key:
                    counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _safe_unlink(path: str | Path | None) -> bool:
        if not path:
            return False
        text = str(path)
        if text.startswith("inline:"):
            return False
        try:
            p = Path(text)
            if p.is_file():
                p.unlink()
                return True
        except OSError as exc:
            logger.warning("Failed to delete file %s: %s", path, exc)
        return False

    async def _clear_empty_corpus(self) -> None:
        """Wipe derived memory/graph/vectors when no OpenIE docs remain."""
        old_ids = await self._previous_vector_ids()
        for ids, store in (
            (sorted(old_ids["passages"]), self.chunks_vdb),
            (sorted(old_ids["passages"]), self.chunks_kv),
            (sorted(old_ids["facts"]), self.facts_vdb),
            (sorted(old_ids["entities"]), self.entities_vdb),
            (sorted(old_ids["schemas"]), self.schemas_vdb),
        ):
            if ids:
                await store.delete(ids)
        await self.memory_kv.delete(["memory"])
        await self.graph.clear()
        self.memory = None
        self.ready_to_retrieve = False
        self._passage_ids = []
        self._fact_ids = []
        self._entity_ids = []
        self._schema_ids = []
        self._passage_id_to_content = {}
        self._fact_id_to_triple = {}
        self._schema_id_to_idx = {}
        self._entity_to_passages = {}

    async def _rebuild_corpus_after_delete(self, *, exclude_doc_ids: set[str]) -> dict[str, Any]:
        """Rebuild memory/graph from remaining docs' cached OpenIE (no LLM)."""
        openie_docs = await self._load_corpus_openie_docs(exclude_doc_ids=exclude_doc_ids)
        if not openie_docs:
            await self._clear_empty_corpus()
            return {
                "num_passages": 0,
                "num_facts": 0,
                "num_schemas": 0,
                "openie_docs": 0,
            }

        memory, _conflicts, _resolution = await self._rebuild_memory_from_openie(
            openie_docs, run_conflicts=False
        )
        await self._embed_and_store_memory(memory, openie_docs)
        await self._install_memory_graph(memory)
        await self.memory_kv.upsert({"memory": memory.to_dict()})
        self.memory = memory
        self.ready_to_retrieve = False
        return {
            "num_passages": len(memory.passage_layer),
            "num_facts": len(memory.fact_layer),
            "num_schemas": len(memory.schema_layer),
            "openie_docs": len(openie_docs),
        }

    async def adelete_by_doc_ids(
        self,
        doc_ids: Sequence[str],
        *,
        delete_file: bool = False,
    ) -> dict[str, Any]:
        """Delete documents and rebuild corpus memory from remaining OpenIE cache.

        Returns LightRAG-style per-id results plus a rebuild summary.
        """
        if not self._initialized:
            await self.initialize_storages()
        ids = [str(d) for d in doc_ids if d]
        main_step(logger, "admin.delete", docs=len(ids), delete_file=delete_file)

        results: dict[str, dict[str, Any]] = {}
        to_remove_status: list[str] = []
        chunks_to_drop: set[str] = set()
        processed_removed: set[str] = set()
        files_deleted: list[str] = []

        all_docs = await self._doc_status_all()
        refcounts = await self._chunk_refcount(exclude_doc_ids=set(ids))

        for doc_id in ids:
            record = all_docs.get(doc_id)
            if record is None:
                results[doc_id] = {
                    "status": "not_found",
                    "message": "Document not found",
                }
                continue

            status = str(record.get("status") or "")
            chunk_ids = [str(c) for c in (record.get("chunk_ids") or []) if c]
            dropped_chunks: list[str] = []
            for cid in chunk_ids:
                if refcounts.get(cid, 0) > 0:
                    continue
                chunks_to_drop.add(cid)
                dropped_chunks.append(cid)

            if delete_file:
                fp = record.get("file_path")
                if self._safe_unlink(fp):
                    files_deleted.append(str(fp))
                bp = record.get("blocks_path")
                if self._safe_unlink(bp):
                    files_deleted.append(str(bp))

            to_remove_status.append(doc_id)
            if status == DocStatus.PROCESSED.value:
                processed_removed.add(doc_id)
            results[doc_id] = {
                "status": "deleted",
                "message": f"Removed ({status or 'unknown'})",
                "previous_status": status,
                "chunks_dropped": dropped_chunks,
                "chunks_shared": [c for c in chunk_ids if c not in dropped_chunks],
            }

        if chunks_to_drop:
            drop_list = sorted(chunks_to_drop)
            await self.openie_kv.delete(drop_list)
            await self.chunks_kv.delete(drop_list)
            await self.chunks_vdb.delete(drop_list)
            sub_step(logger, "admin.delete.chunks", dropped=len(drop_list))

        if to_remove_status:
            await self.doc_status.delete(to_remove_status)

        rebuild: dict[str, Any] = {"skipped": True}
        if processed_removed or any(
            r.get("status") == "deleted"
            and r.get("previous_status")
            in {
                DocStatus.PROCESSED.value,
                DocStatus.PROCESSING.value,
                DocStatus.FAILED.value,
            }
            for r in results.values()
        ):
            # Rebuild whenever any doc with possible corpus impact was removed.
            rebuild = await self._rebuild_corpus_after_delete(exclude_doc_ids=set(to_remove_status))
            rebuild["skipped"] = False

        done_step(
            logger,
            "admin.delete",
            deleted=sum(1 for r in results.values() if r.get("status") == "deleted"),
            not_found=sum(1 for r in results.values() if r.get("status") == "not_found"),
            chunks_dropped=len(chunks_to_drop),
            files_deleted=len(files_deleted),
        )
        return {
            "status": "ok",
            "results": results,
            "chunks_dropped": sorted(chunks_to_drop),
            "files_deleted": files_deleted,
            "rebuild": rebuild,
        }

    async def aclear_all(
        self,
        *,
        delete_files: bool = False,
        input_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Drop all storages (and optionally input/parsed files) for this workspace."""
        if not self._initialized:
            await self.initialize_storages()
        main_step(logger, "admin.clear_all", delete_files=delete_files)

        files_deleted = 0
        if delete_files:
            all_docs = await self._doc_status_all()
            for record in all_docs.values():
                if self._safe_unlink(record.get("file_path")):
                    files_deleted += 1
                if self._safe_unlink(record.get("blocks_path")):
                    files_deleted += 1
            if input_dir:
                root = Path(input_dir)
                if root.is_dir():
                    for path in root.rglob("*"):
                        if path.is_file() and not path.name.startswith("."):
                            if self._safe_unlink(path):
                                files_deleted += 1

        dropped: list[str] = []
        for store in self._storages:
            name = getattr(store, "namespace", type(store).__name__)
            drop_fn = getattr(store, "drop", None)
            if callable(drop_fn):
                await drop_fn()
            elif hasattr(store, "clear") and callable(store.clear):
                await store.clear()
            else:
                # Best-effort: delete all known keys when get_all exists
                get_all = getattr(store, "get_all", None)
                if callable(get_all):
                    try:
                        data = await get_all()
                        if data:
                            await store.delete(list(data.keys()))
                    except Exception as exc:
                        logger.warning("clear_all fallback failed for %s: %s", name, exc)
            dropped.append(str(name))

        self.memory = None
        self.ready_to_retrieve = False
        self._passage_ids = []
        self._fact_ids = []
        self._entity_ids = []
        self._schema_ids = []
        self._passage_id_to_content = {}
        self._fact_id_to_triple = {}
        self._schema_id_to_idx = {}
        self._entity_to_passages = {}
        self._ppr = None

        done_step(
            logger,
            "admin.clear_all",
            storages=len(dropped),
            files_deleted=files_deleted,
        )
        return {
            "status": "ok",
            "dropped": dropped,
            "files_deleted": files_deleted,
        }

    def delete_by_doc_ids(
        self, doc_ids: Sequence[str], *, delete_file: bool = False
    ) -> dict[str, Any]:
        return _run_sync(self.adelete_by_doc_ids(doc_ids, delete_file=delete_file))

    def clear_all(
        self, *, delete_files: bool = False, input_dir: str | Path | None = None
    ) -> dict[str, Any]:
        return _run_sync(self.aclear_all(delete_files=delete_files, input_dir=input_dir))

    # ------------------------------------------------------------------
    # Retrieval prep
    # ------------------------------------------------------------------

    async def prepare_retrieval(self) -> None:
        """Load memory + graph adjacency needed for PPR retrieval."""
        phase(logger, RETRIEVE_PHASE, step="prepare")
        stage(logger, "Preparing for fast retrieval.")
        if not self._initialized:
            await self.initialize_storages()

        if self.memory is None:
            stored = await self.memory_kv.get_by_id("memory")
            if stored:
                self.memory = ThreeLayerMemory.from_dict(stored)
                stage(logger, "Loading keys.", loaded_memory=True)
            else:
                stage(logger, "Loading keys.", loaded_memory=False)
        else:
            # Refresh from KV when in-memory object is empty but KV may have caught up
            # (e.g. concurrent ingest finished after a stale prepare).
            if not self.memory.passage_layer:
                stored = await self.memory_kv.get_by_id("memory")
                if stored:
                    self.memory = ThreeLayerMemory.from_dict(stored)
                    stage(
                        logger,
                        "Loading keys.",
                        loaded_memory=True,
                        refreshed=True,
                    )

        self._passage_id_to_content = {}
        self._passage_ids = []
        self._passage_id_to_source: dict[str, str] = {}
        self._fact_ids = []
        self._schema_ids = []
        self._entity_ids = []
        self._schema_id_to_idx = {}
        self._fact_id_to_triple = {}
        self._entity_to_passages = {}
        self._passage_density = {}

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
                    eid = compute_mdhash_id(canonical_key(ent), prefix="entity-")
                    for pidx in f.passage_indices:
                        passage = self.memory.get_passage_by_idx(pidx)
                        if passage is None:
                            continue
                        pid = passage.chunk_id
                        self._entity_to_passages.setdefault(eid, set()).add(pid)
            for s in self.memory.schema_layer:
                sid = compute_mdhash_id(_triple_str(s.content), prefix="schema-")
                self._schema_ids.append(sid)
                self._schema_id_to_idx[sid] = s.idx

            # Rebuild `_entity_ids` from the same source the indexing path uses.
            # Leaving it empty made `_previous_vector_ids` report zero existing
            # entities, so the first insert after any restart re-embedded (and
            # re-billed) every entity in the corpus and never purged orphans.
            try:
                openie_docs = await self._load_corpus_openie_docs()
            except Exception as exc:
                fail_step(logger, "retrieve.prepare.openie_docs", exc=exc)
                openie_docs = []
            self._entity_ids = [
                compute_mdhash_id(e, prefix="entity-")
                for e in _corpus_entity_surfaces(self.memory, openie_docs)
            ]

            # Eq.19 density factor; corpus-wide, so it is computed once here rather
            # than per query.
            self._passage_density = _information_density(
                self._entity_to_passages, self._passage_ids
            )

        # Map chunk ids → document basenames for LightRAG-style references.
        try:
            for doc_id, record in (await self._doc_status_all()).items():
                fp = str((record or {}).get("file_path") or "")
                if fp.startswith("inline:"):
                    label = fp.split(":", 1)[-1] or str(doc_id)
                else:
                    label = Path(fp).name or fp or str(doc_id)
                for cid in (record or {}).get("chunk_ids") or []:
                    if cid:
                        self._passage_id_to_source[str(cid)] = label
        except Exception as exc:
            fail_step(logger, "retrieve.prepare.sources", exc=exc)

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
            stage(logger, "Loading embeddings.", edges=len(edges))
        except Exception as exc:
            fail_step(logger, "retrieve.prepare.edges", exc=exc)

        try:
            self._ppr = get_ppr_engine(
                self.ppr_engine_name,
                edges=edges,
                edge_weights=weights,
                passage_ids=self._passage_ids,
            )
            stage(logger, "PPR engine ready", engine=self.ppr_engine_name)
        except Exception as exc:
            fail_step(logger, "retrieve.prepare.ppr", exc=exc)
            self._ppr = IgraphPPREngine(
                edges=edges, edge_weights=weights, passage_ids=self._passage_ids
            )

        # Mark ready only when we have passage content to return; otherwise force
        # future queries to re-prepare (avoids sticky empty mid-ingest prepares).
        self.ready_to_retrieve = bool(self._passage_id_to_content)
        done_step(
            logger,
            "Preparing for fast retrieval",
            passages=len(self._passage_ids),
            facts=len(self._fact_ids),
            schemas=len(self._schema_ids),
            edges=len(edges),
            ready=self.ready_to_retrieve,
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
        phase(
            logger,
            RETRIEVE_PHASE,
            queries=len(query_list),
            mode=param.mode,
            top_k=param.top_k,
        )
        need_prepare = not self.ready_to_retrieve
        if self.ready_to_retrieve and not self._passage_id_to_content:
            # Sticky empty prepare — reload if memory_kv has caught up.
            need_prepare = True
            stage(
                logger,
                "Reprepare retrieval",
                reason="empty_content_map",
            )
        if need_prepare:
            # Same lock as ainsert: wait for an in-flight rewrite to finish rather
            # than reading a torn corpus.
            async with self._corpus_lock:
                # Drop stale empty in-memory handle so prepare reloads from KV.
                if self.memory is not None and not self.memory.passage_layer:
                    self.memory = None
                await self.prepare_retrieval()
        if self.embedding_func is None:
            raise PipelineError("aretrieve requires embedding_func")

        results: list[QuerySolution] = []
        for query in query_list:
            results.append(await self._retrieve_one(query, param))
        done_step(
            logger,
            RETRIEVE_PHASE,
            queries=len(query_list),
            results=len(results),
        )
        return results

    def retrieve(
        self,
        queries: str | Sequence[str],
        param: QueryParam | None = None,
    ) -> list[QuerySolution]:
        return _run_sync(self.aretrieve(queries, param=param))

    async def _retrieve_one(self, query: str, param: QueryParam) -> QuerySolution:
        with observation(
            "memgraphrag.retrieve",
            as_type="retriever",
            input={"query": query, "mode": param.mode},
            metadata={
                "top_k": param.top_k,
                "linking_top_k": param.linking_top_k,
                "damping": param.damping,
                "skip_fact_rerank": param.skip_fact_rerank,
            },
        ) as root_span:
            stage(
                logger,
                "Retrieving",
                mode=param.mode,
                query=truncate(query),
                top_k=param.top_k,
                linking_top_k=param.linking_top_k,
            )
            # Embed query for facts
            with observation(
                "memgraphrag.fact_linking",
                as_type="span",
                input={"query": query, "linking_top_k": param.linking_top_k},
            ) as fact_span:
                stage(logger, "Encoding queries for query_to_fact.")
                q_fact = await self.embedding_func(
                    [query],
                    context="query",
                    instruction=get_query_instruction("query_to_fact"),
                )
                q_fact_vec = np.asarray(q_fact[0], dtype=np.float64).tolist()

                fact_hits = await self.facts_vdb.query(q_fact_vec, top_k=param.linking_top_k)
                # Backends publish cosine similarity under "score" (see the
                # BaseVectorStorage contract). No guessing, no renormalisation: the
                # old "invert if it looks like a distance" heuristic tested the whole
                # list at once, so a single negative cosine flipped every ranking.
                sim_scores = [_hit_score(h) for h in fact_hits]

                if param.skip_fact_rerank:
                    kept = self.fact_filter.threshold_filter(
                        sim_scores, param.fact_similarity_threshold
                    )
                    stage(
                        logger,
                        "Fact filtering stats",
                        method="threshold",
                        hits=len(fact_hits),
                        kept=len(kept),
                    )
                else:
                    kept = await self.fact_filter.allm_filter(
                        query,
                        [h.get("triple") or h.get("content") for h in fact_hits],
                        list(range(len(fact_hits))),
                        scores=sim_scores,
                        threshold=param.fact_similarity_threshold,
                        llm_model_func=self.llm_model_func,
                    )
                    stage(
                        logger,
                        "Fact reranking stats",
                        method="llm",
                        hits=len(fact_hits),
                        kept=len(kept),
                    )

                kept_hits = [fact_hits[i] for i in kept if i < len(fact_hits)]
                update_observation(
                    fact_span,
                    output={
                        "fact_hits": len(fact_hits),
                        "kept_facts": len(kept_hits),
                        "threshold": param.fact_similarity_threshold,
                    },
                )

            # Schema linking (hierarchical retrieval) — runs even when facts empty
            seed_weights: dict[str, float] = {}
            schema_hits_n = 0
            if (
                self.memory is not None
                and self.memory.schema_layer
                and getattr(param, "schema_top_k", 0)
            ):
                with observation(
                    "memgraphrag.schema_linking",
                    as_type="span",
                    input={"schema_top_k": param.schema_top_k},
                ) as schema_span:
                    stage(
                        logger,
                        "Schema linking",
                        schema_top_k=param.schema_top_k,
                    )
                    try:
                        schema_hits = await self.schemas_vdb.query(
                            q_fact_vec, top_k=param.schema_top_k
                        )
                    except Exception as exc:
                        fail_step(logger, "retrieve.one.schema_linking", exc=exc)
                        schema_hits = []
                    schema_hits_n = len(schema_hits)
                    for hit in schema_hits:
                        sid = hit.get("id") or hit.get("__id__")
                        score = _hit_score(hit)
                        schema_idx = None
                        if sid and sid in self._schema_id_to_idx:
                            schema_idx = self._schema_id_to_idx[str(sid)]
                        elif hit.get("schema_idx") is not None:
                            schema_idx = int(hit["schema_idx"])
                        else:
                            triple = hit.get("triple")
                            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                                cand = compute_mdhash_id(_triple_str(triple), prefix="schema-")
                                schema_idx = self._schema_id_to_idx.get(cand)
                                sid = cand
                        if schema_idx is None or self.memory is None:
                            continue
                        schema = self.memory.get_schema_by_idx(schema_idx)
                        if schema is None:
                            continue
                        if sid:
                            seed_weights[str(sid)] = (
                                seed_weights.get(str(sid), 0.0) + score * param.schema_node_weight
                            )
                        for fidx in schema.fact_indices:
                            fact = self.memory.get_fact_by_idx(fidx)
                            if fact is None:
                                continue
                            for ent in (fact.content[0], fact.content[2]):
                                eid = compute_mdhash_id(canonical_key(ent), prefix="entity-")
                                seed_weights[eid] = (
                                    seed_weights.get(eid, 0.0) + score * param.schema_node_weight
                                )
                                for pid in self._entity_to_passages.get(eid, set()):
                                    seed_weights[pid] = (
                                        seed_weights.get(pid, 0.0)
                                        + score
                                        * param.schema_node_weight
                                        * param.passage_node_weight
                                    )
                    update_observation(
                        schema_span,
                        output={
                            "schema_hits": schema_hits_n,
                            "seed_nodes": len(seed_weights),
                        },
                    )
                    stage(
                        logger,
                        "Schema linking done",
                        schema_hits=schema_hits_n,
                        seed_nodes=len(seed_weights),
                    )

            if not kept_hits and not seed_weights:
                stage(logger, "Dense fallback", reason="no_facts")
                sol = await self._dense_passage_retrieve(query, param)
                update_observation(
                    root_span,
                    output={
                        "path": "dense_fallback_no_facts",
                        "n_docs": len(sol.docs),
                        "docs": truncate_docs(sol.docs),
                    },
                )
                done_step(
                    logger,
                    "Retrieving",
                    path="dense_fallback_no_facts",
                    docs=len(sol.docs),
                )
                return sol

            # Seed PPR from entities in filtered facts
            entity_score_acc: dict[str, list[float]] = {}
            for hit, score in zip(kept_hits, [sim_scores[i] for i in kept if i < len(sim_scores)]):
                triple = hit.get("triple")
                if not triple:
                    content = hit.get("content", "")
                    try:
                        # ast.literal_eval, not eval: `content` comes back from the vector
                        # store, so eval() there is arbitrary code execution in the API
                        # worker on the /query path.
                        triple = ast.literal_eval(content) if isinstance(content, str) else content
                    except Exception:
                        triple = None
                if not (isinstance(triple, (list, tuple)) and len(triple) == 3):
                    continue
                for ent in (triple[0], triple[2]):
                    eid = compute_mdhash_id(canonical_key(ent), prefix="entity-")
                    # Eq.17 is a MEAN over the retrieved facts that mention the
                    # entity, not a sum. Summing rewarded entities merely for
                    # appearing in many retrieved facts, amplifying hubs — the exact
                    # opposite of what the paper's hub suppression is for. Accumulate
                    # here, divide by the count below.
                    acc = entity_score_acc.setdefault(eid, [0.0, 0])
                    acc[0] += float(score)
                    acc[1] += 1

            for eid, (total, count) in entity_score_acc.items():
                mean_score = total / float(count or 1)
                seed_weights[eid] = seed_weights.get(eid, 0.0) + mean_score
                for pid in self._entity_to_passages.get(eid, set()):
                    seed_weights[pid] = (
                        seed_weights.get(pid, 0.0) + mean_score * param.passage_node_weight
                    )

            # Blend dense passage seeds
            with observation(
                "memgraphrag.passage_seed",
                as_type="span",
                input={"top_k": param.top_k},
            ) as seed_span:
                stage(logger, "Encoding queries for query_to_passage.")
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
                        content = hit.get("content", "")
                        for cand_id, cand_text in self._passage_id_to_content.items():
                            if cand_text == content:
                                pid = cand_id
                                break
                    if not pid:
                        continue
                    score = _hit_score(hit)
                    density = self._passage_density.get(str(pid), 0.5)
                    seed_weights[str(pid)] = (
                        seed_weights.get(str(pid), 0.0)
                        + score * param.passage_node_weight * density
                    )
                update_observation(
                    seed_span,
                    output={
                        "passage_hits": len(passage_hits),
                        "seed_nodes": len(seed_weights),
                    },
                )

            stage(
                logger,
                "Running PPR",
                seed_nodes=len(seed_weights),
                damping=param.damping,
            )
            passage_scores = await self._run_ppr(seed_weights, damping=param.damping)
            if not passage_scores:
                stage(logger, "Dense fallback", reason="empty_ppr")
                sol = await self._dense_passage_retrieve(query, param)
                update_observation(
                    root_span,
                    output={
                        "path": "dense_fallback_empty_ppr",
                        "n_docs": len(sol.docs),
                        "docs": truncate_docs(sol.docs),
                    },
                )
                done_step(
                    logger,
                    "Retrieving",
                    path="dense_fallback_empty_ppr",
                    docs=len(sol.docs),
                )
                return sol

            ranked = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
            top = ranked[: param.top_k]
            docs: list[str] = []
            doc_scores: list[float] = []
            passage_ids: list[str] = []
            sources: list[str] = []
            source_map = getattr(self, "_passage_id_to_source", {}) or {}
            missing_ids: list[str] = []
            for pid, score in top:
                content = self._passage_id_to_content.get(pid, "")
                if not content:
                    missing_ids.append(pid)
                    continue
                docs.append(content)
                doc_scores.append(float(score))
                passage_ids.append(pid)
                sources.append(str(source_map.get(pid) or "unknown"))
            # Fallback: resolve PPR hits via chunks_kv when content map is incomplete
            if missing_ids:
                try:
                    records = await self.chunks_kv.get_by_ids(missing_ids)
                except Exception:
                    records = []
                by_id = {
                    missing_ids[i]: records[i]
                    for i in range(min(len(missing_ids), len(records)))
                    if isinstance(records[i], dict)
                }
                for pid, score in top:
                    if self._passage_id_to_content.get(pid):
                        continue
                    rec = by_id.get(pid)
                    content = str((rec or {}).get("content") or "")
                    if not content:
                        continue
                    self._passage_id_to_content[pid] = content
                    docs.append(content)
                    doc_scores.append(float(score))
                    passage_ids.append(pid)
                    src = str(
                        source_map.get(pid)
                        or (rec or {}).get("file_path")
                        or (rec or {}).get("source")
                        or "unknown"
                    )
                    sources.append(src)
            sol = QuerySolution(
                question=query,
                docs=docs,
                doc_scores=doc_scores,
                sources=sources,
                passage_ids=passage_ids,
            )
            sol.ensure_references()
            update_observation(
                root_span,
                output={
                    "path": "ppr",
                    "n_docs": len(docs),
                    "top_scores": doc_scores[:5],
                    "docs": truncate_docs(docs),
                },
            )
            done_step(
                logger,
                "Retrieving",
                path="ppr",
                docs=len(docs),
                top_score=f"{doc_scores[0]:.4f}" if doc_scores else None,
            )
            return sol

    async def _dense_passage_retrieve(self, query: str, param: QueryParam) -> QuerySolution:
        with observation(
            "memgraphrag.dense_retrieve",
            as_type="retriever",
            input={"query": query, "top_k": param.top_k},
        ) as span:
            stage(logger, "Retrieving (naive dense)", top_k=param.top_k)
            q_pass = await self.embedding_func(
                [query],
                context="query",
                instruction=get_query_instruction("query_to_passage"),
            )
            q_pass_vec = np.asarray(q_pass[0], dtype=np.float64).tolist()
            hits = await self.chunks_vdb.query(q_pass_vec, top_k=param.top_k)
            docs: list[str] = []
            scores: list[float] = []
            passage_ids: list[str] = []
            sources: list[str] = []
            source_map = getattr(self, "_passage_id_to_source", {}) or {}
            for hit in hits:
                content = hit.get("content", "")
                if not content:
                    continue
                pid = str(hit.get("id") or hit.get("__id__") or "")
                docs.append(content)
                scores.append(float(hit.get("score", hit.get("distance", 0.0))))
                passage_ids.append(pid)
                sources.append(
                    str(
                        source_map.get(pid)
                        or hit.get("file_path")
                        or hit.get("source")
                        or "unknown"
                    )
                )
            sol = QuerySolution(
                question=query,
                docs=docs,
                doc_scores=scores,
                sources=sources,
                passage_ids=passage_ids,
            )
            sol.ensure_references()
            update_observation(
                span,
                output={"n_docs": len(docs), "docs": truncate_docs(docs)},
            )
            stage(logger, "Retrieving (naive dense) done", docs=len(docs))
            return sol

    async def _run_ppr(self, seed_weights: dict[str, float], damping: float) -> dict[str, float]:
        engine_name = type(self._ppr).__name__ if self._ppr is not None else "fallback"
        with observation(
            "memgraphrag.ppr",
            as_type="span",
            input={
                "seed_nodes": len(seed_weights),
                "damping": damping,
                "engine": engine_name,
            },
        ) as span:
            if self._ppr is None:
                fail_step(logger, "retrieve.ppr", reason="no_engine")
                out: dict[str, float] = {}
                for nid, w in seed_weights.items():
                    if nid.startswith(("chunk-", "passage-", "doc-")):
                        out[nid] = out.get(nid, 0.0) + float(w)
                    else:
                        for pid in self._entity_to_passages.get(nid, set()):
                            out[pid] = out.get(pid, 0.0) + float(w)
                update_observation(span, output={"scored_passages": len(out)})
                return out

            if asyncio.iscoroutinefunction(getattr(self._ppr, "run", None)):
                result = await self._ppr.run(seed_weights, damping=damping)  # type: ignore
            else:
                result = self._ppr.run(seed_weights, damping=damping)
            update_observation(span, output={"scored_passages": len(result) if result else 0})
            stage(
                logger,
                "PPR done",
                engine=engine_name,
                scored_passages=len(result) if result else 0,
            )
            return result

    async def aquery(
        self,
        query: str,
        param: QueryParam | None = None,
    ) -> Union[str, QuerySolution]:
        """Query with modes: ppr / naive / context / bypass."""
        param = param or QueryParam()
        mode = param.mode
        phase(
            logger,
            RETRIEVE_PHASE,
            mode=mode,
            query=truncate(query),
            only_need_context=bool(param.only_need_context),
        )

        with observation(
            "memgraphrag.query",
            as_type="span",
            input={"query": query, "mode": mode},
            metadata={"workspace": self.workspace or ""},
        ) as root_span:
            try:
                if mode == "bypass":
                    if not self.llm_model_func:
                        raise PipelineError("bypass mode requires llm_model_func")
                    with observation(
                        "memgraphrag.llm_bypass",
                        as_type="generation",
                        input={"query": query},
                        model=os.getenv("LLM_MODEL"),
                    ) as gen_span:
                        stage(logger, "mode=bypass")
                        answer = await self.llm_model_func(
                            query,
                            system_prompt=param.user_prompt,
                            model=param.model,
                            agent="qa.bypass",
                            llm_action="complete",
                        )
                        sol = QuerySolution(question=query, docs=[], answer=str(answer), sources=[])
                        sol.ensure_references()
                        update_observation(gen_span, output={"answer": str(answer)[:2000]})
                    update_observation(root_span, output={"mode": mode, "n_docs": 0})
                    done_step(
                        logger,
                        RETRIEVE_PHASE,
                        mode=mode,
                        answer_chars=len(sol.answer or ""),
                    )
                    return sol

                if mode == "naive":
                    stage(logger, "mode=naive")
                    if not self.ready_to_retrieve or not getattr(
                        self, "_passage_id_to_source", None
                    ):
                        await self.prepare_retrieval()
                    sol = await self._dense_passage_retrieve(query, param)
                else:
                    # ppr or context — both retrieve via PPR path
                    stage(
                        logger,
                        f"mode={mode}",
                        path="ppr" if mode != "context" else "context",
                    )
                    sols = await self.aretrieve(query, param=param)
                    sol = sols[0]

                if mode == "context" or param.only_need_context:
                    sol.ensure_references()
                    update_observation(
                        root_span,
                        output={
                            "mode": mode,
                            "n_docs": len(sol.docs),
                            "docs": truncate_docs(sol.docs),
                        },
                    )
                    done_step(
                        logger,
                        RETRIEVE_PHASE,
                        mode=mode,
                        docs=len(sol.docs),
                        qa=False,
                    )
                    return sol

                if not self.llm_model_func:
                    sol.ensure_references()
                    update_observation(
                        root_span,
                        output={"mode": mode, "n_docs": len(sol.docs), "llm": False},
                    )
                    done_step(
                        logger,
                        RETRIEVE_PHASE,
                        mode=mode,
                        docs=len(sol.docs),
                        qa=False,
                        reason="no_llm",
                    )
                    return sol

                system, user = render_rag_qa(query, sol.docs, sol.sources)
                if param.user_prompt:
                    user = f"{user}\n\n{param.user_prompt}"
                history = param.conversation_history or None
                with observation(
                    "memgraphrag.rag_qa",
                    as_type="generation",
                    input={
                        "query": query,
                        "n_docs": len(sol.docs),
                        "docs": truncate_docs(sol.docs),
                    },
                    model=os.getenv("LLM_MODEL"),
                    metadata={"system_prompt_chars": len(system or "")},
                ) as gen_span:
                    stage(
                        logger,
                        "QA Reading",
                        docs=len(sol.docs),
                        history_turns=len(history or []),
                        agent="qa.reading",
                    )
                    answer = await self.llm_model_func(
                        user,
                        system_prompt=system,
                        history_messages=history,
                        model=param.model,
                        agent="qa.reading",
                        llm_action="complete",
                    )
                    sol.answer = str(answer)
                    sol.ensure_references()
                    update_observation(gen_span, output={"answer": str(answer)[:2000]})
                update_observation(
                    root_span,
                    output={
                        "mode": mode,
                        "n_docs": len(sol.docs),
                        "answer_chars": len(sol.answer or ""),
                    },
                )
                done_step(
                    logger,
                    RETRIEVE_PHASE,
                    mode=mode,
                    docs=len(sol.docs),
                    answer_chars=len(sol.answer or ""),
                )
                return sol
            finally:
                flush_langfuse()

    def query(self, query: str, param: QueryParam | None = None) -> Union[str, QuerySolution]:
        return _run_sync(self.aquery(query, param=param))

    async def astream_qa(
        self,
        query: str,
        param: QueryParam | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Retrieve, then yield the answer as the model produces it.

        Frames, in order: one ``{"references": [...]}``, then one ``{"token": "..."}``
        per chunk, then a final ``{"done": True, "answer": "..."}``.

        This is a parallel route rather than a rewrite of ``aquery``: the buffered
        path is the one every existing caller, test and script depends on, and a
        regression there would be far more expensive than the duplication here.

        Retrieval itself is not streamed — PPR has no partial result to emit — so the
        first token still costs a full retrieval. What this removes is the second
        wait, which on a long answer is the one the user actually feels.
        """
        param = param or QueryParam()
        mode = param.mode
        if not self.llm_model_func:
            raise PipelineError("streaming requires llm_model_func")

        phase(logger, RETRIEVE_PHASE, mode=mode, query=truncate(query), stream=True)
        try:
            if mode == "bypass":
                stage(logger, "mode=bypass (stream)")
                sol = QuerySolution(question=query, docs=[], answer=None, sources=[])
                system: str | None = param.user_prompt
                user = query
                agent = "qa.bypass"
            else:
                if mode == "naive":
                    stage(logger, "mode=naive (stream)")
                    if not self.ready_to_retrieve or not getattr(
                        self, "_passage_id_to_source", None
                    ):
                        await self.prepare_retrieval()
                    sol = await self._dense_passage_retrieve(query, param)
                else:
                    stage(logger, f"mode={mode} (stream)")
                    sols = await self.aretrieve(query, param=param)
                    sol = sols[0]
                sol.ensure_references()
                system, user = render_rag_qa(query, sol.docs, sol.sources)
                if param.user_prompt:
                    user = f"{user}\n\n{param.user_prompt}"
                agent = "qa.reading"

            # References go out before the first token so the UI can render its
            # sources panel while the answer is still being written.
            yield {"references": list(getattr(sol, "references", None) or [])}

            stage(logger, "QA Reading (stream)", docs=len(sol.docs), agent=agent)
            produced = await self.llm_model_func(
                user,
                system_prompt=system,
                history_messages=param.conversation_history or None,
                model=param.model,
                stream=True,
                agent=agent,
                llm_action="stream",
            )

            parts: list[str] = []
            if hasattr(produced, "__aiter__"):
                async for chunk in produced:
                    text = str(chunk)
                    if not text:
                        continue
                    parts.append(text)
                    yield {"token": text}
            else:
                # An llm_model_func that ignores `stream` (a test double, or a binding
                # without streaming support) still works: it just arrives in one frame.
                text = str(produced)
                if text:
                    parts.append(text)
                    yield {"token": text}

            sol.answer = "".join(parts)
            done_step(
                logger,
                RETRIEVE_PHASE,
                mode=mode,
                docs=len(sol.docs),
                answer_chars=len(sol.answer or ""),
                stream=True,
            )
            yield {"done": True, "answer": sol.answer}
        finally:
            flush_langfuse()

    # Research-engine aliases (MemGraphRAG/code naming)
    async def aindex_with_memory(
        self, chunks: Sequence[str] | Sequence[dict[str, str]]
    ) -> dict[str, Any]:
        return await self.ainsert(chunks)

    def index_with_memory(self, chunks: Sequence[str] | Sequence[dict[str, str]]) -> dict[str, Any]:
        return _run_sync(self.aindex_with_memory(chunks))

    async def arag_qa(
        self, query: str, param: QueryParam | None = None
    ) -> Union[str, QuerySolution]:
        return await self.aquery(query, param=param)

    def rag_qa(self, query: str, param: QueryParam | None = None) -> Union[str, QuerySolution]:
        return _run_sync(self.arag_qa(query, param=param))
