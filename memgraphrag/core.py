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
    CONFLICT_ENABLED,
    CONFLICT_MAX_GROUPS,
    DAMPING,
    EMBEDDING_DIM,
    FACT_SIMILARITY_THRESHOLD,
    LINKING_TOP_K,
    MAX_ASYNC_LLM,
    ONTOLOGY_BATCH_SIZE,
    ONTOLOGY_MIN_FREQUENCY,
    PASSAGE_NODE_WEIGHT,
    PPR_ENGINE,
    SCHEMA_NODE_WEIGHT,
    SCHEMA_TOP_K,
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
from memgraphrag.observability.langfuse_trace import (
    flush_langfuse,
    observation,
    truncate_docs,
    update_observation,
)
from memgraphrag.utils.env import get_env_value
from memgraphrag.utils.hashing import compute_mdhash_id
from memgraphrag.utils.json_llm import extract_json_object
from memgraphrag.utils.misc import QuerySolution
from memgraphrag.utils.step_log import done_step, fail_step, main_step, sub_step, truncate

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
        self.schemas_vdb: BaseVectorStorage = vec_cls(
            namespace=NameSpace.VECTOR_SCHEMAS, **common
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
        h, r, t = MemGraphRAG._normalize_triple_key(triple)
        return (h.lower(), r.lower(), t.lower())

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

    async def _apply_cached_ontology(
        self, memory: ThreeLayerMemory
    ) -> int:
        """Apply ``extracted_triple_ontology`` from openie_kv when present."""
        linked = 0
        try:
            cached = await self.openie_kv.get_all()
        except Exception:
            cached = {}
        if not cached:
            return 0

        fact_by_key = {
            self._triple_lookup_key(f.content): f.idx for f in memory.fact_layer
        }
        for _doc_id, doc in cached.items():
            if not isinstance(doc, dict):
                continue
            ont_map = doc.get("extracted_triple_ontology") or {}
            if not isinstance(ont_map, dict) or not ont_map:
                continue
            for triple_key, ontology in ont_map.items():
                try:
                    import ast

                    triple_tuple = ast.literal_eval(str(triple_key))
                    if not (
                        isinstance(triple_tuple, (list, tuple))
                        and len(triple_tuple) == 3
                    ):
                        continue
                except Exception:
                    continue
                if not (isinstance(ontology, (list, tuple)) and len(ontology) == 3):
                    continue
                fact_idx = fact_by_key.get(self._triple_lookup_key(triple_tuple))
                if fact_idx is None:
                    continue
                memory.link_fact_to_schema(
                    fact_idx, self._normalize_triple_key(ontology)
                )
                linked += 1
        return linked

    async def _persist_ontology_to_openie(
        self, memory: ThreeLayerMemory
    ) -> None:
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
                by_chunk.setdefault(str(passage.chunk_id), {})[key] = list(
                    schema.content
                )
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
            sub_step(
                logger,
                "index.schema.skip",
                reason="empty_facts",
                facts=0,
            )
            return memory

        cached_links = await self._apply_cached_ontology(memory)
        unlinked = [f for f in memory.fact_layer if f.schema_idx < 0]
        if cached_links and not unlinked:
            memory.recompute_schema_frequencies()
            done_step(
                logger,
                "index.schema.extract",
                source="cache",
                linked=cached_links,
                schemas=len(memory.schema_layer),
            )
            return memory

        if not self.llm_model_func:
            sub_step(
                logger,
                "index.schema.skip",
                reason="no_llm",
                facts=len(memory.fact_layer),
                cached_links=cached_links,
            )
            return memory

        batch_size = max(
            1, get_env_value("ONTOLOGY_BATCH_SIZE", ONTOLOGY_BATCH_SIZE, int)
        )
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

        sub_step(
            logger,
            "index.schema.extract",
            batches=len(batches),
            unlinked=len(unlinked),
            cached_links=cached_links,
            batch_size=batch_size,
        )

        sem = asyncio.Semaphore(max(1, self.max_async_llm))
        fact_by_key = {
            self._triple_lookup_key(f.content): f.idx for f in memory.fact_layer
        }
        linked = 0
        failed_batches = 0

        async def _run_batch(passage: str, fact_idxs: list[int]) -> int:
            nonlocal failed_batches
            triples = [list(memory.fact_layer[i].content) for i in fact_idxs]
            user = ONTOLOGY_EXTRACTION_USER_TEMPLATE.substitute(
                passage=passage, triples=str(triples)
            )
            try:
                async with sem:
                    raw = await self.llm_model_func(
                        user, system_prompt=ONTOLOGY_EXTRACTION_SYSTEM
                    )
            except Exception as exc:
                fail_step(logger, "index.schema.extract_batch", exc=exc)
                failed_batches += 1
                return 0
            pairs = self._parse_ontology_triples(str(raw))
            if not pairs:
                failed_batches += 1
                fail_step(
                    logger,
                    "index.schema.extract_batch",
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

        results = await asyncio.gather(
            *[_run_batch(p, idxs) for p, idxs in batches]
        )
        linked = sum(results)
        memory.recompute_schema_frequencies()
        try:
            await self._persist_ontology_to_openie(memory)
        except Exception as exc:
            fail_step(logger, "index.schema.cache_write", exc=exc)

        done_step(
            logger,
            "index.schema.extract",
            linked=linked,
            schemas=len(memory.schema_layer),
            failed_batches=failed_batches,
            facts_untyped=sum(1 for f in memory.fact_layer if f.schema_idx < 0),
        )
        return memory

    async def filter_ontology(self, memory: ThreeLayerMemory) -> ThreeLayerMemory:
        """Frequency-based ontology filter with schema reindexing."""
        min_freq = get_env_value(
            "ONTOLOGY_MIN_FREQUENCY", ONTOLOGY_MIN_FREQUENCY, int
        )
        before = len(memory.schema_layer)
        stats = memory.filter_schemas_by_frequency(min_freq)
        sub_step(
            logger,
            "index.ontology.filter",
            before=before,
            kept=stats["kept"],
            dropped=stats["dropped"],
            min_frequency=min_freq,
            noop=min_freq <= 0,
        )
        return memory

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
        groups: list[list[int]] = []
        for bucket in list(hr.values()) + list(rt.values()):
            uniq = sorted(set(bucket))
            if len(uniq) < 2:
                continue
            key = frozenset(uniq)
            if key in seen:
                continue
            seen.add(key)
            groups.append(uniq)
            if len(groups) >= max_groups:
                break
        return groups

    async def detect_conflicts(self, memory: ThreeLayerMemory) -> dict[str, Any]:
        """Detect hard conflicts among fact groups via LLM."""
        result: dict[str, Any] = {
            "has_conflict": False,
            "conflicts": [],
            "summary": {"hard_conflicts": 0, "groups_checked": 0},
        }
        enabled = get_env_value("CONFLICT_ENABLED", CONFLICT_ENABLED, bool)
        if not enabled:
            sub_step(logger, "index.conflict.detect_skip", reason="disabled")
            return result
        if not self.llm_model_func or len(memory.fact_layer) < 2:
            sub_step(
                logger,
                "index.conflict.detect_skip",
                facts=len(memory.fact_layer),
            )
            return result

        max_groups = max(
            1, get_env_value("CONFLICT_MAX_GROUPS", CONFLICT_MAX_GROUPS, int)
        )
        groups = self._conflict_candidate_groups(memory, max_groups)
        sub_step(
            logger,
            "index.conflict.detect",
            groups=len(groups),
            max_groups=max_groups,
        )
        if not groups:
            return result

        sem = asyncio.Semaphore(max(1, self.max_async_llm))
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
                        user, system_prompt=CONFLICT_DETECTION_SYSTEM
                    )
            except Exception as exc:
                fail_step(logger, "index.conflict.detect_group", exc=exc)
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
        done_step(
            logger,
            "index.conflict.detect",
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
            sub_step(logger, "index.conflict.resolve_skip", hard_conflicts=0)
            return memory, resolution
        if not self.llm_model_func:
            return memory, resolution

        # Build evidence bundles keyed by conflict pairs / groups
        bundles: list[str] = []
        for item in hard:
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
                    sources.append(
                        f"fact={list(fact.content)} fact_idx={fidx} passage={preview}"
                    )
            bundles.append(
                f"conflict_type={item.get('conflict_type')}\n"
                f"triple1={t1}\ntriple2={t2}\n"
                f"reason={item.get('conflict_reason')}\n"
                f"sources:\n" + "\n---\n".join(sources)
            )

        user = CONFLICT_RESOLUTION_USER_TEMPLATE.substitute(
            conflicting_triples_with_sources="\n\n====\n\n".join(bundles)
        )
        sub_step(
            logger,
            "index.conflict.resolve",
            conflicts=len(hard),
        )
        try:
            raw = await self.llm_model_func(
                user, system_prompt=CONFLICT_RESOLUTION_SYSTEM
            )
        except Exception as exc:
            fail_step(logger, "index.conflict.resolve", exc=exc)
            return memory, resolution

        data = extract_json_object(str(raw))
        resolved_items = data.get("resolved_triples") or []
        if not isinstance(resolved_items, list):
            fail_step(
                logger,
                "index.conflict.resolve",
                reason="unparsed",
                response_chars=len(str(raw)),
            )
            return memory, resolution

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
        done_step(
            logger,
            "index.conflict.resolve",
            **resolution["summary"],
            facts=len(memory.fact_layer),
        )
        return memory, resolution

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def ainsert(self, chunks: Sequence[str] | Sequence[dict[str, str]]) -> dict[str, Any]:
        """Index already-chunked texts: OpenIE → memory → vectors → graph."""
        main_step(logger, "index.ainsert", chunks=len(chunks))
        if not self._initialized:
            await self.initialize_storages()
        if not chunks:
            raise ValueError("ainsert requires at least one chunk")
        if self.llm_model_func is None or self.embedding_func is None:
            raise PipelineError("ainsert requires llm_model_func and embedding_func")

        assert self.openie is not None
        sub_step(logger, "index.ainsert.openie", chunks=len(chunks))
        openie_docs = await self.openie.batch_openie(chunks)
        await self.openie_kv.upsert(
            {d["idx"]: d for d in openie_docs}
        )
        sub_step(logger, "index.ainsert.openie_done", docs=len(openie_docs))

        memory = ThreeLayerMemory()
        sub_step(logger, "index.ainsert.memory_build", openie_docs=len(openie_docs))
        memory.build_from_raw_openie_results({"docs": openie_docs})
        memory = await self.extract_schema(memory)
        memory = await self.filter_ontology(memory)
        conflicts = await self.detect_conflicts(memory)
        memory, resolution = await self.resolve_conflicts(memory, conflicts)

        sub_step(
            logger,
            "index.ainsert.embed_store",
            passages=len(memory.passage_layer),
            facts=len(memory.fact_layer),
        )
        await self._embed_and_store_memory(memory, openie_docs)
        sub_step(logger, "index.ainsert.graph_install")
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
        done_step(
            logger,
            "index.ainsert",
            passages=stats["num_passages"],
            facts=stats["num_facts"],
            schemas=stats["num_schemas"],
        )
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

        # Schemas (ontology type triples)
        schema_texts = ["\t".join(s.content) for s in memory.schema_layer]
        schema_ids = [
            compute_mdhash_id(_triple_str(s.content), prefix="schema-")
            for s in memory.schema_layer
        ]
        if schema_texts:
            emb = await self.embedding_func(schema_texts)
            emb = np.asarray(emb)
            schema_payload = {
                schema_ids[i]: {
                    "content": _triple_str(memory.schema_layer[i].content),
                    "embedding": emb[i].tolist(),
                    "triple": list(memory.schema_layer[i].content),
                    "schema_idx": memory.schema_layer[i].idx,
                }
                for i in range(len(schema_ids))
            }
            await self.schemas_vdb.upsert(schema_payload)

        self._passage_ids = passage_ids
        self._fact_ids = fact_ids
        self._entity_ids = entity_ids
        self._schema_ids = schema_ids
        self._schema_id_to_idx = {
            schema_ids[i]: memory.schema_layer[i].idx for i in range(len(schema_ids))
        }
        self._passage_id_to_content = dict(zip(passage_ids, passage_texts))
        self._fact_id_to_triple = {
            fact_ids[i]: tuple(memory.fact_layer[i].content)
            for i in range(len(fact_ids))
        }

    async def _install_memory_graph(self, memory: ThreeLayerMemory) -> None:
        """Write schema / entity / fact / passage nodes and typed edges."""
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
            hid = compute_mdhash_id(str(h_type).strip().lower(), prefix="type-")
            tid = compute_mdhash_id(str(t_type).strip().lower(), prefix="type-")
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

        for fact in memory.fact_layer:
            h, _r, t = fact.content
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

        self._entity_to_passages = entity_to_passages

    # ------------------------------------------------------------------
    # Retrieval prep
    # ------------------------------------------------------------------

    async def prepare_retrieval(self) -> None:
        """Load memory + graph adjacency needed for PPR retrieval."""
        main_step(logger, "retrieve.prepare")
        if not self._initialized:
            await self.initialize_storages()

        if self.memory is None:
            stored = await self.memory_kv.get_by_id("memory")
            if stored:
                self.memory = ThreeLayerMemory.from_dict(stored)
                sub_step(logger, "retrieve.prepare.load_memory", loaded=True)
            else:
                sub_step(logger, "retrieve.prepare.load_memory", loaded=False)

        self._passage_id_to_content = {}
        self._passage_ids = []
        self._fact_ids = []
        self._schema_ids = []
        self._schema_id_to_idx = {}
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
            for s in self.memory.schema_layer:
                sid = compute_mdhash_id(_triple_str(s.content), prefix="schema-")
                self._schema_ids.append(sid)
                self._schema_id_to_idx[sid] = s.idx

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
            sub_step(logger, "retrieve.prepare.edges", edges=len(edges))
        except Exception as exc:
            fail_step(logger, "retrieve.prepare.edges", exc=exc)

        try:
            self._ppr = get_ppr_engine(
                self.ppr_engine_name,
                edges=edges,
                edge_weights=weights,
                passage_ids=self._passage_ids,
            )
            sub_step(
                logger,
                "retrieve.prepare.ppr",
                engine=self.ppr_engine_name,
            )
        except Exception as exc:
            fail_step(logger, "retrieve.prepare.ppr", exc=exc)
            self._ppr = IgraphPPREngine(
                edges=edges, edge_weights=weights, passage_ids=self._passage_ids
            )

        self.ready_to_retrieve = True
        done_step(
            logger,
            "retrieve.prepare",
            passages=len(self._passage_ids),
            facts=len(self._fact_ids),
            schemas=len(self._schema_ids),
            edges=len(edges),
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
        main_step(
            logger,
            "retrieve.aretrieve",
            queries=len(query_list),
            mode=param.mode,
            top_k=param.top_k,
        )
        if not self.ready_to_retrieve:
            await self.prepare_retrieval()
        if self.embedding_func is None:
            raise PipelineError("aretrieve requires embedding_func")

        results: list[QuerySolution] = []
        for query in query_list:
            results.append(await self._retrieve_one(query, param))
        done_step(
            logger,
            "retrieve.aretrieve",
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
            main_step(
                logger,
                "retrieve.one",
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
                sub_step(
                    logger,
                    "retrieve.one.fact_linking",
                    linking_top_k=param.linking_top_k,
                )
                q_fact = await self.embedding_func(
                    [query],
                    context="query",
                    instruction=get_query_instruction("query_to_fact"),
                )
                q_fact_vec = np.asarray(q_fact[0], dtype=np.float64).tolist()

                fact_hits = await self.facts_vdb.query(
                    q_fact_vec, top_k=param.linking_top_k
                )
                scores = [
                    float(h.get("score", h.get("distance", 0.0))) for h in fact_hits
                ]
                # nano-vectordb often returns similarity; if distance-like, invert
                if scores and max(scores) <= 1.0 and min(scores) >= 0:
                    sim_scores = scores
                else:
                    sim_scores = [1.0 / (1.0 + abs(s)) for s in scores]

                if param.skip_fact_rerank:
                    kept = self.fact_filter.threshold_filter(
                        sim_scores, param.fact_similarity_threshold
                    )
                    sub_step(
                        logger,
                        "retrieve.one.fact_rerank",
                        method="threshold",
                        hits=len(fact_hits),
                        kept=len(kept),
                    )
                else:
                    kept = self.fact_filter.llm_filter(
                        query,
                        [h.get("content") for h in fact_hits],
                        list(range(len(fact_hits))),
                        scores=sim_scores,
                        threshold=param.fact_similarity_threshold,
                    )
                    sub_step(
                        logger,
                        "retrieve.one.fact_rerank",
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
                    sub_step(
                        logger,
                        "retrieve.one.schema_linking",
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
                        score = float(hit.get("score", hit.get("distance", 0.0)))
                        if score > 1.0 or score < 0:
                            score = 1.0 / (1.0 + abs(score))
                        schema_idx = None
                        if sid and sid in self._schema_id_to_idx:
                            schema_idx = self._schema_id_to_idx[str(sid)]
                        elif hit.get("schema_idx") is not None:
                            schema_idx = int(hit["schema_idx"])
                        else:
                            triple = hit.get("triple")
                            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                                cand = compute_mdhash_id(
                                    _triple_str(triple), prefix="schema-"
                                )
                                schema_idx = self._schema_id_to_idx.get(cand)
                                sid = cand
                        if schema_idx is None or self.memory is None:
                            continue
                        schema = self.memory.get_schema_by_idx(schema_idx)
                        if schema is None:
                            continue
                        if sid:
                            seed_weights[str(sid)] = (
                                seed_weights.get(str(sid), 0.0)
                                + score * param.schema_node_weight
                            )
                        for fidx in schema.fact_indices:
                            fact = self.memory.get_fact_by_idx(fidx)
                            if fact is None:
                                continue
                            for ent in (fact.content[0], fact.content[2]):
                                eid = compute_mdhash_id(
                                    str(ent).strip().lower(), prefix="entity-"
                                )
                                seed_weights[eid] = (
                                    seed_weights.get(eid, 0.0)
                                    + score * param.schema_node_weight
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
                    sub_step(
                        logger,
                        "retrieve.one.schema_linking_done",
                        schema_hits=schema_hits_n,
                        seed_nodes=len(seed_weights),
                    )

            if not kept_hits and not seed_weights:
                sub_step(logger, "retrieve.one.dense_fallback", reason="no_facts")
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
                    "retrieve.one",
                    path="dense_fallback_no_facts",
                    docs=len(sol.docs),
                )
                return sol

            # Seed PPR from entities in filtered facts
            for hit, score in zip(
                kept_hits, [sim_scores[i] for i in kept if i < len(sim_scores)]
            ):
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
                    for pid in self._entity_to_passages.get(eid, set()):
                        seed_weights[pid] = (
                            seed_weights.get(pid, 0.0)
                            + float(score) * param.passage_node_weight
                        )

            # Blend dense passage seeds
            with observation(
                "memgraphrag.passage_seed",
                as_type="span",
                input={"top_k": param.top_k},
            ) as seed_span:
                sub_step(logger, "retrieve.one.passage_seed", top_k=param.top_k)
                q_pass = await self.embedding_func(
                    [query],
                    context="query",
                    instruction=get_query_instruction("query_to_passage"),
                )
                q_pass_vec = np.asarray(q_pass[0], dtype=np.float64).tolist()
                passage_hits = await self.chunks_vdb.query(
                    q_pass_vec, top_k=param.top_k
                )
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
                    score = float(hit.get("score", hit.get("distance", 0.0)))
                    if score > 1.0 or score < 0:
                        score = 1.0 / (1.0 + abs(score))
                    seed_weights[str(pid)] = (
                        seed_weights.get(str(pid), 0.0)
                        + score * param.passage_node_weight
                    )
                update_observation(
                    seed_span,
                    output={
                        "passage_hits": len(passage_hits),
                        "seed_nodes": len(seed_weights),
                    },
                )
                sub_step(
                    logger,
                    "retrieve.one.passage_seed_done",
                    passage_hits=len(passage_hits),
                    seed_nodes=len(seed_weights),
                )

            sub_step(
                logger,
                "retrieve.one.ppr",
                seed_nodes=len(seed_weights),
                damping=param.damping,
            )
            passage_scores = await self._run_ppr(seed_weights, damping=param.damping)
            if not passage_scores:
                sub_step(logger, "retrieve.one.dense_fallback", reason="empty_ppr")
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
                    "retrieve.one",
                    path="dense_fallback_empty_ppr",
                    docs=len(sol.docs),
                )
                return sol

            ranked = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
            top = ranked[: param.top_k]
            docs = [
                self._passage_id_to_content.get(pid, "")
                for pid, _ in top
                if self._passage_id_to_content.get(pid)
            ]
            doc_scores = [float(s) for _, s in top[: len(docs)]]
            sol = QuerySolution(question=query, docs=docs, doc_scores=doc_scores)
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
                "retrieve.one",
                path="ppr",
                docs=len(docs),
                top_score=f"{doc_scores[0]:.4f}" if doc_scores else None,
            )
            return sol

    async def _dense_passage_retrieve(
        self, query: str, param: QueryParam
    ) -> QuerySolution:
        with observation(
            "memgraphrag.dense_retrieve",
            as_type="retriever",
            input={"query": query, "top_k": param.top_k},
        ) as span:
            sub_step(logger, "retrieve.dense", top_k=param.top_k)
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
            sol = QuerySolution(question=query, docs=docs, doc_scores=scores)
            update_observation(
                span,
                output={"n_docs": len(docs), "docs": truncate_docs(docs)},
            )
            sub_step(logger, "retrieve.dense_done", docs=len(docs))
            return sol

    async def _run_ppr(
        self, seed_weights: dict[str, float], damping: float
    ) -> dict[str, float]:
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
            update_observation(
                span, output={"scored_passages": len(result) if result else 0}
            )
            sub_step(
                logger,
                "retrieve.ppr_done",
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
        main_step(
            logger,
            "query.aquery",
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
                        sub_step(logger, "query.aquery.bypass")
                        answer = await self.llm_model_func(
                            query, system_prompt=param.user_prompt
                        )
                        sol = QuerySolution(
                            question=query, docs=[], answer=str(answer)
                        )
                        update_observation(
                            gen_span, output={"answer": str(answer)[:2000]}
                        )
                    update_observation(
                        root_span, output={"mode": mode, "n_docs": 0}
                    )
                    done_step(
                        logger,
                        "query.aquery",
                        mode=mode,
                        answer_chars=len(sol.answer or ""),
                    )
                    return sol

                if mode == "naive":
                    sub_step(logger, "query.aquery.mode_select", path="naive")
                    sol = await self._dense_passage_retrieve(query, param)
                else:
                    # ppr or context — both retrieve via PPR path
                    sub_step(
                        logger,
                        "query.aquery.mode_select",
                        path="ppr" if mode != "context" else "context",
                    )
                    sols = await self.aretrieve(query, param=param)
                    sol = sols[0]

                if mode == "context" or param.only_need_context:
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
                        "query.aquery",
                        mode=mode,
                        docs=len(sol.docs),
                        qa=False,
                    )
                    return sol

                if not self.llm_model_func:
                    update_observation(
                        root_span,
                        output={"mode": mode, "n_docs": len(sol.docs), "llm": False},
                    )
                    done_step(
                        logger,
                        "query.aquery",
                        mode=mode,
                        docs=len(sol.docs),
                        qa=False,
                        reason="no_llm",
                    )
                    return sol

                system, user = render_rag_qa(query, sol.docs)
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
                    sub_step(
                        logger,
                        "query.aquery.rag_qa",
                        docs=len(sol.docs),
                        history_turns=len(history or []),
                    )
                    answer = await self.llm_model_func(
                        user, system_prompt=system, history_messages=history
                    )
                    sol.answer = str(answer)
                    update_observation(
                        gen_span, output={"answer": str(answer)[:2000]}
                    )
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
                    "query.aquery",
                    mode=mode,
                    docs=len(sol.docs),
                    answer_chars=len(sol.answer or ""),
                )
                return sol
            finally:
                flush_langfuse()

    def query(
        self, query: str, param: QueryParam | None = None
    ) -> Union[str, QuerySolution]:
        return _run_sync(self.aquery(query, param=param))

    # Research-engine aliases (MemGraphRAG/code naming)
    async def aindex_with_memory(
        self, chunks: Sequence[str] | Sequence[dict[str, str]]
    ) -> dict[str, Any]:
        return await self.ainsert(chunks)

    def index_with_memory(
        self, chunks: Sequence[str] | Sequence[dict[str, str]]
    ) -> dict[str, Any]:
        return _run_sync(self.aindex_with_memory(chunks))

    async def arag_qa(
        self, query: str, param: QueryParam | None = None
    ) -> Union[str, QuerySolution]:
        return await self.aquery(query, param=param)

    def rag_qa(
        self, query: str, param: QueryParam | None = None
    ) -> Union[str, QuerySolution]:
        return _run_sync(self.arag_qa(query, param=param))
