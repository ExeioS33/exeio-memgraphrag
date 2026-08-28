"""Retrieval state manager for MemGraphRAG PPR warm-up and refresh.

Provenance: replaces the research repo's one-shot ``prepare_retrieval_objects``
warm-up (``MemGraphRAG/code/src/MemGraphRAG.py``) with a fuller lifecycle: eager
hydration, incremental igraph updates after indexing batches, and a versioned
full-reload fallback.

Status: **not wired into the running service.** The API server warms up through
``MemGraphRAG.prepare_retrieval()`` in its lifespan, so nothing outside
``tests/`` imports this module today. It is kept because the planned refresh
path (re-hydrate after an ingest batch instead of re-preparing the whole engine)
builds on it; treat it as unproven until then. The refresh signals it exchanges
through ``memgraphrag.storage.shared`` are process-local, not cross-worker —
which is one reason ``WORKERS > 1`` is refused for file-backed storage (see
``memgraphrag.api.config.validate_worker_count``).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

from memgraphrag.exceptions import NotReadyError
from memgraphrag.ppr import get_ppr_engine
from memgraphrag.ppr.base import PPREngine
from memgraphrag.ppr.igraph_engine import IgraphPPREngine
from memgraphrag.storage import shared as shared_storage

logger = logging.getLogger(__name__)


def _edge_endpoints(edge: Mapping[str, Any]) -> tuple[str, str] | None:
    src = edge.get("source") or edge.get("src") or edge.get("source_node_id")
    tgt = edge.get("target") or edge.get("tgt") or edge.get("target_node_id")
    if src is None or tgt is None:
        return None
    return str(src), str(tgt)


def _is_passage_node(node: Mapping[str, Any], node_id: str) -> bool:
    label = str(node.get("label") or node.get("node_type") or node.get("layer") or "")
    if label.lower() in {"passage", "chunk"}:
        return True
    return node_id.startswith(("chunk-", "passage-", "doc-"))


class RetrievalStateManager:
    """Owns PPR readiness for a MemGraphRAG workspace.

    Can be constructed with a ``MemGraphRAG`` instance (uses its ``graph`` /
    ``_ppr`` / ``workspace``) or with explicit graph storage + engine kwargs.
    """

    def __init__(
        self,
        rag: Any | None = None,
        *,
        graph: Any | None = None,
        ppr_engine: PPREngine | None = None,
        ppr_engine_name: str = "igraph",
        workspace: str | None = None,
        passage_ids: Iterable[str] | None = None,
        directed: bool = False,
    ) -> None:
        self.rag = rag
        self.graph = graph if graph is not None else getattr(rag, "graph", None)
        self.workspace = (
            workspace if workspace is not None else str(getattr(rag, "workspace", "") or "")
        )
        self.ppr_engine_name = ppr_engine_name or getattr(rag, "ppr_engine_name", None) or "igraph"
        self._ppr: PPREngine | None = ppr_engine
        if self._ppr is None and rag is not None:
            self._ppr = getattr(rag, "_ppr", None)

        self._passage_ids: set[str] = set(passage_ids or [])
        self._directed = directed
        self._ready = False
        self._version = 0

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def version(self) -> int:
        return self._version

    @property
    def ppr(self) -> PPREngine | None:
        return self._ppr

    def require_ready(self) -> None:
        """Raise ``NotReadyError`` when retrieval state is not hydrated."""
        if not self._ready:
            raise NotReadyError(
                "Retrieval state is not ready; call warm_up() or full_reload() first"
            )

    def get_ppr_or_raise(self) -> PPREngine:
        self.require_ready()
        if self._ppr is None:
            raise NotReadyError("PPR engine is missing after warm-up")
        return self._ppr

    async def warm_up(self) -> None:
        """Hydrate the PPR engine from graph storage and mark ready."""
        if self.graph is None:
            raise NotReadyError("No graph storage available for warm-up")
        await self._hydrate_from_graph()
        self._ready = True
        self._version = await shared_storage.bump_retrieval_version(self.workspace)
        self._sync_rag()
        logger.info(
            "Retrieval warm-up complete (workspace=%r version=%d passages=%d)",
            self.workspace,
            self._version,
            len(self._passage_ids),
        )

    async def full_reload(self) -> None:
        """Re-hydrate PPR from graph storage (versioned full reload)."""
        self._ready = False
        await self.warm_up()

    async def refresh_incremental(
        self,
        nodes: Sequence[Mapping[str, Any]] | None = None,
        edges: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Apply new nodes/edges to an in-memory igraph engine when possible.

        Falls back to ``full_reload`` for non-igraph engines or when not yet
        ready. Bumps the retrieval version and sets the shared refresh flag.
        """
        nodes = list(nodes or [])
        edges = list(edges or [])

        if not self._ready or not isinstance(self._ppr, IgraphPPREngine):
            await self.full_reload()
            await shared_storage.set_refresh_flag(self.workspace)
            return

        for node in nodes:
            nid = str(node.get("id") or node.get("entity_id") or node.get("node_id") or "")
            if not nid:
                continue
            if _is_passage_node(node, nid):
                self._passage_ids.add(nid)
            self._ensure_igraph_vertex(nid)

        for edge in edges:
            ends = _edge_endpoints(edge)
            if ends is None:
                continue
            src, tgt = ends
            weight = float(edge.get("weight", 1.0) or 1.0)
            self._ensure_igraph_vertex(src)
            self._ensure_igraph_vertex(tgt)
            self._add_igraph_edge(src, tgt, weight)

        if isinstance(self._ppr, IgraphPPREngine):
            self._ppr.set_passage_ids(self._passage_ids)

        self._version = await shared_storage.bump_retrieval_version(self.workspace)
        await shared_storage.set_refresh_flag(self.workspace)
        self._sync_rag()
        logger.info(
            "Incremental retrieval refresh (workspace=%r version=%d +nodes=%d +edges=%d)",
            self.workspace,
            self._version,
            len(nodes),
            len(edges),
        )

    async def consume_cross_worker_signal(self) -> bool:
        """If a refresh was signaled, run ``full_reload``.

        Despite the name the flag never crosses a process boundary:
        ``memgraphrag.storage.shared`` is an in-process dict, so only signals
        raised by this interpreter are ever seen.

        Returns:
            ``True`` if a reload was performed.
        """
        pending = await shared_storage.consume_refresh_flag(self.workspace)
        if not pending:
            return False
        logger.info("Retrieval refresh signal consumed for workspace=%r", self.workspace)
        await self.full_reload()
        return True

    async def signal_refresh(self) -> None:
        """Mark this workspace as needing a retrieval-state refresh."""
        await shared_storage.set_refresh_flag(self.workspace)

    async def get_shared_version(self) -> int:
        return await shared_storage.get_retrieval_version(self.workspace)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _hydrate_from_graph(self) -> None:
        assert self.graph is not None
        edges: list[tuple[str, str]] = []
        weights: list[float] = []
        passage_ids = set(self._passage_ids)

        try:
            raw_nodes = await self.graph.get_all_nodes()
            for node in raw_nodes:
                nid = str(node.get("id") or node.get("entity_id") or node.get("node_id") or "")
                if nid and _is_passage_node(node, nid):
                    passage_ids.add(nid)
        except Exception as exc:
            logger.warning("warm_up: could not load nodes: %s", exc)

        try:
            raw_edges = await self.graph.get_all_edges()
            for edge in raw_edges:
                ends = _edge_endpoints(edge)
                if ends is None:
                    continue
                edges.append(ends)
                weights.append(float(edge.get("weight", 1.0) or 1.0))
        except Exception as exc:
            logger.warning("warm_up: could not load edges: %s", exc)

        # Also pick up passage ids from a MemGraphRAG instance if present
        if self.rag is not None:
            rag_passages = getattr(self.rag, "_passage_ids", None) or []
            passage_ids.update(str(p) for p in rag_passages)

        self._passage_ids = passage_ids
        try:
            self._ppr = get_ppr_engine(
                self.ppr_engine_name,
                edges=edges,
                edge_weights=weights,
                passage_ids=self._passage_ids,
                directed=self._directed,
            )
        except Exception as exc:
            logger.warning("PPR engine init failed (%s); falling back to IgraphPPREngine", exc)
            self._ppr = IgraphPPREngine(
                edges=edges,
                edge_weights=weights,
                passage_ids=self._passage_ids,
                directed=self._directed,
            )

    def _sync_rag(self) -> None:
        if self.rag is None:
            return
        self.rag._ppr = self._ppr
        self.rag.ready_to_retrieve = self._ready
        if self._passage_ids:
            existing = getattr(self.rag, "_passage_ids", None)
            if not existing:
                self.rag._passage_ids = list(self._passage_ids)

    def _ensure_igraph_vertex(self, node_id: str) -> None:
        assert isinstance(self._ppr, IgraphPPREngine)
        if node_id in self._ppr.name_to_idx:
            return
        g = self._ppr.graph
        g.add_vertex(name=node_id)
        self._ppr.name_to_idx[node_id] = g.vcount() - 1

    def _add_igraph_edge(self, src: str, tgt: str, weight: float) -> None:
        assert isinstance(self._ppr, IgraphPPREngine)
        g = self._ppr.graph
        si = self._ppr.name_to_idx[src]
        ti = self._ppr.name_to_idx[tgt]
        # Avoid duplicate undirected edges when possible
        if g.are_adjacent(si, ti):
            # Update weight on first matching edge
            for e in g.es.select(_between=([si], [ti])):
                e["weight"] = weight
                break
            return
        g.add_edge(si, ti)
        edge = g.es[g.ecount() - 1]
        if "weight" not in g.es.attributes():
            g.es["weight"] = [1.0] * (g.ecount() - 1) + [weight]
        else:
            edge["weight"] = weight
