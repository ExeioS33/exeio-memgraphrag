"""Graph exploration routes for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/graph_routes.py`` (heavily slimmed).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("memgraphrag.api.graphs")

# Passage nodes carry the chunk text in ``content``. Returning it turned
# ``GET /graphs?limit=5000`` into a plaintext dump of the corpus, so the body is
# replaced by its length — visualisation clients need the topology, not the text.
_REDACTED_NODE_ATTRS = ("content", "passage", "text")

# Edges are capped relative to the node page so one request cannot pull the whole
# adjacency list of a dense graph into a single JSON body.
EDGE_LIMIT_FACTOR = 4


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("id") or node.get("node_id") or "")


def _edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    source = str(edge.get("source") or edge.get("src") or "")
    target = str(edge.get("target") or edge.get("tgt") or "")
    return source, target


def redact_node(node: dict[str, Any]) -> dict[str, Any]:
    """Return ``node`` without its document body, keeping a length for each field.

    ``props`` is scrubbed too: the storage backends splat props over the node dict, so
    the same text is present twice.
    """
    cleaned = dict(node)
    props = cleaned.get("props")
    cleaned_props = dict(props) if isinstance(props, dict) else None
    for attr in _REDACTED_NODE_ATTRS:
        for holder in (cleaned, cleaned_props):
            if holder is None or attr not in holder:
                continue
            body = holder.pop(attr)
            holder[f"{attr}_length"] = len(body) if isinstance(body, str) else 0
    if cleaned_props is not None:
        cleaned["props"] = cleaned_props
    return cleaned

try:
    from fastapi import APIRouter, Depends, Query, Request
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    Query = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]


def create_graphs_router(api_key: Optional[str] = None) -> Any:
    if APIRouter is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import get_combined_auth_dependency

    router = APIRouter(tags=["graph"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/graphs", dependencies=[Depends(combined_auth)])
    async def explore_graph(
        request: Request,
        label: Optional[str] = Query(default=None, description="Filter nodes by label"),
        limit: int = Query(default=200, ge=1, le=5000),
    ):
        rag = request.app.state.rag
        try:
            nodes = await rag.graph.get_all_nodes()
        except Exception as exc:
            logger.warning("get_all_nodes failed: %s", exc)
            nodes = []
        try:
            edges = await rag.graph.get_all_edges()
        except Exception as exc:
            logger.warning("get_all_edges failed: %s", exc)
            edges = []

        if label:
            label_l = label.lower()
            nodes = [
                n
                for n in nodes
                if str(n.get("label", "")).lower() == label_l
                or str(n.get("layer", "")).lower() == label_l
            ]

        total_nodes = len(nodes)
        page = nodes[:limit]
        kept_ids = {_node_id(n) for n in page}
        kept_ids.discard("")

        # Both endpoints must survive the node cut. Slicing nodes and edges
        # independently used to emit edges pointing at nodes that were never sent, so
        # every visualisation client received dangling endpoints.
        max_edges = limit * EDGE_LIMIT_FACTOR
        kept_edges: list[dict[str, Any]] = []
        for edge in edges:
            source, target = _edge_endpoints(edge)
            if source in kept_ids and target in kept_ids:
                kept_edges.append(edge)
                if len(kept_edges) >= max_edges:
                    break

        return {
            "nodes": [redact_node(n) for n in page],
            "edges": kept_edges,
            "label": label,
            "total_nodes": total_nodes,
            "returned_nodes": len(page),
            "returned_edges": len(kept_edges),
            "truncated": total_nodes > len(page) or len(kept_edges) >= max_edges,
        }

    @router.get("/graph/label/list", dependencies=[Depends(combined_auth)])
    async def list_labels(request: Request):
        rag = request.app.state.rag
        try:
            nodes = await rag.graph.get_all_nodes()
        except Exception as exc:
            logger.warning("get_all_nodes failed: %s", exc)
            nodes = []
        labels: set[str] = set()
        for n in nodes:
            for key in ("label", "layer", "entity_type"):
                val = n.get(key)
                if val:
                    labels.add(str(val))
        return {"labels": sorted(labels)}

    return router
