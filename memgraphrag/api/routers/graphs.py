"""Graph exploration routes for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/graph_routes.py`` (heavily slimmed).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("memgraphrag.api.graphs")

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
            node_ids = {
                str(n.get("id") or n.get("node_id") or "")
                for n in nodes
            }
            edges = [
                e
                for e in edges
                if str(e.get("source") or e.get("src") or "") in node_ids
                or str(e.get("target") or e.get("tgt") or "") in node_ids
            ]

        return {
            "nodes": nodes[:limit],
            "edges": edges[: limit * 2],
            "label": label,
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
