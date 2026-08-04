"""igraph GraphML graph storage for the typed memory graph.

Adapted from LightRAG ``lightrag/kg/networkx_impl.py`` (file-backed GraphML
pattern) and MemGraphRAG research graph construction. Nodes store
``id`` / ``label`` / ``props``; edges store ``source`` / ``target`` /
``type`` / ``weight`` / ``props``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import igraph as ig

from memgraphrag.base import BaseGraphStorage

logger = logging.getLogger(__name__)


def _workspace_dir(working_dir: str, workspace: str) -> str:
    if workspace:
        return os.path.join(working_dir, workspace)
    return working_dir


def _dumps_props(props: Any) -> str:
    if props is None:
        return "{}"
    if isinstance(props, str):
        return props
    return json.dumps(props, ensure_ascii=False)


def _loads_props(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {"value": value}
        except json.JSONDecodeError:
            return {"value": raw}
    return {"value": raw}


@dataclass
class IgraphStorage(BaseGraphStorage):
    """Persist a typed memory graph as GraphML via python-igraph."""

    _graph: Any = field(default=None, init=False, repr=False)
    _file_name: str = field(default="", init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _dirty: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        working_dir = self.global_config.get("working_dir", "./data/rag_storage")
        workspace_dir = _workspace_dir(working_dir, self.workspace or "")
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(
            workspace_dir, f"graph_{self.namespace}.graphml"
        )
        directed = bool(self.global_config.get("is_directed_graph", False))
        self._graph = ig.Graph(directed=directed)
        self._dirty = False

    async def initialize(self) -> None:
        async with self._lock:
            if os.path.exists(self._file_name):
                self._graph = ig.Graph.Read_GraphML(self._file_name)
                logger.info(
                    "Loaded graph from %s (%d nodes, %d edges)",
                    self._file_name,
                    self._graph.vcount(),
                    self._graph.ecount(),
                )
            else:
                directed = bool(self.global_config.get("is_directed_graph", False))
                self._graph = ig.Graph(directed=directed)
            self._dirty = False

    async def finalize(self) -> None:
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._dirty:
                return
            os.makedirs(os.path.dirname(self._file_name) or ".", exist_ok=True)
            tmp = f"{self._file_name}.tmp"
            self._graph.write_graphml(tmp)
            os.replace(tmp, self._file_name)
            self._dirty = False

    def _find_vertex(self, node_id: str):
        try:
            return self._graph.vs.find(name=node_id)
        except (ValueError, KeyError):
            return None

    def _node_dict(self, vertex) -> dict[str, Any]:
        props = _loads_props(vertex.attributes().get("props"))
        return {
            "id": vertex["name"],
            "label": vertex.attributes().get("label", ""),
            "props": props,
            **props,
        }

    def _edge_dict(self, edge) -> dict[str, Any]:
        attrs = edge.attributes()
        props = _loads_props(attrs.get("props"))
        source = self._graph.vs[edge.source]["name"]
        target = self._graph.vs[edge.target]["name"]
        return {
            "source": source,
            "target": target,
            "type": attrs.get("type", ""),
            "weight": float(attrs.get("weight", 1.0) or 1.0),
            "props": props,
            **props,
        }

    async def has_node(self, node_id: str) -> bool:
        async with self._lock:
            return self._find_vertex(node_id) is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self._lock:
            src = self._find_vertex(source_node_id)
            tgt = self._find_vertex(target_node_id)
            if src is None or tgt is None:
                return False
            return self._graph.are_adjacent(src.index, tgt.index)

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        label = str(node_data.get("label", node_data.get("layer", "")))
        props = {
            k: v
            for k, v in node_data.items()
            if k not in ("id", "label", "props")
        }
        if isinstance(node_data.get("props"), dict):
            props.update(node_data["props"])
        async with self._lock:
            vertex = self._find_vertex(node_id)
            if vertex is None:
                self._graph.add_vertex(
                    name=node_id,
                    label=label,
                    props=_dumps_props(props),
                )
            else:
                vertex["label"] = label
                vertex["props"] = _dumps_props(props)
            self._dirty = True
        await self._flush()

    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict[str, Any],
    ) -> None:
        edge_type = str(edge_data.get("type", ""))
        weight = float(edge_data.get("weight", 1.0) or 1.0)
        props = {
            k: v
            for k, v in edge_data.items()
            if k not in ("source", "target", "type", "weight", "props")
        }
        if isinstance(edge_data.get("props"), dict):
            props.update(edge_data["props"])

        async with self._lock:
            if self._find_vertex(source_node_id) is None:
                self._graph.add_vertex(
                    name=source_node_id, label="", props="{}"
                )
            if self._find_vertex(target_node_id) is None:
                self._graph.add_vertex(
                    name=target_node_id, label="", props="{}"
                )
            src = self._find_vertex(source_node_id)
            tgt = self._find_vertex(target_node_id)
            assert src is not None and tgt is not None

            existing = None
            for edge in self._graph.es.select(_between=([src.index], [tgt.index])):
                if str(edge.attributes().get("type", "")) == edge_type:
                    existing = edge
                    break

            if existing is not None:
                existing["type"] = edge_type
                existing["weight"] = weight
                existing["props"] = _dumps_props(props)
            else:
                self._graph.add_edge(
                    src.index,
                    tgt.index,
                    type=edge_type,
                    weight=weight,
                    props=_dumps_props(props),
                )
            self._dirty = True
        await self._flush()

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        async with self._lock:
            vertex = self._find_vertex(node_id)
            if vertex is None:
                return None
            return self._node_dict(vertex)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        async with self._lock:
            src = self._find_vertex(source_node_id)
            tgt = self._find_vertex(target_node_id)
            if src is None or tgt is None:
                return None
            edges = self._graph.es.select(_between=([src.index], [tgt.index]))
            if not edges:
                return None
            return self._edge_dict(edges[0])

    async def get_all_nodes(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [self._node_dict(v) for v in self._graph.vs]

    async def get_all_edges(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [self._edge_dict(e) for e in self._graph.es]

    async def clear(self) -> None:
        async with self._lock:
            directed = self._graph.is_directed()
            self._graph = ig.Graph(directed=directed)
            self._dirty = True
        await self._flush()

    async def node_degree(self, node_id: str) -> int:
        async with self._lock:
            vertex = self._find_vertex(node_id)
            if vertex is None:
                return 0
            return int(self._graph.degree(vertex.index))
