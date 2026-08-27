"""Neo4j-backed typed memory-graph storage for MemGraphRAG.

Provenance: adapted from LightRAG ``lightrag/kg/neo4j_impl.py`` for the
MemGraphRAG memory graph (Type / Entity / Passage nodes; typed weighted
edges). Connection and workspace-isolation patterns follow LightRAG;
schema and return shapes match ``IgraphStorage`` / ``BaseGraphStorage``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from memgraphrag.utils.env import get_env_value
from memgraphrag.base import BaseGraphStorage

logger = logging.getLogger(__name__)

try:
    from neo4j import AsyncGraphDatabase  # type: ignore
    from neo4j import exceptions as neo4jExceptions  # type: ignore

    _NEO4J_AVAILABLE = True
except ImportError:  # pragma: no cover
    AsyncGraphDatabase = None  # type: ignore[assignment, misc]
    neo4jExceptions = None  # type: ignore[assignment, misc]
    _NEO4J_AVAILABLE = False

NODE_LABELS = frozenset({"Type", "Entity", "Passage", "Fact", "Schema"})
EDGE_TYPES = frozenset(
    {
        "ENTITY_RELATION",
        "PASSAGE_ENTITY",
        "ENTITY_SIMILARITY",
        "ENTITY_TO_TYPE",
        "TYPE_RELATION",
        "FACT_SCHEMA",
        "FACT_PASSAGE",
    }
)
MGR_OWNED_PROPERTY = "mgr_owned"

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_neo4j() -> None:
    if not _NEO4J_AVAILABLE or AsyncGraphDatabase is None:
        raise ImportError(
            "neo4j package is required for Neo4JStorage. "
            "Install with: pip install neo4j  (or memgraphrag[api])"
        )


def _escape_label(label: str) -> str:
    """Escape backticks for safe use inside Cypher backtick-quoted labels."""
    return (label or "base").replace("`", "``")


def _normalize_node_label(node_data: dict[str, Any]) -> str:
    raw = node_data.get("node_type") or node_data.get("label") or node_data.get("layer") or "Entity"
    label = str(raw).strip()
    if not label:
        return "Entity"
    # Title-case common layer names (passage → Passage)
    titled = label[:1].upper() + label[1:]
    if titled in NODE_LABELS:
        return titled
    if label in NODE_LABELS:
        return label
    # Allow custom labels if Cypher-safe; otherwise fall back to Entity
    if _SAFE_IDENT.match(label):
        return label
    return "Entity"


def _normalize_edge_type(edge_data: dict[str, Any]) -> str:
    raw = str(edge_data.get("type") or edge_data.get("edge_type") or "ENTITY_RELATION")
    raw = raw.strip().upper().replace("-", "_").replace(" ", "_")
    if raw in EDGE_TYPES:
        return raw
    if _SAFE_IDENT.match(raw):
        return raw
    return "ENTITY_RELATION"


def _sanitize_props(data: dict[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    """Flatten node/edge props to Neo4j-safe scalars / lists of scalars."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in exclude or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
            out[key] = value
        else:
            out[key] = str(value)
    return out


@dataclass
class Neo4JStorage(BaseGraphStorage):
    """Async Neo4j storage for the typed MemGraphRAG memory graph."""

    _driver: Any = field(default=None, init=False, repr=False)
    _DATABASE: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Defer ImportError until initialize() so the class can still be imported.
        neo4j_workspace = os.environ.get("NEO4J_WORKSPACE")
        if neo4j_workspace and neo4j_workspace.strip():
            object.__setattr__(self, "workspace", neo4j_workspace.strip())
        if not self.workspace or not str(self.workspace).strip():
            object.__setattr__(self, "workspace", "base")

    def _workspace_label(self) -> str:
        return _escape_label(str(self.workspace).strip() or "base")

    def _raw_workspace(self) -> str:
        return str(self.workspace).strip() or "base"

    async def initialize(self) -> None:
        _require_neo4j()
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "neo4j")
        database = os.environ.get("NEO4J_DATABASE")
        if not database:
            # Fallback: sanitize namespace like LightRAG does
            database = re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace) or "neo4j"

        pool_size = int(os.environ.get("NEO4J_MAX_CONNECTION_POOL_SIZE", "50"))
        conn_timeout = float(os.environ.get("NEO4J_CONNECTION_TIMEOUT", "30.0"))

        self._driver = AsyncGraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_pool_size=pool_size,
            connection_timeout=conn_timeout,
        )
        self._DATABASE = database

        # Probe connectivity; fall back to default DB if named DB missing.
        for candidate in (database, None):
            try:
                async with self._driver.session(database=candidate) as session:
                    result = await session.run("RETURN 1 AS ok")
                    await result.consume()
                self._DATABASE = candidate
                logger.info(
                    "[%s] Connected to Neo4j database=%s at %s",
                    self.workspace,
                    candidate,
                    uri,
                )
                break
            except Exception as exc:
                if neo4jExceptions and isinstance(
                    exc, getattr(neo4jExceptions, "AuthError", type(None))
                ):
                    raise
                logger.warning(
                    "[%s] Neo4j probe failed for database=%s: %s",
                    self.workspace,
                    candidate,
                    exc,
                )
        else:
            await self.finalize()
            raise ConnectionError(f"Unable to connect to Neo4j at {uri}")

        await self._ensure_indexes()
        await self._assert_workspace_not_shared()

    async def _assert_workspace_not_shared(self) -> None:
        """Refuse to start on a workspace another engine already populated.

        `clear()` is scoped to MemGraphRAG-owned nodes, so nothing would be deleted —
        but sharing a workspace label still mixes two knowledge graphs in every
        traversal, and PPR would walk a foreign engine's edges. Fail loudly instead,
        unless the operator opts in.
        """
        if get_env_value("MEMGRAPHRAG_ALLOW_SHARED_NEO4J_WORKSPACE", False, bool):
            return
        try:
            foreign = await self.foreign_node_count()
        except Exception as exc:  # a probe failure must not block startup
            logger.warning("[%s] Could not audit workspace ownership: %s", self.workspace, exc)
            return
        if foreign:
            raise RuntimeError(
                f"Neo4j workspace {self._raw_workspace()!r} already holds {foreign} nodes "
                f"that MemGraphRAG did not create (no {MGR_OWNED_PROPERTY!r} marker). "
                "Another engine — LightRAG uses the same workspace-as-label convention — "
                "is very likely using it. Pick a distinct WORKSPACE / NEO4J_WORKSPACE, or "
                "set MEMGRAPHRAG_ALLOW_SHARED_NEO4J_WORKSPACE=true to share it knowingly."
            )

    async def _ensure_indexes(self) -> None:
        ws = self._workspace_label()
        queries = [
            f"CREATE INDEX IF NOT EXISTS FOR (n:`{ws}`) ON (n.entity_id)",
            f"CREATE INDEX IF NOT EXISTS FOR (n:`{ws}`) ON (n.node_id)",
        ]
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                for q in queries:
                    result = await session.run(q)
                    await result.consume()
        except Exception as exc:
            logger.warning("[%s] Index creation skipped/failed: %s", self.workspace, exc)

    async def finalize(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    def _session(self, **kwargs: Any):
        assert self._driver is not None, "Neo4JStorage not initialized"
        return self._driver.session(database=self._DATABASE, **kwargs)

    def _node_dict(self, props: dict[str, Any], labels: list[str] | None = None) -> dict[str, Any]:
        node_id = props.get("entity_id") or props.get("node_id") or props.get("id")
        # Prefer typed label over workspace label
        typed = None
        if labels:
            for lab in labels:
                if lab in NODE_LABELS or (
                    lab != self._raw_workspace() and lab != self._workspace_label()
                ):
                    typed = lab
                    break
        label = typed or props.get("node_type") or props.get("label") or ""
        out = {
            "id": node_id,
            "label": label,
            "node_type": props.get("node_type") or label,
            "content": props.get("content", ""),
            **{k: v for k, v in props.items() if k not in ("id",)},
        }
        out["entity_id"] = node_id
        out["node_id"] = node_id
        return out

    def _edge_dict(
        self,
        source: str,
        target: str,
        rel_type: str,
        props: dict[str, Any],
    ) -> dict[str, Any]:
        weight = float(props.get("weight", 1.0) or 1.0)
        return {
            "source": source,
            "target": target,
            "type": rel_type or props.get("type", ""),
            "weight": weight,
            **{k: v for k, v in props.items() if k not in ("source", "target", "type")},
        }

    async def has_node(self, node_id: str) -> bool:
        ws = self._workspace_label()
        query = f"MATCH (n:`{ws}` {{entity_id: $entity_id}}) RETURN count(n) > 0 AS exists"
        async with self._session(default_access_mode="READ") as session:
            result = await session.run(query, entity_id=node_id)
            record = await result.single()
            await result.consume()
            return bool(record and record["exists"])

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        ws = self._workspace_label()
        query = (
            f"MATCH (a:`{ws}` {{entity_id: $src}})-[r]-(b:`{ws}` {{entity_id: $tgt}}) "
            "RETURN count(r) > 0 AS exists"
        )
        async with self._session(default_access_mode="READ") as session:
            result = await session.run(query, src=source_node_id, tgt=target_node_id)
            record = await result.single()
            await result.consume()
            return bool(record and record["exists"])

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        ws = self._workspace_label()
        node_label = _normalize_node_label(node_data)
        props = _sanitize_props(
            node_data,
            exclude={"id", "label", "props", "labels"},
        )
        if isinstance(node_data.get("props"), dict):
            props.update(_sanitize_props(node_data["props"], exclude={"id", "label", "props"}))
        props["entity_id"] = node_id
        props["node_id"] = node_id
        props["node_type"] = node_label
        props["label"] = node_label
        props["workspace"] = self._raw_workspace()
        # Ownership marker. `clear()` deletes only nodes carrying it, so pointing
        # MemGraphRAG at a Neo4j workspace that another engine already populated can
        # never wipe that engine's graph. Without it, `clear()` matched the whole
        # workspace label and a single ainsert against LightRAG's `default` workspace
        # would have destroyed 14 556 nodes and 26 938 relationships.
        props[MGR_OWNED_PROPERTY] = True

        # MERGE on workspace + entity_id, then attach typed label (no APOC).
        merge_q = f"""
        MERGE (n:`{ws}` {{entity_id: $entity_id}})
        SET n += $properties
        """
        async with self._session() as session:
            result = await session.run(merge_q, entity_id=node_id, properties=props)
            await result.consume()
            if _SAFE_IDENT.match(node_label):
                label_q = f"""
                MATCH (n:`{ws}` {{entity_id: $entity_id}})
                SET n:`{node_label}`
                """
                result = await session.run(label_q, entity_id=node_id)
                await result.consume()

    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict[str, Any],
    ) -> None:
        ws = self._workspace_label()
        rel_type = _normalize_edge_type(edge_data)
        props = _sanitize_props(
            edge_data,
            exclude={"source", "target", "type", "edge_type", "props"},
        )
        if isinstance(edge_data.get("props"), dict):
            props.update(
                _sanitize_props(
                    edge_data["props"],
                    exclude={"source", "target", "type", "edge_type", "props"},
                )
            )
        props.setdefault("weight", 1.0)
        props["weight"] = float(props.get("weight", 1.0) or 1.0)
        props["type"] = rel_type
        props["workspace"] = self._raw_workspace()

        # Ensure endpoints exist (mirror IgraphStorage behaviour)
        for nid in (source_node_id, target_node_id):
            if not await self.has_node(nid):
                await self.upsert_node(nid, {"id": nid, "label": "Entity", "content": ""})

        query = f"""
        MATCH (source:`{ws}` {{entity_id: $src}})
        MATCH (target:`{ws}` {{entity_id: $tgt}})
        MERGE (source)-[r:`{rel_type}`]->(target)
        SET r += $properties
        """
        async with self._session() as session:
            result = await session.run(
                query,
                src=source_node_id,
                tgt=target_node_id,
                properties=props,
            )
            await result.consume()

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        ws = self._workspace_label()
        query = f"""
        MATCH (n:`{ws}` {{entity_id: $entity_id}})
        RETURN n, labels(n) AS labels
        LIMIT 1
        """
        async with self._session(default_access_mode="READ") as session:
            result = await session.run(query, entity_id=node_id)
            record = await result.single()
            await result.consume()
            if record is None:
                return None
            return self._node_dict(dict(record["n"]), list(record["labels"] or []))

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, Any] | None:
        ws = self._workspace_label()
        query = f"""
        MATCH (a:`{ws}` {{entity_id: $src}})-[r]-(b:`{ws}` {{entity_id: $tgt}})
        RETURN type(r) AS rel_type, properties(r) AS props
        LIMIT 1
        """
        async with self._session(default_access_mode="READ") as session:
            result = await session.run(query, src=source_node_id, tgt=target_node_id)
            record = await result.single()
            await result.consume()
            if record is None:
                return None
            return self._edge_dict(
                source_node_id,
                target_node_id,
                record["rel_type"],
                dict(record["props"] or {}),
            )

    async def get_all_nodes(self) -> list[dict[str, Any]]:
        ws = self._workspace_label()
        query = f"""
        MATCH (n:`{ws}`)
        RETURN n, labels(n) AS labels
        """
        nodes: list[dict[str, Any]] = []
        async with self._session(default_access_mode="READ") as session:
            result = await session.run(query)
            async for record in result:
                nodes.append(self._node_dict(dict(record["n"]), list(record["labels"] or [])))
            await result.consume()
        return nodes

    async def get_all_edges(self) -> list[dict[str, Any]]:
        ws = self._workspace_label()
        # Directed return; DISTINCT avoids undirected double-count
        query = f"""
        MATCH (a:`{ws}`)-[r]->(b:`{ws}`)
        RETURN a.entity_id AS source, b.entity_id AS target,
               type(r) AS rel_type, properties(r) AS props
        """
        edges: list[dict[str, Any]] = []
        async with self._session(default_access_mode="READ") as session:
            result = await session.run(query)
            async for record in result:
                edges.append(
                    self._edge_dict(
                        record["source"],
                        record["target"],
                        record["rel_type"],
                        dict(record["props"] or {}),
                    )
                )
            await result.consume()
        return edges

    async def clear(self) -> None:
        """Delete this workspace's MemGraphRAG nodes, and only those.

        Scoped by the ownership marker rather than by the workspace label: the label
        is just the workspace name, and other engines (LightRAG in particular) use the
        same convention, so an unscoped delete silently destroys their graph.
        """
        ws = self._workspace_label()
        query = f"MATCH (n:`{ws}`) WHERE n.`{MGR_OWNED_PROPERTY}` = true DETACH DELETE n"
        async with self._session() as session:
            result = await session.run(query)
            await result.consume()

    async def foreign_node_count(self) -> int:
        """Nodes in this workspace label that MemGraphRAG did not create."""
        ws = self._workspace_label()
        query = f"MATCH (n:`{ws}`) WHERE n.`{MGR_OWNED_PROPERTY}` IS NULL RETURN count(n) AS c"
        async with self._session(default_access_mode="READ") as session:
            result = await session.run(query)
            record = await result.single()
            await result.consume()
            return int((record or {}).get("c") or 0)

    async def node_degree(self, node_id: str) -> int:
        ws = self._workspace_label()
        query = f"""
        MATCH (n:`{ws}` {{entity_id: $entity_id}})
        OPTIONAL MATCH (n)-[r]-()
        RETURN count(r) AS degree
        """
        async with self._session(default_access_mode="READ") as session:
            result = await session.run(query, entity_id=node_id)
            record = await result.single()
            await result.consume()
            if record is None:
                return 0
            return int(record["degree"] or 0)
