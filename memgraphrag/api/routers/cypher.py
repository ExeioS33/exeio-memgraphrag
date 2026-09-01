"""Read-only Cypher console and workspace graph introspection.

Protection is layered, in this order:

1. the graph backend must be ``Neo4JStorage`` (the igraph default has no session) —
   otherwise 503;
2. the statement is scanned for write keywords, with string literals and comments
   stripped first so a keyword inside a quoted value is not a false positive — 400;
3. execution happens inside ``Neo4JStorage._session(default_access_mode="READ")``,
   which is the real enforcement: Neo4j itself refuses a write in a READ
   transaction even if the scanner is fooled;
4. a default LIMIT is injected when the statement has none, a record cap bounds the
   stream, a transaction timeout bounds the wall clock, and long string properties
   are truncated so a ``Passage.content`` dump cannot blow up the response.

Every query is scoped to the workspace label. This Neo4j instance is shared with
other engines — LightRAG uses the same workspace-as-label convention — so an
unscoped query silently mixes two unrelated graphs. For the same reason the schema
route derives labels and property keys from the workspace instead of calling
``db.labels()`` / ``db.propertyKeys()``, which are database-wide.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Optional

logger = logging.getLogger("memgraphrag.api.cypher")

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    Query = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    status = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]  # noqa: E731

try:
    from neo4j import exceptions as neo4j_exceptions  # type: ignore
    from neo4j.graph import Node as Neo4jNode  # type: ignore
    from neo4j.graph import Path as Neo4jPath  # type: ignore
    from neo4j.graph import Relationship as Neo4jRelationship  # type: ignore
except ImportError:  # pragma: no cover
    neo4j_exceptions = None  # type: ignore[assignment]
    Neo4jNode = None  # type: ignore[misc, assignment]
    Neo4jPath = None  # type: ignore[misc, assignment]
    Neo4jRelationship = None  # type: ignore[misc, assignment]

# ``isinstance`` against an empty tuple is False, which is what we want when the
# driver is not installed: nothing is ever recognised as a graph value.
_NODE_TYPES: tuple[type, ...] = (Neo4jNode,) if Neo4jNode is not None else ()
_REL_TYPES: tuple[type, ...] = (Neo4jRelationship,) if Neo4jRelationship is not None else ()
_PATH_TYPES: tuple[type, ...] = (Neo4jPath,) if Neo4jPath is not None else ()
_DRIVER_ERRORS: tuple[type, ...] = ()
if neo4j_exceptions is not None:  # pragma: no branch
    _DRIVER_ERRORS = tuple(
        exc
        for exc in (
            getattr(neo4j_exceptions, "Neo4jError", None),
            getattr(neo4j_exceptions, "DriverError", None),
        )
        if isinstance(exc, type)
    )

DEFAULT_LIMIT = 200
MAX_LIMIT = 5000
# Passage.content is the full chunk text; anything longer than this is cut and marked.
MAX_STRING_PROPERTY = 500
# Transaction-level guard: a runaway traversal fails instead of pinning a worker.
QUERY_TIMEOUT_SECONDS = 30.0
# Label / property-key derivation is a full workspace scan, so it is cached briefly.
SCHEMA_CACHE_TTL_SECONDS = 60.0
SCHEMA_SAMPLE = 5000
HIGHLIGHT_TOP_K = 5

# workspace -> (expiry timestamp, payload)
_schema_cache: dict[str, tuple[float, dict[str, Any]]] = {}

BACKEND_UNAVAILABLE = (
    "La console Cypher nécessite le backend de graphe Neo4j "
    "(MEMGRAPHRAG_GRAPH_STORAGE=Neo4JStorage). Backend actuel : {backend}."
)

WRITE_KEYWORDS = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "FOREACH",
    "GRANT",
    "REVOKE",
    "ALTER",
    "RENAME",
    "TERMINATE",
)

# apoc namespaces that only read. Anything else under ``apoc.`` is refused: the
# namespace holds file loaders, exporters, triggers and background writers.
READONLY_APOC_PREFIXES = (
    "apoc.agg.",
    "apoc.algo.",
    "apoc.any.",
    "apoc.coll.",
    "apoc.convert.",
    "apoc.date.",
    "apoc.label.",
    "apoc.map.",
    "apoc.math.",
    "apoc.meta.",
    "apoc.neighbors.",
    "apoc.node.",
    "apoc.nodes.",
    "apoc.number.",
    "apoc.path.",
    "apoc.rel.",
    "apoc.rels.",
    "apoc.schema.",
    "apoc.temporal.",
    "apoc.text.",
)

_WRITE_KEYWORD_RE = re.compile(
    r"(?<![\w.$])(" + "|".join(WRITE_KEYWORDS) + r")(?![\w.])",
    re.IGNORECASE,
)
_LOAD_CSV_RE = re.compile(r"(?<![\w.$])LOAD\s+CSV(?![\w])", re.IGNORECASE)
_CALL_PROC_RE = re.compile(r"(?<![\w.$])CALL\s+([A-Za-z_][\w.]*)", re.IGNORECASE)
_WRITE_PROC_HINT_RE = re.compile(
    r"(create|drop|delete|remove|set|write|import|export|load|trigger"
    r"|periodic|install|uninstall|merge|refactor|atomic)",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(r"[()\[\]{}]|(?<![\w.$])(RETURN|LIMIT)(?![\w.])", re.IGNORECASE)


class CypherRequest(BaseModel):
    query: str = Field(..., min_length=1)
    params: Optional[dict[str, Any]] = None
    limit: Optional[int] = Field(default=None, ge=1, le=MAX_LIMIT)


# --------------------------------------------------------------------------- #
# Statement sanitising
# --------------------------------------------------------------------------- #


def strip_literals_and_comments(query: str) -> str:
    """Blank out quoted spans and comments so keyword scanning cannot false-positive.

    ``MATCH (n) WHERE n.content = 'DELETE me' RETURN n`` is a read; scanning the raw
    text would refuse it. Backtick-quoted identifiers are blanked too — a keyword
    used as an identifier is not a write either. Only the scanned copy is modified;
    the executed statement is always the caller's original text.
    """
    out: list[str] = []
    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            while i < n:
                cur = query[i]
                if cur == "\\" and quote != "`" and i + 1 < n:
                    i += 2
                    continue
                if cur == quote:
                    # A doubled backtick escapes itself inside an identifier.
                    if quote == "`" and i + 1 < n and query[i + 1] == "`":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        if ch == "/" and i + 1 < n and query[i + 1] == "/":
            while i < n and query[i] != "\n":
                i += 1
            out.append(" ")
            continue
        if ch == "/" and i + 1 < n and query[i + 1] == "*":
            i += 2
            while i + 1 < n and not (query[i] == "*" and query[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def find_write_violation(sanitized: str) -> Optional[str]:
    """Return the offending token when the statement is not a pure read."""
    csv_match = _LOAD_CSV_RE.search(sanitized)
    if csv_match:
        return "LOAD CSV"
    for match in _CALL_PROC_RE.finditer(sanitized):
        procedure = match.group(1)
        lowered = procedure.lower()
        if lowered.startswith("apoc."):
            if not any(lowered.startswith(prefix) for prefix in READONLY_APOC_PREFIXES):
                return procedure
            continue
        if _WRITE_PROC_HINT_RE.search(lowered):
            return procedure
    keyword = _WRITE_KEYWORD_RE.search(sanitized)
    if keyword:
        return keyword.group(1).upper()
    return None


def scan_clauses(sanitized: str) -> tuple[bool, bool]:
    """``(has top-level RETURN, has top-level LIMIT after it)``.

    Depth tracking keeps a ``LIMIT`` inside a ``CALL { … }`` subquery or a pattern
    comprehension from being mistaken for the statement's own bound.
    """
    depth = 0
    last_return = -1
    limits: list[int] = []
    for match in _CLAUSE_RE.finditer(sanitized):
        token = match.group(0)
        if token in ("(", "[", "{"):
            depth += 1
        elif token in (")", "]", "}"):
            depth = max(0, depth - 1)
        elif depth == 0:
            if token.upper() == "RETURN":
                last_return = match.start()
            else:
                limits.append(match.start())
    if last_return < 0:
        return False, bool(limits)
    return True, any(pos > last_return for pos in limits)


def apply_limit(query: str, sanitized: str, limit: int) -> tuple[str, bool]:
    """Append ``LIMIT n`` when the statement returns rows without a bound of its own.

    A statement with no top-level ``RETURN`` (a bare ``CALL … YIELD``) is left alone:
    appending ``LIMIT`` there is a syntax error. Those are still bounded, because the
    reader stops after ``limit`` records regardless.
    """
    has_return, has_limit = scan_clauses(sanitized)
    if not has_return or has_limit:
        return query, False
    trimmed = query.rstrip().rstrip(";").rstrip()
    return f"{trimmed}\nLIMIT {limit}", True


# --------------------------------------------------------------------------- #
# Value shaping
# --------------------------------------------------------------------------- #


def _json_safe(value: Any) -> Any:
    """Coerce driver values (temporal, spatial, bytes) to JSON-encodable scalars."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # NaN / inf would be emitted as invalid JSON by the default encoder.
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def _truncate(value: Any) -> tuple[Any, bool]:
    """Cut over-long strings, returning the value and whether anything was cut."""
    if isinstance(value, str):
        if len(value) > MAX_STRING_PROPERTY:
            return value[:MAX_STRING_PROPERTY] + "…", True
        return value, False
    if isinstance(value, list):
        cut = False
        items = []
        for item in value:
            new_item, hit = _truncate(item)
            cut = cut or hit
            items.append(new_item)
        return items, cut
    if isinstance(value, dict):
        cut = False
        out: dict[str, Any] = {}
        for key, item in value.items():
            new_item, hit = _truncate(item)
            cut = cut or hit
            out[key] = new_item
        return out, cut
    return value, False


def _properties(raw: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe, length-bounded properties with a ``_truncated`` marker list."""
    out: dict[str, Any] = {}
    truncated: list[str] = []
    for key, value in raw.items():
        safe, cut = _truncate(_json_safe(value))
        if cut:
            truncated.append(str(key))
        out[str(key)] = safe
    if truncated:
        out["_truncated"] = sorted(truncated)
    return out


def _workspace_of(graph: Any) -> str:
    raw = getattr(graph, "_raw_workspace", None)
    if callable(raw):
        try:
            return str(raw())
        except Exception:  # noqa: BLE001 - fall back to the plain attribute
            pass
    return str(getattr(graph, "workspace", "") or "base").strip() or "base"


def _label_of(graph: Any) -> str:
    """Backtick-safe workspace label for inlining into Cypher."""
    escaped = getattr(graph, "_workspace_label", None)
    if callable(escaped):
        try:
            return str(escaped())
        except Exception:  # noqa: BLE001
            pass
    return _workspace_of(graph).replace("`", "``")


def _node_identity(node: Any) -> str:
    """Nodes carry no ``id`` property — ``entity_id`` is the identity, then element id."""
    props = dict(node)
    return str(props.get("entity_id") or props.get("node_id") or node.element_id)


def _sorted_labels(node: Any, workspace: str) -> list[str]:
    labels = [str(label) for label in node.labels]
    # Typed label first, workspace label last: clients colour by the typed one.
    return sorted(labels, key=lambda label: (label == workspace, label))


def _describe(value: Any, workspace: str) -> Any:
    """Flatten a record value into something a table cell can show."""
    if isinstance(value, _NODE_TYPES):
        labels = [label for label in _sorted_labels(value, workspace) if label != workspace]
        return f"({_node_identity(value)}:{labels[0] if labels else 'Node'})"
    if isinstance(value, _REL_TYPES):
        start = _node_identity(value.start_node) if value.start_node is not None else "?"
        end = _node_identity(value.end_node) if value.end_node is not None else "?"
        return f"({start})-[:{value.type}]->({end})"
    if isinstance(value, _PATH_TYPES):
        return f"path({len(value.nodes)} nœuds, {len(value.relationships)} relations)"
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_describe(item, workspace) for item in value]
    if isinstance(value, dict):
        return {str(key): _describe(item, workspace) for key, item in value.items()}
    safe, _cut = _truncate(_json_safe(value))
    return safe


class _GraphCollector:
    """Deduplicating sink for the nodes and relationships seen in a result."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def add_node(self, node: Any) -> str:
        node_id = _node_identity(node)
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "labels": _sorted_labels(node, self.workspace),
                "properties": _properties(dict(node)),
            }
        return node_id

    def add_edge(self, rel: Any) -> None:
        key = str(rel.element_id)
        if key in self.edges:
            return
        # Endpoints are hydrated as (possibly property-less) stubs. Adding them keeps
        # a relationship-only query from returning edges that point at nothing.
        source = self.add_node(rel.start_node) if rel.start_node is not None else ""
        target = self.add_node(rel.end_node) if rel.end_node is not None else ""
        self.edges[key] = {
            "id": key,
            "type": str(rel.type),
            "source": source,
            "target": target,
            "properties": _properties(dict(rel)),
        }

    def harvest(self, value: Any) -> None:
        if isinstance(value, _NODE_TYPES):
            self.add_node(value)
        elif isinstance(value, _REL_TYPES):
            self.add_edge(value)
        elif isinstance(value, _PATH_TYPES):
            for node in value.nodes:
                self.add_node(node)
            for rel in value.relationships:
                self.add_edge(rel)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                self.harvest(item)
        elif isinstance(value, dict):
            for item in value.values():
                self.harvest(item)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _require_neo4j_graph(request: Any) -> Any:
    """Layer (a): only ``Neo4JStorage`` exposes a session; the default backend does not."""
    rag = getattr(request.app.state, "rag", None)
    graph = getattr(rag, "graph", None)
    backend = type(graph).__name__ if graph is not None else "aucun"
    if graph is None or backend != "Neo4JStorage" or not hasattr(graph, "_session"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=BACKEND_UNAVAILABLE.format(backend=backend),
        )
    if getattr(graph, "_driver", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le pilote Neo4j n'est pas initialisé. Réessayez après le démarrage.",
        )
    return graph


async def _read_records(graph: Any, query: str, params: dict[str, Any], cap: int) -> Any:
    """Run ``query`` in a READ transaction, returning ``(columns, records, truncated)``.

    Layer (c): ``default_access_mode="READ"`` is the enforcement that matters — the
    server refuses a write in this transaction whatever the scanner concluded. The
    transaction is never committed, so it rolls back on exit.
    """
    columns: list[str] = []
    records: list[dict[str, Any]] = []
    truncated = False
    async with graph._session(default_access_mode="READ") as session:
        tx = await session.begin_transaction(timeout=QUERY_TIMEOUT_SECONDS)
        async with tx:
            result = await tx.run(query, params)
            try:
                columns = [str(key) for key in result.keys()]
            except Exception:  # noqa: BLE001 - keys are a convenience, not the payload
                columns = []
            async for record in result:
                if len(records) >= cap:
                    truncated = True
                    break
                if not columns:
                    columns = [str(key) for key in record.keys()]
                records.append(dict(record))
            try:
                await result.consume()
            except Exception:  # noqa: BLE001 - discarding a capped stream may fail
                pass
    return columns, records, truncated


async def _graph_response(
    graph: Any,
    query: str,
    params: dict[str, Any],
    cap: int,
    limit_applied: int,
) -> dict[str, Any]:
    """Execute and shape the ``{columns, rows, nodes, edges, stats}`` payload."""
    workspace = _workspace_of(graph)
    started = time.perf_counter()
    columns, records, truncated = await _read_records(graph, query, params, cap)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

    collector = _GraphCollector(workspace)
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for key, value in record.items():
            collector.harvest(value)
            row[str(key)] = _describe(value, workspace)
        rows.append(row)

    return {
        "columns": columns,
        "rows": rows,
        "nodes": list(collector.nodes.values()),
        "edges": list(collector.edges.values()),
        "stats": {
            "records": len(rows),
            "elapsed_ms": elapsed_ms,
            "truncated": truncated,
            "limit_applied": limit_applied,
        },
    }


def _driver_error(exc: Exception) -> Any:
    """Surface the driver's own message so a syntax error is actionable."""
    message = str(getattr(exc, "message", "") or exc) or exc.__class__.__name__
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Erreur Cypher : {message}",
    )


# --------------------------------------------------------------------------- #
# Highlights
# --------------------------------------------------------------------------- #


def _fallback_suggestions() -> list[dict[str, str]]:
    """Generic French cards. An empty home screen is worse than a generic one."""
    return [
        {
            "kind": "entity",
            "title": "Vue d'ensemble",
            "body": "Découvrez les sujets principaux du corpus indexé.",
            "prompt": "Quels sont les principaux sujets abordés dans les documents indexés ?",
        },
        {
            "kind": "schema",
            "title": "Faits marquants",
            "body": "Les informations les plus souvent répétées.",
            "prompt": "Quels sont les faits les plus importants à retenir du corpus ?",
        },
        {
            "kind": "type",
            "title": "Synthèse",
            "body": "Un résumé des documents disponibles.",
            "prompt": "Peux-tu résumer le contenu des documents disponibles ?",
        },
    ]


def _join_fr(names: list[str]) -> str:
    """``a``, ``a et b``, ``a, b et c`` — a French enumeration."""
    cleaned = [name for name in names if name]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return f"{', '.join(cleaned[:-1])} et {cleaned[-1]}"


async def _scalar_rows(graph: Any, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    _columns, records, _truncated = await _read_records(graph, query, params, HIGHLIGHT_TOP_K)
    return records


def create_cypher_router(api_key: Optional[str] = None) -> Any:
    if APIRouter is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import get_combined_auth_dependency

    router = APIRouter(tags=["graph"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("/graph/cypher", dependencies=[Depends(combined_auth)])
    async def run_cypher(request: Request, body: CypherRequest):
        graph = _require_neo4j_graph(request)

        raw_query = (body.query or "").strip()
        if not raw_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La requête Cypher est vide.",
            )

        # Layer (b): scan the statement with literals and comments blanked out.
        sanitized = strip_literals_and_comments(raw_query)
        violation = find_write_violation(sanitized)
        if violation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Requête refusée : la console est en lecture seule "
                    f"(élément interdit : {violation})."
                ),
            )

        cap = min(max(int(body.limit or DEFAULT_LIMIT), 1), MAX_LIMIT)
        query, _injected = apply_limit(raw_query, sanitized, cap)
        params = body.params if isinstance(body.params, dict) else {}

        try:
            return await _graph_response(graph, query, params, cap, cap)
        except HTTPException:
            raise
        except _DRIVER_ERRORS as exc:
            raise _driver_error(exc) from exc
        except Exception as exc:  # noqa: BLE001 - a console must not answer 500 blindly
            logger.warning("Cypher console query failed: %s", exc)
            raise _driver_error(exc) from exc

    @router.get("/graph/schema", dependencies=[Depends(combined_auth)])
    async def graph_schema(request: Request):
        graph = _require_neo4j_graph(request)
        workspace = _workspace_of(graph)
        label = _label_of(graph)

        cached = _schema_cache.get(workspace)
        now = time.monotonic()
        if cached and cached[0] > now:
            return cached[1]

        # Derived from the workspace, never from db.labels() / db.propertyKeys():
        # those are database-wide and would surface the other engine's schema.
        label_query = (
            f"MATCH (n:`{label}`) UNWIND labels(n) AS l "
            "WITH l WHERE l <> $ws "
            "RETURN l AS label, count(*) AS count ORDER BY count DESC, label ASC"
        )
        rel_query = (
            f"MATCH (:`{label}`)-[r]->(:`{label}`) "
            "RETURN type(r) AS type, count(*) AS count ORDER BY count DESC, type ASC"
        )
        node_key_query = (
            f"MATCH (n:`{label}`) WITH n LIMIT $sample UNWIND keys(n) AS k RETURN DISTINCT k AS key"
        )
        rel_key_query = (
            f"MATCH (:`{label}`)-[r]->(:`{label}`) WITH r LIMIT $sample "
            "UNWIND keys(r) AS k RETURN DISTINCT k AS key"
        )
        count_query = f"MATCH (n:`{label}`) RETURN count(n) AS count"

        try:
            _c, count_rows, _t = await _read_records(graph, count_query, {}, 1)
            _c, label_rows, _t = await _read_records(graph, label_query, {"ws": workspace}, 200)
            _c, rel_rows, _t = await _read_records(graph, rel_query, {}, 200)
            _c, node_keys, _t = await _read_records(
                graph, node_key_query, {"sample": SCHEMA_SAMPLE}, 500
            )
            _c, rel_keys, _t = await _read_records(
                graph, rel_key_query, {"sample": SCHEMA_SAMPLE}, 500
            )
        except _DRIVER_ERRORS as exc:
            raise _driver_error(exc) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph schema introspection failed: %s", exc)
            raise _driver_error(exc) from exc

        labels = [
            {"label": str(row.get("label")), "count": int(row.get("count") or 0)}
            for row in label_rows
        ]
        relationship_types = [
            {"type": str(row.get("type")), "count": int(row.get("count") or 0)} for row in rel_rows
        ]
        keys = {str(row.get("key")) for row in node_keys if row.get("key")}
        keys.update(str(row.get("key")) for row in rel_keys if row.get("key"))
        payload = {
            "workspace": workspace,
            "node_count": int(count_rows[0].get("count") or 0) if count_rows else 0,
            "relationship_count": sum(item["count"] for item in relationship_types),
            "labels": labels,
            "relationship_types": relationship_types,
            "property_keys": sorted(keys),
        }
        _schema_cache[workspace] = (now + SCHEMA_CACHE_TTL_SECONDS, payload)
        return payload

    @router.get("/graph/neighborhood", dependencies=[Depends(combined_auth)])
    async def neighborhood(
        request: Request,
        entity_id: str = Query(..., min_length=1, description="entity_id du nœud central"),
        hops: int = Query(default=1, ge=1, le=3),
        limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ):
        graph = _require_neo4j_graph(request)
        label = _label_of(graph)
        workspace = _workspace_of(graph)

        # ``hops`` is a validated 1..3 integer: Cypher forbids a parameter inside a
        # variable-length pattern, so it is interpolated after the bound check.
        query = (
            f"MATCH (n:`{label}` {{entity_id: $entity_id}}) "
            f"OPTIONAL MATCH p = (n)-[*1..{int(hops)}]-(:`{label}`) "
            "RETURN n AS center, p AS path "
            f"LIMIT {int(limit)}"
        )
        try:
            payload = await _graph_response(graph, query, {"entity_id": entity_id}, limit, limit)
        except _DRIVER_ERRORS as exc:
            raise _driver_error(exc) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neighborhood expansion failed: %s", exc)
            raise _driver_error(exc) from exc

        if not payload["rows"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Aucun nœud avec entity_id « {entity_id} » "
                    f"dans l'espace de travail « {workspace} »."
                ),
            )
        return payload

    @router.get("/graph/highlights", dependencies=[Depends(combined_auth)])
    async def highlights(request: Request):
        try:
            graph = _require_neo4j_graph(request)
            label = _label_of(graph)
        except HTTPException:
            # A missing or non-Neo4j backend still deserves a usable home screen.
            return {"suggestions": _fallback_suggestions()}

        suggestions = _fallback_suggestions()
        by_kind = {item["kind"]: item for item in suggestions}

        entity_query = (
            f"MATCH (e:`{label}`:Entity)-[r:ENTITY_RELATION]-(:`{label}`) "
            "WITH e, count(r) AS degree "
            "RETURN coalesce(e.content, e.entity_id) AS name, degree "
            f"ORDER BY degree DESC LIMIT {HIGHLIGHT_TOP_K}"
        )
        schema_query = (
            f"MATCH (s:`{label}`:Schema) WHERE s.frequency IS NOT NULL "
            "RETURN coalesce(s.content, s.entity_id) AS name, s.frequency AS frequency "
            f"ORDER BY frequency DESC LIMIT {HIGHLIGHT_TOP_K}"
        )
        type_query = (
            f"MATCH (t:`{label}`:Type) "
            f"OPTIONAL MATCH (:`{label}`:Entity)-[rel:ENTITY_TO_TYPE]->(t) "
            "WITH t, count(rel) AS usage "
            "RETURN coalesce(t.content, t.entity_id) AS name, usage "
            f"ORDER BY usage DESC LIMIT {HIGHLIGHT_TOP_K}"
        )

        try:
            rows = await _scalar_rows(graph, entity_query, {})
            names = [str(row.get("name") or "").strip() for row in rows]
            names = [name for name in names if name][:3]
            if names:
                by_kind["entity"] = {
                    "kind": "entity",
                    "title": "Entités centrales",
                    "body": "Les entités les plus connectées du graphe de mémoire.",
                    "prompt": f"Quels sont les liens entre {_join_fr(names)} ?",
                }
        except Exception as exc:  # noqa: BLE001 - a generic card beats an error
            logger.debug("Highlight (entity) unavailable: %s", exc)

        try:
            rows = await _scalar_rows(graph, schema_query, {})
            names = [str(row.get("name") or "").strip() for row in rows]
            names = [name for name in names if name][:2]
            if names:
                by_kind["schema"] = {
                    "kind": "schema",
                    "title": "Schémas dominants",
                    "body": "Les schémas de faits les plus fréquents du corpus.",
                    "prompt": f"Que disent les documents à propos de {_join_fr(names)} ?",
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Highlight (schema) unavailable: %s", exc)

        try:
            rows = await _scalar_rows(graph, type_query, {})
            names = [str(row.get("name") or "").strip() for row in rows]
            names = [name for name in names if name][:2]
            if names:
                by_kind["type"] = {
                    "kind": "type",
                    "title": "Types principaux",
                    "body": "Les catégories d'entités les plus représentées.",
                    "prompt": (
                        f"Quels sont les principaux éléments de type {_join_fr(names)} "
                        "mentionnés dans les documents ?"
                    ),
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Highlight (type) unavailable: %s", exc)

        return {"suggestions": [by_kind["entity"], by_kind["schema"], by_kind["type"]]}

    return router
