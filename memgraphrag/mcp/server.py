"""MCP server exposing MemGraphRAG's read surface to third-party clients.

Mounted inside the existing FastAPI app rather than run as its own service, and the
reason is memory, not tidiness: ``prepare_retrieval()`` loads the whole graph — on
the reference corpus 1 715 passages, 21 961 facts and 121 591 edges in the igraph
engine. A second process would pay that footprint and that start-up again to serve
the same bytes. Mounted, MCP reuses an engine that is already warm and adds no
service to compose.

Three traps, all of them fatal if missed:

1. **Starlette does not run a mounted sub-app's lifespan.** The host app has to
   enter ``session_manager.run()`` in its own lifespan or the *first* request dies
   on ``RuntimeError: Task group is not initialized`` — start-up looks fine.
   ``session_manager`` also only exists after ``streamable_http_app()`` is called,
   and it is single-use, so construction order is not free.
2. **DNS-rebinding protection only allows localhost by default.** Behind a real
   hostname the server answers 421 Invalid Host header, which reads like a routing
   fault. ``MCP_ALLOWED_HOSTS`` is matched exactly, so a deployment needs an entry
   for the bare host *and* one for host:port.
3. **Auth is a ``TokenVerifier``**, wired to the API's own credentials — see
   ``memgraphrag.mcp.auth``.

Only read tools are exposed. A third-party client must not be able to ingest into
or modify the graph, so nothing here writes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from memgraphrag.agent.tools import MAX_TOOL_TOP_K, ToolBox
from memgraphrag.base import QueryParam
from memgraphrag.mcp.auth import ApiTokenVerifier

logger = logging.getLogger(__name__)

MOUNT_PATH = "/mcp"
DEFAULT_CYPHER_LIMIT = 200


def mcp_enabled(args: Any | None = None) -> bool:
    """Whether to mount the MCP surface at all."""
    raw = getattr(args, "mcp_enabled", None)
    if raw is None:
        raw = os.getenv("MCP_ENABLED", "false")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def allowed_hosts(args: Any | None = None) -> list[str]:
    """Hosts the MCP transport will answer for.

    Empty means localhost only, which is the SDK's default and the reason a first
    remote call comes back 421. An entry is an exact match, so ``example.com`` does
    not cover ``example.com:9621``; both are usually needed.
    """
    raw = getattr(args, "mcp_allowed_hosts", None)
    if raw is None:
        raw = os.getenv("MCP_ALLOWED_HOSTS", "")
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def build_mcp_server(
    rag: Any, *, auth_handler: Any = None, api_key: str | None = None, args: Any = None
):
    """Build the FastMCP server. Returns ``None`` when the SDK is not installed."""
    try:
        from mcp.server.auth.settings import AuthSettings
        from mcp.server.fastmcp import FastMCP
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("MCP_ENABLED is set but the `mcp` package is not installed")
        return None

    hosts = allowed_hosts(args)
    origins = [f"http://{h}" for h in hosts] + [f"https://{h}" for h in hosts]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts or ["localhost", "127.0.0.1", "localhost:9621", "127.0.0.1:9621"],
        allowed_origins=origins or ["http://localhost:9621", "http://127.0.0.1:9621"],
    )

    # Auth is installed only when the server actually has credentials. Passing a
    # verifier unconditionally would put `RequireAuthMiddleware` in front of the
    # transport, and a server started with no auth at all — the laptop default —
    # would answer 401 on MCP while answering every HTTP route in the clear. One
    # surface refusing what the others allow reads as a bug, not as a policy.
    has_credentials = bool(api_key) or bool(getattr(auth_handler, "accounts", None))
    auth_kwargs: dict[str, Any] = {}
    if has_credentials:
        issuer = os.getenv("MCP_ISSUER_URL") or "http://localhost:9621"
        auth_kwargs = {
            "token_verifier": ApiTokenVerifier(auth_handler=auth_handler, api_key=api_key),
            "auth": AuthSettings(issuer_url=issuer, resource_server_url=issuer),
        }
    else:
        logger.warning(
            "MCP is mounted on a server with no credentials configured; it is as open "
            "as the HTTP API is. Set MEMGRAPHRAG_API_KEY or AUTH_ACCOUNTS before "
            "exposing either beyond localhost."
        )

    server = FastMCP(
        name="memgraphrag",
        instructions=(
            "Search a MemGraphRAG corpus. `retrieve` runs graph retrieval and returns "
            "fenced passages with their source documents; `search_documents` and "
            "`read_document` browse the ingested files; `cypher` runs a read-only "
            "query against the memory graph. Nothing here writes."
        ),
        transport_security=security,
        # The sub-app is mounted at /mcp, so its own path must be the root or the
        # endpoint ends up at /mcp/mcp.
        streamable_http_path="/",
        stateless_http=True,
        **auth_kwargs,
    )

    _register_tools(server, rag)
    return server


def _register_tools(server: Any, rag: Any) -> None:
    """Register the read-only tool surface.

    ``retrieve`` and ``search_documents`` share their bodies with the in-process
    agent loop through :mod:`memgraphrag.agent.tools` — one definition, two
    exposures. ``cypher`` is here and *not* in the agent's tool list: a human-driven
    MCP client typing graph code is a different threat model from a loop whose
    arguments are influenced by the corpus it just read.
    """

    @server.tool(description="Search the corpus and return the passages that answer a question.")
    async def retrieve(query: str, top_k: int | None = None) -> str:
        toolbox = ToolBox(rag, QueryParam())
        arguments: dict[str, Any] = {"query": query}
        if top_k is not None:
            arguments["top_k"] = max(1, min(MAX_TOOL_TOP_K, int(top_k)))
        result = await toolbox.run("retrieve", arguments)
        return result.text

    @server.tool(description="List ingested documents whose path matches a term.")
    async def search_documents(term: str = "") -> str:
        toolbox = ToolBox(rag, QueryParam())
        result = await toolbox.run("search_documents", {"term": term})
        return result.text

    @server.tool(description="Return the graph passages extracted from one document.")
    async def read_document(file_path: str, limit: int = 50) -> str:
        records = await rag._doc_status_all()
        wanted = str(file_path or "").strip()
        if not wanted:
            return "read_document needs a file_path."
        chunk_ids: list[str] = []
        for record in (records or {}).values():
            path = str((record or {}).get("file_path") or "")
            if path == wanted or path.endswith("/" + wanted) or path.endswith("\\" + wanted):
                chunk_ids.extend(str(c) for c in (record or {}).get("chunk_ids") or [])
        if not chunk_ids:
            return f"No ingested document matched {wanted!r}."
        contents = getattr(rag, "_passage_id_to_content", {}) or {}
        blocks = []
        for chunk_id in chunk_ids[: max(1, min(200, int(limit)))]:
            body = contents.get(chunk_id)
            if body:
                blocks.append(f"[{chunk_id}]\n{body}")
        if not blocks:
            return (
                f"{wanted} has {len(chunk_ids)} chunks on record but none are loaded in "
                "the retrieval index; run a query first or check that ingestion finished."
            )
        return "\n\n".join(blocks)

    @server.tool(description="Run a read-only Cypher query against the memory graph.")
    async def cypher(query: str, limit: int = DEFAULT_CYPHER_LIMIT) -> str:
        # Exactly the guards the HTTP console uses, imported rather than restated:
        # a second copy of a security check is a second thing to forget to update.
        from memgraphrag.api.routers.cypher import (
            apply_limit,
            find_write_violation,
            strip_literals_and_comments,
        )

        statement = str(query or "").strip()
        if not statement:
            return "cypher needs a query."
        stripped = strip_literals_and_comments(statement)
        violation = find_write_violation(stripped)
        if violation is not None:
            return f"Refused: this console is read-only and the query uses {violation}."

        graph = getattr(rag, "chunk_entity_relation_graph", None)
        session_factory = getattr(graph, "_session", None)
        if session_factory is None:
            return (
                "Cypher is only available with the Neo4j graph backend "
                "(MEMGRAPHRAG_GRAPH_STORAGE=Neo4JStorage)."
            )
        rewritten, _ = apply_limit(statement, stripped, max(1, min(1000, int(limit))))
        rows: list[str] = []
        async with session_factory(default_access_mode="READ") as session:
            result = await session.run(rewritten)  # type: ignore[union-attr]
            async for record in result:
                rows.append(str(dict(record)))
                if len(rows) >= 1000:
                    break
        if not rows:
            return "The query returned no rows."
        return "\n".join(rows)
