# MCP server

A read-only MCP surface over the same corpus the web UI queries, so a third-party
client — Claude Desktop, an IDE, another agent — can search it without going
through the HTTP API by hand.

## Turn it on

```bash
MCP_ENABLED=true \
MCP_ALLOWED_HOSTS=rag.example.com,rag.example.com:9621 \
uv run memgraphrag-server
```

The endpoint is `POST /mcp/` on the API's own port, Streamable HTTP transport. In
Compose, `MCP_ENABLED` and `MCP_ALLOWED_HOSTS` are already passed through; nothing
else changes, because there is no new service.

**Why it is mounted rather than run separately.** `prepare_retrieval()` loads the
whole graph into memory — on the reference corpus 1 715 passages, 21 961 facts and
121 591 edges in the igraph engine. A second process would pay that footprint and
that start-up again to serve exactly the same bytes. Mounted, MCP reuses an engine
that is already warm. If you ever do need an independent deployment, the same image
serves it with a different command — but point both processes at the same
PostgreSQL and Neo4j rather than duplicating the index.

## Client configuration

```json
{
  "mcpServers": {
    "memgraphrag": {
      "type": "http",
      "url": "https://rag.example.com/mcp/",
      "headers": { "Authorization": "Bearer <your API key or JWT>" }
    }
  }
}
```

## Exposed tools

All read-only. A third-party client cannot ingest into or modify the graph.

| Tool | Arguments | Returns |
|---|---|---|
| `retrieve` | `query`, optional `top_k` (≤ 30) | Fenced passages with their source documents |
| `search_documents` | `term` | Ingested document paths matching the term |
| `read_document` | `file_path`, `limit` | The graph passages extracted from one document |
| `cypher` | `query`, `limit` | Rows from a read-only Cypher query |

`retrieve` and `search_documents` share their bodies with the in-process agent loop
(`memgraphrag/agent/tools.py`): one definition, two exposures. `cypher` runs behind
the same three guards as the HTTP console — imported from
`memgraphrag/api/routers/cypher.py` rather than restated, because a second copy of a
security check is a second thing to forget to update.

Note which tool the **agent loop** does not get: `cypher`. A retrieval tool takes a
search string, so a poisoned passage that steers it stays bounded by what the corpus
can answer. A Cypher tool takes graph code, and a passage that steers *that* is not
bounded. Exposing it to a human-driven MCP client is a different threat model from
exposing it to a loop that reads the corpus.

## Three traps

Each of these looks fine at start-up and fails on the first call.

**1. The lifespan.** Starlette does not run a mounted sub-app's lifespan, so the
host app enters `session_manager.run()` in its own — see `create_app` in
`memgraphrag/api/server.py`. Without it the server boots cleanly and the first
request dies on `RuntimeError: Task group is not initialized`. `session_manager`
also only exists *after* `streamable_http_app()` is called, and it is single-use, so
the construction order in `create_app` is not free. `tests/mcp/test_mcp_mount.py`
exercises a real request rather than an import for exactly this reason.

**2. Allowed hosts.** The transport's DNS-rebinding protection accepts `localhost`
only until told otherwise. Behind a real hostname every call comes back **421
Misdirected Request / Invalid Host header**, which reads like a routing fault.
Entries are exact matches, so `rag.example.com` does not cover
`rag.example.com:9621` — list both. A reverse proxy that forwards the public host
triggers the same refusal.

**3. Authentication.** A `TokenVerifier` sits on the API's own credentials:
`AuthHandler.validate_token` for a JWT, a constant-time comparison for
`MEMGRAPHRAG_API_KEY`. There is one identity system, so one place to revoke a
credential. A missing or wrong token gets 401 with the `WWW-Authenticate` header a
compliant client expects.

On a server started with **no** credentials at all, no auth middleware is installed
and MCP is as open as every HTTP route already is. That is deliberate: one surface
refusing what the others allow reads as a bug rather than as a policy. Set
`MEMGRAPHRAG_API_KEY` or `AUTH_ACCOUNTS` before exposing either beyond localhost.

## Checking a deployment

In this order — each step is one of the traps above:

```bash
# 1. the FIRST request must succeed, not just the boot
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:9621/mcp/ \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'

# 2. remote access: the public Host must not come back 421
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://rag.example.com/mcp/ ...

# 3. auth: no token → 401 with WWW-Authenticate; a valid one → 200
```

Then connect a real client, list the tools, and call `retrieve` on a question whose
answer you know from the corpus. Confirm `cypher` refuses a write
(`MATCH (n) DETACH DELETE n`) exactly as the HTTP route does.

## Variables

| Variable | Meaning |
|---|---|
| `MCP_ENABLED` | Mount the server. Default `false`. |
| `MCP_ALLOWED_HOSTS` | Comma-separated exact host matches. Empty means localhost only. |
| `MCP_ISSUER_URL` | Advertised issuer / resource URL in the OAuth metadata. |
