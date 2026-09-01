# MemGraphRAG API Server

Production FastAPI server for the MemGraphRAG memory-based GraphRAG engine.

## Features

- Three-layer memory indexing (schema / fact / passage) with ontology extraction, frequency filter, and conflict-aware construction
- Hierarchical PPR retrieval with schema linking (`SCHEMA_TOP_K`, `SCHEMA_NODE_WEIGHT`; `PPR_ENGINE=igraph|neo4j_gds`)
- Pluggable storage via `MEMGRAPHRAG_*_STORAGE` (JSON/nano/igraph defaults or PostgreSQL + Neo4j)
- OpenAI-compatible LLM and embedding bindings
- Document upload/scan/delete/clear with legacy + Docling parsers and F/R/P chunkers
- Corpus-accumulating memory: each ingest merges OpenIE from all PROCESSED docs; delete rebuilds from cached OpenIE (no LLM re-run)
- Checkpointed extraction (`OPENIE_CHECKPOINT_SIZE`), per-chunk and per-batch retries, JSON repair, bounded embedding requests — see [IngestionResilience.md](IngestionResilience.md)
- Corpus language pin (`MEMGRAPHRAG_LANGUAGE`) and accent/case-folded entity keys, so one concept is one node
- Neo4j workspace ownership guard (`mgr_owned`) and UNWIND-batched graph writes
- JWT and/or API-key authentication
- Ollama-compatible `/api` emulation

## Quick start (local, file-backed)

```bash
uv sync --extra api
cp env.example .env
# set LLM_BINDING_API_KEY / EMBEDDING_BINDING_API_KEY
uv run memgraphrag-server
```

Open `http://localhost:9621/docs`.

## Compose stack

```bash
cp env.example .env
docker compose up -d --build
# optional Docling:
docker compose --profile docling up -d
```

Compose wires `PGKVStorage` / `PGVectorStorage` / `PGDocStatusStorage` / `Neo4JStorage`.

## LLM bindings (including self-hosted vLLM)

MemGraphRAG talks to any OpenAI-compatible chat API via `LLM_BINDING_*`. Pointing it
at a self-hosted vLLM (say `--served-model-name mistral` on port `8001`) needs
exactly four variables:

```bash
LLM_BINDING=openai
LLM_BINDING_HOST=http://localhost:8001/v1
LLM_BINDING_API_KEY=EMPTY
LLM_MODEL=mistral
```

`VLLM_*` / `HF_TOKEN` style settings belong to the vLLM container's own compose
file: this repository ships no vLLM service and reads none of those names.

Keep a **separate** embedding endpoint (`EMBEDDING_BINDING_*`); a chat-only vLLM
service cannot serve embeddings.

The request timeout is not configurable: `memgraphrag/llm/openai_compatible.py`
hard-codes `httpx.Timeout(150.0, connect=30.0)`. A slow local model needs a code
change, not an `LLM_TIMEOUT` variable.

## Auth

| Mode | Env |
|------|-----|
| API key | `MEMGRAPHRAG_API_KEY` → header `X-API-Key` |
| JWT | `AUTH_ACCOUNTS=user:pass` + `TOKEN_SECRET` → `POST /login` then `Authorization: Bearer …` |
| Open (dev) | neither set — safe only on loopback, and only when `REQUIRE_AUTH=false` |

`TOKEN_SECRET` is mandatory whenever `AUTH_ACCOUNTS` is set: the fallback secret is
published in this repository, so tokens signed with it are forgeable. `JWT_ALGORITHM=none`
is rejected outright.

### Hardening currently enforced

| Control | Behaviour |
|---------|-----------|
| `WHITELIST_PATHS` | Defaults to `/health,/docs,/openapi.json` — **`/api/*` is deliberately excluded.** The Ollama router is mounted on `/api`, and `/api/chat`, `/api/generate` reach the billed LLM (`/bypass` skips retrieval entirely). The whitelist short-circuits before any token or key check, so whitelisting `/api/*` publishes an open LLM proxy. |
| `REQUIRE_AUTH=true` | Fail-closed switch: unauthenticated requests get 403 even when neither `AUTH_ACCOUNTS` nor `MEMGRAPHRAG_API_KEY` resolved — a `.env` that failed to load degrades into a locked server instead of an open one. |
| `CORS_ORIGINS=*` | Credentialed cross-origin requests are disabled while the origin list is `*`. Name explicit origins to allow cookies / `Authorization` from a browser. |
| `LOGIN_MAX_ATTEMPTS` / `LOGIN_WINDOW_SECONDS` | `POST /login` is rate-limited per client IP (default 10 attempts / 60 s, then 429). |
| `MAX_UPLOAD_SIZE` | Per-file upload cap, default 100 MiB; larger bodies are rejected with 413. |
| Secrets in Compose | `POSTGRES_PASSWORD` and `NEO4J_PASSWORD` have no defaults — `docker compose up` fails unless they are set. Auth variables are passed through to the container. |
| `WORKERS` | Startup **refuses** `WORKERS > 1` while any file-backed backend is selected (see below). |
| API key / password comparison | Constant-time (`hmac.compare_digest`), so response timing does not leak a prefix. |

## Workers and concurrency

Run one worker. `validate_worker_count` in `memgraphrag/api/config.py` aborts
startup when `WORKERS > 1` and any of `JsonKVStorage`, `NanoVectorDBStorage`,
`IgraphStorage`, `JsonDocStatusStorage` is selected: those backends guard their
files with `asyncio` locks that exist inside a single interpreter, so two
processes sharing `WORKING_DIR` interleave rewrites of the same JSON / GraphML
file. The check runs both in the `memgraphrag-gunicorn` entry point and in
`gunicorn_config.py`, before any worker forks.

Moving to `PGKVStorage` / `PGVectorStorage` / `PGDocStatusStorage` /
`Neo4JStorage` lifts the refusal but not the whole limitation: the ingest
`pipeline_lock` is an `asyncio.Lock` on `app.state`, so the "409 while busy"
guarantee below only holds within the worker that owns the running ingest.
`memgraphrag/storage/shared.py` is likewise per-process, not a cross-worker bus.

## Main endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness — 200 while the process answers, plus `ready` / `retrieval_status`. No `working_dir`/`workspace`: the path is whitelisted. |
| GET | `/health/ready` | Readiness — 503 until retrieval is warmed up (unauthenticated, carries only the bit) |
| GET | `/metrics` | Prometheus exposition (authenticated) |
| POST | `/login` | JWT login |
| POST | `/documents/upload` | Upload file (locks pipeline) |
| POST | `/documents/text` | Insert raw text (locks pipeline) |
| GET | `/documents/` | List doc statuses — paginated (`limit` 1..1000, default 100, `offset`, optional `status`); returns `total`/`returned`/`next_offset` and never the document body |
| GET | `/documents/{doc_id}` | Document detail (chunk ids, paths) |
| DELETE | `/documents/{doc_id}` | Delete one doc + rebuild corpus |
| POST | `/documents/delete` | Batch delete `{doc_ids, delete_file}` |
| POST | `/documents/{doc_id}/requeue` | Reset failed/stuck doc to PENDING |
| DELETE | `/documents/?confirm=true` | Clear all storages (optional `delete_files`) |
| POST | `/documents/scan` | Scan `INPUT_DIR` |
| POST | `/query` | Retrieve + QA → `{response, references}` |
| POST | `/query/data` | Retrieval evidence only (`response`/`references` + docs) |
| POST | `/query/stream` | SSE framing, **not token streaming** — see below |
| GET | `/query/params` | Tunable-parameter registry (bounds + presets) for clients |
| GET | `/documents/{doc_id}/chunks` | Passages a document was split into (paged) |
| GET | `/graphs` | Explore memory graph |
| GET | `/graph/label/list` | List node labels |
| POST | `/graph/cypher` | Run a **read-only** Cypher statement against the memory graph |
| GET | `/graph/schema` | Workspace-scoped labels, relationship types and property keys |
| GET | `/graph/neighborhood` | N-hop expansion around one node |
| GET | `/graph/highlights` | Corpus-derived suggestions for the UI's empty state |
| GET | `/library/tree` | Browse the on-disk document library (`LIBRARY_ROOT`) |
| GET | `/library/file` | Serve one library file inline |
| GET | `/library/preview` | Per-page extracted text |
| GET | `/library/passages` | Graph passages whose `file_path` matches a library file |
| GET | `/models` | Providers and models selectable per request |
| POST | `/chat/threads` | Create a conversation |
| GET | `/chat/threads` | List conversations (paged, owner-scoped) |
| GET | `/chat/threads/{thread_id}` | One conversation with its messages |
| PATCH | `/chat/threads/{thread_id}` | Rename / retarget a conversation |
| DELETE | `/chat/threads/{thread_id}` | Delete a conversation and its messages |
| POST | `/chat/threads/{thread_id}/messages` | Append a message |
| GET/POST | `/api/*` | Ollama emulation (`/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/version`) |

That is the whole surface: 40 operations in total.

`POST /graph/cypher` is read-only and enforced in three layers: the backend must be
`Neo4JStorage`, write keywords are rejected after string literals and comments are
stripped, and execution runs in a `default_access_mode="READ"` transaction — which
Neo4j itself refuses to write from. A `LIMIT` is injected when the statement has
none. The graph is shared with other tools, so this is not optional hardening.

Per-request `provider` / `model` on `/query` route completions to any
OpenAI-compatible endpoint (Together AI, Ollama, vLLM). **Embeddings are never
routed**: the corpus is indexed with one model at one dimension, and sending query
embeddings elsewhere returns vectors from a different space.

The `/chat/*` routes are backed by a **separate** application database
(`APP_DATABASE_URL`, the `postgres-app` compose service), never by a RAG storage
backend. With that variable unset they answer **503** and everything else keeps
working; the web UI then holds conversations in the browser tab only.

Admin mutate endpoints return **409** while `/health` reports `pipeline_busy=true`.
Clear-all requires `confirm=true` (400 otherwise). Per-doc delete needs content-hash
`chunk_ids` on the status record (re-ingest legacy docs, or use clear-all).

The busy check is read-then-acquire, so two requests arriving at the same instant
can both pass the test; the loser then waits on the lock instead of receiving 409.
Treat 409 as the common case, not as a guarantee.

### `/query/stream` is chunked delivery, not streaming

The handler awaits the complete answer, then emits three SSE frames:
`{"references": …}`, `{"response": …}`, `data: [DONE]`. `QueryParam.stream` is set
on the request object but the engine never reads it (`grep -c stream
memgraphrag/core.py` → 0), and no token-level path exists between the LLM binding
and the response. Time-to-first-byte therefore equals total query latency — the
endpoint is SSE-shaped for LightRAG client compatibility, nothing more.

## Query modes (MemGraphRAG-native)

Unlike LightRAG's local/global/hybrid modes:

| Mode | Behavior |
|------|----------|
| `ppr` (default) | Fact linking + PPR + QA |
| `naive` | Dense passage retrieval only |
| `context` | Retrieval without QA |
| `bypass` | Direct LLM |

Ollama chat messages may prefix `/naive`, `/context`, or `/bypass`.

## Structured server logging

Ingest and query flows emit framework-aligned lines for the MemGraphRAG engine
(`[INDEX]` / `[RETRIEVE]` / `[STAGE]` / `[LLM]` / `[EMBED]`) plus `[MAIN]` /
`[STEP]` for HTTP and file-pipeline boundaries (see [Logging.md](Logging.md)).
Filter with:

```bash
docker compose logs memgraphrag 2>&1 | grep -E '\[(INDEX|RETRIEVE|STAGE|LLM|EMBED)\]'
docker compose logs memgraphrag 2>&1 | grep '\[LLM\]'   # agent / chat calls
docker compose logs memgraphrag 2>&1 | grep -E '\[(MAIN|STEP)\]'  # API / parse / chunk
```

## Langfuse observability

Optional retrieval tracing: set `LANGFUSE_ENABLE_TRACE=true` plus `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` (or `LANGFUSE_HOST`). Each `/query` emits nested spans for fact linking, PPR, dense retrieval, and RAG generation. Details: [LangfuseObservability.md](LangfuseObservability.md).

## Storage selection

```bash
MEMGRAPHRAG_KV_STORAGE=JsonKVStorage|PGKVStorage
MEMGRAPHRAG_VECTOR_STORAGE=NanoVectorDBStorage|PGVectorStorage
MEMGRAPHRAG_GRAPH_STORAGE=IgraphStorage|Neo4JStorage
MEMGRAPHRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage|PGDocStatusStorage
```

Always call `await rag.initialize_storages()` after constructing `MemGraphRAG` in Python.

These variables are read by the API server (`memgraphrag/api/config.py`). The
`MemGraphRAG` constructor itself defaults to the file backends whatever the
environment says — an embedded script must pass the backends explicitly, e.g.
`MemGraphRAG(..., **resolve_storage_backends())`.

### Sharing a Neo4j server

`Neo4JStorage` labels every node with the workspace name — the same convention
LightRAG uses — so two engines pointed at one workspace would mix their graphs in
every traversal, and a naive `clear()` would delete the other engine's nodes.
Three rules apply:

- every node MemGraphRAG writes carries `mgr_owned = true`; `clear()` matches only
  those, so it can never destroy a foreign graph;
- startup refuses a workspace that already holds unmarked nodes, with their count
  in the error; set `MEMGRAPHRAG_ALLOW_SHARED_NEO4J_WORKSPACE=true` only to share
  one knowingly;
- an empty `WORKSPACE` falls back to `base` — name the workspace explicitly on a
  shared server.

Graph installs run inside `graph.batch()`: `Neo4JStorage` buffers the writes and
flushes them as `UNWIND` statements of 1 000 rows grouped by label and
relationship type, so an install costs a handful of statements instead of two
round trips per node and three per edge. Reads issued inside a batch flush first.

## Clients (CLI + Streamlit)

Optional lightweight clients talk to this API over HTTP (not embedded in the service image). Install with `uv sync --extra client`, then:

```bash
uv run memgraphrag-cli health
uv run streamlit run memgraphrag/client/app.py
```

Full coverage matrix, optimizer notes, and UI screenshot: [Clients.md](Clients.md) / [images/memgraphrag_webui.png](images/memgraphrag_webui.png).

## See also

- [developer_api_guide.md](developer_api_guide.md) — curl examples and `{response, references}` contract
- [Clients.md](Clients.md)
- [DockerDeployment.md](DockerDeployment.md)
- [FileProcessingPipeline.md](FileProcessingPipeline.md)
- [Logging.md](Logging.md)
- [LangfuseObservability.md](LangfuseObservability.md)
- [ProgramingWithCore.md](ProgramingWithCore.md)
