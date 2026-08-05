# MemGraphRAG API Server

Production FastAPI server for the MemGraphRAG memory-based GraphRAG engine.

## Features

- Three-layer memory indexing (schema / fact / passage) with ontology extraction, frequency filter, and conflict-aware construction
- Hierarchical PPR retrieval with schema linking (`SCHEMA_TOP_K`, `SCHEMA_NODE_WEIGHT`; `PPR_ENGINE=igraph|neo4j_gds`)
- Pluggable storage via `MEMGRAPHRAG_*_STORAGE` (JSON/nano/igraph defaults or PostgreSQL + Neo4j)
- OpenAI-compatible LLM and embedding bindings
- Document upload/scan/delete/clear with legacy + Docling parsers and F/R/P chunkers
- Corpus-accumulating memory: each ingest merges OpenIE from all PROCESSED docs; delete rebuilds from cached OpenIE (no LLM re-run)
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

MemGraphRAG talks to any OpenAI-compatible chat API via `LLM_BINDING_*`. For a local/remote **vLLM** Mistral service (Compose `vllm-mistal`, host port `8001`, `--served-model-name mistral`):

```bash
# Helper vars (see env.example) — used by your vLLM compose stack
VLLM_HOST=localhost
VLLM_PORT=8001
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_SERVED_MODEL_NAME=mistral
VLLM_API_KEY=EMPTY
HF_TOKEN=...
MISTRAL_MODEL_7B_GPTQ_4=...

# Wire MemGraphRAG to that endpoint
LLM_BINDING=openai
LLM_BINDING_HOST=http://localhost:8001/v1
LLM_BINDING_API_KEY=EMPTY
LLM_MODEL=mistral
```

Keep a **separate** embedding endpoint (`EMBEDDING_BINDING_*`); the Mistral vLLM service is chat/completions only.

## Auth

| Mode | Env |
|------|-----|
| API key | `MEMGRAPHRAG_API_KEY` → header `X-API-Key` |
| JWT | `AUTH_ACCOUNTS=user:pass` + `TOKEN_SECRET` → `POST /login` then `Authorization: Bearer …` |
| Open (dev) | neither set — safe only on loopback |

`WHITELIST_PATHS` defaults include `/health` and `/api/*`.

## Main endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/login` | JWT login |
| POST | `/documents/upload` | Upload file (locks pipeline) |
| POST | `/documents/text` | Insert raw text (locks pipeline) |
| GET | `/documents` | List doc statuses |
| GET | `/documents/{doc_id}` | Document detail (chunk ids, paths) |
| DELETE | `/documents/{doc_id}` | Delete one doc + rebuild corpus |
| POST | `/documents/delete` | Batch delete `{doc_ids, delete_file}` |
| POST | `/documents/{doc_id}/requeue` | Reset failed/stuck doc to PENDING |
| DELETE | `/documents/?confirm=true` | Clear all storages (optional `delete_files`) |
| POST | `/documents/scan` | Scan `INPUT_DIR` |

Admin mutate endpoints return **409** while `/health` reports `pipeline_busy=true`. Clear-all requires `confirm=true` (400 otherwise). Per-doc delete needs content-hash `chunk_ids` on the status record (re-ingest legacy docs, or use clear-all).
| POST | `/query` | Retrieve + QA (structured JSON answer by default) |
| POST | `/query/data` | Retrieval evidence only |
| POST | `/query/stream` | SSE answer stream |

Developer curl cookbook (login, modes, structured response fields, errors): [developer_api_guide.md](developer_api_guide.md).
| GET | `/graphs` | Explore memory graph |
| GET/POST | `/api/*` | Ollama emulation |

## Query modes (MemGraphRAG-native)

Unlike LightRAG's local/global/hybrid modes:

| Mode | Behavior |
|------|----------|
| `ppr` (default) | Fact linking + PPR + QA |
| `naive` | Dense passage retrieval only |
| `context` | Retrieval without QA |
| `bypass` | Direct LLM |

Ollama chat messages may prefix `/naive`, `/context`, or `/bypass`.

### Structured QA output

`POST /query` defaults to `structured_output=true`. The QA system prompt
(`RAG_QA_STRUCTURED_SYSTEM` in `memgraphrag/prompts/templates.py`) asks the LLM
for JSON `{thought, answer, citations, sources, confidence}` and labels each
passage with `Source: <filename>`. The API always returns a `references` array
(`reference_id` + `file_path`) built from retrieved passage sources, plus
`response` (= `answer`) for LightRAG-style clients. Set
`structured_output=false` for the legacy freeform `Thought:` / `Answer:` prompt
(references still included).

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

## Clients (CLI + Streamlit)

Optional lightweight clients talk to this API over HTTP (not embedded in the service image). Install with `uv sync --extra client`, then:

```bash
uv run memgraphrag-cli health
uv run streamlit run memgraphrag/client/app.py
```

Full coverage matrix, optimizer notes, and UI screenshot: [Clients.md](Clients.md) / [images/memgraphrag_webui.png](images/memgraphrag_webui.png).

## See also

- [developer_api_guide.md](developer_api_guide.md)
- [Clients.md](Clients.md)
- [DockerDeployment.md](DockerDeployment.md)
- [FileProcessingPipeline.md](FileProcessingPipeline.md)
- [Logging.md](Logging.md)
- [LangfuseObservability.md](LangfuseObservability.md)
- [ProgramingWithCore.md](ProgramingWithCore.md)
