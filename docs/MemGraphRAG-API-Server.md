# MemGraphRAG API Server

Production FastAPI server for the MemGraphRAG memory-based GraphRAG engine.

## Features

- Three-layer memory indexing (schema / fact / passage) with conflict-aware construction
- Personalized PageRank retrieval (`PPR_ENGINE=igraph|neo4j_gds`)
- Pluggable storage via `MEMGRAPHRAG_*_STORAGE` (JSON/nano/igraph defaults or PostgreSQL + Neo4j)
- OpenAI-compatible LLM and embedding bindings
- Document upload/scan with legacy + Docling parsers and F/R/P chunkers
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
| POST | `/documents/upload` | Upload file |
| POST | `/documents/text` | Insert raw text |
| GET | `/documents` | List doc statuses |
| POST | `/documents/scan` | Scan `INPUT_DIR` |
| POST | `/query` | Retrieve + QA |
| POST | `/query/data` | Retrieval evidence only |
| POST | `/query/stream` | SSE answer stream |
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

## Storage selection

```bash
MEMGRAPHRAG_KV_STORAGE=JsonKVStorage|PGKVStorage
MEMGRAPHRAG_VECTOR_STORAGE=NanoVectorDBStorage|PGVectorStorage
MEMGRAPHRAG_GRAPH_STORAGE=IgraphStorage|Neo4JStorage
MEMGRAPHRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage|PGDocStatusStorage
```

Always call `await rag.initialize_storages()` after constructing `MemGraphRAG` in Python.

## See also

- [DockerDeployment.md](DockerDeployment.md)
- [FileProcessingPipeline.md](FileProcessingPipeline.md)
- [ProgramingWithCore.md](ProgramingWithCore.md)
