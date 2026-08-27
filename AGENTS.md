# Repository Guidelines

## Project Overview

MemGraphRAG is an industrialized memory-based GraphRAG API server inspired by [Wu et al., arXiv:2606.00610](https://arxiv.org/abs/2606.00610); ownership of this repo remains EXEIO. The research engine builds a three-layer memory (schema / fact / passage), detects and resolves conflicts, then retrieves with embedding similarity and Personalized PageRank (PPR).

This repository packages that engine as a production FastAPI service inspired by LightRAG patterns (pluggable storage, env-driven bindings, parser/chunker engines, Docker Compose) while remaining MemGraphRAG-native in its data model and retrieval semantics.

## Tech Stack

| Area | Choice |
|------|--------|
| Language / packaging | Python >= 3.10, **uv**, exact-pinned direct deps in `pyproject.toml` + `uv.lock`, extras `[api]` / `[pytest]` / `[client]` / `[test]` |
| Container image | Compose build tags **`exeio-memgraphrag:<MEMGRAPHRAG_VERSION>`** and `:latest` |
| API | FastAPI + Uvicorn / Gunicorn |
| Clients | `memgraphrag-cli` (Typer+Rich) + Streamlit playground (`memgraphrag/client/`) — see `docs/Clients.md` and screenshot `docs/images/memgraphrag_webui.png` |
| Auth | JWT (`AUTH_ACCOUNTS`) and/or API key (`MEMGRAPHRAG_API_KEY`) |
| LLM / embeddings | OpenAI-compatible bindings only (`LLM_*`, `EMBEDDING_*`) |
| Vector / KV / doc-status | PostgreSQL + pgvector (`PG*Storage`) or JSON / nano-vectordb defaults |
| Graph | Neo4j 5 + GDS plugin (`Neo4JStorage`) or igraph GraphML default |
| PPR | `PPR_ENGINE=igraph` (default) or `neo4j_gds` — see "Divergences from the paper" |
| File processing | Parser registry (`legacy`, `docling`) + chunkers F / R / P |
| Orchestration | Docker Compose: `memgraphrag` + `postgres` + `neo4j` (+ optional `docling` profile) |
| Observability | Optional Langfuse (`LANGFUSE_*`) on the retrieval / query path |
| Tests | pytest + pytest-asyncio, `./scripts/test.sh` |

## Confirmed Architecture Decisions

- Local git first; GitHub publication deferred.
- OpenAI-compatible bindings only for POC (no local torch/HF embedders in the service image).
- Storage selected by `MEMGRAPHRAG_{KV,VECTOR,GRAPH,DOC_STATUS}_STORAGE`.
- PPR hybrid: igraph default + Neo4j GDS alternative.
- LightRAG-*shaped* API, not a drop-in replacement: the same four surfaces (documents / query / graph / Ollama `/api`) with 23 operations against LightRAG's ~47, and differing response shapes on several of them. A LightRAG client can be pointed at this server for the common query/ingest calls, but expect to adapt; treat parity as partial and unverified per route.
- Optional Streamlit/CLI clients talk to the API (not embedded in the service image).
- Docling via optional compose profile; VLM ANALYZING stage reserved, not implemented.
- One local commit per advancement; single-line messages; details live in `docs/`.

## Module Layout (`memgraphrag/`)

- **`core.py`**: Engine class `MemGraphRAG` — index_with_memory, retrieve, rag_qa. Always `await rag.initialize_storages()` after construction.
- **`memory.py`**: `ThreeLayerMemory` (schema / fact / passage).
- **`pipeline.py`**: Async ingestion (PENDING → PARSING → PROCESSING → PROCESSED). It also writes seven memory sub-stage labels, but they are emitted in one tight loop *before* the corresponding work runs, so a status poll shows the last label (`graph_install`) for essentially the whole run. Do not treat the sub-stage field as progress.
- **`base.py`**: Storage ABCs (`BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `DocStatusStorage`).
- **`storage/`**: Registry + factory + backends (`*_impl.py`). Renamed from LightRAG's `kg/` because it holds all storage types.
- **`parser/`**, **`chunker/`**, **`sidecar/`**: File processing (LightRAG-inspired, MemGraphRAG-adapted).
- **`ppr/`**: `IgraphPPREngine`, `Neo4jGDSPPREngine`.
- **`llm/`**, **`openie/`**, **`prompts/`**, **`rerank.py`**: Bindings and extraction.
- **`observability/`**: Optional Langfuse tracing for query/retrieval (`langfuse_trace.py`).
- **`evaluation/`**: Offline scoring harness — metrics, dataset loaders, LLM judge, multi-run campaigns, golden-set gate. Every definition is frozen in `docs/Evaluation.md`; change a metric there and in code together, or a golden set stops being comparable.
- **`client/`**: HTTP client, Typer CLI (`memgraphrag-cli`), Streamlit playground, hybrid optimizer.
- **`api/`**: FastAPI app (`server.py`), `config.py`, `auth.py`, `dependencies.py`, `routers/`.

## Naming Conventions

- Modules never repeat their parent package name (`api/server.py`, not `api/memgraphrag_server.py`).
- Subsystem packages are singular (`parser`, `storage`, `openie`); plural only for `routers/`.
- Suffixes: `_impl.py` for storage backends, `_engine.py` for PPR engines (also avoids shadowing `neo4j` / `igraph`).
- Public API re-exported from `memgraphrag/__init__.py`. Sync `insert`/`query` with async `ainsert`/`aquery`.
- Storage class names match LightRAG env values (`PGKVStorage`, `Neo4JStorage`).
- Project env vars use `MEMGRAPHRAG_` prefix; infrastructure/binding vars stay unprefixed (`POSTGRES_*`, `NEO4J_*`, `LLM_*`, `DOCLING_*`, `CHUNK_*`).
- Provenance: adapted modules cite their LightRAG or research-repo source in the module docstring.

## Adaptation Principles

- Chunks become `PassageNode`s at the engine boundary; graph is the typed memory graph, not a flat entity/relation KG.
- PROCESSING = `openie → memory_build → schema_extraction → ontology_filter → conflict_detection → conflict_resolution → graph_install` (ontology + conflict stages are implemented; disable conflicts with `CONFLICT_ENABLED=false`).
- Ingest accumulates corpus memory: OpenIE is keyed by content-hash `chunk-…` ids stored on doc_status; each `ainsert` rebuilds memory from all PROCESSED docs' cached OpenIE (new chunks only hit the LLM).
- Document admin: `DELETE /documents/{id}`, `POST /documents/delete`, `DELETE /documents/?confirm=true`, `POST /documents/{id}/requeue`. Delete drops exclusive chunks (shared-chunk refcount), then rebuilds memory/graph from remaining OpenIE without conflict LLM. Concurrent ingest/delete share `pipeline_lock`; the busy check is read-then-acquire, so a request that loses that race blocks on the lock instead of returning 409. 409 is the usual answer, not a guarantee — and `pipeline_lock` is per-process, so it means nothing across workers.
- Single worker only: startup refuses `WORKERS > 1` while a file-backed backend (`JsonKVStorage` / `NanoVectorDBStorage` / `IgraphStorage` / `JsonDocStatusStorage`) is selected, because their locks are `asyncio` locks inside one process and two workers on one `WORKING_DIR` corrupt the JSON/GraphML files. Shared-database backends lift the refusal but not the per-process ingest lock.
- Query params are MemGraphRAG-native (`LINKING_TOP_K`, `PASSAGE_NODE_WEIGHT`, `DAMPING`, `FACT_SIMILARITY_THRESHOLD`, `SKIP_FACT_RERANK`, `SCHEMA_TOP_K`, `SCHEMA_NODE_WEIGHT`, `PPR_ENGINE`).
- Index-time ontology/conflict knobs: `ONTOLOGY_BATCH_SIZE`, `ONTOLOGY_MIN_FREQUENCY`, `CONFLICT_ENABLED`, `CONFLICT_MAX_GROUPS`.
- Ollama prefixes: `/naive` dense passages; default PPR+QA; `/context` passages only; `/bypass` direct LLM.

## Development Workflow

```bash
uv sync --extra api --extra pytest --extra client
cp env.example .env          # env.example is the only template in the repo
./scripts/test.sh tests
uv run memgraphrag-server
uv run memgraphrag-cli health
uv run streamlit run memgraphrag/client/app.py
docker compose up -d
docker compose --profile docling up -d   # optional Docling
```

`env.example` documents only variables the code actually reads; a setting that is
not implemented is labelled as such rather than listed as a knob. Before adding a
row to it, prove the read: `grep -rn "NAME" memgraphrag --include='*.py'`.

Environment loading is explicit: no module under `memgraphrag/api/` calls
`load_dotenv` at import; the entry points (`server.main`, `gunicorn_runner.main`,
`gunicorn_config.py`) call `memgraphrag.api.config.load_env_file()` instead.
Importing the API package used to inject the developer's `.env` — real provider
keys included — into every process, which turned `pytest --run-integration` into a
false green (79 passed / 0 skipped instead of skipping for want of credentials).
`tests/api/test_env_loading.py` pins this per module; keep new API modules out of
`load_dotenv`.

## Divergences from the paper

Say what is implemented, not what the paper describes:

- The igraph PPR engine is **not** paper-exact, and no document may call it that.
  Seeding and scoring are simplified adaptations, several equations of the paper
  have no counterpart in `core.py`, and no result has ever been reproduced against
  the published numbers. Describe the behaviour a test pins, not the equation a
  paper prescribes.
- The VLM ANALYZING stage is reserved, not implemented.
- `memgraphrag/retrieval.py` (`RetrievalStateManager`) and
  `memgraphrag/storage/shared.py` are not on any request path; they are scaffolding
  for a future incremental-refresh mode and are imported only by tests.

## Security posture (enforced today)

- Whitelist defaults to `/health,/docs,/openapi.json`; `/api/*` must never be added
  — it fronts the billed LLM and the whitelist short-circuits every auth check.
- `CORS_ORIGINS=*` disables credentialed cross-origin requests.
- `REQUIRE_AUTH=true` fails closed when no credential resolved.
- `TOKEN_SECRET` is mandatory with `AUTH_ACCOUNTS`; `JWT_ALGORITHM=none` is rejected.
- `POST /login` is rate-limited per IP (`LOGIN_MAX_ATTEMPTS` / `LOGIN_WINDOW_SECONDS`).
- Uploads are capped by `MAX_UPLOAD_SIZE` (413 beyond it).
- Compose has no default `POSTGRES_PASSWORD` / `NEO4J_PASSWORD` and passes the auth
  variables through to the container.
- API keys and passwords are compared with `hmac.compare_digest`.

## Testing

- Layout mirrors the package: `tests/storage/`, `tests/parser/`, `tests/api/`, …
- Unit tests are offline by default; integration tests require `--run-integration`.
- Cover edge cases: empty corpus, corrupt files, auth failures, query-before-ready, storage retry, Docling timeouts, mid-pipeline resume.

## Commits

- One commit per advancement.
- Single-line subject; details in `docs/`.
- English only for code, commits, logs, and documentation.
