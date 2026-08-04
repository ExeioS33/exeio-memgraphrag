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
| PPR | `PPR_ENGINE=igraph` (default, paper-exact) or `neo4j_gds` |
| File processing | Parser registry (`legacy`, `docling`) + chunkers F / R / P |
| Orchestration | Docker Compose: `memgraphrag` + `postgres` + `neo4j` (+ optional `docling` profile) |
| Observability | Optional Langfuse (`LANGFUSE_*`) on the retrieval / query path |
| Tests | pytest + pytest-asyncio, `./scripts/test.sh` |

## Confirmed Architecture Decisions

- Local git first; GitHub publication deferred.
- OpenAI-compatible bindings only for POC (no local torch/HF embedders in the service image).
- Storage selected by `MEMGRAPHRAG_{KV,VECTOR,GRAPH,DOC_STATUS}_STORAGE`.
- PPR hybrid: igraph default + Neo4j GDS alternative.
- Full API parity with LightRAG (documents / query / graph / Ollama `/api`); optional Streamlit/CLI clients talk to the API (not embedded in the service image).
- Docling via optional compose profile; VLM ANALYZING stage reserved, not implemented.
- One local commit per advancement; single-line messages; details live in `docs/`.

## Module Layout (`memgraphrag/`)

- **`core.py`**: Engine class `MemGraphRAG` — index_with_memory, retrieve, rag_qa. Always `await rag.initialize_storages()` after construction.
- **`memory.py`**: `ThreeLayerMemory` (schema / fact / passage).
- **`pipeline.py`**: Async ingestion (PENDING → PARSING → PROCESSING → PROCESSED) with memory sub-stage tracking.
- **`base.py`**: Storage ABCs (`BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `DocStatusStorage`).
- **`storage/`**: Registry + factory + backends (`*_impl.py`). Renamed from LightRAG's `kg/` because it holds all storage types.
- **`parser/`**, **`chunker/`**, **`sidecar/`**: File processing (LightRAG-inspired, MemGraphRAG-adapted).
- **`ppr/`**: `IgraphPPREngine`, `Neo4jGDSPPREngine`.
- **`llm/`**, **`openie/`**, **`prompts/`**, **`rerank.py`**: Bindings and extraction.
- **`observability/`**: Optional Langfuse tracing for query/retrieval (`langfuse_trace.py`).
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
- Document admin: `DELETE /documents/{id}`, `POST /documents/delete`, `DELETE /documents/?confirm=true`, `POST /documents/{id}/requeue`. Delete drops exclusive chunks (shared-chunk refcount), then rebuilds memory/graph from remaining OpenIE without conflict LLM. Concurrent ingest/delete share `pipeline_lock` (409 when busy).
- Query params are MemGraphRAG-native (`LINKING_TOP_K`, `PASSAGE_NODE_WEIGHT`, `DAMPING`, `FACT_SIMILARITY_THRESHOLD`, `SKIP_FACT_RERANK`, `SCHEMA_TOP_K`, `SCHEMA_NODE_WEIGHT`, `PPR_ENGINE`).
- Index-time ontology/conflict knobs: `ONTOLOGY_BATCH_SIZE`, `ONTOLOGY_MIN_FREQUENCY`, `CONFLICT_ENABLED`, `CONFLICT_MAX_GROUPS`.
- Ollama prefixes: `/naive` dense passages; default PPR+QA; `/context` passages only; `/bypass` direct LLM.

## Development Workflow

```bash
uv sync --extra api --extra pytest --extra client
cp env.example .env
./scripts/test.sh tests
uv run memgraphrag-server
uv run memgraphrag-cli health
uv run streamlit run memgraphrag/client/app.py
docker compose up -d
docker compose --profile docling up -d   # optional Docling
```

## Testing

- Layout mirrors the package: `tests/storage/`, `tests/parser/`, `tests/api/`, …
- Unit tests are offline by default; integration tests require `--run-integration`.
- Cover edge cases: empty corpus, corrupt files, auth failures, query-before-ready, storage retry, Docling timeouts, mid-pipeline resume.

## Commits

- One commit per advancement.
- Single-line subject; details in `docs/`.
- English only for code, commits, logs, and documentation.
