# Docker Deployment

## Services

| Service | Image / build | Role |
|---------|---------------|------|
| `memgraphrag` | **`exeio-memgraphrag:<version>`** (local `Dockerfile`) | API server on port 9621 |
| `postgres` | `pgvector/pgvector:pg16` | KV, vectors (pgvector), doc-status |
| `neo4j` | `neo4j:5-community` + GDS plugin | Memory graph |
| `docling` (profile) | `ghcr.io/docling-project/docling-serve` | Optional remote parser |

### App image tags

Compose builds and tags the API image as:

- `exeio-memgraphrag:${MEMGRAPHRAG_VERSION:-0.1.0}`
- `exeio-memgraphrag:latest`

```bash
# optional override
export MEMGRAPHRAG_VERSION=0.1.0
docker compose build memgraphrag
docker images 'exeio-memgraphrag*'
```

The Dockerfile also sets OCI labels (`org.opencontainers.image.title=exeio-memgraphrag`, version, source, license).

## Bring up

```bash
cp env.example .env
# set LLM_* / EMBEDDING_* / secrets
docker compose up -d --build
docker compose logs -f memgraphrag
```

`env.example` is the only environment template in the repository (`.env.example`
was a divergent duplicate and has been removed). `POSTGRES_PASSWORD` and
`NEO4J_PASSWORD` have **no defaults**: `docker compose up` fails outright until
both are set in `.env`.

With Docling:

```bash
# in .env:
# DOCLING_ENDPOINT=http://docling:5001
docker compose --profile docling up -d --build
```

## Volumes

- `./data/rag_storage` — working dir / caches
- `./data/inputs` — uploads
- `postgres_data`, `neo4j_data` — named volumes

## Healthchecks

Postgres uses `pg_isready`; Neo4j probes HTTP `:7474`. The app waits for both via `depends_on` conditions.

## Clients against the container

The Streamlit / CLI clients run on the host (or another machine) and call the published API port — they are **not** part of the Compose image. After `docker compose up`:

```bash
uv sync --extra client
export MEMGRAPHRAG_SERVER_URL=http://localhost:9621
uv run memgraphrag-cli health
uv run streamlit run memgraphrag/client/app.py
```

See [Clients.md](Clients.md) and the playground screenshot [images/memgraphrag_webui.png](images/memgraphrag_webui.png).

## Workers

The compose service pins `WORKERS=1`. The image runs a single process on purpose:

- With the file-backed defaults, startup **refuses** `WORKERS > 1` outright —
  those backends lock with `asyncio` locks inside one interpreter, so two
  processes sharing `/app/data/rag_storage` would interleave rewrites of the same
  JSON / GraphML file.
- The compose stack uses Postgres + Neo4j, which lifts that refusal, but the
  ingest `pipeline_lock` still lives on `app.state` in one process. Extra workers
  would silently break the "409 while a pipeline is busy" behaviour. Scale by
  running separate workspaces or separate stacks, not by raising `WORKERS`.

## Security notes

- Bind `HOST=0.0.0.0` only with `MEMGRAPHRAG_API_KEY` or `AUTH_ACCOUNTS`, and
  prefer `REQUIRE_AUTH=true` so a `.env` that fails to load produces 403 instead
  of an open server.
- Compose passes `MEMGRAPHRAG_API_KEY`, `AUTH_ACCOUNTS`, `TOKEN_SECRET`,
  `REQUIRE_AUTH`, `CORS_ORIGINS` and `WHITELIST_PATHS` into the container. They
  used not to be passed at all, which published :9621 with no credential no matter
  what the host `.env` said.
- Never put `/api/*` in `WHITELIST_PATHS`: the Ollama emulation router lives there
  and reaches the billed LLM, and the whitelist short-circuits before every auth
  check.
- `CORS_ORIGINS=*` (the default) disables credentialed cross-origin requests; name
  explicit origins when a browser must send cookies or `Authorization`.
- `POST /login` is rate-limited per client IP (`LOGIN_MAX_ATTEMPTS` /
  `LOGIN_WINDOW_SECONDS`); uploads above `MAX_UPLOAD_SIZE` are rejected with 413.
- Postgres and Neo4j publish on `127.0.0.1` only. Override
  `POSTGRES_PUBLISH_ADDR` / `NEO4J_PUBLISH_ADDR` deliberately — the Neo4j instance
  runs `gds.*` unrestricted.
- Do not commit `.env`. Prefer secrets managers in production.
- Entrypoint drops to UID 1000 after fixing bind-mount ownership.
