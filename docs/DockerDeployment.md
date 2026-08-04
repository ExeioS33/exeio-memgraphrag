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

## Security notes

- Bind `HOST=0.0.0.0` only with `MEMGRAPHRAG_API_KEY` or `AUTH_ACCOUNTS`.
- Do not commit `.env`. Prefer secrets managers in production.
- Entrypoint drops to UID 1000 after fixing bind-mount ownership.
