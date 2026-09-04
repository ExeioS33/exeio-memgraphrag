# Docker deployment

Two compose files, because there are two genuinely different deployments and they
disagree about one thing: who owns the RAG databases.

| File | Starts | Use it when |
|---|---|---|
| `docker-compose.yml` | app + **its own** Postgres/pgvector + Neo4j/GDS + chat database | you want a self-contained stack on one machine |
| `docker-compose.app.yml` | app + chat database only | Postgres and Neo4j already run elsewhere — a LAN host, a managed instance |

Both build the same image, `exeio-memgraphrag:${MEMGRAPHRAG_VERSION:-0.1.0}` (also
tagged `:latest`), whose `node` stage compiles the React bundle into
`memgraphrag/api/static/`. Nothing extra is needed for the web UI.

## The app on databases you already run

```bash
cp env.example .env          # then fill in the sections below
docker compose -f docker-compose.app.yml up -d --build
#   → http://localhost:9621/
```

`.env` supplies everything about those databases:

```bash
POSTGRES_HOST=192.168.6.2
POSTGRES_PORT=5432
POSTGRES_USER=…
POSTGRES_PASSWORD=…
POSTGRES_DATABASE=rag_db
NEO4J_URI=bolt://192.168.6.2:7688
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=…
WORKSPACE=rfe_mgr            # the Neo4j label your corpus lives under
MEMGRAPHRAG_KV_STORAGE=PGKVStorage
MEMGRAPHRAG_VECTOR_STORAGE=PGVectorStorage
MEMGRAPHRAG_GRAPH_STORAGE=Neo4JStorage
MEMGRAPHRAG_DOC_STATUS_STORAGE=PGDocStatusStorage

APP_POSTGRES_PASSWORD=…      # REQUIRED — the chat database container
LIBRARY_HOST_DIR=/path/to/your/corpus
```

A database on the Docker host itself is reachable as `host.docker.internal`; a LAN
address needs nothing special.

## The self-contained stack

```bash
docker compose up -d --build
```

Here the app **must not** use the `POSTGRES_HOST` / `NEO4J_URI` from `.env` — those
name your machine, and the container needs the compose network — so
`docker-compose.yml` overrides them with `postgres` and `neo4j://neo4j:7687`. Putting
a LAN address in `.env` has no effect on this file. That is the difference between
the two, and the only one that matters.

## How configuration reaches the container

Both files load the whole of `.env` with `env_file:`, so adding a setting no longer
means editing compose. What each still pins in `environment:` — which wins over
`env_file:` — is only what must differ *inside* the container:

| Overridden by both | Why |
|---|---|
| `HOST`, `PORT` | bind inside the container, not the host publish address |
| `WORKING_DIR`, `INPUT_DIR`, `LIBRARY_ROOT` | `.env` holds host-relative paths (`./data/…`) that mean nothing in `/app` |
| `MEMGRAPHRAG_CORP_CA_FILE`, `MEMGRAPHRAG_SSL_CERT_FILE`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` | the CA is mounted at a fixed path |
| `APP_DATABASE_URL` | points at the `postgres-app` service over the compose network, not the host publish port |

and, **in `docker-compose.yml` only**, `POSTGRES_HOST`, `POSTGRES_PORT` and
`NEO4J_URI`.

That precedence is load-bearing rather than stylistic. A working `.env` legitimately
contains `POSTGRES_HOST=192.168.6.2`, `WORKING_DIR=./data/rag_storage` and
`SSL_CERT_FILE=./certs/corporate-ca.crt`. Injected verbatim, those point the
container at the wrong database and at paths that do not exist in it.

One caveat worth knowing: `.env` is *also* bind-mounted at `/app/.env`, and the entry
points read it through `load_env_file()`. That is safe only because
`memgraphrag/api/config.py` calls `load_dotenv(..., override=False)` — real
environment variables win. Change that flag and the host file silently reclaims
every override above.

`VLLM_HOST` reaches the container unmodified and is read nowhere in the code; the
provider registry uses `VLLM_BASE_URL`.

## Volumes

| Host | Container | Mode |
|---|---|---|
| `./data/rag_storage` | `/app/data/rag_storage` | rw — file-backed state, unused with PG/Neo4j backends |
| `./data/inputs` | `/app/data/inputs` | rw — what `docs scan` picks up |
| `${LIBRARY_HOST_DIR}` | `/app/data/library` | **ro** — what the library browses |
| `${MEMGRAPHRAG_HOST_CA_FILE}` | `/app/certs/corporate-ca.crt` | ro — corporate TLS interception |
| `./.env` | `/app/.env` | ro |

The library mount is why `LIBRARY_ROOT` in `.env` does nothing under Docker: the
container path is fixed, and `LIBRARY_HOST_DIR` chooses what appears there. Without
it the panel renders an empty tree — the directory exists and holds nothing.

## Three things that stop a first run

Each fails loudly, but not obviously.

**`APP_POSTGRES_PASSWORD` unset.** Guarded with `:?` in both files, so compose
refuses before creating anything:

```
error while interpolating services.memgraphrag.environment.APP_DATABASE_URL:
required variable APP_POSTGRES_PASSWORD is missing a value: set APP_POSTGRES_PASSWORD in .env
```

Nobody types this password; generate one.

**Port 5433 already taken.** `postgres-app` publishes on `127.0.0.1:5433` because
5432 is usually the RAG database. A second project — another checkout, a git
worktree — takes it first and the second stack fails to bind. `docker ps --filter
publish=5433` names the holder; removing that container keeps its named volume.

**A stale image.** `docker compose up -d` alone reuses whatever was built last. An
image predating the web UI serves `Web UI not built; serving API only` and every API
route still works, which reads as a UI bug rather than a stale build. Use `--build`
after pulling.

## Why the app used to crash-loop on startup

Seen in the wild, and fixed rather than documented away. `docker-compose.yml` waits
for Neo4j with `depends_on: condition: service_healthy`, but the healthcheck used to
be `wget http://localhost:7474`. Neo4j answers on HTTP well before Bolt can serve
routing, so the container reported healthy, the app started, and the driver failed:

```
ERROR [neo4j.pool] Unable to retrieve routing information
ConnectionError: Unable to connect to Neo4j at neo4j://neo4j:7687
Application startup failed. Exiting.
```

Five startups in ninety seconds, and only `restart: unless-stopped` eventually caught
it. The healthcheck now runs `cypher-shell … 'RETURN 1'` over Bolt — it proves the
thing the app is about to do — with a `start_period` that stops the retries counting
against a database that is merely still opening its store files.

## Checks

```bash
docker compose -f docker-compose.app.yml ps          # every service Up
curl -s localhost:9621/health | jq .retrieval_status # "ready"
curl -s localhost:9621/ -o /dev/null -w '%{http_code}\n'   # 200 = bundle served
```

`retrieval_status` other than `ready` means the engine could not warm up — the
databases are unreachable or the workspace is empty. `/health` carries the reason in
`retrieval_error`; the server starts anyway, on purpose, so an empty corpus is not a
boot failure.

To confirm the container really talks to the databases you meant:

```bash
docker compose -f docker-compose.app.yml config \
  | grep -E 'POSTGRES_HOST|NEO4J_URI|WORKING_DIR|LIBRARY_ROOT'
```

Hosts should be yours, paths should be `/app/…`. If a path shows up as `./data/…`,
an override is missing and the container is about to use a directory that is not
there.

## Optional services

**Docling parser** — a profile on the self-contained stack:

```bash
docker compose --profile docling up -d
```

**MCP** — no new service: the server mounts at `/mcp` in the existing container, on
the port already published. Two variables in `.env`, and `docs/MCP.md` for the rest:

```bash
MCP_ENABLED=true
MCP_ALLOWED_HOSTS=rag.example.com,rag.example.com:9621
```

`MCP_ALLOWED_HOSTS` is not optional off localhost: unlisted hosts get **421 Invalid
Host header**, and the entries are exact matches — list the host with and without
its port.
