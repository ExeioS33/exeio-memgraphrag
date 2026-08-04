# MemGraphRAG

Industrialized API server for [MemGraphRAG](https://arxiv.org/abs/2606.00610) — a memory-enhanced GraphRAG engine with a three-layer memory (schema / fact / passage), conflict-aware construction, and Personalized PageRank retrieval.

This repository packages the research engine as a LightRAG-style production service: FastAPI REST API, pluggable storage (PostgreSQL + pgvector, Neo4j + GDS), OpenAI-compatible LLM/embedding bindings, Docling-capable file processing, Docker Compose, and uv-based tooling.

## Quick start

```bash
# Install (requires uv)
uv sync --extra api

# Copy and edit environment
cp env.example .env

# Run API server (file-based defaults; no external DB required)
uv run memgraphrag-server

# Or full stack
docker compose up -d
```

API docs: `http://localhost:9621/docs`

## Documentation

See [`docs/MemGraphRAG-API-Server.md`](docs/MemGraphRAG-API-Server.md) and other guides under [`docs/`](docs/).

## Agent maintenance

This repository is maintained by AI agents. Conventions, tech stack, and architecture decisions live in [`AGENTS.md`](AGENTS.md).

## License

MIT — see [LICENSE](LICENSE).
