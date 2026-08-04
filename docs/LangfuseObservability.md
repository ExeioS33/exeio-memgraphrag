# Langfuse observability (retrieval layer)

MemGraphRAG can emit [Langfuse](https://langfuse.com/) traces for the **query / retrieval** path so you can inspect fact linking, PPR, dense fallbacks, and RAG answer generation in your Langfuse project (cloud or self-hosted).

## Enable

1. Install the API extra (includes `langfuse`):

```bash
uv sync --extra api
```

2. Set environment variables (see `env.example`):

| Variable | Purpose |
|----------|---------|
| `LANGFUSE_ENABLE_TRACE` | `true` to opt in |
| `LANGFUSE_PUBLIC_KEY` | Project public key |
| `LANGFUSE_SECRET_KEY` | Project secret key |
| `LANGFUSE_BASE_URL` | Self-hosted or cloud base URL (preferred) |
| `LANGFUSE_HOST` | Legacy alias for the base URL |

When the flag is off or keys are missing, all helpers no-op — retrieval still works without Langfuse.

Docker Compose loads `.env` into the app container; rebuild after adding the dependency:

```bash
docker compose up -d --build memgraphrag
```

## What is traced

Root span: `memgraphrag.query`

| Observation | Type | When |
|-------------|------|------|
| `memgraphrag.retrieve` | retriever | PPR / graph retrieval path |
| `memgraphrag.fact_linking` | span | Query→fact embedding + filter |
| `memgraphrag.schema_linking` | span | Query→schema embedding + fact/entity seed expansion |
| `memgraphrag.passage_seed` | span | Dense passage seeds for PPR |
| `memgraphrag.ppr` | span | Personalized PageRank run |
| `memgraphrag.dense_retrieve` | retriever | Naive mode or PPR fallback |
| `memgraphrag.rag_qa` | generation | Final LLM answer from passages |
| `memgraphrag.llm_bypass` | generation | `mode=bypass` (no retrieval) |

Passage text in traces is truncated (few short snippets) to keep payloads small.

## Smoke test

With a document already indexed:

```bash
curl -s -X POST http://localhost:9621/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is MemGraphRAG?","mode":"ppr"}' | jq -r .answer
```

Then open your Langfuse UI (`LANGFUSE_BASE_URL`) and confirm a new trace named `memgraphrag.query` with nested retrieval spans.

Server logs should include:

```text
Langfuse tracing enabled (host=…)
```

## Implementation notes

- Module: `memgraphrag/observability/langfuse_trace.py`
- Instrumentation: `MemGraphRAG.aquery` / `_retrieve_one` / `_run_ppr` / `_dense_passage_retrieve`
- Flush: after each query (`aquery` finally + `/query` router)
- Optional dependency: listed under `[project.optional-dependencies] api`
