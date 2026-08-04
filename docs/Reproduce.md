# Reproduce / Benchmark Protocol

Neither LightRAG nor MemGraphRAG ships head-to-head numbers in-tree. With a shared industrial stack (same parser/chunker/bindings/storage/REST), the engine is the only variable.

## Per-layer measurements

Use the same corpus and the same `LLM_*` / `EMBEDDING_*` endpoints for both systems.

| Layer | Metrics | Expected shape |
|-------|---------|----------------|
| File processing | docs/min | Identical (shared code) |
| Indexing | LLM calls/tokens per 1k chunks, wall time, graph size | LightRAG cheaper; MemGraphRAG richer (schema + conflicts) |
| Storage | upsert throughput, ANN p95 | MemGraphRAG has an extra **facts** vector namespace |
| Retrieval | Recall@k, latency, $/query | MemGraphRAG default has no query-time LLM call |
| QA | EM/F1 (multi-hop); LLM-judged comprehensiveness (thematic) | Different strengths — report both |
| Serving | RPS, p50/p95, streaming TTFB | Same gunicorn settings |

## Harness hooks

- Ported evaluation helpers live under research provenance; wire Recall@k / EM / F1 against `QuerySolution` outputs.
- `docs/` and `AGENTS.md` describe the adaptation boundary for fair A/B scripts.
