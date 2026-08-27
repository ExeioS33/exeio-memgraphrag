# Reproduce / Benchmark Protocol

> **Status: protocol only — no harness ships in this repository.**
> There is no benchmark script, no evaluation dataset loader, no metrics module,
> and no "research provenance" directory. Every number below has to be produced
> by a harness you write. Nothing here has been run against LightRAG, so this
> document must not be cited as evidence of a result.

Neither LightRAG nor MemGraphRAG ships head-to-head numbers in-tree. A fair A/B
therefore needs a shared industrial stack (same corpus, parser, chunker, LLM and
embedding endpoints, storage backends and REST surface) so that the engine is the
only variable.

## What the repository actually gives you

| Available today | Where |
|-----------------|-------|
| Retrieval evidence per query (`response`, `references`, scored docs) | `POST /query/data` |
| Per-stage timings and LLM/embed call logs | structured logs, see [Logging.md](Logging.md) |
| Nested spans per query stage (fact linking, PPR, dense fallback, generation) | Langfuse, see [LangfuseObservability.md](LangfuseObservability.md) |
| `QuerySolution` fields for scoring (`docs`, `doc_scores`, `passage_ids`, `answer`, `gold_answers`, `gold_docs`) | `memgraphrag.utils.misc` |

`QuerySolution` carries `gold_answers` / `gold_docs` slots, but nothing in the
repository fills them: a labelled dataset and the code that loads it are yours to
add.

## What a harness would still have to build

- **Load generation.** No load tool is bundled, so RPS / p50 / p95 need an
  external driver (`k6`, `locust`, `hey`, …) against `POST /query`.
- **Recall@k, EM, F1.** No metric implementation exists. Score
  `QuerySolution.passage_ids` against your own gold set.
- **Cost per query.** Token counts are not aggregated anywhere; derive them from
  the provider's usage reporting or from the `[LLM]` / `[EMBED]` log lines.
- **Time-to-first-token.** Not measurable today: `POST /query/stream` awaits the
  complete answer before emitting its first SSE frame (see
  [MemGraphRAG-API-Server.md](MemGraphRAG-API-Server.md)), so TTFB equals total
  latency by construction. Comparing it to a truly token-streaming system
  measures the transport, not the engine.

## Per-layer measurements, once a harness exists

Use the same corpus and the same `LLM_*` / `EMBEDDING_*` endpoints for both
systems, and pin `WORKERS=1` on both sides (this server refuses more with
file-backed storage, and its ingest lock is per-process either way).

| Layer | Metrics | Expected shape (hypothesis, unmeasured) |
|-------|---------|------------------------------------------|
| File processing | docs/min | Identical when the parser/chunker config is shared |
| Indexing | LLM calls & tokens per 1k chunks, wall time, graph size | LightRAG cheaper; MemGraphRAG does extra schema + conflict passes |
| Storage | upsert throughput, ANN p95 | MemGraphRAG carries an extra **facts** vector namespace |
| Retrieval | Recall@k, latency, $/query | MemGraphRAG's default path makes no query-time LLM call for retrieval itself |
| QA | EM/F1 (multi-hop); LLM-judged comprehensiveness (thematic) | Report both; the two systems optimise different things |
| Serving | RPS, p50/p95 | Same worker count and same gunicorn settings on both sides |

## Adaptation boundary

[AGENTS.md](../AGENTS.md) and the `docs/` guides describe where this repository
departs from both upstreams — read them before declaring any measurement
"engine vs engine" rather than "deployment vs deployment".
