# Reproduce / Benchmark Protocol

> **Status: harness implemented, no head-to-head result measured.**
> Metrics, dataset loaders, the judge prompt, the variance-aware runner, the
> golden-set check and a latency/cost benchmark now ship — see
> [Evaluation.md](Evaluation.md), `memgraphrag/evaluation/`, `scripts/evaluate.py`
> and `scripts/bench.py`. What is still absent is a **result**: nothing in this
> repository has been run against LightRAG, so this document must not be cited as
> evidence of one. It describes the protocol for producing that comparison.

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
| `QuerySolution` fields for scoring (`docs`, `doc_scores`, `passage_ids`, `answer`) | `memgraphrag.utils.misc` |
| Str-Acc, LLM-Acc, Context Relevance, Evidence Recall, EM, F1, Recall@k | `memgraphrag/evaluation/`, defined in [Evaluation.md](Evaluation.md) |
| Loaders for HotpotQA / 2WikiMultihopQA / MuSiQue / Medical | `memgraphrag/evaluation/datasets.py` |
| Multi-run campaign with mean / stdev / 95% CI, and a golden-set regression gate | `scripts/evaluate.py` |
| Retrieval p50/p95/p99, RPS, LLM calls and tokens per query | `scripts/bench.py` |

`QuerySolution` still carries unfilled `gold_answers` / `gold_docs` slots; the
harness does not use them — gold labels travel in
`evaluation.EvaluationExample` instead.

## What is still missing

- **Load generation against the HTTP surface.** `scripts/bench.py` drives the
  engine in-process at a fixed concurrency; RPS through the REST layer (auth,
  serialisation, workers) still needs an external driver (`k6`, `locust`, `hey`,
  …) against `POST /query`.
- **Billed cost per query.** `bench.py` counts LLM calls and estimates tokens
  with the project tokenizer, because `llm/openai_compatible.py` does not surface
  `response.usage`. Reconcile against the provider's usage reporting before
  quoting a euro figure.
- **A LightRAG-side harness.** Nothing here runs the other engine; the A/B needs
  the same corpus and endpoints wired on both sides.
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
| QA | EM/F1 and Str-Acc/LLM-Acc via `scripts/evaluate.py` | Report both; the two systems optimise different things |
| Serving | RPS, p50/p95 | Same worker count and same gunicorn settings on both sides |

## Adaptation boundary

[AGENTS.md](../AGENTS.md) and the `docs/` guides describe where this repository
departs from both upstreams — read them before declaring any measurement
"engine vs engine" rather than "deployment vs deployment".
