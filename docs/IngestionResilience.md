# Ingestion resilience and cost control

Extraction is the billed part of ingestion: two LLM calls per chunk (NER, then
triples), then one ontology call per passage batch and up to `CONFLICT_MAX_GROUPS`
conflict calls. This page describes what the engine does so that a corpus-sized
run survives a crash, a provider hiccup or a kill without re-billing what it has
already paid for — and the knobs that govern it.

## What happens during `ainsert`

```
chunks ──► OpenIE (checkpointed) ──► memory build ──► ontology ──► conflicts ──► embed ──► graph
                 │ cache: openie_kv                      │ cache: openie_kv (per-doc ontology map)
```

| Stage | Cached where | Re-billed on relaunch |
|-------|--------------|-----------------------|
| OpenIE (NER + triples) | `openie_kv`, one record per chunk id | only chunks missing from the cache |
| Ontology (schema extraction) | written back into each chunk's `openie_kv` record | only facts without a cached link |
| Conflict detection / resolution | not cached | yes (bounded by `CONFLICT_MAX_GROUPS`) |
| Embeddings | vector store, diffed against the previous memory snapshot | yes, when the memory snapshot was not written |
| Graph install | graph store, rebuilt from `clear()` | yes (seconds with batched writes) |

The stage log line `[STAGE] Performing OpenIE | chunks=N cached=C extract=E`
says exactly what a relaunch will pay: `E` chunks.

## Checkpoints

`OPENIE_CHECKPOINT_SIZE` (default 64) is the number of chunks extracted between
two cache writes. Each sub-batch is upserted into `openie_kv` as soon as it
completes, so a kill at minute 35 keeps minutes 0 to 35 — with the file backends
(`JsonKVStorage` writes on every upsert outside a batch) as much as with
PostgreSQL.

The cache used to be written once, after the last chunk of the corpus; a run
that died before that point had nothing durable to show for its calls.

## Retries before abort

- **Chunks.** A chunk whose extraction raises (network error, provider 5xx after
  the binding's own retries) is reported as `failed=True`, never cached, and
  collected. After the whole corpus has been attempted, the failed chunks are
  extracted once more. Only a chunk that fails twice stops the run with
  `PipelineError`, and every other chunk is already in the cache. A malformed
  answer is **not** a failure — see the next section.
- **Ontology batches.** A batch whose answer cannot be parsed is asked again once
  before its facts are left untyped (`facts_untyped` in the
  `Extracting schema done` line).
- **Embedding requests.** A request the provider refuses (`BadRequestError`) is
  split in two and retried recursively down to a single text.

## Repaired JSON

Small instruction-tuned models routinely break JSON on non-English text: an
unescaped quote inside a string, a trailing comma, a truncated last element.
`memgraphrag.utils.json_llm.extract_json_object` tries the strict parser, then
the first `{…}` span, then `json_repair` (the same fallback LightRAG uses). A
repaired answer is logged at DEBUG; only a hopeless one is logged as
`Failed to parse LLM JSON` and treated as empty.

## Bounded embedding requests

| Variable | Default | Role |
|----------|---------|------|
| `EMBEDDING_MAX_TOKENS` | unset | Cap per **text** (provider context window); tiktoken undercounts e5/bge, so `EMBEDDING_TOKEN_SAFETY` shrinks the budget |
| `EMBEDDING_BATCH_SIZE` | 64 | Max texts per **request** |
| `EMBEDDING_BATCH_MAX_TOKENS` | 100 000 | Max tokens per **request** |

These are safety rails, not tuning knobs: the bisection fallback makes any value
converge, the defaults only decide how many requests a corpus costs.

## One language, one node per concept

Two index-time behaviours decide whether the ontology filter can work at all:

- `MEMGRAPHRAG_LANGUAGE` (default `auto`) is injected into the NER, triple,
  ontology and answer prompts. Left on `auto` with a non-English corpus, the
  model emits `("Entreprise","doit émettre","facture")` next to
  `("Company","must issue","invoice")`: every concept gets two schemas at
  frequency 1.
- Every id derived from a label goes through `canonical_key`
  (`memgraphrag/utils/canonical.py`): NFKC, typographic folding (dashes,
  quotes, ligatures), accents stripped, case folded, whitespace collapsed.
  `Réforme de la Facture Électronique` and `REFORME DE LA FACTURE ELECTRONIQUE`
  are one entity; display text keeps its accents.

`ONTOLOGY_MAX_DEACTIVATION_RATIO` (default 0.5) is the safety valve behind both:
if pruning schemas below `ONTOLOGY_MIN_FREQUENCY` would deactivate more than that
share of the facts, the filter stands down for the build and logs the ratio
instead of emptying the fact graph.

## Concurrency

`MAX_ASYNC_LLM` (default 4) bounds every outbound LLM call. Extraction throughput
scales almost linearly with it until the provider starts refusing; measure with
a short `--limit` run before raising it on a paid endpoint.

## Knobs at a glance

| Variable | Default | Stage |
|----------|---------|-------|
| `OPENIE_CHECKPOINT_SIZE` | 64 | OpenIE cache write frequency |
| `MAX_ASYNC_LLM` | 4 | all LLM calls |
| `MEMGRAPHRAG_LANGUAGE` | auto | extraction + answers |
| `ONTOLOGY_BATCH_SIZE` | 20 | facts per ontology call |
| `ONTOLOGY_MIN_FREQUENCY` | 2 | schema frequency filter |
| `ONTOLOGY_MAX_DEACTIVATION_RATIO` | 0.5 | filter safety valve |
| `CONFLICT_ENABLED` / `CONFLICT_MAX_GROUPS` | true / 50 | conflict detection budget |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_BATCH_MAX_TOKENS` | 64 / 100 000 | embedding request bounds |

All of them are documented with their rationale in `env.example`.
