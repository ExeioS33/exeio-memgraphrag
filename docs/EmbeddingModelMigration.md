# Changing the embedding model

Changing `EMBEDDING_MODEL` on a corpus that is already indexed does **not** re-embed
it, and on most model pairs nothing will tell you. This document explains why, and
how `scripts/reembed.py` fixes it without re-ingesting anything.

## Why a model swap silently breaks retrieval

Three behaviours combine into one failure:

1. **Indexing is delta-only.** `_embed_and_store_memory` (`memgraphrag/core.py`)
   embeds only the ids it has not seen before, and ids are content hashes
   (`memgraphrag/utils/hashing.py`). Re-running `ainsert` over an unchanged corpus
   therefore produces an empty work list: no vector is recomputed, and the old
   model's vectors stay exactly where they were.
2. **Nothing records which model wrote a vector.** `PGVectorStorage.upsert` writes
   `id`, `content`, `embedding` and `metadata` — no model name, no timestamp.
3. **The only guard is dimensional.** `_assert_embedding_dim`
   (`memgraphrag/storage/postgres_impl.py`) compares the column's declared width
   against `EMBEDDING_DIM`. Two models of the *same* width pass it without a word.

So a swap between two 1024-dimension models "works": the service starts, ingestion
succeeds, no warning is logged — and every query vector is compared against vectors
from a foreign space.

### What that costs, measured

Take any indexed corpus, embed a document's own text with the new model, and compare
it against that same document's stored vector:

| Comparison | Cosine similarity |
|---|---|
| new-model vector vs the **same text** re-embedded by the new model | `1.0000` |
| new-model vector vs the **same document's** stored old-model vector | `~ -0.02` |

Across a query set, top-5 overlap between the two models against the old index came
out at **0 out of 5** on every question, with the best similarity falling from ~0.9
to ~0.05. That is not degradation, it is an orthogonal vector: the top hit is
indistinguishable from a random row.

Pointing only *new* ingestions at the new model is worse, not safer. New documents
land in the same tables, are the only rows genuinely comparable to the query, and so
are always retrieved — while the existing corpus becomes noise.

There is no supported way to run two embedding models at once, by design:
`_embed_client()` takes no arguments, and unlike completions, embeddings are never
routed per request. A second endpoint would recreate exactly the mixture above.

## Re-embedding without re-ingesting

The text that was embedded is already stored next to each vector, so it can be
recomputed directly — no parsing, no extraction, no LLM call:

| Namespace | Text to re-embed | Queried at retrieval |
|---|---|---|
| `chunks` | the `content` column, verbatim | yes |
| `facts` | `"\t".join(metadata["triple"])` | yes |
| `schemas` | `"\t".join(metadata["triple"])` | yes |
| `entities` | the `content` column | no — written only |

The trap is `facts` and `schemas`: their `content` column holds `str(tuple(...))`
(`_triple_str`), while what was actually embedded is the tab-joined triple. Only
`metadata["triple"]` reproduces the original input, and re-embedding `content`
instead would look correct while indexing a different string.

Everything outside the vector tables stays valid — ids are content hashes and the
content does not change — so the memory row, the OpenIE cache, the chunk texts, the
doc-status rows and the graph are untouched. The graph stores no vector at all.

## Procedure

### 1. Capture a reference first

Do this **before** switching, while the old model still answers. Once it is gone you
can no longer tell a migration problem from a pre-existing one. Any repeatable query
set works.

### 2. Point the bindings at the new model

MemGraphRAG speaks only the OpenAI-compatible API, so an Ollama server is addressed
through its `/v1` prefix — locally:

```
EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=http://localhost:11434/v1
EMBEDDING_BINDING_API_KEY=ollama
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIM=1024
EMBEDDING_SEND_DIMENSIONS=false
```

or, for a remote one, `http://<host>:11434/v1`. Confirm the dimension from the
server rather than trusting a model card:

```bash
curl -s http://<host>:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding:0.6b","input":"probe"}' \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["data"][0]["embedding"]))'
```

Three things bite here:

- **Never leave `EMBEDDING_BINDING_API_KEY` empty.** The resolution order is
  `EMBEDDING_BINDING_API_KEY → LLM_BINDING_API_KEY → OPENAI_API_KEY → "no-key"`, and
  it does not check that the two hosts differ — an empty value sends your completion
  provider's key to the embedding host. `EMBEDDING_BINDING_HOST` falls back to
  `LLM_BINDING_HOST` the same way. Ollama ignores the value but wants the header, so
  any non-empty string does.
- **`EMBEDDING_BINDING` is not read.** It is parsed into config and used nowhere; the
  client is always the OpenAI SDK. A native-Ollama style host without `/v1` will not
  work here even though other tools accept it.
- **`EMBEDDING_MAX_TOKENS` changes meaning with the model name.**
  `_embedding_token_safety` applies a 0.60 factor to names containing `e5`, `bge`,
  `gte-` or `nomic-embed`, and 0.95 to everything else. The same
  `EMBEDDING_MAX_TOKENS=480` yields a 288-token budget on an e5 model and 456 on a
  name it does not recognise — so passages that used to be truncated may stop being
  truncated. That is usually an improvement, but it is a change in what gets indexed.

### 3. Rebuild the vectors

```bash
uv run python scripts/reembed.py --workspace <workspace> --dry-run   # counts only
uv run python scripts/reembed.py --workspace <workspace> --shadow
```

`--shadow` writes into `<table>__reembed` beside each live table and builds the
vector index once the rows have landed. The live tables keep serving the old model
throughout, so there is no window in which an index mixes two vector spaces — which
is exactly what `--in-place` would create, and why it should only be used on a
service that is stopped.

The run is idempotent and resumable: it pages by id and continues past the highest
one already written, so an interrupted run picks up where it stopped.

### 4. Verify, then promote

Before serving anything, check that the new vectors actually match their own text —
embed a stored `content` with the new model and confirm the similarity against its
shadow row is ~1.0. Then:

```bash
uv run python scripts/reembed.py --workspace <workspace> --promote
```

`--promote` refuses to run unless each shadow table holds exactly as many rows as the
live one, then renames tables *and their indexes* inside one transaction, keeping the
previous vectors as `<table>__prev`. Renaming the indexes matters: a table rename
leaves index names behind, and the engine's
`CREATE INDEX IF NOT EXISTS <table>_embedding_idx` would otherwise build a second
index over the same column.

Replay your reference query set and compare. Roll back by renaming the tables the
other way and restoring the previous `EMBEDDING_*` block; drop the `__prev` tables
only once you are satisfied.

## What changes afterwards

- **Query latency.** Each query embeds through the new endpoint. The default PPR path
  makes two query embeddings (facts and passages; the schema search reuses the fact
  vector), so a slower endpoint is paid twice per query. `/naive` pays it once.
- **Availability.** Embeddings are never routed and have no fallback, so the
  embedding endpoint becomes a hard dependency of both ingestion and querying. A
  self-hosted server trades a managed dependency for one you operate.
- **Cost.** Re-embedding makes no LLM call at all — only embedding calls, against
  whatever endpoint you have just configured.
