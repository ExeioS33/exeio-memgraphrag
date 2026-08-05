# MemGraphRAG structured logging

Server operators follow two conventions:

1. **Framework engine logs** (MemGraphRAG indexing / retrieval) — aligned with
   [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) stage
   wording plus explicit agent / LLM / embed markers.
2. **API / file-pipeline logs** — `[MAIN]` / `[STEP]` for HTTP and parse/chunk
   boundaries.

Helpers live in `memgraphrag.utils.step_log`.

## Prefixes

| Prefix | Meaning |
|--------|---------|
| `[INDEX]` | Banner: **Memory-based Indexing Graph Construction** |
| `[RETRIEVE]` | Banner: **Memory-guided Online Retrieval** |
| `[STAGE]` | Framework stage (OpenIE, schema, conflict, PPR, QA, …) |
| `[LLM]` | Agent / chat-completion call (role id + sizes, never full prompts) |
| `[EMBED]` | Embedding call (`context`, `n`, `model`) |
| `[MAIN]` | API / pipeline flow start (upload, parse, chunk, admin) |
| `[STEP]` | API / pipeline sub-step |
| `[DONE]` | Successful completion |
| `[FAIL]` | Failure (warning, or traceback when `exc_info=True`) |

## Multi-agent / LLM roles (`agent=`)

These ids appear on `[LLM]` lines when the corresponding stage runs:

| Agent id | When |
|----------|------|
| `openie.ner` | Named-entity extraction per passage |
| `openie.triple` | Relation triple extraction per passage |
| `schema.extract` | Ontology / schema typing batches |
| `conflict.detect` | Hard-conflict detection groups |
| `conflict.resolve` | Conflict resolution with passage evidence |
| `qa.reading` | RAG answer generation over retrieved docs |
| `qa.bypass` | Direct LLM answer (`mode=bypass`) |

Fact “LLM rerank” is currently a **threshold stub** (`FactFilter.llm_filter`); logs show `method=threshold` / `method=llm` accordingly — no live LLM call until that stub is wired.

## Example: Memory-based Indexing Graph Construction

```text
[INDEX] Memory-based Indexing Graph Construction | chunks=16
[STAGE] Indexing Documents | chunks=16
[STAGE] Performing OpenIE | chunks=16 cached=10 extract=6
[STAGE] Performing OpenIE | docs=6 concurrency=4
[STAGE] NER | docs=6
[STAGE] Extracting triples | docs=6
[LLM] agent=openie.ner action=complete model=… prompt_chars=… preview=…
[LLM] agent=openie.ner action=complete_done model=… response_chars=…
[LLM] agent=openie.triple action=complete …
[STAGE] OpenIE completed | docs=6 entities=42 triples=38
[STAGE] Building three-layer memory | openie_docs=16 run_conflicts=True
[STAGE] Built memory structure | passages=16 facts=41 schemas=0
[STAGE] Extracting schema | batches=3 unlinked=41 agent=schema.extract
[LLM] agent=schema.extract action=complete …
[STAGE] Extracting schema done | linked=40 schemas=12 failed_batches=0
[STAGE] Ontology filtering | before=12 kept=8 dropped=4 min_frequency=2
[STAGE] Detecting conflicts | groups=10 agent=conflict.detect
[LLM] agent=conflict.detect action=complete …
[STAGE] Detecting conflicts done | hard_conflicts=1 groups_checked=10
[STAGE] Resolving conflicts | conflicts=1 agent=conflict.resolve
[LLM] agent=conflict.resolve action=complete …
[STAGE] Resolving conflicts done | resolved=1 discarded=1 …
[STAGE] Encoding Entities | passages=16 facts=41 …
[EMBED] context=document model=… n=… dim=…
[STAGE] Encoding Facts | add_passages=… add_facts=…
[STAGE] Constructing Graph
[STAGE] Graph construction completed! | passages=16 facts=41 schemas=8
[DONE] Memory-based Indexing Graph Construction | passages=16 facts=41 schemas=8
```

## Example: Memory-guided Online Retrieval

```text
[MAIN] api.query | mode=ppr query=What is MemGraphRAG? stream=False
[RETRIEVE] Memory-guided Online Retrieval | queries=1 mode=ppr top_k=5
[STAGE] Preparing for fast retrieval.
[STAGE] Loading keys. | loaded_memory=True
[STAGE] Loading embeddings. | edges=5720
[STAGE] PPR engine ready | engine=igraph
[DONE] Preparing for fast retrieval | passages=99 facts=2115 schemas=191 ready=True
[STAGE] Retrieving | mode=ppr query=What is MemGraphRAG? top_k=5 linking_top_k=10
[STAGE] Encoding queries for query_to_fact.
[EMBED] context=query model=… n=1 dim=…
[STAGE] Fact filtering stats | method=threshold hits=10 kept=4
[STAGE] Schema linking | schema_top_k=5
[STAGE] Schema linking done | schema_hits=3 seed_nodes=9
[STAGE] Encoding queries for query_to_passage.
[STAGE] Running PPR | seed_nodes=18 damping=0.5
[DONE] Retrieving | path=ppr docs=5 top_score=0.1832
[STAGE] QA Reading | docs=5 history_turns=0 agent=qa.reading
[LLM] agent=qa.reading action=complete model=… prompt_chars=…
[LLM] agent=qa.reading action=complete_done … response_chars=312
[DONE] Memory-guided Online Retrieval | mode=ppr docs=5 answer_chars=312
[DONE] api.query | mode=ppr docs=5 answer_chars=312
```

## Filtering Compose logs

```bash
# Framework engine only
docker compose logs memgraphrag 2>&1 | grep -E '\[(INDEX|RETRIEVE|STAGE|LLM|EMBED|DONE|FAIL)\]'

# LLM / agent calls only
docker compose logs memgraphrag 2>&1 | grep '\[LLM\]'

# Indexing vs retrieval banners
docker compose logs memgraphrag 2>&1 | grep -E '\[(INDEX|RETRIEVE)\]'

# API / parse / chunk plumbing
docker compose logs memgraphrag 2>&1 | grep -E '\[(MAIN|STEP)\]'

# Failures
docker compose logs memgraphrag 2>&1 | grep '\[FAIL\]'
```

After code changes, rebuild the service image:

```bash
docker compose up -d --build memgraphrag
```

## Privacy

Do not log API keys, full LLM prompts, or entire document bodies. Prefer `doc_id`,
file names, byte/char counts, chunk counts, scores, agent ids, and truncated
previews (`truncate`, default 160 chars; LLM preview default 80).
