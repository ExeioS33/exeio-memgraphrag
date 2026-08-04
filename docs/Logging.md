# MemGraphRAG structured logging

Server operators can follow ingest and query flows in `docker compose logs memgraphrag` using a two-level convention: **Main step** and **sub-steps**.

## Prefixes

| Prefix | Meaning |
|--------|---------|
| `[MAIN]` | Start of a major flow (API request, document process, index, retrieve, parse, chunk) |
| `[STEP]` | Sub-step inside a main flow |
| `[DONE]` | Successful completion of a main flow |
| `[FAIL]` | Failure (warning, or exception with traceback when `exc_info=True`) |

Helpers live in `memgraphrag.utils.step_log`:

- `main_step(logger, name, **fields)`
- `sub_step(logger, name, **fields)`
- `done_step(logger, name, **fields)`
- `fail_step(logger, name, *, exc=None, exc_info=False, **fields)`
- `truncate(value, limit=160)` for safe text previews

Field values are appended as `key=value` pairs (no secrets, no full prompts, no huge bodies).

## Example: document ingest

```text
[MAIN] api.documents.upload | doc_id=doc-abc filename=report.pdf bytes=102400
[STEP] api.documents.upload.enqueue | doc_id=doc-abc filename=report.pdf
[MAIN] ingest.enqueue | doc_id=doc-abc file=report.pdf content_chars=0
[DONE] ingest.enqueue | doc_id=doc-abc engine=legacy status=pending
[MAIN] ingest.process | pending=1
[MAIN] ingest.doc | doc_id=doc-abc file=report.pdf
[STEP] ingest.doc.parse | doc_id=doc-abc engine=legacy source_exists=True
[MAIN] parse.legacy | doc_id=doc-abc file=report.pdf
[STEP] parse.legacy.extract_text | suffix=pdf bytes=102400
[DONE] parse.legacy | doc_id=doc-abc chars=18420 suffix=pdf
[STEP] ingest.doc.chunk | doc_id=doc-abc strategy=F chunk_token_size=1200 overlap=100
[MAIN] chunk.run | strategy=F content_chars=18420 chunk_token_size=1200 overlap=100
[DONE] chunk.run | strategy=F chunks=16
[STEP] ingest.doc.index | doc_id=doc-abc chunks=16
[MAIN] index.ainsert | chunks=16
[STEP] index.ainsert.openie | chunks=16
[STEP] index.ainsert.memory_build | openie_docs=16
[STEP] index.ainsert.graph_install
[DONE] index.ainsert | passages=16 facts=42 schemas=0
[DONE] ingest.doc | doc_id=doc-abc chunks=16 status=processed
[DONE] ingest.process | processed=1 failed=0
[DONE] api.documents.upload | doc_id=doc-abc filename=report.pdf processed=1 failed=0
```

## Example: query

```text
[MAIN] api.query | mode=ppr query=What is MemGraphRAG? stream=False only_need_context=False
[MAIN] query.aquery | mode=ppr query=What is MemGraphRAG? only_need_context=False
[STEP] query.aquery.mode_select | path=ppr
[MAIN] retrieve.aretrieve | queries=1 mode=ppr top_k=5
[MAIN] retrieve.one | mode=ppr query=What is MemGraphRAG? top_k=5 linking_top_k=10
[STEP] retrieve.one.fact_linking | linking_top_k=10
[STEP] retrieve.one.fact_rerank | method=threshold hits=10 kept=4
[STEP] retrieve.one.ppr | seed_nodes=12 damping=0.5
[DONE] retrieve.one | path=ppr docs=5 top_score=0.1832
[STEP] query.aquery.rag_qa | docs=5 history_turns=0
[DONE] query.aquery | mode=ppr docs=5 answer_chars=312
[DONE] api.query | mode=ppr docs=5 answer_chars=312
```

## Filtering Compose logs

```bash
# All structured steps
docker compose logs memgraphrag 2>&1 | grep -E '\[(MAIN|STEP|DONE|FAIL)\]'

# Ingest only
docker compose logs memgraphrag 2>&1 | grep -E '\[(MAIN|STEP|DONE|FAIL)\].*ingest\.|parse\.|chunk\.|index\.'

# Query / retrieve only
docker compose logs memgraphrag 2>&1 | grep -E '\[(MAIN|STEP|DONE|FAIL)\].*(api\.query|query\.|retrieve\.)'

# Failures
docker compose logs memgraphrag 2>&1 | grep '\[FAIL\]'
```

After code changes, rebuild the service image:

```bash
docker compose up -d --build memgraphrag
```

## Privacy

Do not log API keys, full LLM prompts, or entire document bodies. Prefer `doc_id`, file names, byte/char counts, chunk counts, scores, and truncated previews (`truncate`, default 160 chars).
