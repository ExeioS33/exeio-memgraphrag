# Programming with the Core

Prefer the REST API for product integration. Use the core for embedded/research scripts.

The bindings are plain async functions in `memgraphrag.llm.openai_compatible`
(`openai_complete`, `openai_embed`) — there is no `create_llm_func` /
`create_embedding_func` factory. Both read `LLM_*` / `EMBEDDING_*` from the
environment; bind extra arguments with `functools.partial` when you need to pin a
model or a dimension.

```python
import asyncio
from functools import partial

from memgraphrag import MemGraphRAG, QueryParam
from memgraphrag.api.config import load_env_file
from memgraphrag.llm.openai_compatible import openai_complete, openai_embed


async def main():
    load_env_file()  # importing the package no longer reads .env by itself
    rag = MemGraphRAG(
        working_dir="./data/rag_storage",
        workspace="demo",
        llm_model_func=openai_complete,
        embedding_func=partial(openai_embed, embedding_dim=1536),
    )
    await rag.initialize_storages()
    try:
        await rag.aindex_with_memory(
            [
                "Paris is the capital of France.",
                "The Seine river flows through Paris.",
            ]
        )
        # arag_qa takes ONE query string and returns a QuerySolution.
        result = await rag.arag_qa(
            "What is the capital of France?",
            param=QueryParam(mode="ppr", top_k=5),
        )
        print(result.answer)
        print(result.references)
    finally:
        await rag.finalize_storages()


asyncio.run(main())
```

## Critical patterns

- Always `await rag.initialize_storages()` before insert/query, and
  `await rag.finalize_storages()` on the way out.
- Retrieval warm-up is `await rag.prepare_retrieval()`. The API server calls it
  once in its lifespan, and `aquery` re-runs it on demand when the engine is not
  ready. `memgraphrag/retrieval.py` (`RetrievalStateManager`) is **not** on that
  path — it is unused scaffolding for a future incremental-refresh mode and is
  imported only by tests. Do not build on it yet.
- Chunk ids are content hashes. A plain string is stored verbatim and keyed by
  `chunk-<md5(content)>`, so a `"0:"`-style prefix ends up *inside* the passage
  text — it does not name the chunk. To choose ids yourself, pass
  `{"idx": "chunk-…", "content": …}` dicts; any `idx` not starting with `chunk-`
  is replaced by the hash. Those ids are what delete/rebuild refcount on.
- Storage backends are selected by constructor args or env
  (`MEMGRAPHRAG_*_STORAGE`). The file-backed defaults are single-process only.
- Sync wrappers `insert` / `query` exist; prefer async `ainsert` / `aquery` /
  `aindex_with_memory` / `arag_qa`.
- Indexing raises `PipelineError` when OpenIE fails for every chunk, rather than
  storing a fact-free document that no query could ever reach.
