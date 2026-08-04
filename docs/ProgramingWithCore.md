# Programming with the Core

Prefer the REST API for product integration. Use the core for embedded/research scripts.

```python
import asyncio
from memgraphrag import MemGraphRAG, QueryParam
from memgraphrag.llm.openai_compatible import create_llm_func, create_embedding_func

async def main():
    rag = MemGraphRAG(
        working_dir="./data/rag_storage",
        workspace="demo",
        llm_model_func=create_llm_func(),
        embedding_func=create_embedding_func(),
    )
    await rag.initialize_storages()
    await rag.aindex_with_memory([
        "0:Paris is the capital of France.",
        "1:The Seine river flows through Paris.",
    ])
    result = await rag.arag_qa(
        ["What is the capital of France?"],
        param=QueryParam(mode="ppr", top_k=5),
    )
    print(result)
    await rag.finalize_storages()

asyncio.run(main())
```

## Critical patterns

- Always `await rag.initialize_storages()` before insert/query.
- Warm retrieval via `RetrievalStateManager` (server lifespan does this).
- Storage backends are selected by constructor args / env (`MEMGRAPHRAG_*_STORAGE`).
- Sync wrappers `insert` / `query` exist; prefer async `ainsert` / `aquery` / `aindex_with_memory` / `arag_qa`.
