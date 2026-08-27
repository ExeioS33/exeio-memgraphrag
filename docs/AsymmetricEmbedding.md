# Asymmetric Embedding

MemGraphRAG retrieval uses instruction prefixes when scoring queries against facts/passages (research linking prompts).

The prefix is not an environment setting. `core.py` passes
`instruction=get_query_instruction(...)` (from `memgraphrag/prompts/`) to
`openai_embed` with `context="query"`; `EMBEDDING_QUERY_PREFIX` is read nowhere.
To change the instruction, edit the linking prompts, or call `openai_embed`
yourself with an explicit `query_prefix=` when embedding the core directly.

Environment settings that do apply to embedding:

```bash
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_MAX_TOKENS=512
EMBEDDING_TOKEN_SAFETY=0.60
EMBEDDING_SEND_DIMENSIONS=true
```

Document chunks are embedded without the query instruction; queries use the asymmetric prefix so cosine/ANN scores align with the trained instruction format of the embedding endpoint.
