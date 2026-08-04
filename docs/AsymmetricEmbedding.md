# Asymmetric Embedding

MemGraphRAG retrieval uses instruction prefixes when scoring queries against facts/passages (research linking prompts).

Configure via:

```bash
# Optional explicit query prefix (otherwise prompts/linking defaults apply)
EMBEDDING_QUERY_PREFIX=
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

Document chunks are embedded without the query instruction; queries use the asymmetric prefix so cosine/ANN scores align with the trained instruction format of the embedding endpoint.
