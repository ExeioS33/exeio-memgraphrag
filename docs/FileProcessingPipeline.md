# File Processing Pipeline

Adapted from LightRAG's file layer for MemGraphRAG's memory pipeline.

## Status ladder

```
PENDING → PARSING → PROCESSING → PROCESSED
                 ↘ FAILED ↗ (retry re-enqueues as PENDING)
```

`PROCESSING` writes MemGraphRAG memory sub-stage labels into doc-status metadata:

`openie → memory_build → schema_extraction → ontology_filter → conflict_detection → conflict_resolution → graph_install`

**These labels are not progress.** `pipeline.py` writes all seven in one tight loop
*before* the corresponding work starts, so a status poll shows the final label
(`graph_install`) for effectively the entire run.

The `ANALYZING` stage is reserved for a future VLM extension (not implemented in this POC).

## Flow

1. `POST /documents/upload` or `/documents/scan` saves files under `INPUT_DIR`.
2. Parser routing resolves engine from `MEMGRAPHRAG_PARSER` rules and filename `[hints]`.
3. Default engine `legacy` extracts text in-process (pdf/docx/pptx/xlsx/txt/md).
4. Optional `docling` engine calls `DOCLING_ENDPOINT` and writes a sidecar under `__parsed__/`.
5. Chunkers F (fixed-token), R (recursive), or P (paragraph-semantic over sidecar headings) produce `{doc_id}-chunk-{order}` chunks.
6. Chunks become `PassageNode`s and enter `index_with_memory`.

## Env knobs

| Variable | Role |
|----------|------|
| `MEMGRAPHRAG_PARSER` | Extension → engine[-options] rules |
| `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE` | Size and overlap for **every** chunker (F, R and P alike) |
| `DOCLING_ENDPOINT` / `DOCLING_POLL_INTERVAL_SECONDS` / `DOCLING_MAX_POLLS` / `DOCLING_ADDITIONAL_SUFFIXES` | Remote Docling service |
| `PDF_DECRYPT_PASSWORD` | Password for encrypted PDFs (legacy parser) |

Not implemented, despite having been documented: `CHUNK_F_SIZE` / `CHUNK_R_SIZE` /
`CHUNK_P_SIZE` and their `*_OVERLAP_SIZE` (nothing under `memgraphrag/chunker/`
reads any environment variable — the pipeline passes the two global values down),
and `MAX_PARALLEL_INSERT` (ingest concurrency is fixed in `pipeline.py`; only
`MAX_ASYNC_LLM` bounds outbound LLM concurrency).

See also [ParserServiceDeployment.md](ParserServiceDeployment.md), [ParagraphSemanticChunking.md](ParagraphSemanticChunking.md), [MemGraphRAGSidecarFormat.md](MemGraphRAGSidecarFormat.md).
