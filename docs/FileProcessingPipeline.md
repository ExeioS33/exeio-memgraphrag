# File Processing Pipeline

Adapted from LightRAG's file layer for MemGraphRAG's memory pipeline.

## Status ladder

```
PENDING → PARSING → PROCESSING → PROCESSED
                 ↘ FAILED ↗ (retry re-enqueues as PENDING)
```

`PROCESSING` tracks MemGraphRAG memory sub-stages in doc-status metadata:

`openie → memory_build → schema_extraction → ontology_filter → conflict_detection → conflict_resolution → graph_install`

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
| `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE` | Global F defaults |
| `CHUNK_F_*` / `CHUNK_R_*` / `CHUNK_P_*` | Per-strategy sizing |
| `DOCLING_*` | Remote Docling service |
| `MAX_PARALLEL_INSERT` | Pipeline concurrency |

See also [ParserServiceDeployment.md](ParserServiceDeployment.md), [ParagraphSemanticChunking.md](ParagraphSemanticChunking.md), [MemGraphRAGSidecarFormat.md](MemGraphRAGSidecarFormat.md).
