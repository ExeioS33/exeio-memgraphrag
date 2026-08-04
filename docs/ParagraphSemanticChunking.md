# Paragraph Semantic Chunking (P)

The P chunker consumes Docling/native sidecar `blocks.jsonl` headings to form paragraph-aware chunks.

- If no sidecar exists, P **falls back to R** (recursive character).
- Size/overlap: `CHUNK_P_SIZE`, `CHUNK_P_OVERLAP_SIZE`.
- Select via routing options (`docling-P`) or chunk strategy `P`.

This is MemGraphRAG's heading-aware path; F and R remain the default text strategies.
