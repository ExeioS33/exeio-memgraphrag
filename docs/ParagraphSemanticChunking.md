# Paragraph Semantic Chunking (P)

The P chunker consumes Docling/native sidecar `blocks.jsonl` headings to form paragraph-aware chunks.

- If no sidecar exists, P **falls back to R** (recursive character).
- Size/overlap: the global `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`. Per-chunker
  `CHUNK_P_SIZE` / `CHUNK_P_OVERLAP_SIZE` were documented but never implemented —
  `memgraphrag/chunker/` reads no environment variable at all.
- Select via routing options (`docling-P`) or chunk strategy `P`.

This is MemGraphRAG's heading-aware path; F and R remain the default text strategies.
