# MemGraphRAG Sidecar Format

Adapted from LightRAG's `*.parsed/` interchange for structured parsers (Docling).

## Layout

```
INPUT_DIR/__parsed__/<basename>.parsed/
  blocks.jsonl      # ordered content blocks (headings, paragraphs, …)
  tables.json       # optional
  *.assets/         # optional binary assets
```

- Legacy parser never creates sidecars (`parse_format=raw`).
- Docling writes `parse_format=lightrag` sidecars used by the P chunker.
- Resume may reuse an existing sidecar without re-parsing.

Multimodal VLM consumption of drawings/tables is reserved for a future ANALYZING stage.
