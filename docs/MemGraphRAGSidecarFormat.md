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
- The pipeline does not resume from an existing sidecar: `process_pending` parses
  a file and then chunks it. To index sidecars you already have, chunk them
  directly with `chunking_by_paragraph_semantic(..., blocks_path=...)` and call
  `ainsert` once (see [ProgramingWithCore.md](ProgramingWithCore.md)).

Multimodal VLM consumption of drawings/tables is reserved for a future ANALYZING stage.

## Importing LightRAG artefacts

The block format is the one LightRAG's structured parsers emit (`format:
lightrag` in the `meta` header), so a corpus LightRAG has already parsed with
Docling and captioned with a VLM can be reused as is. `scripts/import_lightrag_parsed.py`
walks a LightRAG `__parsed__/<doc>.parsed/` tree and writes MemGraphRAG sidecars:

| LightRAG artefact | What the importer does |
|---|---|
| `<table format="json">` inline tables | Rendered as Markdown pipe tables, so extraction sees rows and headers, not a JSON matrix |
| `<drawing id=…>` tags + `*.drawings.json` VLM analyses | Tag replaced by the caption (`name`, `description`) plus OCR text when present; decorative images under 64 px are dropped |
| Captions in another language than the corpus | Optional batch translation through the configured LLM, cached on disk so it is never paid twice |
| Letter-spaced headings (`V ID A · E XI GE NC ES`) | Re-kerned before chunking |
| `blocks.jsonl` | Copied with normalised paths; `type == "content"` blocks are what the P chunker reads |

```bash
# size the run, write nothing
uv run python scripts/import_lightrag_parsed.py --source <lightrag>/data/inputs/<ws>/__parsed__ --out data/<corpus>/parsed --dry-run
# convert; add --translate to translate captions through LLM_MODEL (cached in --translation-cache)
uv run python scripts/import_lightrag_parsed.py --source <lightrag>/data/inputs/<ws>/__parsed__ --out data/<corpus>/parsed
```

The importer makes no LLM call unless `--translate` is given, and never touches
the source tree.
