# Parser Service Deployment (Docling)

MemGraphRAG's Docling parser is a remote HTTP adapter (same pattern as LightRAG).

## Compose profile

```bash
# .env
DOCLING_ENDPOINT=http://docling:5001
DOCLING_DO_OCR=true

docker compose --profile docling up -d
```

The `docling` service uses `ghcr.io/docling-project/docling-serve:latest` and is **not** started by default (keeps the POC at three services).

## Routing

```bash
# Prefer Docling for PDFs, legacy elsewhere
MEMGRAPHRAG_PARSER=pdf:docling-P,*:legacy-F
```

Filename hints also work: `report.[docling-P].pdf`.

The Docling engine is only selectable when `DOCLING_ENDPOINT` is set; otherwise `get_parser("docling")` raises `ParserUnavailableError`.

## Operational notes

- OCR-heavy PDFs can take minutes — parsing runs in background workers.
- Set `MAX_PARALLEL_PARSE_DOCLING` to cap concurrency.
- `MEMGRAPHRAG_FORCE_REPARSE_DOCLING=true` bypasses cached parse artifacts.
