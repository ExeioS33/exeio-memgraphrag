# Parser Service Deployment (Docling)

MemGraphRAG's Docling parser is a remote HTTP adapter (same pattern as LightRAG).

## Compose profile

```bash
# .env
DOCLING_ENDPOINT=http://docling:5001

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

- OCR-heavy PDFs can take minutes — parsing runs in background workers. Raise
  `DOCLING_MAX_POLLS` (× `DOCLING_POLL_INTERVAL_SECONDS`) rather than expecting a
  timeout knob.
- OCR is **not** steerable from here. The adapter posts no conversion options, so
  the docling-serve instance's own defaults decide; `DOCLING_DO_OCR` and
  `DOCLING_FORCE_OCR` are read nowhere in this repository. Configure OCR on the
  docling-serve side.
- `MAX_PARALLEL_PARSE_DOCLING` and `MEMGRAPHRAG_FORCE_REPARSE_DOCLING` are likewise
  not implemented — `grep -rn` finds no reader for either, and the adapter has no
  parse-artifact cache to bypass in the first place.
