#!/usr/bin/env python
"""Import LightRAG `__parsed__` sidecars into MemGraphRAG's input directory.

Why this exists
---------------
The RFE corpus (23 French PDFs, 770 pages) was already parsed once by LightRAG:
Docling with forced OCR on a GPU, then a VLM caption for every image, table and
equation. That run cost ~36 hours and produced 776 MB of artefacts that are still on
disk. Re-parsing would repeat the whole bill for the same result.

The two sidecar formats are the same by design — `docs/MemGraphRAGSidecarFormat.md`
says the MemGraphRAG format is *"Adapted from LightRAG's `*.parsed/` interchange"*,
and the P chunker (`memgraphrag/chunker/paragraph_semantic.py`) reads exactly these
`blocks.jsonl` files. So the import is mostly a copy; the work is in the three
transforms below, which turn markup the LLM cannot read into text it can:

* `<table format="json">[[…]]</table>` → a Markdown pipe table. A raw JSON matrix
  extracts as noise; the pipe table extracts as "Cas 9 / Qui vend ? / Vendeur".
* `<drawing id="…" />` → the VLM description plus its OCR text. This is what recovers
  the 193 of 770 pages (25 %) whose knowledge is trapped in an image — without paying
  for a single vision call.
* Kerned titles (`V ID A · E XI GE NC ES`) → `VIDA · EXIGENCES`. Left alone they are
  unsearchable and inflate the token count.

Usage:
    uv run python scripts/import_lightrag_parsed.py --dry-run
    uv run python scripts/import_lightrag_parsed.py --out data/inputs/rfe
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_SOURCE = Path(
    "/home/sanda/Desktop/project/lightrag/cf_lightrag/data/inputs/default/__parsed__"
)

# `<table id="…" format="json">[[…]]</table>` — the payload is a JSON matrix.
TABLE_RE = re.compile(
    r'<table\b[^>]*\bid="(?P<id>[^"]*)"[^>]*>(?P<payload>.*?)</table>',
    re.DOTALL,
)
# `<drawing id="…" format="png" path="…" src="" />` — self-closing.
DRAWING_RE = re.compile(r'<drawing\b[^>]*\bid="(?P<id>[^"]*)"[^>]*/?>')
EQUATION_RE = re.compile(
    r'<equation\b[^>]*\bid="(?P<id>[^"]*)"[^>]*>(?P<payload>.*?)</equation>',
    re.DOTALL,
)

# A run of >= 3 single letters separated by spaces is kerning, not words.
KERNING_RE = re.compile(r"(?<![^\W\d_])(?:[^\W\d_] ){2,}[^\W\d_](?![^\W\d_])")


@dataclass
class Stats:
    documents: int = 0
    blocks: int = 0
    tables_inlined: int = 0
    drawings_inlined: int = 0
    equations_inlined: int = 0
    drawings_skipped_tiny: int = 0
    drawings_without_caption: int = 0
    kerning_fixed: int = 0
    characters: int = 0
    per_document: list[tuple[str, int, int, int]] = field(default_factory=list)


@lru_cache(maxsize=1)
def _tokenizer():
    """One tokenizer for the whole run: building it per block dominated the runtime."""
    from memgraphrag.utils.tokenizer import TiktokenTokenizer

    return TiktokenTokenizer()


CAPTION_TRANSLATION_PROMPT = (
    "Tu traduis en français des légendes d'images extraites de documents sur la "
    "Réforme de la Facture Électronique. Traduis fidèlement, garde les sigles "
    "(RFE, PA, PDP, TVA, Factur-X, UBL, CII, B2B, e-invoicing) et les noms propres "
    "tels quels. Ne commente pas, ne reformule pas.\n\n"
    'Entrée : un objet JSON {"1": "...", "2": "..."}.\n'
    "Sortie : le MÊME objet JSON avec les mêmes clés et les valeurs traduites. "
    "Rien d'autre."
)


def _caption_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


async def translate_captions(
    texts: list[str],
    cache: dict[str, str],
    batch_size: int = 3,
    concurrency: int = 12,
) -> dict[str, str]:
    """Translate VLM captions to French, caching by content hash.

    LightRAG generated all 3 378 captions with SUMMARY_LANGUAGE=English on a French
    corpus, so entities extracted from images ("invoice", "seller") would never join
    the ones extracted from the surrounding French text ("facture", "vendeur") — which
    is exactly what fragments the schema layer and neutralises the ontology filter.

    Batches are small on purpose, and smaller than they look like they should be. At
    40 captions per call all 31 batches came back empty; at 8 the response still cut
    off mid-sentence around 5 200 characters — the served model stops well before the
    requested `max_tokens`, so the JSON was never closed and `json_repair` salvaged
    nothing. Three captions (~2 000 characters out) leaves a wide margin.

    The cache is keyed by the source text, so re-running the import is free.
    """
    from memgraphrag.llm.openai_compatible import openai_complete
    from memgraphrag.utils.json_llm import extract_json_object

    pending = [t for t in dict.fromkeys(texts) if _caption_key(t) not in cache]
    if not pending:
        return cache

    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    print(f"traduction : {len(pending)} légendes en {len(batches)} appels LLM")

    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    failed = 0

    async def _translate(batch: list[str]) -> dict[str, str]:
        nonlocal done, failed
        payload = {str(i): t for i, t in enumerate(batch, start=1)}
        async with semaphore:
            try:
                raw = await openai_complete(
                    json.dumps(payload, ensure_ascii=False),
                    system_prompt=CAPTION_TRANSLATION_PROMPT,
                    agent="import.translate_caption",
                    llm_action="complete",
                    max_tokens=4096,
                )
                translated = extract_json_object(str(raw))
            except Exception as exc:
                failed += 1
                print(f"  lot en échec ({type(exc).__name__}); légendes gardées en anglais")
                return {}
        out: dict[str, str] = {}
        if isinstance(translated, dict):
            for i, source in enumerate(batch, start=1):
                value = translated.get(str(i))
                if isinstance(value, str) and value.strip():
                    out[_caption_key(source)] = value.strip()
        if len(out) < len(batch):
            # A truncated response parses to nothing and used to be counted as a
            # success, so the run reported "0 failures" while translating 15 %.
            failed += 1
        done += len(out)
        return out

    for result in await asyncio.gather(*(_translate(b) for b in batches)):
        cache.update(result)
    print(
        f"traduites : {done}/{len(pending)} légendes ; "
        f"{failed}/{len(batches)} lots incomplets ou en échec"
    )
    return cache


def unkern(text: str) -> tuple[str, int]:
    """Reglue letter-spaced runs. Returns (text, number of runs fixed)."""
    fixed = 0

    def _join(match: re.Match[str]) -> str:
        nonlocal fixed
        fixed += 1
        return match.group(0).replace(" ", "")

    return KERNING_RE.sub(_join, text), fixed


def table_to_markdown(payload: str) -> str | None:
    """Render a JSON matrix as a Markdown pipe table, or None if unusable."""
    try:
        rows = json.loads(payload.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(rows, list) or not rows:
        return None
    matrix = [
        [str(c) if c is not None else "" for c in row] for row in rows if isinstance(row, list)
    ]
    if not matrix:
        return None
    width = max(len(r) for r in matrix)
    matrix = [r + [""] * (width - len(r)) for r in matrix]

    header, *body = matrix
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def drawing_to_text(entry: dict | None, translations: dict[str, str] | None = None) -> str | None:
    """Render a VLM-captioned drawing as prose the extractor can read."""
    if not entry:
        return None
    parts: list[str] = []
    result = entry.get("llm_analyze_result") or {}
    if isinstance(result, dict) and str(result.get("status", "success")) == "success":
        name = str(result.get("name") or "").strip()
        description = str(result.get("description") or "").strip()
        if name:
            # Names are machine slugs ("schema_des_4_coins"); readable, not worth a call.
            parts.append(name if name.endswith(".") else f"{name}.")
        if description:
            if translations:
                description = translations.get(_caption_key(description), description)
            parts.append(description)
    caption = str(entry.get("caption") or "").strip()
    if caption:
        parts.append(caption)
    ocr = entry.get("ocr_texts")
    if isinstance(ocr, list) and ocr:
        texts = [str(t).strip() for t in ocr if str(t).strip()]
        if texts:
            # OCR is already in the document's language; keep it verbatim.
            parts.append("Texte lu dans l'image : " + " ; ".join(texts))
    if not parts:
        return None
    return "[Image] " + " ".join(parts)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def transform_block(
    content: str,
    drawings: dict,
    tables: dict,
    equations: dict,
    stats: Stats,
    translations: dict[str, str] | None = None,
) -> str:
    """Replace markup with readable text, in place, preserving block order."""

    def _table(match: re.Match[str]) -> str:
        payload = match.group("payload")
        rendered = table_to_markdown(payload)
        if rendered is None:
            entry = tables.get(match.group("id")) or {}
            rendered = table_to_markdown(json.dumps(entry.get("cells") or []))
        if rendered is None:
            return ""
        stats.tables_inlined += 1
        return "\n" + rendered + "\n"

    def _drawing(match: re.Match[str]) -> str:
        entry = drawings.get(match.group("id")) or {}
        rendered = drawing_to_text(entry, translations)
        if rendered is None:
            # LightRAG skips anything under 64px: bullets, rules, logos. They carry
            # no knowledge, so dropping them is correct, not a loss.
            status = str((entry.get("llm_analyze_result") or {}).get("status") or "")
            if status == "skipped":
                stats.drawings_skipped_tiny += 1
            else:
                stats.drawings_without_caption += 1
            return ""
        stats.drawings_inlined += 1
        return "\n" + rendered + "\n"

    def _equation(match: re.Match[str]) -> str:
        entry = equations.get(match.group("id")) or {}
        latex = str(entry.get("latex") or match.group("payload") or "").strip()
        if not latex:
            return ""
        stats.equations_inlined += 1
        return f"\n[Équation] {latex}\n"

    content = TABLE_RE.sub(_table, content)
    content = EQUATION_RE.sub(_equation, content)
    content = DRAWING_RE.sub(_drawing, content)
    content, fixed = unkern(content)
    stats.kerning_fixed += fixed
    # Collapse the blank lines the substitutions leave behind, but never touch
    # intra-line spacing: it is the only thing holding table columns apart.
    return re.sub(r"\n{3,}", "\n\n", content).strip()


def import_document(
    parsed_dir: Path,
    out_dir: Path,
    stats: Stats,
    dry_run: bool,
    translations: dict[str, str] | None = None,
) -> None:
    blocks_files = sorted(parsed_dir.glob("*.blocks.jsonl"))
    if not blocks_files:
        return
    blocks_path = blocks_files[0]
    stem = blocks_path.name[: -len(".blocks.jsonl")]

    drawings = (_load_json(parsed_dir / f"{stem}.drawings.json") or {}).get("drawings") or {}
    tables = (_load_json(parsed_dir / f"{stem}.tables.json") or {}).get("tables") or {}
    equations = (_load_json(parsed_dir / f"{stem}.equations.json") or {}).get("equations") or {}

    rows = []
    for line in blocks_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    out_rows = []
    doc_chars = 0
    doc_blocks = 0
    for row in rows:
        if row.get("type") != "content":
            out_rows.append(row)
            continue
        transformed = transform_block(
            str(row.get("content") or ""), drawings, tables, equations, stats, translations
        )
        if not transformed:
            continue
        row = dict(row)
        row["content"] = transformed
        out_rows.append(row)
        doc_blocks += 1
        doc_chars += len(transformed)

    stats.documents += 1
    stats.blocks += doc_blocks
    stats.characters += doc_chars
    encode = _tokenizer().encode
    doc_tokens = sum(
        len(encode(str(r.get("content") or ""))) for r in out_rows if r.get("type") == "content"
    )
    stats.per_document.append((stem, doc_blocks, doc_chars, doc_tokens))

    if dry_run:
        return
    target = out_dir / f"{stem}.parsed"
    target.mkdir(parents=True, exist_ok=True)
    with (target / f"{stem}.blocks.jsonl").open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_captions(parsed_dirs: list[Path]) -> list[str]:
    """Every VLM description that will actually reach the index."""
    out: list[str] = []
    for d in parsed_dirs:
        for blocks_path in sorted(d.glob("*.blocks.jsonl")):
            stem = blocks_path.name[: -len(".blocks.jsonl")]
            drawings = (_load_json(d / f"{stem}.drawings.json") or {}).get("drawings") or {}
            for entry in drawings.values():
                result = entry.get("llm_analyze_result") or {}
                if str(result.get("status", "")) != "success":
                    continue
                description = str(result.get("description") or "").strip()
                if description:
                    out.append(description)
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=REPO / "data/inputs/rfe/__parsed__")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--translate",
        action="store_true",
        help="translate VLM captions to French (cached; costs LLM calls once)",
    )
    parser.add_argument(
        "--translation-cache",
        type=Path,
        default=REPO / "data/rfe/caption_translations.json",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"source introuvable : {args.source}")
        return 2

    parsed_dirs = sorted(d for d in args.source.iterdir() if d.name.endswith(".parsed"))
    # Only the PDF corpus; the HTML side-documents were ingested separately.
    parsed_dirs = [d for d in parsed_dirs if d.name.endswith(".pdf.parsed")]
    if args.limit:
        parsed_dirs = parsed_dirs[: args.limit]

    translations: dict[str, str] = {}
    cache_path = Path(args.translation_cache)
    if cache_path.exists():
        translations = _load_json(cache_path) or {}
        print(f"cache de traduction : {len(translations)} légendes déjà traduites")

    if args.translate:
        from memgraphrag.api.config import load_env_file

        load_env_file(str(REPO / ".env"))
        captions = collect_captions(parsed_dirs)
        translations = asyncio.run(translate_captions(captions, translations))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(translations, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        print(f"cache écrit : {cache_path} ({len(translations)} entrées)")

    stats = Stats()
    for d in parsed_dirs:
        import_document(d, args.out, stats, args.dry_run, translations or None)

    print(f"{'DRY RUN — ' if args.dry_run else ''}source: {args.source}")
    if not args.dry_run:
        print(f"sortie : {args.out}")
    print(f"\ndocuments {stats.documents} · blocs {stats.blocks} · {stats.characters:,} caractères")
    print(
        f"tableaux inlinés {stats.tables_inlined} · images inlinées "
        f"{stats.drawings_inlined} · décoratives ignorées (<64px) "
        f"{stats.drawings_skipped_tiny} · sans légende exploitable "
        f"{stats.drawings_without_caption} · équations {stats.equations_inlined} · "
        f"titres décrénés {stats.kerning_fixed}"
    )

    # Tokens are what drives the ingestion bill, so report them, not characters.
    total_tokens = sum(t for _, _, _, t in stats.per_document)
    print(f"total tokens (cl100k) : {total_tokens:,}")
    for size in (400, 1200, 2000):
        print(f"  -> ~{total_tokens // size + stats.documents} chunks a CHUNK_SIZE={size}")

    print(f"\n{'document':52} {'blocs':>6} {'car.':>10} {'tokens':>9}")
    for name, blocks, chars, tokens in sorted(stats.per_document, key=lambda x: -x[3]):
        print(f"{name[:52]:52} {blocks:>6} {chars:>10,} {tokens:>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
