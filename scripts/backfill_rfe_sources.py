#!/usr/bin/env python
"""Restore document provenance for an RFE corpus that was ingested without any.

`scripts/ingest_rfe.py` bypasses the file pipeline and calls `ainsert` directly, so
no doc-status record was ever written. `MemGraphRAG.prepare_retrieval` builds
`_passage_id_to_source` **exclusively** from doc-status — `file_path` + `chunk_ids`,
`core.py` around line 2121 — which is exactly why every citation in an answer reads
"unknown". No amount of re-querying fixes that; the mapping has to be put back.

It is recoverable because a chunk id is a pure function of the chunk body:
`_normalize_chunks` (core.py) drops every key except idx/content, and
`_assign_chunk_ids` (pipeline.py) hashes that content into `chunk-<md5>`. Re-running
the *same* chunking over the *same* sidecars therefore reproduces the *same* ids.
This script re-runs it by importing `build_chunks` from `ingest_rfe` — never by
copying the logic — because a chunking option that drifted apart from the ingest
script would still produce a confident-looking backfill, with the wrong filenames.

Nothing is written before the regenerated ids have been checked against the Passage
nodes actually present in Neo4j. Below `--min-overlap` the script refuses and says
so: a misaligned backfill attaches wrong sources to passages, and an answer that
cites the wrong document is worse than one that cites none.

Usage:
    uv run python scripts/backfill_rfe_sources.py                  # verify only
    uv run python scripts/backfill_rfe_sources.py --write --dry-run
    uv run python scripts/backfill_rfe_sources.py --write
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))

from memgraphrag.api.config import load_env_file, resolve_storage_backends  # noqa: E402
from memgraphrag.utils.canonical import canonical_key  # noqa: E402

DEFAULT_PARSED_ROOT = REPO / "data/rfe/parsed"
DEFAULT_PDF_ROOT = Path("~/Desktop/project/lightrag/cf_lightrag/data/rfe-igor")
# ingest_rfe.py hardcodes the same working dir; the file-backed backends resolve
# their JSON under it, so a different value here would read an empty doc-status.
WORKING_DIR = REPO / "data/rfe/storage"

#: Extensions worth considering as the origin of a sidecar. Anything else under the
#: PDF root (thumbnails, .DS_Store, exports) is neither matched nor reported orphan.
DOCUMENT_SUFFIXES = frozenset(
    {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".md", ".txt", ".html", ".htm"}
)

#: Rows per UNWIND. Same order of magnitude as Neo4JStorage's own batched writer.
GRAPH_BATCH_SIZE = 1000


# ----------------------------------------------------------------------------------
# Chunk regeneration
# ----------------------------------------------------------------------------------


@dataclass
class SourceChunks:
    """Everything one source document contributed to the corpus."""

    ids: list[str] = field(default_factory=list)
    content_length: int = 0
    preview: str = ""


def regenerate_chunk_map(
    parsed_root: Path, chunk_size: int, overlap: int
) -> tuple[dict[str, SourceChunks], list[str], int]:
    """Return ``{source stem: SourceChunks}``, the flat id list, and any id mismatch.

    The ids come out of `compute_mdhash_id(content, prefix="chunk-")`, which is the
    single derivation `pipeline._assign_chunk_ids` uses. Recomputing it here rather
    than trusting the ``idx`` `build_chunks` already set is what makes the two paths
    provably identical: a mismatch means the invariant this backfill rests on has
    broken, and it is counted and reported instead of silently absorbed.
    """
    from ingest_rfe import build_chunks

    from memgraphrag.utils.hashing import compute_mdhash_id

    by_source: dict[str, SourceChunks] = {}
    all_ids: list[str] = []
    mismatches = 0

    for chunk in build_chunks(parsed_root, chunk_size, overlap):
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        chunk_id = compute_mdhash_id(content, prefix="chunk-")
        if str(chunk.get("idx") or "") != chunk_id:
            mismatches += 1
        all_ids.append(chunk_id)
        source = str(chunk.get("source") or "").strip()
        if not source:
            continue
        entry = by_source.setdefault(source, SourceChunks())
        entry.ids.append(chunk_id)
        entry.content_length += len(content)
        if not entry.preview:
            entry.preview = content

    return by_source, all_ids, mismatches


# ----------------------------------------------------------------------------------
# Source stem -> real file
# ----------------------------------------------------------------------------------


@dataclass
class Matching:
    """Outcome of pairing sidecar stems with files under the PDF root."""

    matched: dict[str, Path] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    ambiguous: dict[str, list[Path]] = field(default_factory=dict)
    orphans: list[Path] = field(default_factory=list)


def _match_keys(name: str) -> list[str]:
    """Normalised lookup keys for a name, with and without a trailing extension."""
    keys = [canonical_key(name)]
    stripped = Path(name).stem
    if stripped and stripped != name:
        keys.append(canonical_key(stripped))
    return [k for k in keys if k]


def match_sources(stems: list[str], pdf_root: Path) -> Matching:
    """Pair each source stem with one real file, searching ``pdf_root`` recursively.

    Exact stem equality wins; otherwise the comparison is folded through
    `canonical_key` (NFKC, accents, case) with the extension stripped — the same fold
    the engine uses on labels. A stem that resolves to more than one file is reported
    ambiguous and left unmatched: guessing would attach a plausible wrong filename.
    """
    files = [
        path
        for path in sorted(pdf_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    ]

    by_stem: dict[str, list[Path]] = {}
    by_key: dict[str, list[Path]] = {}
    for path in files:
        by_stem.setdefault(path.stem, []).append(path)
        for key in {*_match_keys(path.name), *_match_keys(path.stem)}:
            bucket = by_key.setdefault(key, [])
            if path not in bucket:
                bucket.append(path)

    result = Matching()
    claimed: set[Path] = set()
    for stem in stems:
        candidates = by_stem.get(stem)
        if not candidates:
            for key in _match_keys(stem):
                candidates = by_key.get(key)
                if candidates:
                    break
        if not candidates:
            result.missing.append(stem)
            continue
        if len(candidates) > 1:
            result.ambiguous[stem] = list(candidates)
            result.missing.append(stem)
            continue
        result.matched[stem] = candidates[0]
        claimed.add(candidates[0])

    result.orphans = [path for path in files if path not in claimed]
    return result


# ----------------------------------------------------------------------------------
# Storage handles
# ----------------------------------------------------------------------------------


def _build_storage(backend: str, namespace: str, workspace: str) -> Any:
    """Instantiate one storage backend the way `MemGraphRAG.__init__` does."""
    from memgraphrag.storage.factory import get_storage_class

    storage_cls = get_storage_class(backend)
    return storage_cls(
        namespace=namespace,
        workspace=workspace,
        global_config={"working_dir": str(WORKING_DIR), "workspace": workspace},
        embedding_func=None,
    )


async def open_doc_status(backend: str, workspace: str) -> Any:
    from memgraphrag.namespace import NameSpace

    storage = _build_storage(backend, NameSpace.DOC_STATUS, workspace)
    await storage.initialize()
    return storage


async def open_graph(backend: str, workspace: str) -> Any:
    """Open the graph backend.

    `Neo4JStorage` is reused rather than a hand-rolled driver on purpose: it is what
    resolves NEO4J_URI, the database probe (named database, then the default) and the
    NEO4J_WORKSPACE override. A second connection helper here would drift from the
    engine's and verify against a database the engine never wrote to.
    """
    from memgraphrag.namespace import NameSpace

    storage = _build_storage(backend, NameSpace.GRAPH_MEMORY, workspace)
    await storage.initialize()
    return storage


# ----------------------------------------------------------------------------------
# Neo4j read / write
# ----------------------------------------------------------------------------------


async def graph_passage_ids(graph: Any) -> set[str]:
    """Return the ``entity_id`` of every Passage node in the workspace."""
    workspace_label = graph._workspace_label()
    query = f"MATCH (n:`{workspace_label}`:Passage) RETURN n.entity_id AS id"
    ids: set[str] = set()
    async with graph._session(default_access_mode="READ") as session:
        result = await session.run(query)
        async for record in result:
            value = record["id"]
            if value:
                ids.add(str(value))
        await result.consume()
    return ids


async def set_passage_file_paths(graph: Any, rows: list[dict[str, str]]) -> int:
    """Stamp ``file_path`` on Passage nodes, ~1 000 rows per round trip."""
    workspace_label = graph._workspace_label()
    query = (
        "UNWIND $rows AS row\n"
        f"MATCH (n:`{workspace_label}`:Passage {{entity_id: row.id}})\n"
        "SET n.file_path = row.path\n"
        "RETURN count(n) AS updated"
    )
    updated = 0
    for start in range(0, len(rows), GRAPH_BATCH_SIZE):
        batch = rows[start : start + GRAPH_BATCH_SIZE]
        async with graph._session(default_access_mode="WRITE") as session:
            result = await session.run(query, rows=batch)
            record = await result.single()
            await result.consume()
        if record is not None:
            updated += int(record["updated"] or 0)
    return updated


# ----------------------------------------------------------------------------------
# Doc-status records
# ----------------------------------------------------------------------------------


def build_doc_record(
    path: Path,
    chunk_ids: list[str],
    preview: str,
    content_length: int,
    created_at: int | None = None,
) -> dict[str, Any]:
    """Build one PROCESSED doc-status record, shaped exactly like the pipeline's.

    Same keys as `pipeline.enqueue_document`, plus the two that
    `pipeline.process_pending` adds on the PROCESSED transition (``chunk_count`` /
    ``chunk_ids``) and ``metadata.memory_sub_stage = "done"``.
    """
    from memgraphrag.base import DocStatus

    # Imported, never re-declared: the preview cap has to be the pipeline's own.
    from memgraphrag.pipeline import CONTENT_SUMMARY_LIMIT
    from memgraphrag.utils.step_log import truncate

    engine = "legacy"
    process_options = ""
    try:
        from memgraphrag.parser.routing import resolve_parser_directives

        directives = resolve_parser_directives(str(path))
        engine = directives.engine
        process_options = directives.process_options
    except Exception:
        # Routing is a convenience here: the document is already parsed and indexed,
        # so an unroutable name must not cost the provenance. Keep the defaults.
        pass

    now = int(time.time())
    return {
        "status": DocStatus.PROCESSED.value,
        "file_path": str(path),
        "content_summary": truncate(preview, CONTENT_SUMMARY_LIMIT) if preview else "",
        "content_length": content_length,
        "parse_engine": engine,
        "process_options": process_options,
        "chunk_options": {},
        "created_at": created_at or now,
        "updated_at": now,
        "metadata": {"memory_sub_stage": "done"},
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------


def _percent(part: int, whole: int) -> float:
    return (part / whole * 100.0) if whole else 0.0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="vérification seule (défaut)")
    parser.add_argument("--write", action="store_true", help="écrire la provenance")
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_PARSED_ROOT)
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.95,
        help="recouvrement minimal exigé avant toute écriture (0-1)",
    )
    parser.add_argument("--workspace", default="rfe_mgr")
    parser.add_argument(
        "--no-write-graph",
        dest="write_graph",
        action="store_false",
        help="ne pas poser file_path sur les nœuds Passage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tout calculer et vérifier, n'écrire nulle part",
    )
    args = parser.parse_args()

    if args.verify and args.write:
        print("--verify et --write sont exclusifs : choisissez l'un ou l'autre.")
        return 2

    load_env_file(str(REPO / ".env"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    from memgraphrag.constants import CHUNK_OVERLAP_SIZE, CHUNK_SIZE
    from memgraphrag.utils.env import get_env_value
    from memgraphrag.utils.hashing import compute_mdhash_id

    parsed_root = args.parsed_root.expanduser()
    pdf_root = args.pdf_root.expanduser()
    will_write = bool(args.write) and not args.dry_run

    if not parsed_root.exists():
        print(f"sidecars introuvables : {parsed_root}")
        return 2
    if not pdf_root.exists():
        print(f"racine PDF introuvable : {pdf_root}")
        return 2

    # Same resolution as ingest_rfe: the chunking must be byte-identical to the run
    # that produced the graph, otherwise every regenerated id is a different hash.
    chunk_size = get_env_value("CHUNK_SIZE", CHUNK_SIZE, int)
    overlap = get_env_value("CHUNK_OVERLAP_SIZE", CHUNK_OVERLAP_SIZE, int)

    os.environ["WORKSPACE"] = args.workspace
    neo4j_workspace = (os.getenv("NEO4J_WORKSPACE") or "").strip()
    if neo4j_workspace and neo4j_workspace != args.workspace:
        print(
            f"ATTENTION : NEO4J_WORKSPACE={neo4j_workspace!r} l'emporte sur "
            f"--workspace={args.workspace!r} côté graphe."
        )

    backends = resolve_storage_backends()
    print(f"sidecars   : {parsed_root}")
    print(f"PDF        : {pdf_root}")
    print(f"chunking   : P, {chunk_size} tokens / {overlap} de recouvrement")
    print(f"workspace  : {args.workspace}")
    print("stockage   : " + ", ".join(f"{k.split('_')[0]}={v}" for k, v in backends.items()))
    print(f"mode       : {'ÉCRITURE' if will_write else 'VÉRIFICATION seule'}")
    print()

    by_source, all_ids, mismatches = regenerate_chunk_map(parsed_root, chunk_size, overlap)
    if not all_ids:
        print("aucun chunk régénéré — rien à rétablir")
        return 1
    if mismatches:
        print(
            f"ATTENTION : {mismatches} chunk(s) dont l'idx de build_chunks diffère du hash "
            "recalculé — la dérivation des ids a changé, le backfill n'est plus fiable."
        )

    regenerated = set(all_ids)
    matching = match_sources(sorted(by_source), pdf_root)

    for stem in matching.missing:
        if stem in matching.ambiguous:
            names = ", ".join(str(p) for p in matching.ambiguous[stem])
            print(f"AMBIGU   : {stem} -> {names}")
        else:
            print(f"SANS PDF : {stem}")
    for path in matching.orphans:
        print(f"SANS SIDECAR : {path}")
    if matching.missing or matching.orphans:
        print()

    # ---- verification against the live graph (always, before any write) ----------
    graph: Any = None
    doc_status: Any = None
    graph_ids: set[str] = set()
    coverage = 0.0
    fidelity = 0.0
    records_written = 0
    nodes_updated = 0
    exit_code = 0

    try:
        if backends["graph_storage"] != "Neo4JStorage":
            print(
                f"graphe configuré sur {backends['graph_storage']} et non Neo4JStorage : "
                "la vérification exigée avant écriture est impossible."
            )
            return 2

        graph = await open_graph(backends["graph_storage"], args.workspace)
        graph_ids = await graph_passage_ids(graph)
        common = regenerated & graph_ids
        coverage = _percent(len(common), len(graph_ids))
        fidelity = _percent(len(common), len(regenerated))
        print(
            f"vérification : {len(common)} ids communs / {len(graph_ids)} passages Neo4j "
            f"/ {len(regenerated)} ids régénérés"
        )
        print(f"  couverture des passages du graphe : {coverage:.1f} %")
        print(f"  ids régénérés retrouvés           : {fidelity:.1f} %")

        threshold = args.min_overlap * 100.0
        overlap_ok = coverage >= threshold and fidelity >= threshold
        if not overlap_ok:
            print(
                f"\nREFUS : recouvrement sous le seuil de {threshold:.1f} %. Un backfill "
                "désaligné attacherait de mauvais noms de fichiers aux passages, ce qui est "
                "pire que l'absence de citation. Vérifiez que CHUNK_SIZE / "
                "CHUNK_OVERLAP_SIZE et les sidecars sont ceux de l'ingestion."
            )
            if will_write:
                exit_code = 1
            will_write = False

        # ---- write ---------------------------------------------------------------
        if will_write:
            doc_status = await open_doc_status(backends["doc_status_storage"], args.workspace)
            existing: dict[str, dict[str, Any]] = {}
            try:
                existing = await doc_status.get_all()
            except NotImplementedError:
                existing = {}

            records: dict[str, dict[str, Any]] = {}
            rows: list[dict[str, str]] = []
            for stem, path in sorted(matching.matched.items()):
                resolved = path.resolve()
                entry = by_source.get(stem)
                if entry is None or not entry.ids:
                    continue
                doc_id = compute_mdhash_id(str(resolved), prefix="doc-")
                previous = existing.get(doc_id) or {}
                records[doc_id] = build_doc_record(
                    resolved,
                    entry.ids,
                    preview=entry.preview,
                    content_length=entry.content_length,
                    created_at=previous.get("created_at"),
                )
                rows.extend({"id": cid, "path": str(resolved)} for cid in entry.ids)

            if records:
                await doc_status.upsert(records)
                records_written = len(records)
            print(f"\ndoc-status : {records_written} enregistrement(s) écrit(s)")

            if args.write_graph and rows:
                nodes_updated = await set_passage_file_paths(graph, rows)
                print(f"graphe     : {nodes_updated} nœud(s) Passage mis à jour")
            elif not args.write_graph:
                print("graphe     : ignoré (--no-write-graph)")
        elif args.dry_run and args.write:
            planned = sum(len(by_source[s].ids) for s in matching.matched if s in by_source)
            print(
                f"\nDRY RUN : {len(matching.matched)} enregistrement(s) et {planned} "
                "nœud(s) auraient été écrits."
            )
    finally:
        if doc_status is not None:
            await doc_status.finalize()
        if graph is not None:
            await graph.finalize()

    # ---- French summary ----------------------------------------------------------
    unmapped = len(graph_ids - regenerated)
    print("\nRÉSUMÉ")
    print(f"  {'documents sidecar':32} {len(by_source):>8}")
    print(f"  {'documents appariés à un PDF':32} {len(matching.matched):>8}")
    print(f"  {'documents sans PDF':32} {len(matching.missing):>8}")
    print(f"  {'PDF sans sidecar':32} {len(matching.orphans):>8}")
    print(f"  {'chunks régénérés':32} {len(regenerated):>8}")
    print(f"  {'passages dans le graphe':32} {len(graph_ids):>8}")
    print(f"  {'passages non identifiés':32} {unmapped:>8}")
    print(f"  {'couverture du graphe':32} {coverage:>7.1f} %")
    print(f"  {'ids régénérés retrouvés':32} {fidelity:>7.1f} %")
    print(f"  {'enregistrements écrits':32} {records_written:>8}")
    print(f"  {'nœuds Passage mis à jour':32} {nodes_updated:>8}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
