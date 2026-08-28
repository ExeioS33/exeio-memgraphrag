#!/usr/bin/env python
"""Ingest the RFE corpus into MemGraphRAG from imported LightRAG sidecars.

Bypasses the file pipeline on purpose. `process_pending` parses a file and then
chunks it, and nothing in the repository reads an *existing* sidecar — despite what
`docs/MemGraphRAGSidecarFormat.md` claims about resume. Since the whole point is not
to re-parse, this script chunks the sidecars directly with the P chunker and calls
`ainsert` **once** for the entire corpus.

That single call matters as much as the reuse. `ainsert` reloads the whole corpus,
replays conflict detection and rebuilds the graph from `clear()` every time it runs,
so feeding 23 documents one at a time costs ~1 100 redundant conflict LLM calls and
23 full graph rebuilds. One call pays that once.

Usage:
    uv run python scripts/ingest_rfe.py --dry-run          # size it, no LLM
    uv run python scripts/ingest_rfe.py --workspace rfe_mgr
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memgraphrag.api.config import load_env_file, resolve_storage_backends  # noqa: E402

DEFAULT_SIDECARS = REPO / "data/rfe/parsed"


def build_chunks(sidecar_root: Path, chunk_size: int, overlap: int) -> list[dict]:
    """Chunk every imported sidecar with the P (paragraph-semantic) chunker.

    P is the only chunker that respects the document's own structure: it merges whole
    blocks up to the budget and carries their heading, so a chunk boundary never lands
    in the middle of a table. F would slice this corpus blind — the profile measured
    that a page averages 212 tokens, so a fixed window crosses tables constantly.
    """
    from memgraphrag.chunker.paragraph_semantic import chunking_by_paragraph_semantic
    from memgraphrag.utils.hashing import compute_mdhash_id
    from memgraphrag.utils.tokenizer import TiktokenTokenizer

    tokenizer = TiktokenTokenizer()
    chunks: list[dict] = []
    per_doc: list[tuple[str, int, int]] = []

    for parsed_dir in sorted(sidecar_root.glob("*.parsed")):
        blocks = sorted(parsed_dir.glob("*.blocks.jsonl"))
        if not blocks:
            continue
        blocks_path = blocks[0]
        stem = blocks_path.name[: -len(".blocks.jsonl")]

        pieces = chunking_by_paragraph_semantic(
            tokenizer,
            "",  # content is unused when a sidecar is supplied
            chunk_token_size=chunk_size,
            blocks_path=str(blocks_path),
            chunk_overlap_token_size=overlap,
            doc_id=stem,
        )
        tokens = 0
        for piece in pieces:
            content = str(piece.get("content") or "").strip()
            if not content:
                continue
            # Prefix the document title so a retrieved passage can be traced back and
            # cited; `sources` is otherwise empty on the ainsert path.
            body = f"{stem}\n\n{content}"
            chunks.append(
                {
                    "idx": compute_mdhash_id(body, prefix="chunk-"),
                    "content": body,
                    "source": stem,
                }
            )
            tokens += int(piece.get("tokens") or 0)
        per_doc.append((stem, len(pieces), tokens))

    print(f"{'document':52} {'chunks':>7} {'tokens':>9}")
    for name, n, tokens in sorted(per_doc, key=lambda x: -x[1]):
        print(f"{name[:52]:52} {n:>7} {tokens:>9,}")
    print(f"\ntotal : {len(chunks)} chunks sur {len(per_doc)} documents")
    return chunks


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecars", type=Path, default=DEFAULT_SIDECARS)
    parser.add_argument("--workspace", default="rfe_mgr")
    parser.add_argument("--chunk-size", type=int, default=0, help="0 = CHUNK_SIZE from env")
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="size the run, no LLM call")
    parser.add_argument("--limit", type=int, default=0, help="ingest only the first N chunks")
    parser.add_argument("--no-conflicts", action="store_true")
    args = parser.parse_args()

    load_env_file(str(REPO / ".env"))
    # Nothing in the engine configures logging. On file backends nano-vectordb's
    # import-time basicConfig happened to show INFO; on database backends the
    # stage and checkpoint lines silently vanished. Configure it explicitly.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    from memgraphrag.constants import CHUNK_OVERLAP_SIZE, CHUNK_SIZE
    from memgraphrag.utils.env import get_env_value

    chunk_size = args.chunk_size or get_env_value("CHUNK_SIZE", CHUNK_SIZE, int)
    overlap = args.overlap or get_env_value("CHUNK_OVERLAP_SIZE", CHUNK_OVERLAP_SIZE, int)

    if not args.sidecars.exists():
        print(f"sidecars introuvables : {args.sidecars} — lancez d'abord import_lightrag_parsed.py")
        return 2

    print(f"sidecars   : {args.sidecars}")
    print(f"chunking   : P, {chunk_size} tokens / {overlap} de recouvrement")
    chunks = build_chunks(args.sidecars, chunk_size, overlap)
    if args.limit:
        chunks = chunks[: args.limit]
        print(f"limité à {len(chunks)} chunks")
    if not chunks:
        print("aucun chunk produit")
        return 1

    # The workspace is what keeps this run away from LightRAG's graph; Neo4JStorage
    # refuses to start on a workspace another engine populated, but say it out loud.
    os.environ["WORKSPACE"] = args.workspace
    print(f"workspace  : {args.workspace}")
    print(f"langue     : {os.getenv('MEMGRAPHRAG_LANGUAGE', 'auto')}")
    print(f"LLM        : {os.getenv('LLM_MODEL')} @ {os.getenv('LLM_BINDING_HOST')}")
    print(f"concurrence: MAX_ASYNC_LLM={os.getenv('MAX_ASYNC_LLM', '4')}")

    from memgraphrag.core import MemGraphRAG
    from memgraphrag.llm.openai_compatible import openai_complete, openai_embed

    async def llm_model_func(prompt: str, **kwargs):
        return str(await openai_complete(prompt, model=os.getenv("LLM_MODEL"), **kwargs))

    async def embedding_func(texts, **kwargs):
        return await openai_embed(
            texts,
            model=os.getenv("EMBEDDING_MODEL"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM") or 1024),
            **kwargs,
        )

    # MemGraphRAG's constructor defaults are literals (core.py:233-236); only
    # api/config.py reads MEMGRAPHRAG_*_STORAGE. A script that omits them silently
    # ignores the configured backends and always writes files.
    backends = resolve_storage_backends()
    print("stockage   : " + ", ".join(f"{k.split('_')[0]}={v}" for k, v in backends.items()))
    rag = MemGraphRAG(
        working_dir=str(REPO / "data/rfe/storage"),
        workspace=args.workspace,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        embedding_dim=int(os.getenv("EMBEDDING_DIM") or 1024),
        max_async_llm=int(os.getenv("MAX_ASYNC_LLM") or 4),
        **backends,
    )
    await rag.initialize_storages()

    # --dry-run used to return before this point, so it checked chunking and nothing
    # else. That is exactly how a run announced as "Postgres + Neo4j" went to files
    # instead: the only evidence anyone had was a dry-run that never built a storage.
    # It now constructs the real backends and reports what it actually reached.
    if args.dry_run:
        nodes = len(await rag.graph.get_all_nodes())
        print(f"\nDRY RUN — {nodes} noeuds dans le workspace, aucun appel LLM")
        await rag.finalize_storages()
        return 0

    started = time.perf_counter()
    stats = await rag.ainsert(chunks, run_conflicts=not args.no_conflicts)
    elapsed = time.perf_counter() - started

    print(f"\ningestion terminée en {elapsed / 60:.1f} min")
    # `ainsert` also returns the memory object and every OpenIE record; only the
    # counters are worth printing.
    print(json.dumps(stats.get("stats", stats), indent=2, ensure_ascii=False, default=str))

    await rag.prepare_retrieval()
    edges = await rag.graph.get_all_edges()
    by_type: dict[str, int] = {}
    for e in edges:
        by_type[str(e.get("type"))] = by_type.get(str(e.get("type")), 0) + 1
    print(f"\ngraphe : {len(await rag.graph.get_all_nodes())} noeuds, {len(edges)} aretes")
    for k in sorted(by_type):
        print(f"  {k:16} {by_type[k]}")

    await rag.finalize_storages()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
