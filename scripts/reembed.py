#!/usr/bin/env python
"""Recompute stored vectors after an embedding-model change, without re-ingesting.

Changing ``EMBEDDING_MODEL`` re-embeds nothing. ``_embed_and_store_memory``
(``memgraphrag/core.py``) only embeds ids it has not seen, and ids are content
hashes (``memgraphrag/utils/hashing.py``), so an unchanged corpus yields an empty
work list. The stored vectors stay in the old model's space while queries are
embedded in the new one. When both models share a dimension nothing raises:
``_assert_embedding_dim`` compares integers, and no model name is stored beside a
vector. Retrieval then degrades silently and completely -- measured on one corpus,
top-5 overlap fell to zero and the best cosine similarity from 0.9 to 0.05, which
is what an orthogonal vector looks like.

Re-ingesting is not needed to fix it: the text that was embedded is already in the
table, so every vector can be recomputed from it.

    chunks / entities  ->  the ``content`` column, verbatim
    facts  / schemas   ->  "\\t".join(metadata["triple"])

For facts and schemas ``content`` holds ``str(tuple(...))`` (``_triple_str``), not
the tab-joined string that was actually embedded, so only ``metadata["triple"]``
reproduces the original input.

Everything else stays valid and untouched -- the memory row, the OpenIE cache, the
chunk texts, the doc-status rows and the graph, which stores no vector. No LLM call
is made.

Documents are embedded with no instruction prefix, matching the engine
(``core.py`` calls ``embedding_func(texts)`` and ``openai_embed`` only prefixes
when ``context="query"``).

Usage:
    uv run python scripts/reembed.py --workspace default --dry-run
    uv run python scripts/reembed.py --workspace default --shadow
    uv run python scripts/reembed.py --workspace default --promote
    uv run python scripts/reembed.py --workspace default --in-place

``--shadow`` builds ``<table>__reembed`` beside the live table and leaves it there;
``--promote`` swaps it in atomically, keeping the old table as ``<table>__prev``.
Splitting the two means the new vectors can be checked before anything is served.
``--in-place`` rewrites the live table directly: it is faster and needs no extra
storage, but the table mixes both vector spaces while it runs, so retrieval returns
nonsense until it finishes.
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
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memgraphrag.api.config import load_env_file  # noqa: E402

logger = logging.getLogger("reembed")

#: Vector namespaces, in ascending cost order so a run fails early and cheaply.
VECTOR_NAMESPACES = ("chunks", "schemas", "entities", "facts")
#: Namespaces whose embedded text is the tab-joined triple, not ``content``.
TRIPLE_NAMESPACES = frozenset({"facts", "schemas"})

SHADOW_SUFFIX = "__reembed"
BACKUP_SUFFIX = "__prev"
#: PostgreSQL truncates identifiers beyond this, which would silently alias tables.
PG_MAX_IDENTIFIER = 63


def embedded_text(namespace: str, row: Any) -> str | None:
    """Return the exact text the engine embedded for this row, or None to skip.

    Rows with no usable text are skipped rather than embedded as an empty string:
    an empty input yields an arbitrary vector that would pollute the index.
    """
    if namespace in TRIPLE_NAMESPACES:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata or "{}")
        triple = (metadata or {}).get("triple")
        if not triple:
            return None
        return "\t".join(str(part) for part in triple)
    content = row["content"]
    return content if content else None


def live_table(workspace: str, namespace: str) -> str:
    """Mirror ``PGVectorStorage.__post_init__``'s table naming."""
    from memgraphrag.storage.postgres_impl import _safe_ident

    return f"mgr_vec_{_safe_ident(workspace or 'default')}_{_safe_ident(namespace)}"


def _check_identifier(name: str) -> str:
    if len(name) > PG_MAX_IDENTIFIER:
        raise SystemExit(
            f"Table name {name!r} is {len(name)} characters, over PostgreSQL's "
            f"{PG_MAX_IDENTIFIER}-character limit. Shorten the workspace name."
        )
    return name


async def _table_exists(conn: Any, table: str) -> bool:
    return bool(await conn.fetchval("SELECT to_regclass($1)", f"public.{table}"))


async def _row_count(conn: Any, table: str) -> int:
    return int(await conn.fetchval(f'SELECT count(*) FROM public."{table}"') or 0)


async def _create_shadow(conn: Any, shadow: str, dim: int) -> None:
    """Create the shadow table with the live schema, but no vector index yet.

    The index is built after the rows land: filling an empty HNSW index row by row
    is markedly slower than indexing a populated table once.
    """
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {shadow} (
            id TEXT PRIMARY KEY,
            content TEXT,
            embedding VECTOR({dim}),
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
        )
        """
    )


async def _embed(texts: list[str], model: str, dim: int) -> list[list[float]]:
    """Embed one batch and enforce the width the storage expects.

    ``openai_embed`` returns whatever the provider sent; nothing on the write path
    checks it, so a provider serving another dimension surfaces only as an opaque
    pgvector cast error. Failing here names the real cause.
    """
    from memgraphrag.llm.openai_compatible import openai_embed

    vectors = await openai_embed(texts, model=model, embedding_dim=dim)
    if vectors.shape[0] != len(texts):
        raise SystemExit(f"Provider returned {vectors.shape[0]} vectors for {len(texts)} texts.")
    if vectors.shape[1] != dim:
        raise SystemExit(
            f"Model {model!r} returned {vectors.shape[1]}-dimension vectors but "
            f"EMBEDDING_DIM is {dim}. Fix EMBEDDING_DIM (and re-create the tables "
            f"if it really changed) before re-embedding."
        )
    return [row.tolist() for row in vectors]


async def reembed_namespace(
    pool: Any,
    workspace: str,
    namespace: str,
    *,
    model: str,
    dim: int,
    batch_size: int,
    limit: int,
    shadow_mode: bool,
    dry_run: bool,
) -> dict[str, int]:
    """Re-embed one vector table. Returns per-table counters."""
    table = _check_identifier(live_table(workspace, namespace))
    shadow = _check_identifier(f"{table}{SHADOW_SUFFIX}")
    target = shadow if shadow_mode else table

    async with pool.acquire() as conn:
        if not await _table_exists(conn, table):
            logger.info("%-34s absent, skipped", table)
            return {"total": 0, "done": 0, "skipped": 0}
        total = await _row_count(conn, table)
        if dry_run:
            logger.info("%-34s %7d vectors to re-embed", table, total)
            return {"total": total, "done": 0, "skipped": 0}
        if shadow_mode:
            await _create_shadow(conn, shadow, dim)
            # Resume: rows already written keep their id order, so continue past
            # the highest one instead of re-embedding what a killed run finished.
            after = await conn.fetchval(f'SELECT max(id) FROM public."{shadow}"') or ""
            if after:
                logger.info("%-34s resuming after id %s", shadow, after)
        else:
            after = ""

    done = skipped = 0
    started = time.monotonic()
    while True:
        remaining = (limit - done - skipped) if limit else batch_size
        if limit and remaining <= 0:
            break
        page_size = max(1, min(batch_size, remaining))
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f'SELECT id, content, metadata FROM public."{table}" '
                f"WHERE id > $1 ORDER BY id LIMIT $2",
                after,
                page_size,
            )
        if not rows:
            break
        after = rows[-1]["id"]

        payload: list[tuple[str, str, Any]] = []
        for row in rows:
            text = embedded_text(namespace, row)
            if text is None:
                skipped += 1
                logger.warning("%s: %s has no embeddable text, skipped", table, row["id"])
                continue
            payload.append((row["id"], text, row))
        if not payload:
            continue

        vectors = await _embed([text for _, text, _ in payload], model, dim)

        async with pool.acquire() as conn:
            if shadow_mode:
                await conn.executemany(
                    f'INSERT INTO public."{shadow}" (id, content, embedding, metadata) '
                    f"VALUES ($1, $2, $3::vector, $4::jsonb) "
                    f"ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, "
                    f"embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata",
                    [
                        (
                            row["id"],
                            row["content"],
                            str(vector),
                            row["metadata"]
                            if isinstance(row["metadata"], str)
                            else json.dumps(row["metadata"] or {}),
                        )
                        for (_, _, row), vector in zip(payload, vectors, strict=True)
                    ],
                )
            else:
                await conn.executemany(
                    f'UPDATE public."{table}" SET embedding = $2::vector WHERE id = $1',
                    [
                        (row_id, str(vector))
                        for (row_id, _, _), vector in zip(payload, vectors, strict=True)
                    ],
                )
        done += len(payload)
        elapsed = time.monotonic() - started
        logger.info(
            "%-34s %6d/%-6d  %5.1f vec/s", target, done, total, done / elapsed if elapsed else 0
        )

    if shadow_mode and done and not limit:
        from memgraphrag.storage.postgres_impl import _vector_index_ddl

        ddl = _vector_index_ddl(shadow)
        if ddl:
            logger.info("%-34s building vector index", shadow)
            async with pool.acquire() as conn:
                await conn.execute(ddl)

    return {"total": total, "done": done, "skipped": skipped}


async def promote(pool: Any, workspace: str, namespaces: tuple[str, ...]) -> None:
    """Swap every filled shadow table in for its live table, in one transaction.

    Renaming a table leaves its indexes under their old names, so the constraint and
    the vector index are renamed too. Without that, the engine's
    ``CREATE INDEX IF NOT EXISTS <table>_embedding_idx`` would not find the promoted
    index and would build a second one over the same column.
    """
    plan: list[tuple[str, str, str]] = []
    async with pool.acquire() as conn:
        for namespace in namespaces:
            table = live_table(workspace, namespace)
            shadow = f"{table}{SHADOW_SUFFIX}"
            backup = _check_identifier(f"{table}{BACKUP_SUFFIX}")
            if not await _table_exists(conn, shadow):
                logger.info("%-34s no shadow table, skipped", table)
                continue
            if await _table_exists(conn, backup):
                raise SystemExit(
                    f"{backup} already exists: a previous promote was not cleaned up. "
                    f"Drop or rename it before promoting again."
                )
            live_rows = await _row_count(conn, table)
            shadow_rows = await _row_count(conn, shadow)
            if shadow_rows != live_rows:
                raise SystemExit(
                    f"{shadow} holds {shadow_rows} rows against {live_rows} in {table}. "
                    f"Finish the re-embedding run before promoting."
                )
            plan.append((table, shadow, backup))

    if not plan:
        logger.info("nothing to promote")
        return

    async with pool.acquire() as conn, conn.transaction():
        for table, shadow, backup in plan:
            await conn.execute(f'ALTER TABLE public."{table}" RENAME TO "{backup}"')
            await conn.execute(
                f'ALTER INDEX IF EXISTS public."{table}_pkey" RENAME TO "{backup}_pkey"'
            )
            await conn.execute(
                f'ALTER INDEX IF EXISTS public."{table}_embedding_idx" '
                f'RENAME TO "{backup}_embedding_idx"'
            )
            await conn.execute(f'ALTER TABLE public."{shadow}" RENAME TO "{table}"')
            await conn.execute(
                f'ALTER INDEX IF EXISTS public."{shadow}_pkey" RENAME TO "{table}_pkey"'
            )
            await conn.execute(
                f'ALTER INDEX IF EXISTS public."{shadow}_embedding_idx" '
                f'RENAME TO "{table}_embedding_idx"'
            )
            logger.info("%-34s promoted, previous vectors kept as %s", table, backup)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workspace", default="default", help="WORKSPACE the tables belong to")
    parser.add_argument(
        "--tables",
        default=",".join(VECTOR_NAMESPACES),
        help=f"comma-separated namespaces among {', '.join(VECTOR_NAMESPACES)}",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N rows per table (smoke test)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--shadow", action="store_true", help="fill <table>__reembed, leave live table alone"
    )
    mode.add_argument(
        "--in-place", action="store_true", help="rewrite the live table (unsafe while serving)"
    )
    mode.add_argument(
        "--promote", action="store_true", help="swap filled shadow tables in, atomically"
    )
    mode.add_argument("--dry-run", action="store_true", help="count rows, embed nothing")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env_file(str(REPO / ".env"))

    if not (args.shadow or args.in_place or args.promote or args.dry_run):
        raise SystemExit("Pick one of --dry-run, --shadow, --promote or --in-place.")

    namespaces = tuple(n.strip() for n in args.tables.split(",") if n.strip())
    unknown = [n for n in namespaces if n not in VECTOR_NAMESPACES]
    if unknown:
        raise SystemExit(f"Unknown namespace(s): {', '.join(unknown)}")

    from memgraphrag.storage.postgres_impl import ClientManager

    model = os.getenv("EMBEDDING_MODEL") or ""
    dim = int(os.getenv("EMBEDDING_DIM") or 1024)
    if not args.dry_run and not args.promote and not model:
        raise SystemExit("EMBEDDING_MODEL is not set; nothing to re-embed with.")

    pool = await ClientManager.get_client()
    try:
        if args.promote:
            await promote(pool, args.workspace, namespaces)
            return 0

        if not args.dry_run:
            logger.info(
                "re-embedding workspace %r with %s (%d dimensions), %s mode",
                args.workspace,
                model,
                dim,
                "shadow" if args.shadow else "in-place",
            )
        totals = {"total": 0, "done": 0, "skipped": 0}
        started = time.monotonic()
        for namespace in namespaces:
            counters = await reembed_namespace(
                pool,
                args.workspace,
                namespace,
                model=model,
                dim=dim,
                batch_size=args.batch_size,
                limit=args.limit,
                shadow_mode=args.shadow,
                dry_run=args.dry_run,
            )
            for key, value in counters.items():
                totals[key] += value

        if args.dry_run:
            logger.info("total %d vectors would be re-embedded", totals["total"])
        else:
            logger.info(
                "re-embedded %d vectors in %.1f s (%d skipped)",
                totals["done"],
                time.monotonic() - started,
                totals["skipped"],
            )
            if args.shadow:
                logger.info("shadow tables filled; check them, then run --promote")
    finally:
        await ClientManager.release_client(pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
