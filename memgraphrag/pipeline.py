"""Async document ingestion pipeline for MemGraphRAG.

PENDING → PARSING → PROCESSING → PROCESSED | FAILED, with
``memory_sub_stage`` tracking during PROCESSING.

Adapted conceptually from LightRAG ``lightrag/pipeline.py``, slimmed for
MemGraphRAG's memory-based indexing path.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from memgraphrag.base import DocStatus, DocStatusStorage
from memgraphrag.chunker import get_chunker
from memgraphrag.constants import CHUNK_OVERLAP_SIZE, CHUNK_SIZE
from memgraphrag.parser.base import ParseContext
from memgraphrag.parser.registry import get_parser
from memgraphrag.parser.routing import (
    parse_chunking_strategy,
    resolve_parser_directives,
)
from memgraphrag.utils.tokenizer import TiktokenTokenizer

logger = logging.getLogger(__name__)

# PROCESSING sub-stages (AGENTS.md)
_MEMORY_SUB_STAGES = (
    "openie",
    "memory_build",
    "schema_extraction",
    "ontology_filter",
    "conflict_detection",
    "conflict_resolution",
    "graph_install",
)


def _now() -> int:
    return int(time.time())


async def _set_status(
    storage: DocStatusStorage,
    doc_id: str,
    record: dict[str, Any],
    status: DocStatus,
    *,
    memory_sub_stage: str | None = None,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    updated = dict(record)
    updated["status"] = status.value
    updated["updated_at"] = _now()
    meta = dict(updated.get("metadata") or {})
    if memory_sub_stage is not None:
        meta["memory_sub_stage"] = memory_sub_stage
    if error is not None:
        meta["error"] = error
    updated["metadata"] = meta
    updated.update(extra)
    await storage.upsert({doc_id: updated})
    return updated


async def enqueue_document(
    doc_id: str,
    file_path: str,
    doc_status_storage: DocStatusStorage,
    content: str | None = None,
    parse_engine: str | None = None,
    chunk_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enqueue a document as ``PENDING`` for later :func:`process_pending`."""
    directives = None
    engine = parse_engine
    process_options = ""
    if not engine:
        try:
            directives = resolve_parser_directives(file_path)
            engine = directives.engine
            process_options = directives.process_options
        except Exception as exc:
            logger.warning("enqueue_document: routing failed for %s: %s", file_path, exc)
            engine = "legacy"

    record: dict[str, Any] = {
        "status": DocStatus.PENDING.value,
        "file_path": file_path,
        "content": content,
        "parse_engine": engine,
        "process_options": process_options,
        "chunk_options": chunk_options or {},
        "created_at": _now(),
        "updated_at": _now(),
        "metadata": {"memory_sub_stage": None},
    }
    await doc_status_storage.upsert({doc_id: record})
    logger.info("Enqueued doc_id=%s engine=%s status=pending", doc_id, engine)
    return record


async def _index_chunks(rag_engine: Any, chunks: list[dict[str, Any]]) -> Any:
    """Call ``aindex_with_memory`` / ``ainsert`` on the engine."""
    texts = [c["content"] for c in chunks if c.get("content")]
    if not texts:
        raise ValueError("no chunks to index")

    async_fn = getattr(rag_engine, "aindex_with_memory", None)
    if callable(async_fn):
        return await async_fn(texts)

    async_insert = getattr(rag_engine, "ainsert", None)
    if callable(async_insert):
        return await async_insert(texts)

    sync_fn = getattr(rag_engine, "index_with_memory", None)
    if callable(sync_fn):
        return sync_fn(texts)

    sync_insert = getattr(rag_engine, "insert", None)
    if callable(sync_insert):
        return sync_insert(texts)

    raise AttributeError(
        "rag_engine has no aindex_with_memory / ainsert / index_with_memory / insert"
    )


async def process_pending(
    rag_engine: Any,
    doc_status_storage: DocStatusStorage,
    input_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Process all ``PENDING`` documents through parse → chunk → index.

    Returns a summary ``{processed, failed, doc_ids}``.
    """
    pending = await doc_status_storage.get_docs_by_statuses([DocStatus.PENDING])
    summary: dict[str, Any] = {"processed": 0, "failed": 0, "doc_ids": []}

    tokenizer_model = os.getenv("TIKTOKEN_MODEL", "gpt-4o-mini")
    try:
        tokenizer = TiktokenTokenizer(tokenizer_model)
    except Exception:
        tokenizer = TiktokenTokenizer("gpt-4o-mini")

    for doc_id, record in list(pending.items()):
        file_path = str(record.get("file_path") or "")
        try:
            record = await _set_status(
                doc_status_storage, doc_id, record, DocStatus.PARSING,
                memory_sub_stage=None,
            )

            # Resolve source path (prefer input_dir join for relative paths)
            source = Path(file_path)
            if input_dir and not source.is_file():
                candidate = Path(input_dir) / source.name
                if candidate.is_file():
                    source = candidate

            engine_name = str(record.get("parse_engine") or "legacy")
            content_data: dict[str, Any] = {
                "content": record.get("content"),
                "source_file": str(source) if source.is_file() else file_path,
            }
            # Inline content without a file: feed via content_data only
            if not source.is_file() and record.get("content"):
                content_data["source_file"] = file_path

            ctx = ParseContext(
                doc_id=doc_id,
                file_path=file_path,
                content_data=content_data,
                rag=rag_engine,
            )
            parser = get_parser(engine_name)
            parse_result = await parser.parse(ctx)

            process_options = str(record.get("process_options") or "")
            strategy = parse_chunking_strategy(process_options)
            chunk_opts = dict(record.get("chunk_options") or {})
            chunk_token_size = int(
                chunk_opts.get("chunk_token_size")
                or os.getenv("CHUNK_SIZE", CHUNK_SIZE)
            )
            chunk_overlap = int(
                chunk_opts.get("chunk_overlap_token_size")
                or os.getenv("CHUNK_OVERLAP_SIZE", CHUNK_OVERLAP_SIZE)
            )

            chunker = get_chunker(strategy)
            chunk_kwargs: dict[str, Any] = {
                "chunk_overlap_token_size": chunk_overlap,
            }
            if strategy == "P":
                chunk_kwargs["blocks_path"] = parse_result.blocks_path or None
                chunk_kwargs["doc_id"] = doc_id
            elif strategy == "F":
                if "split_by_character" in chunk_opts:
                    chunk_kwargs["split_by_character"] = chunk_opts["split_by_character"]
                if "split_by_character_only" in chunk_opts:
                    chunk_kwargs["split_by_character_only"] = chunk_opts[
                        "split_by_character_only"
                    ]
            elif strategy == "R" and "separators" in chunk_opts:
                chunk_kwargs["separators"] = chunk_opts["separators"]

            chunks = chunker(
                tokenizer,
                parse_result.content,
                chunk_token_size,
                **chunk_kwargs,
            )
            if not chunks:
                raise ValueError("chunker produced no chunks")

            record = await _set_status(
                doc_status_storage,
                doc_id,
                record,
                DocStatus.PROCESSING,
                memory_sub_stage=_MEMORY_SUB_STAGES[0],
                parse_format=parse_result.parse_format,
                blocks_path=parse_result.blocks_path,
                chunk_count=len(chunks),
            )

            # Track sub-stages around the index call (engine owns the real work).
            for stage in _MEMORY_SUB_STAGES:
                meta = dict(record.get("metadata") or {})
                meta["memory_sub_stage"] = stage
                record["metadata"] = meta
                record["updated_at"] = _now()
                await doc_status_storage.upsert({doc_id: record})

            await _index_chunks(rag_engine, chunks)

            record = await _set_status(
                doc_status_storage,
                doc_id,
                record,
                DocStatus.PROCESSED,
                memory_sub_stage="done",
                chunk_count=len(chunks),
            )
            summary["processed"] += 1
            summary["doc_ids"].append(doc_id)
            logger.info("Processed doc_id=%s chunks=%d", doc_id, len(chunks))

        except Exception as exc:
            logger.exception("Failed processing doc_id=%s: %s", doc_id, exc)
            await _set_status(
                doc_status_storage,
                doc_id,
                record,
                DocStatus.FAILED,
                memory_sub_stage=None,
                error=str(exc),
            )
            summary["failed"] += 1
            summary["doc_ids"].append(doc_id)

    return summary
