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
from memgraphrag.utils.hashing import compute_mdhash_id
from memgraphrag.utils.step_log import done_step, fail_step, main_step, sub_step
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
    main_step(
        logger,
        "ingest.enqueue",
        doc_id=doc_id,
        file=Path(file_path).name,
        content_chars=len(content) if content else 0,
    )
    directives = None
    engine = parse_engine
    process_options = ""
    if not engine:
        try:
            directives = resolve_parser_directives(file_path)
            engine = directives.engine
            process_options = directives.process_options
            sub_step(
                logger,
                "ingest.enqueue.route",
                doc_id=doc_id,
                engine=engine,
                process_options=process_options or "-",
            )
        except Exception as exc:
            fail_step(
                logger,
                "ingest.enqueue.route",
                doc_id=doc_id,
                file=Path(file_path).name,
                exc=exc,
            )
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
    done_step(
        logger,
        "ingest.enqueue",
        doc_id=doc_id,
        engine=engine,
        status="pending",
    )
    return record


def _assign_chunk_ids(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure each chunk has a content-hash ``idx`` (``chunk-…``) for OpenIE/KV keys."""
    prepared: list[dict[str, Any]] = []
    for chunk in chunks:
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        chunk_id = str(chunk.get("idx") or chunk.get("id") or "")
        if not chunk_id.startswith("chunk-"):
            chunk_id = compute_mdhash_id(content, prefix="chunk-")
        prepared.append(
            {
                **chunk,
                "idx": chunk_id,
                "content": content,
            }
        )
    return prepared


async def _index_chunks(rag_engine: Any, chunks: list[dict[str, Any]]) -> list[str]:
    """Call ``aindex_with_memory`` / ``ainsert`` with chunk-id keyed dicts.

    Returns the content-hash chunk ids that were indexed.
    """
    prepared = _assign_chunk_ids(chunks)
    if not prepared:
        raise ValueError("no chunks to index")
    chunk_ids = [c["idx"] for c in prepared]

    sub_step(logger, "ingest.index.call", chunks=len(prepared))
    async_fn = getattr(rag_engine, "aindex_with_memory", None)
    if callable(async_fn):
        await async_fn(prepared)
        return chunk_ids

    async_insert = getattr(rag_engine, "ainsert", None)
    if callable(async_insert):
        await async_insert(prepared)
        return chunk_ids

    sync_fn = getattr(rag_engine, "index_with_memory", None)
    if callable(sync_fn):
        sync_fn(prepared)
        return chunk_ids

    sync_insert = getattr(rag_engine, "insert", None)
    if callable(sync_insert):
        sync_insert(prepared)
        return chunk_ids

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
    main_step(logger, "ingest.process", pending=len(pending))

    tokenizer_model = os.getenv("TIKTOKEN_MODEL", "gpt-4o-mini")
    try:
        tokenizer = TiktokenTokenizer(tokenizer_model)
    except Exception:
        tokenizer = TiktokenTokenizer("gpt-4o-mini")

    for doc_id, record in list(pending.items()):
        file_path = str(record.get("file_path") or "")
        main_step(
            logger,
            "ingest.doc",
            doc_id=doc_id,
            file=Path(file_path).name or file_path,
        )
        try:
            record = await _set_status(
                doc_status_storage, doc_id, record, DocStatus.PARSING,
                memory_sub_stage=None,
            )
            sub_step(logger, "ingest.doc.status", doc_id=doc_id, status="parsing")

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
            sub_step(
                logger,
                "ingest.doc.parse",
                doc_id=doc_id,
                engine=engine_name,
                source_exists=source.is_file(),
            )
            parse_result = await parser.parse(ctx)
            sub_step(
                logger,
                "ingest.doc.parse_result",
                doc_id=doc_id,
                format=parse_result.parse_format,
                content_chars=len(parse_result.content or ""),
                blocks_path=bool(parse_result.blocks_path),
            )

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

            sub_step(
                logger,
                "ingest.doc.chunk",
                doc_id=doc_id,
                strategy=strategy,
                chunk_token_size=chunk_token_size,
                overlap=chunk_overlap,
            )
            chunks = chunker(
                tokenizer,
                parse_result.content,
                chunk_token_size,
                **chunk_kwargs,
            )
            if not chunks:
                raise ValueError("chunker produced no chunks")
            # Attach document source so QA can always cite the originating file.
            source_label = Path(file_path).name or file_path or doc_id
            for chunk in chunks:
                if isinstance(chunk, dict):
                    chunk.setdefault("file_path", source_label)
                    chunk.setdefault("full_doc_id", doc_id)
            sub_step(
                logger,
                "ingest.doc.chunk_result",
                doc_id=doc_id,
                chunks=len(chunks),
                source=source_label,
            )

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
            sub_step(
                logger,
                "ingest.doc.status",
                doc_id=doc_id,
                status="processing",
            )

            # Track sub-stages around the index call (engine owns the real work).
            for stage in _MEMORY_SUB_STAGES:
                meta = dict(record.get("metadata") or {})
                meta["memory_sub_stage"] = stage
                record["metadata"] = meta
                record["updated_at"] = _now()
                await doc_status_storage.upsert({doc_id: record})

            sub_step(logger, "ingest.doc.index", doc_id=doc_id, chunks=len(chunks))
            chunk_ids = await _index_chunks(rag_engine, chunks)

            record = await _set_status(
                doc_status_storage,
                doc_id,
                record,
                DocStatus.PROCESSED,
                memory_sub_stage="done",
                chunk_count=len(chunk_ids),
                chunk_ids=chunk_ids,
            )
            summary["processed"] += 1
            summary["doc_ids"].append(doc_id)
            done_step(
                logger,
                "ingest.doc",
                doc_id=doc_id,
                chunks=len(chunk_ids),
                status="processed",
            )

        except Exception as exc:
            fail_step(
                logger,
                "ingest.doc",
                doc_id=doc_id,
                file=Path(file_path).name or file_path,
                exc=exc,
                exc_info=True,
            )
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

    done_step(
        logger,
        "ingest.process",
        processed=summary["processed"],
        failed=summary["failed"],
    )
    return summary
