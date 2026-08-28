"""Document ingestion and admin routes for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/document_routes.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from memgraphrag.base import DocStatus
from memgraphrag.constants import MAX_UPLOAD_SIZE
from memgraphrag.parser.registry import available_engine_suffixes
from memgraphrag.pipeline import (
    CONTENT_SUMMARY_LIMIT,
    enqueue_document,
    process_pending,
)
from memgraphrag.utils.hashing import compute_mdhash_id
from memgraphrag.utils.step_log import done_step, fail_step, main_step, sub_step, truncate

logger = logging.getLogger("memgraphrag.api.documents")

# Document identity is the hash of the *content*, on every ingestion path. Upload used
# to hash name+size and scan path+size, so the same bytes got two ids depending on how
# they arrived, and two revisions sharing a name and a size collapsed onto one id —
# the second ingest overwrote the first record, orphaning its chunks in openie_kv.
DOC_ID_PREFIX = "doc-"

# Raw text posted to /documents/text is spooled here so doc-status only has to carry a
# path; the body itself never goes into the status record.
INLINE_SUBDIR = "__inline__"

# Directories skipped by /documents/scan: parser sidecars and our own inline spool are
# derived artefacts, not sources.
_SCAN_EXCLUDED_DIRS = ("__parsed__", INLINE_SUBDIR)

# Listing defaults for GET /documents/. Unbounded listing serialized every record —
# including the full document body — into a single response.
DEFAULT_LIST_LIMIT = 100
MAX_LIST_LIMIT = 1000

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")

try:
    from fastapi import (
        APIRouter,
        BackgroundTasks,
        Depends,
        File,
        HTTPException,
        Query,
        Request,
        UploadFile,
    )
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    BackgroundTasks = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    File = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    Query = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    UploadFile = None  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    # Stub so the module still imports without the [api] extra; never called.
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]  # noqa: E731


class TextInsertRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to index")
    doc_id: Optional[str] = Field(default=None, description="Optional document id")


class BatchDeleteRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_length=1, description="Document ids to delete")
    delete_file: bool = Field(
        default=False, description="Also delete source/parsed files when present"
    )


def _pipeline_lock(request: Request) -> asyncio.Lock:
    lock = getattr(request.app.state, "pipeline_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.pipeline_lock = lock
    return lock


def _pipeline_is_busy(request: Request) -> bool:
    lock = _pipeline_lock(request)
    return lock.locked() or bool(getattr(request.app.state, "pipeline_busy", False))


async def _acquire_pipeline(request: Request) -> None:
    """Non-blocking acquire; raises 409 if already held (delete/clear)."""
    if _pipeline_is_busy(request):
        raise HTTPException(
            status_code=409,
            detail="Pipeline busy; retry when /health reports pipeline_busy=false",
        )
    lock = _pipeline_lock(request)
    await lock.acquire()
    request.app.state.pipeline_busy = True


async def _try_acquire_pipeline(request: Request) -> bool:
    """Non-blocking acquire; returns False if busy (upload/scan drain)."""
    if _pipeline_is_busy(request):
        return False
    lock = _pipeline_lock(request)
    await lock.acquire()
    request.app.state.pipeline_busy = True
    return True


def _release_pipeline(request: Request) -> None:
    lock = _pipeline_lock(request)
    request.app.state.pipeline_busy = False
    if lock.locked():
        lock.release()


async def _run_pipeline(rag: Any, input_dir: Path | None) -> dict[str, Any]:
    """Drain PENDING docs through parse → chunk → index."""
    return await process_pending(rag, rag.doc_status, input_dir=input_dir)


async def _drain_until_idle(request: Request, input_dir: Path | None) -> dict[str, Any]:
    """Serialize process_pending until no PENDING remain (or defer if busy)."""
    request.app.state.drain_requested = True
    last: dict[str, Any] = {"processed": 0, "failed": 0, "doc_ids": []}
    while getattr(request.app.state, "drain_requested", False):
        if not await _try_acquire_pipeline(request):
            return {"status": "deferred", **last}
        request.app.state.drain_requested = False
        try:
            rag = request.app.state.rag
            while True:
                last = await _run_pipeline(rag, input_dir)
                pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
                if not pending:
                    break
                # New docs were enqueued while we indexed — keep draining.
                request.app.state.drain_requested = False
        finally:
            _release_pipeline(request)
    return last


def track_background_task(app: Any, coro: Any) -> asyncio.Task:
    """Run ``coro`` as a tracked task registered on ``app.state.background_tasks``.

    The lifespan shutdown hook drains that set, which is the only thing standing
    between a SIGTERM and an indexing run being killed mid-write.
    """
    task = asyncio.ensure_future(coro)
    tasks = getattr(app.state, "background_tasks", None)
    if tasks is None:
        tasks = set()
        app.state.background_tasks = tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


def _schedule_drain(
    request: Request,
    background_tasks: BackgroundTasks,
    input_dir: Path | None,
    *,
    log_name: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Mark drain requested and run a background worker when the lock is free."""
    request.app.state.drain_requested = True
    extra = extra or {}

    async def _drain() -> None:
        try:
            summary = await _drain_until_idle(request, input_dir)
            done_step(
                logger,
                log_name,
                processed=summary.get("processed"),
                failed=summary.get("failed"),
                status=summary.get("status"),
                **{k: v for k, v in extra.items() if v is not None},
            )
        except Exception as exc:
            fail_step(logger, log_name, exc=exc, exc_info=True, **extra)

    async def _bg() -> None:
        # Shielded: Starlette runs background tasks inside the request task, so a
        # disconnecting client or a shutdown-cancelled request would otherwise abort
        # indexing halfway. The task stays registered for the shutdown drain instead.
        await asyncio.shield(track_background_task(request.app, _drain()))

    background_tasks.add_task(_bg)


def _max_upload_size(request: Any) -> int:
    args = getattr(getattr(request, "app", None), "state", None)
    return int(getattr(getattr(args, "args", None), "max_upload_size", MAX_UPLOAD_SIZE))


def _reject_unsupported_suffix(filename: str) -> None:
    """Refuse extensions no configured parser can read, before writing to disk.

    ``available_engine_suffixes()`` is endpoint-aware: Docling-only suffixes are
    accepted only while DOCLING_ENDPOINT is configured.
    """
    suffix = Path(filename).suffix.lower().lstrip(".")
    allowed = available_engine_suffixes()
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type {suffix or '(none)'!r}. "
                f"Supported: {', '.join(sorted(allowed))}"
            ),
        )


async def _spool_upload(file: Any, dest: Path, max_bytes: int, hasher: Any | None = None) -> int:
    """Stream an upload to ``dest`` in chunks, aborting past ``max_bytes``.

    Returns the number of bytes written. On overflow the partial file is removed and
    413 is raised, so a rejected upload leaves nothing behind in the input directory.

    ``hasher`` (any ``hashlib`` object) is fed every chunk, so the caller gets the
    content digest for free instead of reading the spooled file back a second time.
    """
    written = 0
    chunk_size = 1024 * 1024
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds MAX_UPLOAD_SIZE ({max_bytes} bytes).",
                    )
                if hasher is not None:
                    hasher.update(chunk)
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return written


def _hash_bytes_sync(path: Path) -> str:
    """MD5 of a file's bytes, read in chunks (files can be hundreds of MiB)."""
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


async def content_doc_id(path: Path) -> str:
    """Return the content-addressed document id of an on-disk file.

    Matches ``compute_mdhash_id(text, prefix="doc-")`` for a UTF-8 text file, so the
    same document uploaded, scanned, or posted as raw text resolves to one id.
    """
    return DOC_ID_PREFIX + await asyncio.to_thread(_hash_bytes_sync, path)


def inline_source_name(doc_id: str) -> str:
    """Safe basename for the spooled copy of a raw-text ingest.

    ``doc_id`` may come straight from the request body, so it can contain ``/`` or
    ``..``; sanitising is what keeps the write inside the input directory. The digest
    suffix keeps two ids that sanitise identically apart.
    """
    stem = _UNSAFE_FILENAME_CHARS.sub("_", doc_id)[:64].strip("._-") or "inline"
    return f"{stem}-{compute_mdhash_id(doc_id)[:8]}.txt"


def _strip_document_body(record: dict[str, Any]) -> dict[str, Any]:
    """Drop any full body from a status record before it leaves the process.

    Records written by earlier versions still carry ``content``; returning them makes
    a listing proportional to the corpus.
    """
    if "content" not in record:
        return record
    trimmed = dict(record)
    body = trimmed.pop("content", None) or ""
    trimmed.setdefault("content_summary", truncate(body, CONTENT_SUMMARY_LIMIT) if body else "")
    trimmed.setdefault("content_length", len(body))
    return trimmed


def create_documents_router(api_key: Optional[str] = None) -> Any:
    """Build ``/documents`` router."""
    if APIRouter is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import get_combined_auth_dependency

    router = APIRouter(prefix="/documents", tags=["documents"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("/upload", dependencies=[Depends(combined_auth)])
    async def upload_document(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ):
        rag = request.app.state.rag
        input_dir = Path(getattr(request.app.state, "input_dir", "./data/inputs"))
        input_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or f"upload-{uuid.uuid4().hex}").name
        _reject_unsupported_suffix(safe_name)

        # Spool to a scratch name first: the id is only known once the bytes are in,
        # and writing straight to `input_dir/<name>` would clobber the source file of
        # a different document that happens to share the name.
        spool = input_dir / f".upload-{uuid.uuid4().hex}.part"
        hasher = hashlib.md5()
        # Stream to disk with a hard cap: `await file.read()` used to pull the whole
        # body into RAM with no limit, so one large POST could OOM the worker.
        size = await _spool_upload(file, spool, _max_upload_size(request), hasher)
        doc_id = DOC_ID_PREFIX + hasher.hexdigest()

        existing = await rag.doc_status.get_by_id(doc_id)
        if existing is not None:
            # Same bytes, same id: re-enqueuing would overwrite the record and drop the
            # chunk_ids that keep its chunks reachable. Refuse and point at /requeue.
            spool.unlink(missing_ok=True)
            existing_status = str(existing.get("status") or "")
            done_step(
                logger,
                "api.documents.upload",
                doc_id=doc_id,
                filename=safe_name,
                status="duplicate",
                doc_status=existing_status,
            )
            return {
                "status": "duplicate",
                "filename": safe_name,
                "path": existing.get("file_path"),
                "doc_id": doc_id,
                "doc_status": existing_status,
                "message": (
                    "Identical content is already indexed under this doc_id; "
                    f"POST /documents/{doc_id}/requeue to re-index it"
                ),
                "pipeline_busy": _pipeline_is_busy(request),
            }

        dest = input_dir / safe_name
        if dest.exists():
            # A homonym is already on disk. If it is a different revision, park this
            # one under a digest-qualified name so the older document keeps a readable
            # source file (deleting it would strand the record that points at it).
            try:
                same_bytes = await content_doc_id(dest) == doc_id
            except OSError:
                same_bytes = False
            if not same_bytes:
                suffixed = Path(safe_name)
                dest = input_dir / (f"{suffixed.stem}.{hasher.hexdigest()[:12]}{suffixed.suffix}")
        spool.replace(dest)
        busy = _pipeline_is_busy(request)
        main_step(
            logger,
            "api.documents.upload",
            doc_id=doc_id,
            filename=safe_name,
            bytes=size,
            pipeline_busy=busy,
        )
        sub_step(
            logger,
            "api.documents.upload.enqueue",
            doc_id=doc_id,
            filename=safe_name,
        )
        await enqueue_document(
            doc_id=doc_id,
            file_path=str(dest),
            doc_status_storage=rag.doc_status,
        )
        _schedule_drain(
            request,
            background_tasks,
            input_dir,
            log_name="api.documents.upload",
            extra={"doc_id": doc_id, "filename": safe_name},
        )
        return {
            "status": "queued",
            # ``dest.name`` may differ from the uploaded name when a homonym revision
            # was already on disk; report what the server actually stored.
            "filename": dest.name,
            "path": str(dest),
            "doc_id": doc_id,
            "message": (
                "File saved and enqueued; indexing will start when the pipeline is idle"
                if busy
                else "File saved; indexing started in background"
            ),
            "pipeline_busy": busy,
        }

    @router.post("/text", dependencies=[Depends(combined_auth)])
    async def insert_text(
        request: Request,
        background_tasks: BackgroundTasks,
        body: TextInsertRequest,
    ):
        rag = request.app.state.rag
        doc_id = body.doc_id or compute_mdhash_id(body.text, prefix=DOC_ID_PREFIX)
        main_step(
            logger,
            "api.documents.text",
            doc_id=doc_id,
            chars=len(body.text),
            preview=truncate(body.text),
        )
        # The body is spooled to disk instead of being kept in the status record:
        # doc-status is an index, not a document store, and the listing endpoint used
        # to serialize every stored body in one response.
        input_dir = Path(getattr(request.app.state, "input_dir", "./data/inputs"))
        inline_dir = input_dir / INLINE_SUBDIR
        inline_dir.mkdir(parents=True, exist_ok=True)
        source = inline_dir / inline_source_name(doc_id)
        await asyncio.to_thread(source.write_text, body.text, "utf-8")
        await enqueue_document(
            doc_id=doc_id,
            file_path=str(source),
            doc_status_storage=rag.doc_status,
            content=body.text,
            parse_engine="legacy",
        )
        busy = _pipeline_is_busy(request)
        if busy:
            _schedule_drain(
                request,
                background_tasks,
                None,
                log_name="api.documents.text",
                extra={"doc_id": doc_id},
            )
            return {
                "status": "queued",
                "doc_id": doc_id,
                "message": "Enqueued; indexing will start when the pipeline is idle",
                "pipeline_busy": True,
            }
        acquired = False
        try:
            await _acquire_pipeline(request)
            acquired = True
            request.app.state.drain_requested = False
            result = await _run_pipeline(rag, None)
            # Drain any docs enqueued while we ran.
            while True:
                pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
                if not pending:
                    break
                result = await _run_pipeline(rag, None)
            done_step(
                logger,
                "api.documents.text",
                doc_id=doc_id,
                processed=result.get("processed"),
                failed=result.get("failed"),
            )
            return {"status": "ok", "doc_id": doc_id, "result": result}
        except HTTPException:
            raise
        except Exception as exc:
            fail_step(
                logger,
                "api.documents.text",
                doc_id=doc_id,
                exc=exc,
                exc_info=True,
            )
            raise
        finally:
            if acquired:
                _release_pipeline(request)
            if getattr(request.app.state, "drain_requested", False):
                _schedule_drain(
                    request,
                    background_tasks,
                    None,
                    log_name="api.documents.text.drain",
                    extra={"doc_id": doc_id},
                )

    @router.get("/", dependencies=[Depends(combined_auth)])
    async def list_documents(
        request: Request,
        status: str | None = Query(default=None, description="Only list documents in this status"),
        limit: int = Query(default=DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
        offset: int = Query(default=0, ge=0),
    ):
        rag = request.app.state.rag
        if status:
            try:
                wanted = DocStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown status {status!r}. Expected one of: "
                        f"{', '.join(s.value for s in DocStatus)}"
                    ),
                ) from None
            docs = await rag.doc_status.get_docs_by_statuses([wanted])
        else:
            try:
                docs = await rag.doc_status.get_all()
            except NotImplementedError:
                docs = {}
                for known in DocStatus:
                    try:
                        part = await rag.doc_status.get_docs_by_statuses([known])
                        docs.update(part)
                    except Exception:
                        pass

        # Ordered by creation time, which never changes: paging on ``updated_at`` would
        # let a document being indexed jump pages and be skipped by the caller.
        ordered = sorted(
            docs.items(),
            key=lambda item: (int((item[1] or {}).get("created_at") or 0), item[0]),
        )
        total = len(ordered)
        page = ordered[offset : offset + limit]
        return {
            "statuses": {doc_id: _strip_document_body(record or {}) for doc_id, record in page},
            "total": total,
            "limit": limit,
            "offset": offset,
            "returned": len(page),
            "next_offset": offset + len(page) if offset + len(page) < total else None,
            "status": status,
        }

    @router.post("/delete", dependencies=[Depends(combined_auth)])
    async def batch_delete_documents(request: Request, body: BatchDeleteRequest):
        main_step(
            logger,
            "api.documents.delete_batch",
            docs=len(body.doc_ids),
            delete_file=body.delete_file,
        )
        acquired = False
        try:
            await _acquire_pipeline(request)
            acquired = True
            result = await request.app.state.rag.adelete_by_doc_ids(
                body.doc_ids, delete_file=body.delete_file
            )
            done_step(
                logger,
                "api.documents.delete_batch",
                docs=len(body.doc_ids),
            )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            fail_step(
                logger,
                "api.documents.delete_batch",
                exc=exc,
                exc_info=True,
            )
            raise
        finally:
            if acquired:
                _release_pipeline(request)

    @router.post("/scan", dependencies=[Depends(combined_auth)])
    async def scan_input_dir(request: Request, background_tasks: BackgroundTasks):
        rag = request.app.state.rag
        input_dir = Path(getattr(request.app.state, "input_dir", "./data/inputs"))
        input_dir.mkdir(parents=True, exist_ok=True)
        files = [
            p
            for p in input_dir.rglob("*")
            if p.is_file()
            and not p.name.startswith(".")
            and not any(part in _SCAN_EXCLUDED_DIRS for part in p.parts)
        ]
        busy = _pipeline_is_busy(request)
        main_step(
            logger,
            "api.documents.scan",
            files_found=len(files),
            input_dir=str(input_dir),
            pipeline_busy=busy,
        )
        enqueued = 0
        skipped = 0
        for path in files:
            try:
                # Same id as the upload/text paths: one file has one identity however
                # it reached the input directory.
                doc_id = await content_doc_id(path)
                existing = await rag.doc_status.get_by_id(doc_id)
                if (
                    existing is not None
                    and str(existing.get("status") or "") != DocStatus.FAILED.value
                ):
                    # Re-enqueuing rewrites the record from scratch, dropping the
                    # chunk_ids of an already-indexed document and orphaning its
                    # chunks. Only a FAILED document is worth another run.
                    skipped += 1
                    continue
                sub_step(
                    logger,
                    "api.documents.scan.enqueue",
                    doc_id=doc_id,
                    file=path.name,
                )
                await enqueue_document(
                    doc_id=doc_id,
                    file_path=str(path),
                    doc_status_storage=rag.doc_status,
                )
                enqueued += 1
            except Exception as exc:
                fail_step(
                    logger,
                    "api.documents.scan.enqueue",
                    file=path.name,
                    exc=exc,
                )
        _schedule_drain(
            request,
            background_tasks,
            input_dir,
            log_name="api.documents.scan",
            extra={"files_found": len(files)},
        )
        return {
            "status": "queued",
            "files_found": len(files),
            "enqueued": enqueued,
            "skipped": skipped,
            "input_dir": str(input_dir),
            "pipeline_busy": busy,
            "message": (
                "Files enqueued; indexing will start when the pipeline is idle"
                if busy
                else "Files enqueued; indexing started in background"
            ),
        }

    @router.delete("/", dependencies=[Depends(combined_auth)])
    async def clear_documents(
        request: Request,
        confirm: bool = Query(False),
        delete_files: bool = Query(False),
    ):
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Pass confirm=true to clear all documents and storages",
            )
        main_step(
            logger,
            "api.documents.clear",
            delete_files=delete_files,
        )
        input_dir = Path(getattr(request.app.state, "input_dir", "./data/inputs"))
        acquired = False
        try:
            await _acquire_pipeline(request)
            acquired = True
            result = await request.app.state.rag.aclear_all(
                delete_files=delete_files,
                input_dir=input_dir if delete_files else None,
            )
            done_step(
                logger,
                "api.documents.clear",
                files_deleted=result.get("files_deleted"),
            )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            fail_step(logger, "api.documents.clear", exc=exc, exc_info=True)
            raise
        finally:
            if acquired:
                _release_pipeline(request)

    @router.get("/{doc_id}", dependencies=[Depends(combined_auth)])
    async def get_document(request: Request, doc_id: str):
        rag = request.app.state.rag
        record = await rag.doc_status.get_by_id(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return {"doc_id": doc_id, "document": _strip_document_body(record)}

    @router.delete("/{doc_id}", dependencies=[Depends(combined_auth)])
    async def delete_document(
        request: Request,
        doc_id: str,
        delete_file: bool = Query(False),
    ):
        main_step(
            logger,
            "api.documents.delete",
            doc_id=doc_id,
            delete_file=delete_file,
        )
        acquired = False
        try:
            await _acquire_pipeline(request)
            acquired = True
            result = await request.app.state.rag.adelete_by_doc_ids(
                [doc_id], delete_file=delete_file
            )
            done_step(
                logger,
                "api.documents.delete",
                doc_id=doc_id,
                status=(result.get("results") or {}).get(doc_id, {}).get("status"),
            )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            fail_step(
                logger,
                "api.documents.delete",
                doc_id=doc_id,
                exc=exc,
                exc_info=True,
            )
            raise
        finally:
            if acquired:
                _release_pipeline(request)

    @router.post("/{doc_id}/requeue", dependencies=[Depends(combined_auth)])
    async def requeue_document(
        request: Request,
        doc_id: str,
        background_tasks: BackgroundTasks,
    ):
        rag = request.app.state.rag
        record = await rag.doc_status.get_by_id(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        status = str(record.get("status") or "")
        if status not in {
            DocStatus.FAILED.value,
            DocStatus.PROCESSING.value,
            DocStatus.PARSING.value,
            DocStatus.PENDING.value,
        }:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot requeue document in status={status!r}",
            )

        input_dir = Path(getattr(request.app.state, "input_dir", "./data/inputs"))
        updated = dict(record)
        updated["status"] = DocStatus.PENDING.value
        meta = dict(updated.get("metadata") or {})
        meta["memory_sub_stage"] = None
        meta.pop("error", None)
        updated["metadata"] = meta
        updated.pop("chunk_ids", None)
        await rag.doc_status.upsert({doc_id: updated})
        busy = _pipeline_is_busy(request)
        main_step(
            logger,
            "api.documents.requeue",
            doc_id=doc_id,
            from_status=status,
            pipeline_busy=busy,
        )
        _schedule_drain(
            request,
            background_tasks,
            input_dir,
            log_name="api.documents.requeue",
            extra={"doc_id": doc_id},
        )
        return {
            "status": "queued",
            "doc_id": doc_id,
            "pipeline_busy": busy,
            "message": (
                f"Requeued from {status}; indexing will start when the pipeline is idle"
                if busy
                else f"Requeued from {status}; indexing started in background"
            ),
        }

    return router
