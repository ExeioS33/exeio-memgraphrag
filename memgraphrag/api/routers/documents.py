"""Document ingestion and admin routes for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/document_routes.py``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from memgraphrag.base import DocStatus
from memgraphrag.pipeline import enqueue_document, process_pending
from memgraphrag.utils.hashing import compute_mdhash_id
from memgraphrag.utils.step_log import done_step, fail_step, main_step, sub_step, truncate

logger = logging.getLogger("memgraphrag.api.documents")

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
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]


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


async def _drain_until_idle(
    request: Request, input_dir: Path | None
) -> dict[str, Any]:
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
                pending = await rag.doc_status.get_docs_by_statuses(
                    [DocStatus.PENDING]
                )
                if not pending:
                    break
                # New docs were enqueued while we indexed — keep draining.
                request.app.state.drain_requested = False
        finally:
            _release_pipeline(request)
    return last


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

    async def _bg() -> None:
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

    background_tasks.add_task(_bg)


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
        dest = input_dir / safe_name
        content = await file.read()
        dest.write_bytes(content)
        doc_id = compute_mdhash_id(f"{safe_name}:{len(content)}", prefix="doc-")
        busy = _pipeline_is_busy(request)
        main_step(
            logger,
            "api.documents.upload",
            doc_id=doc_id,
            filename=safe_name,
            bytes=len(content),
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
            "filename": safe_name,
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
        doc_id = body.doc_id or compute_mdhash_id(body.text, prefix="doc-")
        main_step(
            logger,
            "api.documents.text",
            doc_id=doc_id,
            chars=len(body.text),
            preview=truncate(body.text),
        )
        await enqueue_document(
            doc_id=doc_id,
            file_path=f"inline:{doc_id}",
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
                pending = await rag.doc_status.get_docs_by_statuses(
                    [DocStatus.PENDING]
                )
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
    async def list_documents(request: Request):
        rag = request.app.state.rag
        try:
            docs = await rag.doc_status.get_all()
        except NotImplementedError:
            docs = {}
            for status in DocStatus:
                try:
                    part = await rag.doc_status.get_docs_by_statuses([status])
                    docs.update(part)
                except Exception:
                    pass
        return {"statuses": docs}

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
            and "__parsed__" not in p.parts
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
        for path in files:
            try:
                doc_id = compute_mdhash_id(
                    f"{path}:{path.stat().st_size}", prefix="doc-"
                )
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
        return {"doc_id": doc_id, "document": record}

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
