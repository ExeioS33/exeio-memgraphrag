"""Document ingestion routes for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/document_routes.py`` (heavily slimmed).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from memgraphrag.base import DocStatus
from memgraphrag.pipeline import enqueue_document, process_pending
from memgraphrag.utils.hashing import compute_mdhash_id

logger = logging.getLogger("memgraphrag.api.documents")

try:
    from fastapi import (
        APIRouter,
        BackgroundTasks,
        Depends,
        File,
        HTTPException,
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
    Request = None  # type: ignore[misc, assignment]
    UploadFile = None  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]


class TextInsertRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to index")
    doc_id: Optional[str] = Field(default=None, description="Optional document id")


async def _run_pipeline(rag: Any, input_dir: Path) -> dict[str, Any]:
    """Drain PENDING docs through parse → chunk → index."""
    return await process_pending(rag, rag.doc_status, input_dir=input_dir)


async def _index_inline_text(rag: Any, text: str, doc_id: str) -> dict[str, Any]:
    """Enqueue inline text with legacy engine and process immediately."""
    await enqueue_document(
        doc_id=doc_id,
        file_path=f"inline:{doc_id}",
        doc_status_storage=rag.doc_status,
        content=text,
        parse_engine="legacy",
    )
    return await process_pending(rag, rag.doc_status, input_dir=None)


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

        async def _bg() -> None:
            try:
                request.app.state.pipeline_busy = True
                await enqueue_document(
                    doc_id=doc_id,
                    file_path=str(dest),
                    doc_status_storage=rag.doc_status,
                )
                summary = await _run_pipeline(rag, input_dir)
                logger.info("Upload pipeline finished for %s: %s", safe_name, summary)
            except Exception as exc:
                logger.exception("Background upload index failed: %s", exc)
            finally:
                request.app.state.pipeline_busy = False

        background_tasks.add_task(_bg)
        return {
            "status": "queued",
            "filename": safe_name,
            "path": str(dest),
            "doc_id": doc_id,
            "message": "File saved; indexing started in background",
        }

    @router.post("/text", dependencies=[Depends(combined_auth)])
    async def insert_text(request: Request, body: TextInsertRequest):
        rag = request.app.state.rag
        doc_id = body.doc_id or compute_mdhash_id(body.text, prefix="doc-")
        result = await _index_inline_text(rag, body.text, doc_id)
        return {"status": "ok", "doc_id": doc_id, "result": result}

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

    @router.delete("/", dependencies=[Depends(combined_auth)])
    async def clear_documents(request: Request):
        # Optional stub — full wipe not wired for POC safety
        return {
            "status": "not_implemented",
            "message": "DELETE /documents clear is a stub in this POC",
        }

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

        async def _bg() -> None:
            try:
                request.app.state.pipeline_busy = True
                for path in files:
                    try:
                        doc_id = compute_mdhash_id(
                            f"{path}:{path.stat().st_size}", prefix="doc-"
                        )
                        await enqueue_document(
                            doc_id=doc_id,
                            file_path=str(path),
                            doc_status_storage=rag.doc_status,
                        )
                    except Exception as exc:
                        logger.warning("Scan enqueue skip %s: %s", path, exc)
                summary = await _run_pipeline(rag, input_dir)
                logger.info("Scan pipeline finished: %s", summary)
            except Exception as exc:
                logger.exception("Scan pipeline failed: %s", exc)
            finally:
                request.app.state.pipeline_busy = False

        background_tasks.add_task(_bg)
        return {
            "status": "queued",
            "files_found": len(files),
            "input_dir": str(input_dir),
        }

    return router
