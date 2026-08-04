"""Document ingestion routes for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/document_routes.py`` (heavily slimmed).
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from memgraphrag.base import DocStatus
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


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


async def _index_texts(rag: Any, texts: list[str], doc_ids: list[str] | None = None) -> dict:
    payloads: list[dict[str, str]] = []
    for i, text in enumerate(texts):
        idx = (doc_ids[i] if doc_ids and i < len(doc_ids) else None) or compute_mdhash_id(
            text, prefix="chunk-"
        )
        payloads.append({"idx": idx, "passage": text, "content": text})
        try:
            await rag.doc_status.upsert(
                {
                    idx: {
                        "status": DocStatus.PROCESSING.value,
                        "content_length": len(text),
                    }
                }
            )
        except Exception as exc:
            logger.debug("doc_status upsert skipped: %s", exc)

    try:
        result = await rag.aindex_with_memory(payloads)
        for item in payloads:
            try:
                await rag.doc_status.upsert(
                    {
                        item["idx"]: {
                            "status": DocStatus.PROCESSED.value,
                            "content_length": len(item.get("content") or item.get("passage", "")),
                        }
                    }
                )
            except Exception:
                pass
        return result if isinstance(result, dict) else {"ok": True}
    except Exception as exc:
        logger.exception("Indexing failed: %s", exc)
        for item in payloads:
            try:
                await rag.doc_status.upsert(
                    {
                        item["idx"]: {
                            "status": DocStatus.FAILED.value,
                            "error": str(exc),
                        }
                    }
                )
            except Exception:
                pass
        raise


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

        async def _bg() -> None:
            try:
                text = _read_text_file(dest)
                if text.strip():
                    await _index_texts(rag, [text], [compute_mdhash_id(text, prefix="doc-")])
            except Exception as exc:
                logger.exception("Background upload index failed: %s", exc)

        background_tasks.add_task(_bg)
        return {
            "status": "queued",
            "filename": safe_name,
            "path": str(dest),
            "message": "File saved; indexing started in background",
        }

    @router.post("/text", dependencies=[Depends(combined_auth)])
    async def insert_text(request: Request, body: TextInsertRequest):
        rag = request.app.state.rag
        doc_id = body.doc_id or compute_mdhash_id(body.text, prefix="doc-")
        result = await _index_texts(rag, [body.text], [doc_id])
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
            if p.is_file() and not p.name.startswith(".")
        ]

        async def _bg() -> None:
            for path in files:
                try:
                    text = _read_text_file(path)
                    if not text.strip():
                        continue
                    await _index_texts(
                        rag,
                        [text],
                        [compute_mdhash_id(str(path) + text[:64], prefix="doc-")],
                    )
                except Exception as exc:
                    logger.warning("Scan skip %s: %s", path, exc)

        background_tasks.add_task(_bg)
        return {
            "status": "queued",
            "files_found": len(files),
            "input_dir": str(input_dir),
        }

    return router
