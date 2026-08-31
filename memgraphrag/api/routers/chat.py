"""Chat thread and message routes.

Backed by the dedicated application database (``APP_DATABASE_URL``), not by any RAG
storage backend. When that database is not configured these routes answer 503 and
every other route keeps working — the UI degrades to single-shot questions rather
than failing to load.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from memgraphrag.utils.step_log import main_step

logger = logging.getLogger("memgraphrag.api.chat")

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    Query = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    status = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]  # noqa: E731

# Identity for API-key callers and for the fully open configuration. The combined auth
# dependency returns nothing, so a Bearer subject is the only real identity available.
DEFAULT_OWNER = "guest"

DEFAULT_THREAD_LIMIT = 100
MAX_THREAD_LIMIT = 500

STORE_UNCONFIGURED = (
    "Chat persistence is not configured. Set APP_DATABASE_URL and start the postgres-app service."
)


class ThreadCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    model: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class ThreadUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    model: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class MessageCreateRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)
    references: Optional[list[dict[str, Any]]] = None


def resolve_owner(request: Any) -> str:
    """Identify the caller from a Bearer subject, falling back to ``guest``.

    API-key authentication carries no identity, so every API-key caller shares the
    guest namespace. That is intentional: an API key is a service credential, not a
    person, and inventing a per-key identity would imply an isolation the key does
    not actually provide.
    """
    from memgraphrag.api.dependencies import resolve_auth_context

    header = request.headers.get("Authorization") or ""
    if not header.lower().startswith("bearer "):
        return DEFAULT_OWNER
    token = header[7:].strip()
    if not token:
        return DEFAULT_OWNER
    handler, _patterns, _configured = resolve_auth_context(request)
    try:
        info = handler.validate_token(token)
    except Exception:  # noqa: BLE001 - an unreadable token is simply not an identity
        return DEFAULT_OWNER
    subject = str(info.get("sub") or "").strip()
    return subject or DEFAULT_OWNER


def _require_store(request: Any) -> Any:
    store = getattr(request.app.state, "chat_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=STORE_UNCONFIGURED,
        )
    return store


def _not_found(thread_id: str) -> Any:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Thread not found: {thread_id}",
    )


def create_chat_router(api_key: Optional[str] = None) -> Any:
    if APIRouter is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import get_combined_auth_dependency

    router = APIRouter(prefix="/chat", tags=["chat"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("/threads", dependencies=[Depends(combined_auth)])
    async def create_thread(request: Request, body: ThreadCreateRequest):
        store = _require_store(request)
        owner = resolve_owner(request)
        thread = await store.create_thread(
            owner,
            title=body.title,
            model=body.model,
            params=body.params,
        )
        main_step(logger, "api.chat.create_thread", thread=thread.id, owner=owner)
        return thread.to_dict(messages=[])

    @router.get("/threads", dependencies=[Depends(combined_auth)])
    async def list_threads(
        request: Request,
        limit: int = Query(DEFAULT_THREAD_LIMIT, ge=1, le=MAX_THREAD_LIMIT),
        offset: int = Query(0, ge=0),
    ):
        store = _require_store(request)
        owner = resolve_owner(request)
        threads, total = await store.list_threads(owner, limit=limit, offset=offset)
        next_offset = offset + len(threads)
        return {
            "threads": [t.to_dict() for t in threads],
            "total": total,
            "limit": limit,
            "offset": offset,
            "returned": len(threads),
            "next_offset": next_offset if next_offset < total else None,
        }

    @router.get("/threads/{thread_id}", dependencies=[Depends(combined_auth)])
    async def get_thread(request: Request, thread_id: str):
        store = _require_store(request)
        owner = resolve_owner(request)
        thread = await store.get_thread(thread_id, owner)
        if thread is None:
            raise _not_found(thread_id)
        messages = await store.list_messages(thread_id, owner) or []
        return thread.to_dict(messages=messages)

    @router.patch("/threads/{thread_id}", dependencies=[Depends(combined_auth)])
    async def update_thread(request: Request, thread_id: str, body: ThreadUpdateRequest):
        store = _require_store(request)
        owner = resolve_owner(request)
        # Only fields the caller actually sent are applied, so PATCH {"title": ...}
        # does not silently blank the thread's model and params.
        provided = body.model_fields_set
        kwargs: dict[str, Any] = {}
        if "title" in provided and body.title is not None:
            kwargs["title"] = body.title
        if "model" in provided:
            kwargs["model"] = body.model
        if "params" in provided:
            kwargs["params"] = body.params
        thread = await store.update_thread(thread_id, owner, **kwargs)
        if thread is None:
            raise _not_found(thread_id)
        return thread.to_dict()

    @router.delete("/threads/{thread_id}", dependencies=[Depends(combined_auth)])
    async def delete_thread(request: Request, thread_id: str):
        store = _require_store(request)
        owner = resolve_owner(request)
        deleted = await store.delete_thread(thread_id, owner)
        if not deleted:
            raise _not_found(thread_id)
        main_step(logger, "api.chat.delete_thread", thread=thread_id, owner=owner)
        return {"status": "deleted", "thread_id": thread_id}

    @router.post("/threads/{thread_id}/messages", dependencies=[Depends(combined_auth)])
    async def add_message(request: Request, thread_id: str, body: MessageCreateRequest):
        store = _require_store(request)
        owner = resolve_owner(request)
        message = await store.add_message(
            thread_id,
            owner,
            role=body.role,
            content=body.content,
            refs=body.references,
        )
        if message is None:
            raise _not_found(thread_id)
        return message.to_dict()

    return router
