"""Dataclasses for application-level chat data.

These are product objects — threads and messages — and deliberately live outside
``memgraphrag/storage``: that package holds the RAG's own knowledge stores, keyed by
namespace and swappable per backend. A conversation is neither knowledge nor
swappable, so it gets its own model and its own database.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# A thread title is derived from its first user message. The mockup renders it on a
# single ellipsised line, so anything past this is never shown and only costs storage.
TITLE_LIMIT = 120

VALID_ROLES = ("user", "assistant")


def new_id() -> str:
    """Return a fresh opaque id. Not content-addressed: two identical questions in
    different threads are different messages."""
    return uuid.uuid4().hex


def now_ts() -> int:
    """Unix seconds, matching the doc-status records written by the pipeline."""
    return int(time.time())


def derive_title(text: str) -> str:
    """Build a thread title from the first user message."""
    collapsed = " ".join((text or "").split())
    if not collapsed:
        return "New chat"
    if len(collapsed) <= TITLE_LIMIT:
        return collapsed
    return collapsed[:TITLE_LIMIT].rstrip() + "…"


@dataclass
class ChatMessage:
    """One turn. ``refs`` carries the ``references`` payload returned by /query."""

    id: str
    thread_id: str
    role: str
    content: str
    refs: list[dict[str, Any]] = field(default_factory=list)
    created_at: int = field(default_factory=now_ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "references": list(self.refs),
            "created_at": self.created_at,
        }


@dataclass
class ChatThread:
    """A conversation. ``params`` stores the retrieval knobs chosen for the thread so
    a reopened conversation answers the way it did when it was created."""

    id: str
    owner: str
    title: str
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=now_ts)
    updated_at: int = field(default_factory=now_ts)

    def to_dict(self, messages: list[ChatMessage] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "owner": self.owner,
            "title": self.title,
            "model": self.model,
            "params": dict(self.params),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if messages is not None:
            payload["messages"] = [m.to_dict() for m in messages]
        return payload
