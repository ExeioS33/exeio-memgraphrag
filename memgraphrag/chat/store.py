"""Chat persistence backed by a dedicated Postgres.

Two implementations share one interface: ``PostgresChatStore`` for real use and
``InMemoryChatStore`` for tests, mirroring how ``create_app(testing=True)`` already
swaps the RAG engine for a mock.

There is deliberately **no file-backed fallback**. A single persistence path is
easier to reason about than two, and a silent degradation to a JSON file would hide
a misconfigured deployment until someone noticed their conversations had stopped
following them between machines. When ``APP_DATABASE_URL`` is unset the store is
simply absent and ``/chat/*`` answers 503; every other route is unaffected.

Every read and write is scoped by ``owner`` so one account cannot reach another's
threads, even by guessing an id.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from memgraphrag.chat.models import (
    VALID_ROLES,
    ChatMessage,
    ChatThread,
    derive_title,
    new_id,
    now_ts,
)

logger = logging.getLogger("memgraphrag.chat.store")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Sentinel for "caller did not pass this field", so update_thread can tell
# "set title to None" apart from "leave title alone".
_UNSET: Any = object()


class ChatStoreUnavailable(RuntimeError):
    """Raised when chat persistence is not configured or not reachable."""


def load_schema_sql() -> str:
    """Read the DDL shipped next to this module.

    Ships only because ``pyproject.toml`` lists ``chat/*.sql`` under package-data —
    without that entry the file is absent from the wheel and startup fails here.
    """
    return SCHEMA_PATH.read_text(encoding="utf-8")


class BaseChatStore:
    """Interface shared by the Postgres and in-memory stores."""

    async def initialize(self) -> None:  # pragma: no cover - trivial
        return None

    async def close(self) -> None:  # pragma: no cover - trivial
        return None

    async def create_thread(
        self,
        owner: str,
        *,
        title: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ChatThread:
        raise NotImplementedError

    async def list_threads(
        self, owner: str, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[ChatThread], int]:
        raise NotImplementedError

    async def get_thread(self, thread_id: str, owner: str) -> ChatThread | None:
        raise NotImplementedError

    async def update_thread(
        self,
        thread_id: str,
        owner: str,
        *,
        title: Any = _UNSET,
        model: Any = _UNSET,
        params: Any = _UNSET,
    ) -> ChatThread | None:
        raise NotImplementedError

    async def delete_thread(self, thread_id: str, owner: str) -> bool:
        raise NotImplementedError

    async def add_message(
        self,
        thread_id: str,
        owner: str,
        *,
        role: str,
        content: str,
        refs: list[dict[str, Any]] | None = None,
    ) -> ChatMessage | None:
        raise NotImplementedError

    async def list_messages(self, thread_id: str, owner: str) -> list[ChatMessage] | None:
        raise NotImplementedError


def _check_role(role: str) -> str:
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
    return role


class InMemoryChatStore(BaseChatStore):
    """Process-local store used by tests. Nothing survives a restart."""

    def __init__(self) -> None:
        self._threads: dict[str, ChatThread] = {}
        self._messages: dict[str, list[ChatMessage]] = {}

    async def create_thread(
        self,
        owner: str,
        *,
        title: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ChatThread:
        thread = ChatThread(
            id=new_id(),
            owner=owner,
            title=title or "New chat",
            model=model,
            params=dict(params or {}),
        )
        self._threads[thread.id] = thread
        self._messages[thread.id] = []
        return thread

    async def list_threads(
        self, owner: str, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[ChatThread], int]:
        owned = [t for t in self._threads.values() if t.owner == owner]
        owned.sort(key=lambda t: (t.updated_at, t.id), reverse=True)
        return owned[offset : offset + limit], len(owned)

    async def get_thread(self, thread_id: str, owner: str) -> ChatThread | None:
        thread = self._threads.get(thread_id)
        if thread is None or thread.owner != owner:
            return None
        return thread

    async def update_thread(
        self,
        thread_id: str,
        owner: str,
        *,
        title: Any = _UNSET,
        model: Any = _UNSET,
        params: Any = _UNSET,
    ) -> ChatThread | None:
        thread = await self.get_thread(thread_id, owner)
        if thread is None:
            return None
        if title is not _UNSET:
            thread.title = str(title)
        if model is not _UNSET:
            thread.model = model
        if params is not _UNSET:
            thread.params = dict(params or {})
        thread.updated_at = now_ts()
        return thread

    async def delete_thread(self, thread_id: str, owner: str) -> bool:
        thread = await self.get_thread(thread_id, owner)
        if thread is None:
            return False
        self._threads.pop(thread_id, None)
        self._messages.pop(thread_id, None)
        return True

    async def add_message(
        self,
        thread_id: str,
        owner: str,
        *,
        role: str,
        content: str,
        refs: list[dict[str, Any]] | None = None,
    ) -> ChatMessage | None:
        thread = await self.get_thread(thread_id, owner)
        if thread is None:
            return None
        message = ChatMessage(
            id=new_id(),
            thread_id=thread_id,
            role=_check_role(role),
            content=content,
            refs=list(refs or []),
        )
        self._messages.setdefault(thread_id, []).append(message)
        # First user turn names the thread, matching the mockup's sidebar.
        if role == "user" and thread.title == "New chat":
            thread.title = derive_title(content)
        thread.updated_at = message.created_at
        return message

    async def list_messages(self, thread_id: str, owner: str) -> list[ChatMessage] | None:
        thread = await self.get_thread(thread_id, owner)
        if thread is None:
            return None
        return list(self._messages.get(thread_id, ()))


class PostgresChatStore(BaseChatStore):
    """asyncpg-backed store against the dedicated application database."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any | None = None

    async def initialize(self) -> None:
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - asyncpg ships with [api]
            raise ChatStoreUnavailable("asyncpg is required; install memgraphrag[api]") from exc

        async def _init_connection(conn: Any) -> None:
            # asyncpg hands back jsonb as raw text unless a codec is registered; without
            # this, params/refs would come out of the database as strings.
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            init=_init_connection,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(load_schema_sql())
        logger.info("Chat store ready (dedicated application database)")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise ChatStoreUnavailable("Chat store is not initialized")
        return self._pool

    @staticmethod
    def _thread_from_row(row: Any) -> ChatThread:
        return ChatThread(
            id=row["id"],
            owner=row["owner"],
            title=row["title"],
            model=row["model"],
            params=row["params"] or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _message_from_row(row: Any) -> ChatMessage:
        return ChatMessage(
            id=row["id"],
            thread_id=row["thread_id"],
            role=row["role"],
            content=row["content"],
            refs=row["refs"] or [],
            created_at=row["created_at"],
        )

    async def create_thread(
        self,
        owner: str,
        *,
        title: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ChatThread:
        thread = ChatThread(
            id=new_id(),
            owner=owner,
            title=title or "New chat",
            model=model,
            params=dict(params or {}),
        )
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_thread (id, owner, title, model, params, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                thread.id,
                thread.owner,
                thread.title,
                thread.model,
                thread.params,
                thread.created_at,
                thread.updated_at,
            )
        return thread

    async def list_threads(
        self, owner: str, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[ChatThread], int]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, owner, title, model, params, created_at, updated_at
                FROM chat_thread
                WHERE owner = $1
                ORDER BY updated_at DESC, id DESC
                LIMIT $2 OFFSET $3
                """,
                owner,
                limit,
                offset,
            )
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM chat_thread WHERE owner = $1",
                owner,
            )
        return [self._thread_from_row(r) for r in rows], int(total or 0)

    async def get_thread(self, thread_id: str, owner: str) -> ChatThread | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, owner, title, model, params, created_at, updated_at
                FROM chat_thread
                WHERE id = $1 AND owner = $2
                """,
                thread_id,
                owner,
            )
        return None if row is None else self._thread_from_row(row)

    async def update_thread(
        self,
        thread_id: str,
        owner: str,
        *,
        title: Any = _UNSET,
        model: Any = _UNSET,
        params: Any = _UNSET,
    ) -> ChatThread | None:
        current = await self.get_thread(thread_id, owner)
        if current is None:
            return None
        new_title = current.title if title is _UNSET else str(title)
        new_model = current.model if model is _UNSET else model
        new_params = current.params if params is _UNSET else dict(params or {})
        updated_at = now_ts()
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE chat_thread
                SET title = $3, model = $4, params = $5, updated_at = $6
                WHERE id = $1 AND owner = $2
                """,
                thread_id,
                owner,
                new_title,
                new_model,
                new_params,
                updated_at,
            )
        current.title = new_title
        current.model = new_model
        current.params = new_params
        current.updated_at = updated_at
        return current

    async def delete_thread(self, thread_id: str, owner: str) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Messages go with it: chat_message.thread_id is ON DELETE CASCADE.
            result = await conn.execute(
                "DELETE FROM chat_thread WHERE id = $1 AND owner = $2",
                thread_id,
                owner,
            )
        return str(result).rsplit(" ", 1)[-1] != "0"

    async def add_message(
        self,
        thread_id: str,
        owner: str,
        *,
        role: str,
        content: str,
        refs: list[dict[str, Any]] | None = None,
    ) -> ChatMessage | None:
        thread = await self.get_thread(thread_id, owner)
        if thread is None:
            return None
        message = ChatMessage(
            id=new_id(),
            thread_id=thread_id,
            role=_check_role(role),
            content=content,
            refs=list(refs or []),
        )
        pool = self._require_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO chat_message (id, thread_id, role, content, refs, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    message.id,
                    message.thread_id,
                    message.role,
                    message.content,
                    message.refs,
                    message.created_at,
                )
                if role == "user" and thread.title == "New chat":
                    await conn.execute(
                        "UPDATE chat_thread SET title = $2, updated_at = $3 WHERE id = $1",
                        thread_id,
                        derive_title(content),
                        message.created_at,
                    )
                else:
                    await conn.execute(
                        "UPDATE chat_thread SET updated_at = $2 WHERE id = $1",
                        thread_id,
                        message.created_at,
                    )
        return message

    async def list_messages(self, thread_id: str, owner: str) -> list[ChatMessage] | None:
        thread = await self.get_thread(thread_id, owner)
        if thread is None:
            return None
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, thread_id, role, content, refs, created_at
                FROM chat_message
                WHERE thread_id = $1
                ORDER BY created_at ASC, id ASC
                """,
                thread_id,
            )
        return [self._message_from_row(r) for r in rows]


def create_chat_store(dsn: str | None) -> BaseChatStore | None:
    """Build a store from ``APP_DATABASE_URL``. ``None`` means chat persistence is off."""
    if not dsn or not str(dsn).strip():
        return None
    return PostgresChatStore(str(dsn).strip())
