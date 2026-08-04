"""PostgreSQL storage backends for MemGraphRAG.

Simplified adaptation of LightRAG ``lightrag/kg/postgres_impl.py`` using
asyncpg directly. Provides ``PGKVStorage``, ``PGVectorStorage`` (pgvector
cosine), and ``PGDocStatusStorage``. Tables are created on ``initialize``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from memgraphrag.base import BaseKVStorage, BaseVectorStorage, DocStatus, DocStatusStorage
from memgraphrag.constants import EMBEDDING_DIM

logger = logging.getLogger(__name__)


def _strip_nul(obj: Any) -> Any:
    """Remove NULs so asyncpg can store values as PostgreSQL text/jsonb."""
    if isinstance(obj, str):
        return obj.replace("\x00", "") if "\x00" in obj else obj
    if isinstance(obj, dict):
        return {
            (_strip_nul(k) if isinstance(k, str) else k): _strip_nul(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_nul(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_nul(v) for v in obj)
    return obj


def _json_dumps_safe(obj: Any) -> str:
    return json.dumps(_strip_nul(obj), ensure_ascii=False)



def _pg_dsn() -> dict[str, Any]:
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", ""),
        "database": os.environ.get("POSTGRES_DATABASE", "memgraphrag"),
    }


def _safe_ident(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned.lower()[:50]


async def _get_pool():
    import asyncpg

    dsn = _pg_dsn()
    return await asyncpg.create_pool(
        host=dsn["host"],
        port=dsn["port"],
        user=dsn["user"],
        password=dsn["password"],
        database=dsn["database"],
        min_size=1,
        max_size=10,
    )


@dataclass
class PGKVStorage(BaseKVStorage):
    """Generic key-value storage backed by a JSONB PostgreSQL table."""

    _pool: Any = field(default=None, init=False, repr=False)
    _table: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        ws = _safe_ident(self.workspace or "default")
        ns = _safe_ident(self.namespace)
        self._table = f"mgr_kv_{ws}_{ns}"

    async def initialize(self) -> None:
        self._pool = await _get_pool()
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )

    async def finalize(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT data FROM {self._table} WHERE id = $1", id
            )
        if row is None:
            return None
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        result = dict(data)
        result["_id"] = id
        return result

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, data FROM {self._table} WHERE id = ANY($1::text[])",
                ids,
            )
        by_id = {}
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            record = dict(data)
            record["_id"] = row["id"]
            by_id[row["id"]] = record
        return [by_id.get(i) for i in ids]  # type: ignore[misc]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id FROM {self._table} WHERE id = ANY($1::text[])",
                list(keys),
            )
        existing = {row["id"] for row in rows}
        return set(keys) - existing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for key, value in data.items():
                    record = dict(value)
                    record["_id"] = key
                    await conn.execute(
                        f"""
                        INSERT INTO {self._table} (id, data)
                        VALUES ($1, $2::jsonb)
                        ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
                        """,
                        key,
                        _json_dumps_safe(record),
                    )

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1::text[])",
                ids,
            )

    async def get_all(self) -> dict[str, dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT id, data FROM {self._table}")
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            record = dict(data)
            record["_id"] = row["id"]
            out[row["id"]] = record
        return out

    async def drop(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table}")


@dataclass
class PGVectorStorage(BaseVectorStorage):
    """pgvector cosine similarity storage."""

    _pool: Any = field(default=None, init=False, repr=False)
    _table: str = field(default="", init=False, repr=False)
    _embedding_dim: int = field(default=EMBEDDING_DIM, init=False, repr=False)

    def __post_init__(self) -> None:
        ws = _safe_ident(self.workspace or "default")
        ns = _safe_ident(self.namespace)
        self._table = f"mgr_vec_{ws}_{ns}"
        emb_dim = None
        if self.embedding_func is not None:
            emb_dim = getattr(self.embedding_func, "embedding_dim", None)
        if emb_dim is None:
            emb_dim = self.global_config.get("embedding_dim", EMBEDDING_DIM)
        self._embedding_dim = int(emb_dim)

    async def initialize(self) -> None:
        self._pool = await _get_pool()
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    embedding VECTOR({self._embedding_dim}),
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )

    async def finalize(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def query(
        self, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        vector_literal = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM {self._table}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vector_literal,
                top_k,
            )
        results = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            results.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "distance": float(row["score"]),
                    **(meta or {}),
                }
            )
        return results

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for doc_id, record in data.items():
                    embedding = record.get("embedding")
                    if embedding is None:
                        raise ValueError(
                            f"PGVectorStorage.upsert requires 'embedding' for id={doc_id}"
                        )
                    content = record.get("content", "")
                    meta = {
                        k: v
                        for k, v in record.items()
                        if k not in ("embedding", "content")
                    }
                    vector_literal = (
                        "[" + ",".join(str(float(x)) for x in embedding) + "]"
                    )
                    await conn.execute(
                        f"""
                        INSERT INTO {self._table} (id, content, embedding, metadata)
                        VALUES ($1, $2, $3::vector, $4::jsonb)
                        ON CONFLICT (id) DO UPDATE
                        SET content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                        """,
                        doc_id,
                        content,
                        vector_literal,
                        _json_dumps_safe(meta),
                    )

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1::text[])",
                ids,
            )

    async def drop(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table}")


@dataclass
class PGDocStatusStorage(DocStatusStorage):
    """Document-status storage backed by PostgreSQL."""

    _pool: Any = field(default=None, init=False, repr=False)
    _table: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        ws = _safe_ident(self.workspace or "default")
        ns = _safe_ident(self.namespace or "doc_status")
        self._table = f"mgr_doc_{ws}_{ns}"

    async def initialize(self) -> None:
        self._pool = await _get_pool()
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    data JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_status_idx "
                f"ON {self._table} (status)"
            )

    async def finalize(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT status, data FROM {self._table} WHERE id = $1", id
            )
        if row is None:
            return None
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        result = dict(data)
        result["status"] = row["status"]
        result["_id"] = id
        return result

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, status, data FROM {self._table} WHERE id = ANY($1::text[])",
                ids,
            )
        by_id = {}
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            record = dict(data)
            record["status"] = row["status"]
            record["_id"] = row["id"]
            by_id[row["id"]] = record
        return [by_id.get(i) for i in ids]  # type: ignore[misc]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id FROM {self._table} WHERE id = ANY($1::text[])",
                list(keys),
            )
        existing = {row["id"] for row in rows}
        return set(keys) - existing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for key, value in data.items():
                    record = dict(value)
                    status = str(record.get("status", DocStatus.PENDING.value))
                    if isinstance(record.get("status"), DocStatus):
                        status = record["status"].value
                    record["status"] = status
                    record["_id"] = key
                    await conn.execute(
                        f"""
                        INSERT INTO {self._table} (id, status, data)
                        VALUES ($1, $2, $3::jsonb)
                        ON CONFLICT (id) DO UPDATE
                        SET status = EXCLUDED.status, data = EXCLUDED.data
                        """,
                        key,
                        status,
                        _json_dumps_safe(record),
                    )

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1::text[])",
                ids,
            )

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, dict[str, Any]]:
        wanted = [s.value if isinstance(s, DocStatus) else str(s) for s in statuses]
        if not wanted:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, status, data FROM {self._table} WHERE status = ANY($1::text[])",
                wanted,
            )
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            record = dict(data)
            record["status"] = row["status"]
            record["_id"] = row["id"]
            result[row["id"]] = record
        return result

    async def get_all(self) -> dict[str, dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT id, status, data FROM {self._table}")
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            record = dict(data)
            record["status"] = row["status"]
            record["_id"] = row["id"]
            result[row["id"]] = record
        return result

    async def drop(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._table}")
