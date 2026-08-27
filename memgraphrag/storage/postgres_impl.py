"""PostgreSQL storage backends for MemGraphRAG.

Simplified adaptation of LightRAG ``lightrag/kg/postgres_impl.py`` using
asyncpg directly. Provides ``PGKVStorage``, ``PGVectorStorage`` (pgvector
cosine), and ``PGDocStatusStorage``. Tables and indexes are created on
``initialize``; all storages sharing a DSN share one reference-counted pool
(``ClientManager``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

from memgraphrag.base import BaseKVStorage, BaseVectorStorage, DocStatus, DocStatusStorage
from memgraphrag.constants import EMBEDDING_DIM
from memgraphrag.utils.env import get_env_value

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


def _pool_config() -> dict[str, Any]:
    config = _pg_dsn()
    # Two connections is the floor: one busy statement plus one for the next acquire.
    config["max_connections"] = max(
        2, get_env_value("POSTGRES_MAX_CONNECTIONS", 10, int)
    )
    return config


async def _create_pool(config: dict[str, Any]) -> Any:
    import asyncpg

    return await asyncpg.create_pool(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        min_size=1,
        max_size=config["max_connections"],
    )


class ClientManager:
    """Reference-counted registry of asyncpg pools shared by the PG storages.

    Every ``initialize()`` used to open a pool of its own. The engine builds nine
    storages, eight of them on PostgreSQL, so a single worker opened ~80
    connections against a server whose default ``max_connections`` is 100 — the
    second worker could not even connect. Storages sharing a DSN now share one
    pool, closed when the last of them calls ``release_client``.

    Mirrors the ``ClientManager`` pattern of LightRAG ``lightrag/kg/postgres_impl.py``.
    """

    _pools: ClassVar[dict[tuple[Any, ...], dict[str, Any]]] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> Any:
        config = _pool_config()
        async with cls._lock:
            key = cls._pool_key(config)
            entry = cls._pools.get(key)
            if entry is None:
                entry = {"pool": await _create_pool(config), "ref_count": 0}
                cls._pools[key] = entry
            entry["ref_count"] += 1
            return entry["pool"]

    @classmethod
    async def release_client(cls, pool: Any) -> None:
        """Drop one reference; close the pool once nobody holds it any more."""
        if pool is None:
            return
        async with cls._lock:
            for key, entry in list(cls._pools.items()):
                if entry["pool"] is not pool:
                    continue
                entry["ref_count"] -= 1
                if entry["ref_count"] <= 0:
                    del cls._pools[key]
                    await pool.close()
                return
        # Untracked pool (already released, or created before this manager existed):
        # close it anyway rather than leaking its connections.
        await pool.close()

    @staticmethod
    def _pool_key(config: dict[str, Any]) -> tuple[Any, ...]:
        # An asyncpg pool belongs to the loop that created it, so the running loop is
        # part of the pool identity: reusing one across loops awaits on a dead loop.
        return (
            id(asyncio.get_running_loop()),
            config["host"],
            config["port"],
            config["user"],
            config["database"],
        )


def _vector_index_ddl(table: str) -> str | None:
    """Return the pgvector index DDL for ``table.embedding``, or ``None`` if disabled.

    ``POSTGRES_VECTOR_INDEX_TYPE`` was documented in ``env.example`` but never read,
    and no index was ever created, so every ``/query`` did a full sequential scan of
    each vector collection. Accepted values: ``hnsw`` (default), ``ivfflat``, ``none``.

    The operator class must be ``vector_cosine_ops``: queries order by ``<=>``, and
    pgvector only uses an index whose operator matches the ordering operator.
    """
    index_type = str(
        get_env_value("POSTGRES_VECTOR_INDEX_TYPE", "hnsw", str)
    ).strip().lower()
    if index_type in ("", "none", "off"):
        return None
    if index_type == "ivfflat":
        lists = max(1, get_env_value("POSTGRES_IVFFLAT_LISTS", 100, int))
        return (
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} "
            f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
        )
    if index_type != "hnsw":
        logger.warning(
            "Unknown POSTGRES_VECTOR_INDEX_TYPE=%r, falling back to hnsw", index_type
        )
    m = max(2, get_env_value("POSTGRES_HNSW_M", 16, int))
    ef_construction = max(4, get_env_value("POSTGRES_HNSW_EF", 64, int))
    return (
        f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} "
        f"USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {m}, ef_construction = {ef_construction})"
    )


_EMBEDDING_COLUMN_SQL = """
SELECT a.atttypmod AS typmod, format_type(a.atttypid, a.atttypmod) AS type_name
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = $1
  AND a.attname = 'embedding'
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND n.nspname = ANY(current_schemas(false))
LIMIT 1
"""


async def _assert_embedding_dim(conn: Any, table: str, expected_dim: int) -> None:
    """Fail fast when an existing table pins a different embedding dimension.

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing column, so raising
    ``EMBEDDING_DIM`` on a populated database left ``VECTOR(<old dim>)`` in place and
    every insert failed later with an opaque pgvector error. For pgvector the column
    ``atttypmod`` holds the declared dimension.
    """
    row = await conn.fetchrow(_EMBEDDING_COLUMN_SQL, table)
    if row is None:
        return
    actual_dim = int(row["typmod"] or 0)
    if actual_dim <= 0 or actual_dim == expected_dim:
        return
    raise RuntimeError(
        f"Embedding dimension mismatch on table {table}: the column is "
        f"{row['type_name']} ({actual_dim} dimensions) while the configured "
        f"embedding dimension is {expected_dim}. CREATE TABLE IF NOT EXISTS cannot "
        f"change it. Either restore EMBEDDING_DIM={actual_dim}, or re-index into a "
        f"fresh workspace, or run: ALTER TABLE {table} ALTER COLUMN embedding TYPE "
        f"VECTOR({expected_dim}) after clearing the stale vectors."
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
        self._pool = await ClientManager.get_client()
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
        pool, self._pool = self._pool, None
        await ClientManager.release_client(pool)

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
        self._pool = await ClientManager.get_client()
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
            await _assert_embedding_dim(conn, self._table, self._embedding_dim)
            await self._ensure_vector_index(conn)

    async def _ensure_vector_index(self, conn: Any) -> None:
        """Create the ANN index; a backend that cannot build it must still serve."""
        ddl = _vector_index_ddl(self._table)
        if ddl is None:
            return
        try:
            await conn.execute(ddl)
        except Exception as exc:  # noqa: BLE001 - an index is an optimisation, not a requirement
            logger.warning(
                "Could not create the vector index on %s, queries fall back to a "
                "sequential scan: %s",
                self._table,
                exc,
            )

    async def finalize(self) -> None:
        pool, self._pool = self._pool, None
        await ClientManager.release_client(pool)

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
            # `1 - (embedding <=> $1)` is cosine similarity in [-1, 1]. Publish it as
            # `score` per the BaseVectorStorage contract; `distance` is the alias.
            results.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "score": float(row["score"]),
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
        self._pool = await ClientManager.get_client()
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
        pool, self._pool = self._pool, None
        await ClientManager.release_client(pool)

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
