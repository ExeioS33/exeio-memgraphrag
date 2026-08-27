"""Unit tests for the PostgreSQL backends (fake asyncpg pool, no server needed)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytestmark = pytest.mark.offline


class FakeConn:
    """Records executed SQL and serves a canned ``embedding`` column row."""

    def __init__(
        self,
        embedding_row: dict[str, Any] | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.executed: list[str] = []
        self.embedding_row = embedding_row
        self.fail_on = fail_on

    async def execute(self, sql: str, *args: Any) -> None:
        self.executed.append(sql)
        if self.fail_on is not None and self.fail_on in sql:
            raise RuntimeError('access method "hnsw" does not exist')

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        return self.embedding_row


class _FakeAcquire:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeConn:
        return self._conn

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class FakePool:
    def __init__(self, conn: FakeConn | None = None) -> None:
        self.conn = conn or FakeConn()
        self.closed = False

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.conn)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def pg(monkeypatch: pytest.MonkeyPatch):
    """Isolate the pool registry and replace asyncpg pool creation."""
    from memgraphrag.storage import postgres_impl

    monkeypatch.setattr(postgres_impl.ClientManager, "_pools", {})
    return postgres_impl


def _install_pool_factory(monkeypatch, postgres_impl, conn: FakeConn | None = None):
    created: list[FakePool] = []

    async def fake_create_pool(config: dict[str, Any]) -> FakePool:
        pool = FakePool(conn)
        created.append(pool)
        return pool

    monkeypatch.setattr(postgres_impl, "_create_pool", fake_create_pool)
    return created


def _storage(cls, namespace: str, **global_config: Any):
    return cls(workspace="ws", namespace=namespace, global_config=global_config)


@pytest.mark.asyncio
async def test_storages_share_one_pool(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    created = _install_pool_factory(monkeypatch, pg)

    kv = _storage(pg.PGKVStorage, "kv")
    doc = _storage(pg.PGDocStatusStorage, "doc_status")
    vec = _storage(pg.PGVectorStorage, "vec", embedding_dim=4)
    for storage in (kv, doc, vec):
        await storage.initialize()

    assert len(created) == 1, "one DSN must yield exactly one asyncpg pool"
    assert kv._pool is doc._pool is vec._pool


@pytest.mark.asyncio
async def test_pool_closes_only_after_the_last_finalize(
    monkeypatch: pytest.MonkeyPatch, pg
) -> None:
    created = _install_pool_factory(monkeypatch, pg)

    kv = _storage(pg.PGKVStorage, "kv")
    doc = _storage(pg.PGDocStatusStorage, "doc_status")
    await kv.initialize()
    await doc.initialize()

    await kv.finalize()
    assert created[0].closed is False, "a pool still in use must stay open"
    assert kv._pool is None

    await doc.finalize()
    assert created[0].closed is True
    assert pg.ClientManager._pools == {}


@pytest.mark.asyncio
async def test_pool_reopened_after_full_release(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    created = _install_pool_factory(monkeypatch, pg)

    kv = _storage(pg.PGKVStorage, "kv")
    await kv.initialize()
    await kv.finalize()
    await kv.initialize()

    assert len(created) == 2
    assert created[1].closed is False


@pytest.mark.asyncio
async def test_double_finalize_is_harmless(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    created = _install_pool_factory(monkeypatch, pg)

    kv = _storage(pg.PGKVStorage, "kv")
    await kv.initialize()
    await kv.finalize()
    await kv.finalize()

    assert len(created) == 1


def test_vector_index_ddl_defaults_to_hnsw_cosine(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    monkeypatch.delenv("POSTGRES_VECTOR_INDEX_TYPE", raising=False)
    ddl = pg._vector_index_ddl("mgr_vec_ws_chunks")
    assert ddl is not None
    assert "USING hnsw" in ddl
    # The query orders by `<=>`, so only a cosine operator class can be used.
    assert "vector_cosine_ops" in ddl


def test_vector_index_ddl_honours_ivfflat(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    monkeypatch.setenv("POSTGRES_VECTOR_INDEX_TYPE", "IVFFlat")
    monkeypatch.setenv("POSTGRES_IVFFLAT_LISTS", "42")
    ddl = pg._vector_index_ddl("mgr_vec_ws_chunks")
    assert ddl is not None
    assert "USING ivfflat" in ddl
    assert "lists = 42" in ddl
    assert "vector_cosine_ops" in ddl


def test_vector_index_ddl_can_be_disabled(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    monkeypatch.setenv("POSTGRES_VECTOR_INDEX_TYPE", "none")
    assert pg._vector_index_ddl("mgr_vec_ws_chunks") is None


def test_unknown_vector_index_type_falls_back_to_hnsw(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    monkeypatch.setenv("POSTGRES_VECTOR_INDEX_TYPE", "diskann")
    ddl = pg._vector_index_ddl("mgr_vec_ws_chunks")
    assert ddl is not None and "USING hnsw" in ddl


@pytest.mark.asyncio
async def test_initialize_creates_the_vector_index(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    monkeypatch.delenv("POSTGRES_VECTOR_INDEX_TYPE", raising=False)
    conn = FakeConn(embedding_row=None)
    _install_pool_factory(monkeypatch, pg, conn)

    vec = _storage(pg.PGVectorStorage, "chunks", embedding_dim=4)
    await vec.initialize()

    index_statements = [sql for sql in conn.executed if "CREATE INDEX" in sql]
    assert index_statements, "no ANN index means a sequential scan on every query"
    assert "vector_cosine_ops" in index_statements[0]
    assert "mgr_vec_ws_chunks" in index_statements[0]


@pytest.mark.asyncio
async def test_index_creation_failure_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch, pg, caplog: pytest.LogCaptureFixture
) -> None:
    conn = FakeConn(embedding_row=None, fail_on="CREATE INDEX")
    _install_pool_factory(monkeypatch, pg, conn)

    vec = _storage(pg.PGVectorStorage, "chunks", embedding_dim=4)
    with caplog.at_level(logging.WARNING):
        await vec.initialize()

    assert "vector index" in caplog.text


@pytest.mark.asyncio
async def test_embedding_dim_mismatch_fails_fast(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    conn = FakeConn(embedding_row={"typmod": 1024, "type_name": "vector(1024)"})
    _install_pool_factory(monkeypatch, pg, conn)

    vec = _storage(pg.PGVectorStorage, "chunks", embedding_dim=768)
    with pytest.raises(RuntimeError) as excinfo:
        await vec.initialize()

    message = str(excinfo.value)
    assert "1024" in message and "768" in message
    assert "mgr_vec_ws_chunks" in message


@pytest.mark.asyncio
async def test_matching_embedding_dim_initializes(monkeypatch: pytest.MonkeyPatch, pg) -> None:
    conn = FakeConn(embedding_row={"typmod": 768, "type_name": "vector(768)"})
    _install_pool_factory(monkeypatch, pg, conn)

    vec = _storage(pg.PGVectorStorage, "chunks", embedding_dim=768)
    await vec.initialize()

    assert any("CREATE TABLE" in sql for sql in conn.executed)
