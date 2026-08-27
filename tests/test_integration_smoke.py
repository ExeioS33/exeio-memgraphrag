"""Integration smoke tests: real connections to the Compose backends.

Gated by ``--run-integration``. These used to assert only that a handful of
environment variables were non-empty, which any developer ``.env`` satisfied, so
they reported green without ever opening a socket — a gate that proved nothing
about the backends it claimed to smoke-test.

They now dial the backend. A missing driver or missing configuration skips (the
caller cannot connect to something they never configured), but a *configured*
backend that refuses the connection fails: passing ``--run-integration`` is an
explicit claim that ``docker compose up -d`` is running.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

#: Keep the dial short — a wedged backend should fail the gate, not hang CI.
CONNECT_TIMEOUT_SECONDS = 10.0


@pytest.mark.requires_db
async def test_postgres_accepts_connections() -> None:
    """PostgreSQL answers a trivial query over a real asyncpg connection."""
    asyncpg = pytest.importorskip("asyncpg", reason="asyncpg not installed (extra: api)")

    host = os.environ.get("POSTGRES_HOST")
    database = os.environ.get("POSTGRES_DATABASE")
    if not host or not database:
        pytest.skip("POSTGRES_HOST / POSTGRES_DATABASE not set")

    connection = await asyncpg.connect(
        host=host,
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        database=database,
        timeout=CONNECT_TIMEOUT_SECONDS,
    )
    try:
        assert await connection.fetchval("SELECT 1") == 1
    finally:
        await connection.close()


@pytest.mark.requires_db
async def test_neo4j_accepts_connections() -> None:
    """Neo4j answers a trivial Cypher query over a real bolt session."""
    neo4j = pytest.importorskip("neo4j", reason="neo4j driver not installed (extra: api)")

    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not username or not password:
        pytest.skip("NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD not set")

    driver = neo4j.AsyncGraphDatabase.driver(
        uri,
        auth=(username, password),
        connection_timeout=CONNECT_TIMEOUT_SECONDS,
        # Without this the driver retries a refused backend for 30s per query,
        # turning a down service into a CI timeout instead of a test failure.
        max_transaction_retry_time=CONNECT_TIMEOUT_SECONDS,
    )
    try:
        await driver.verify_connectivity()
        async with driver.session(database=os.environ.get("NEO4J_DATABASE")) as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
        assert record is not None and record["ok"] == 1
    finally:
        await driver.close()
