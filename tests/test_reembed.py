"""Tests for the re-embedding script.

The gap this script fills is invisible by construction: an embedding-model swap at a
constant dimension raises nothing anywhere. ``_assert_embedding_dim`` compares
integers, no model name is stored beside a vector, and ``_embed_and_store_memory``
skips every id it has already seen -- so a swapped model leaves the old vectors in
place and retrieval silently collapses. Nothing in the suite covered that, which is
why these tests pin the reconstruction rather than the plumbing.

The subtle part is what text to re-embed. For facts and schemas the ``content``
column holds ``_triple_str`` (``str(tuple(...))``), while what was actually embedded
is the tab-joined triple. Re-embedding ``content`` would look right and quietly index
a different string, so the tab-join is asserted here against the exact form
``core.py`` uses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.offline

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "reembed.py"


@pytest.fixture(scope="module")
def reembed():
    """Load the script as a module; it has no import-time side effects."""
    spec = importlib.util.spec_from_file_location("reembed_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["reembed_script"] = module
    spec.loader.exec_module(module)
    return module


class FakeConn:
    """Serves canned rows and records every write it is handed."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[str] = []
        self.written: list[tuple[str, list[Any]]] = []

    async def execute(self, sql: str, *args: Any) -> None:
        self.executed.append(sql)

    async def executemany(self, sql: str, payload: list[Any]) -> None:
        self.written.append((sql, list(payload)))

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if "to_regclass" in sql:
            return "public.some_table"
        if "count(*)" in sql:
            return len(self.rows)
        if "max(id)" in sql:
            return None
        return None

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        after, limit = args[0], args[1]
        later = [r for r in self.rows if r["id"] > after]
        return sorted(later, key=lambda r: r["id"])[:limit]


class FakePool:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn

    def acquire(self) -> Any:
        conn = self.conn

        class _Ctx:
            async def __aenter__(self) -> FakeConn:
                return conn

            async def __aexit__(self, *exc: Any) -> None:
                return None

        return _Ctx()


# --- what text gets re-embedded -------------------------------------------------


def test_triple_namespaces_rebuild_the_tab_joined_form(reembed):
    """facts/schemas must be rebuilt from metadata, not from the content column."""
    row = {
        "id": "fact-1",
        "content": "('Alpha', 'relates to', 'Beta')",
        "metadata": {"triple": ["Alpha", "relates to", "Beta"]},
    }
    assert reembed.embedded_text("facts", row) == "Alpha\trelates to\tBeta"
    assert reembed.embedded_text("schemas", row) == "Alpha\trelates to\tBeta"
    # The content column is precisely what must NOT be used here.
    assert reembed.embedded_text("facts", row) != row["content"]


def test_triple_metadata_may_arrive_as_json_text(reembed):
    """asyncpg hands back jsonb as str unless a codec is registered."""
    row = {
        "id": "fact-2",
        "content": "ignored",
        "metadata": json.dumps({"triple": ["A", "b", "C"]}),
    }
    assert reembed.embedded_text("facts", row) == "A\tb\tC"


def test_plain_namespaces_use_content_verbatim(reembed):
    row = {"id": "chunk-1", "content": "  a passage\nover lines ", "metadata": {}}
    assert reembed.embedded_text("chunks", row) == "  a passage\nover lines "
    assert reembed.embedded_text("entities", row) == "  a passage\nover lines "


@pytest.mark.parametrize(
    ("namespace", "row"),
    [
        ("facts", {"id": "f", "content": "x", "metadata": {}}),
        ("facts", {"id": "f", "content": "x", "metadata": {"triple": []}}),
        ("chunks", {"id": "c", "content": "", "metadata": {}}),
        ("chunks", {"id": "c", "content": None, "metadata": {}}),
    ],
)
def test_rows_without_embeddable_text_are_skipped(reembed, namespace, row):
    """An empty input would be embedded into an arbitrary vector; skip instead."""
    assert reembed.embedded_text(namespace, row) is None


def test_reconstruction_matches_the_engine(reembed):
    """Pin the reconstruction against the expression core.py embeds."""
    triple = ["Some Entity", "contient", "Autre"]
    engine_side = "\t".join(triple)
    row = {"id": "fact-3", "content": str(tuple(triple)), "metadata": {"triple": triple}}
    assert reembed.embedded_text("facts", row) == engine_side


# --- table naming ---------------------------------------------------------------


def test_live_table_matches_storage_naming(reembed):
    assert reembed.live_table("ws", "chunks") == "mgr_vec_ws_chunks"
    # _safe_ident lowercases and folds anything outside [A-Za-z0-9_].
    assert reembed.live_table("My-WS", "facts") == "mgr_vec_my_ws_facts"
    assert reembed.live_table("", "facts") == "mgr_vec_default_facts"


def test_overlong_identifier_is_refused(reembed):
    with pytest.raises(SystemExit, match="63-character limit"):
        reembed._check_identifier("x" * 64)


# --- provider width enforcement -------------------------------------------------


async def test_wrong_dimension_from_provider_is_named(reembed, monkeypatch):
    """Nothing on the write path checks vector width; this script must."""

    async def fake_embed(texts, model=None, embedding_dim=None, **kwargs):
        return np.zeros((len(texts), 768), dtype=np.float32)

    monkeypatch.setattr("memgraphrag.llm.openai_compatible.openai_embed", fake_embed, raising=True)
    with pytest.raises(SystemExit, match="768-dimension"):
        await reembed._embed(["a"], "some-model", 1024)


async def test_short_provider_response_is_refused(reembed, monkeypatch):
    async def fake_embed(texts, model=None, embedding_dim=None, **kwargs):
        return np.zeros((1, 1024), dtype=np.float32)

    monkeypatch.setattr("memgraphrag.llm.openai_compatible.openai_embed", fake_embed, raising=True)
    with pytest.raises(SystemExit, match="1 vectors for 2 texts"):
        await reembed._embed(["a", "b"], "some-model", 1024)


# --- the loop -------------------------------------------------------------------


async def test_dry_run_counts_without_embedding(reembed, monkeypatch):
    async def explode(*args: Any, **kwargs: Any):
        raise AssertionError("--dry-run must not call the provider")

    monkeypatch.setattr("memgraphrag.llm.openai_compatible.openai_embed", explode)
    conn = FakeConn([{"id": f"chunk-{i}", "content": "t", "metadata": {}} for i in range(7)])
    counters = await reembed.reembed_namespace(
        FakePool(conn),
        "ws",
        "chunks",
        model="m",
        dim=4,
        batch_size=64,
        limit=0,
        shadow_mode=False,
        dry_run=True,
    )
    assert counters == {"total": 7, "done": 0, "skipped": 0}
    assert conn.written == []


async def test_in_place_updates_every_row_across_pages(reembed, monkeypatch):
    seen: list[list[str]] = []

    async def fake_embed(texts, model=None, embedding_dim=None, **kwargs):
        seen.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("memgraphrag.llm.openai_compatible.openai_embed", fake_embed)
    rows = [
        {"id": f"fact-{i}", "content": "c", "metadata": {"triple": [f"a{i}", "r", "b"]}}
        for i in range(5)
    ]
    conn = FakeConn(rows)
    counters = await reembed.reembed_namespace(
        FakePool(conn),
        "ws",
        "facts",
        model="m",
        dim=4,
        batch_size=2,
        limit=0,
        shadow_mode=False,
        dry_run=False,
    )
    assert counters["done"] == 5
    # Paged 2 + 2 + 1, and every batch embedded the tab-joined triple.
    assert [len(batch) for batch in seen] == [2, 2, 1]
    assert seen[0] == ["a0\tr\tb", "a1\tr\tb"]
    updated = [payload for sql, payload in conn.written if "UPDATE" in sql]
    assert sum(len(p) for p in updated) == 5


async def test_skipped_rows_are_counted_not_embedded(reembed, monkeypatch):
    async def fake_embed(texts, model=None, embedding_dim=None, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("memgraphrag.llm.openai_compatible.openai_embed", fake_embed)
    rows = [
        {"id": "fact-1", "content": "c", "metadata": {"triple": ["a", "r", "b"]}},
        {"id": "fact-2", "content": "c", "metadata": {}},
    ]
    conn = FakeConn(rows)
    counters = await reembed.reembed_namespace(
        FakePool(conn),
        "ws",
        "facts",
        model="m",
        dim=4,
        batch_size=64,
        limit=0,
        shadow_mode=False,
        dry_run=False,
    )
    assert counters["done"] == 1
    assert counters["skipped"] == 1


async def test_limit_stops_early_for_smoke_runs(reembed, monkeypatch):
    async def fake_embed(texts, model=None, embedding_dim=None, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr("memgraphrag.llm.openai_compatible.openai_embed", fake_embed)
    conn = FakeConn([{"id": f"chunk-{i:03d}", "content": "t", "metadata": {}} for i in range(50)])
    counters = await reembed.reembed_namespace(
        FakePool(conn),
        "ws",
        "chunks",
        model="m",
        dim=4,
        batch_size=64,
        limit=3,
        shadow_mode=False,
        dry_run=False,
    )
    assert counters["done"] == 3


# --- CLI ------------------------------------------------------------------------


def test_a_mode_must_be_chosen(reembed):
    args = reembed.parse_args(["--workspace", "ws"])
    assert not (args.shadow or args.in_place or args.promote or args.dry_run)


def test_modes_are_mutually_exclusive(reembed):
    with pytest.raises(SystemExit):
        reembed.parse_args(["--shadow", "--in-place"])
