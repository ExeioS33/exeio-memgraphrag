"""Behaviour of the in-memory chat store.

The Postgres store shares this interface and is exercised by the API tests through
the in-memory double; a live-database test would need the postgres-app service and
belongs behind --run-integration.
"""

from __future__ import annotations

import pytest

from memgraphrag.chat import InMemoryChatStore
from memgraphrag.chat.models import derive_title

pytestmark = pytest.mark.offline


async def test_thread_lifecycle() -> None:
    store = InMemoryChatStore()
    thread = await store.create_thread("alice", title="Sujet")
    assert thread.owner == "alice"

    fetched = await store.get_thread(thread.id, "alice")
    assert fetched is not None and fetched.title == "Sujet"

    renamed = await store.update_thread(thread.id, "alice", title="Autre")
    assert renamed is not None and renamed.title == "Autre"

    assert await store.delete_thread(thread.id, "alice") is True
    assert await store.get_thread(thread.id, "alice") is None


async def test_threads_are_owner_scoped() -> None:
    """An id is not a capability: guessing one must not reach another owner's thread."""
    store = InMemoryChatStore()
    thread = await store.create_thread("alice")

    assert await store.get_thread(thread.id, "bob") is None
    assert await store.update_thread(thread.id, "bob", title="pirate") is None
    assert await store.delete_thread(thread.id, "bob") is False
    assert await store.add_message(thread.id, "bob", role="user", content="hi") is None
    assert await store.list_messages(thread.id, "bob") is None

    threads, total = await store.list_threads("bob")
    assert threads == [] and total == 0


async def test_first_user_message_names_the_thread() -> None:
    store = InMemoryChatStore()
    thread = await store.create_thread("alice")
    assert thread.title == "New chat"

    await store.add_message(thread.id, "alice", role="user", content="  Quel est le budget ?  ")
    named = await store.get_thread(thread.id, "alice")
    assert named is not None and named.title == "Quel est le budget ?"

    # A later turn must not rename the thread again.
    await store.add_message(thread.id, "alice", role="user", content="Et ensuite ?")
    still = await store.get_thread(thread.id, "alice")
    assert still is not None and still.title == "Quel est le budget ?"


async def test_messages_round_trip_with_references() -> None:
    store = InMemoryChatStore()
    thread = await store.create_thread("alice")
    refs = [{"reference_id": "1", "file_path": "/corpus/a.pdf", "content": None}]

    await store.add_message(thread.id, "alice", role="user", content="Question")
    await store.add_message(thread.id, "alice", role="assistant", content="Réponse", refs=refs)

    messages = await store.list_messages(thread.id, "alice")
    assert messages is not None
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].to_dict()["references"] == refs


async def test_invalid_role_is_rejected() -> None:
    store = InMemoryChatStore()
    thread = await store.create_thread("alice")
    with pytest.raises(ValueError):
        await store.add_message(thread.id, "alice", role="system", content="nope")


async def test_list_threads_paginates_newest_first() -> None:
    store = InMemoryChatStore()
    created = [await store.create_thread("alice", title=f"t{i}") for i in range(5)]
    # Ordering is by updated_at then id, both descending; same-second creation makes
    # the id the tiebreaker, so compare as a set rather than assuming clock spread.
    page, total = await store.list_threads("alice", limit=2, offset=0)
    assert total == 5 and len(page) == 2
    rest, _ = await store.list_threads("alice", limit=10, offset=2)
    assert len(rest) == 3
    assert {t.id for t in page} | {t.id for t in rest} == {t.id for t in created}


def test_derive_title_collapses_and_truncates() -> None:
    assert derive_title("  a   b  ") == "a b"
    assert derive_title("") == "New chat"
    long = derive_title("x" * 400)
    assert long.endswith("…") and len(long) <= 121
