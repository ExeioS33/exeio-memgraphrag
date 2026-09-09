"""Database accounts: sign-up, approval, isolation, and what only an attack reveals.

Every scenario the plan's verification list names is provoked here rather than
described, because most of them fail silently in the wild: a deactivated user whose
token still works, two first sign-ups that both become admin, a sign-up response
that says whether an email exists.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgraphrag.api.config import namespace_from_dict
from memgraphrag.api.server import create_app
from memgraphrag.chat.users import InMemoryUserStore

pytestmark = pytest.mark.offline

SECRET = "a-real-secret-not-the-published-default"
PASSWORD = "correct horse battery"


def _mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    return rag


def _client(**overrides) -> TestClient:
    args = namespace_from_dict({"auth_signup_enabled": True, "token_secret": SECRET, **overrides})
    return TestClient(create_app(args, testing=True, rag=_mock_rag()))


def _signup(client: TestClient, email: str, name: str = "Someone") -> dict:
    response = client.post(
        "/auth/signup", json={"email": email, "name": name, "password": PASSWORD}
    )
    assert response.status_code == 202, response.text
    return response.json()


def _login(client: TestClient, email: str, password: str = PASSWORD):
    return client.post("/login", data={"username": email, "password": password})


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# The approval model
# --------------------------------------------------------------------------- #


def test_the_first_account_is_admin_and_every_later_one_is_pending() -> None:
    with _client() as client:
        first = _signup(client, "Alice@Example.com", "Alice")
        assert first["status"] == "admin"

        second = _signup(client, "bob@example.com", "Bob")
        assert second["status"] == "pending"

        # Admin signs straight in; Bob is told why he cannot.
        assert _login(client, "alice@example.com").status_code == 200
        pending = _login(client, "bob@example.com")
        assert pending.status_code == 403
        assert "approval" in pending.json()["detail"]


def test_approval_from_the_admin_route_unlocks_the_account() -> None:
    with _client() as client:
        _signup(client, "alice@example.com")
        _signup(client, "bob@example.com")
        admin = _login(client, "alice@example.com").json()["access_token"]

        users = client.get("/auth/users", headers=_bearer(admin)).json()["users"]
        bob = next(u for u in users if u["email"] == "bob@example.com")
        assert bob["role"] == "pending"

        approved = client.post(f"/auth/users/{bob['id']}/approve", headers=_bearer(admin))
        assert approved.status_code == 200
        assert approved.json()["role"] == "user"

        token = _login(client, "bob@example.com").json()["access_token"]
        me = client.get("/auth/me", headers=_bearer(token)).json()
        assert me["role"] == "user"
        assert me["email"] == "bob@example.com"


def test_the_login_screen_appears_before_anyone_has_an_account() -> None:
    """A store with zero users still means auth is on: that is the moment the first
    person must be able to sign up rather than being waved through as guest."""
    with _client() as client:
        assert client.get("/health").json()["auth_mode"] == "enabled"
        assert client.get("/chat/threads").status_code in (401, 403)


# --------------------------------------------------------------------------- #
# Isolation and inheritance
# --------------------------------------------------------------------------- #


def test_threads_are_isolated_per_account() -> None:
    """`chat_thread.owner` was always there; this is the first time it has had two
    real owners to keep apart."""
    with _client() as client:
        _signup(client, "alice@example.com")
        _signup(client, "bob@example.com")
        alice = _login(client, "alice@example.com").json()["access_token"]
        admin_headers = _bearer(alice)
        bob_id = next(
            u["id"]
            for u in client.get("/auth/users", headers=admin_headers).json()["users"]
            if u["email"] == "bob@example.com"
        )
        client.post(f"/auth/users/{bob_id}/approve", headers=admin_headers)
        bob = _login(client, "bob@example.com").json()["access_token"]

        created = client.post("/chat/threads", json={"title": "Alice's"}, headers=_bearer(alice))
        assert created.status_code == 200
        thread_id = created.json()["id"]

        assert client.get("/chat/threads", headers=_bearer(bob)).json()["threads"] == []
        assert client.get(f"/chat/threads/{thread_id}", headers=_bearer(bob)).status_code == 404


def test_the_first_admin_inherits_guest_threads() -> None:
    """Everything conversed before accounts existed belongs to `guest`. The instant a
    user logs in those threads are nobody's — unless the first admin takes them."""
    with _client() as client:
        store = client.app.state.chat_store
        # The in-memory store is plain state with no I/O, so a throwaway loop is fine.
        asyncio.run(store.create_thread("guest", title="Old"))

        first = _signup(client, "alice@example.com")
        assert first["threads_adopted"] == 1

        alice = _login(client, "alice@example.com").json()["access_token"]
        titles = [
            t["title"]
            for t in client.get("/chat/threads", headers=_bearer(alice)).json()["threads"]
        ]
        assert titles == ["Old"]


# --------------------------------------------------------------------------- #
# What only an attack reveals
# --------------------------------------------------------------------------- #


def test_a_deactivated_account_with_a_valid_token_is_refused() -> None:
    """The token is stateless and valid for 48 h. Without the per-request reload,
    deactivating a user changes nothing until it expires."""
    with _client() as client:
        _signup(client, "alice@example.com")
        _signup(client, "bob@example.com")
        admin = _login(client, "alice@example.com").json()["access_token"]
        bob_id = next(
            u["id"]
            for u in client.get("/auth/users", headers=_bearer(admin)).json()["users"]
            if u["email"] == "bob@example.com"
        )
        client.post(f"/auth/users/{bob_id}/approve", headers=_bearer(admin))
        bob = _login(client, "bob@example.com").json()["access_token"]
        assert client.get("/chat/threads", headers=_bearer(bob)).status_code == 200

        client.post(f"/auth/users/{bob_id}/deactivate", headers=_bearer(admin))

        # Same token, seconds later.
        assert client.get("/chat/threads", headers=_bearer(bob)).status_code == 401
        assert client.get("/auth/me", headers=_bearer(bob)).status_code == 401
        # And the account itself cannot log back in.
        assert _login(client, "bob@example.com").status_code == 403


@pytest.mark.asyncio
async def test_the_mcp_verifier_applies_the_same_account_check() -> None:
    """If only the FastAPI dependency reloaded the account, MCP would be the back
    door: same handler, same token, must be the same answer."""
    pytest.importorskip("mcp")
    from memgraphrag.api.auth import AuthHandler, hash_password
    from memgraphrag.mcp.auth import ApiTokenVerifier

    handler = AuthHandler(namespace_from_dict({"token_secret": SECRET}))
    store = InMemoryUserStore()
    handler.user_store = store
    user = await store.create("alice@example.com", "Alice", hash_password(PASSWORD))
    token = handler.create_token(username=user.id, role="admin", metadata={"auth_source": "db"})
    verifier = ApiTokenVerifier(auth_handler=handler)

    assert await verifier.verify_token(token) is not None
    await store.set_active(user.id, False)
    assert await verifier.verify_token(token) is None


@pytest.mark.asyncio
async def test_two_concurrent_first_signups_make_exactly_one_admin() -> None:
    store = InMemoryUserStore()
    from memgraphrag.api.auth import hash_password

    hashed = hash_password(PASSWORD)
    a, b = await asyncio.gather(
        store.create("a@example.com", "A", hashed), store.create("b@example.com", "B", hashed)
    )
    assert sorted([a.role, b.role]) == ["admin", "pending"]


def test_signup_does_not_reveal_whether_an_email_exists() -> None:
    with _client() as client:
        _signup(client, "alice@example.com")
        fresh = _signup(client, "bob@example.com")
        again = _signup(client, "bob@example.com")
        assert again == fresh


def test_login_does_not_distinguish_unknown_from_wrong_password() -> None:
    with _client() as client:
        _signup(client, "alice@example.com")
        unknown = _login(client, "nobody@example.com")
        wrong = _login(client, "alice@example.com", "not the password")
        assert unknown.status_code == wrong.status_code == 401
        assert unknown.json() == wrong.json()


def test_signup_is_rate_limited_like_login() -> None:
    with _client(login_max_attempts=2, login_window_seconds=60) as client:
        for i in range(2):
            _signup(client, f"user{i}@example.com")
        third = client.post(
            "/auth/signup", json={"email": "user2@example.com", "password": PASSWORD}
        )
        assert third.status_code == 429
        assert "Retry-After" in third.headers


def test_signup_refuses_to_run_under_the_published_default_secret() -> None:
    """The first real account under a secret anyone can read is an account anyone
    can impersonate."""
    with _client(token_secret=None) as client:
        response = client.post(
            "/auth/signup", json={"email": "alice@example.com", "password": PASSWORD}
        )
        assert response.status_code == 503
        assert "TOKEN_SECRET" in response.json()["detail"]


def test_admin_routes_reject_a_plain_user_and_self_deactivation() -> None:
    with _client() as client:
        _signup(client, "alice@example.com")
        _signup(client, "bob@example.com")
        admin = _login(client, "alice@example.com").json()["access_token"]
        me = client.get("/auth/me", headers=_bearer(admin)).json()
        bob_id = next(
            u["id"]
            for u in client.get("/auth/users", headers=_bearer(admin)).json()["users"]
            if u["email"] == "bob@example.com"
        )
        client.post(f"/auth/users/{bob_id}/approve", headers=_bearer(admin))
        bob = _login(client, "bob@example.com").json()["access_token"]

        assert client.get("/auth/users", headers=_bearer(bob)).status_code == 403
        assert (
            client.post(f"/auth/users/{me['id']}/deactivate", headers=_bearer(admin)).status_code
            == 400
        )


def test_env_accounts_still_work_without_the_database() -> None:
    """The only path that needs no application database must keep working."""
    args = namespace_from_dict({"auth_accounts": "ops:secret-pw", "token_secret": SECRET})
    with TestClient(create_app(args, testing=True, rag=_mock_rag())) as client:
        assert client.get("/health").json()["auth_mode"] == "enabled"
        assert _login(client, "ops", "secret-pw").status_code == 200
        assert (
            client.post("/auth/signup", json={"email": "x@y.z", "password": PASSWORD}).status_code
            == 503
        )


def test_a_bootstrapped_admin_exists_before_anyone_signs_up() -> None:
    """On an exposed port the admin must not be whoever signs up first."""
    with _client(auth_bootstrap_admin="Root@Example.com:root-password") as client:
        # The admin is there, with the admin role, and owns the guest threads.
        root = _login(client, "root@example.com", "root-password")
        assert root.status_code == 200, root.text
        me = client.get("/auth/me", headers=_bearer(root.json()["access_token"])).json()
        assert me["role"] == "admin"

        # So the very first sign-up is already pending, not admin.
        assert _signup(client, "alice@example.com")["status"] == "pending"
        assert _login(client, "alice@example.com").status_code == 403


def test_bootstrap_admin_is_ignored_once_accounts_exist() -> None:
    args = namespace_from_dict(
        {
            "auth_signup_enabled": True,
            "token_secret": SECRET,
            "auth_bootstrap_admin": "a@b.c:12345678",
        }
    )
    app = create_app(args, testing=True, rag=_mock_rag())
    from memgraphrag.api.auth import hash_password

    # An account exists before the lifespan runs: the bootstrap must stand down.
    store = app.state.user_store
    asyncio.run(store.create("first@example.com", "First", hash_password(PASSWORD)))
    with TestClient(app) as client:
        assert _login(client, "a@b.c", "12345678").status_code == 401
        assert _login(client, "first@example.com").status_code == 200


def test_bootstrap_admin_refuses_a_malformed_value() -> None:
    with pytest.raises(RuntimeError, match="email:password"):
        with _client(auth_bootstrap_admin="no-colon-here"):
            pass
