"""Accounts for the web UI, kept in the application database next to the threads.

The shape follows Open WebUI's, which is the part of that project that transposes
without adaptation: a profile table that holds no credential, and an auth table that
holds nothing else. Every read that serves a page — ``get``, ``list_users``, a join on
``chat_thread.owner`` — goes through :class:`AppUser` and cannot carry a password
hash, because the object has no field for one.

Two implementations share one interface, as for the chat store: Postgres for real
use, in-memory for tests. There is no file-backed fallback for the same reason.

The one operation that has to be atomic is :meth:`BaseUserStore.create`. "The first
account becomes admin" is decided by counting then writing, and two simultaneous
first sign-ups must not both win. In Postgres the create runs inside a transaction
that takes a table lock; in memory the event loop serialises it for free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from memgraphrag.chat.models import new_id, now_ts
from memgraphrag.chat.store import ChatStoreUnavailable, load_schema_sql

logger = logging.getLogger("memgraphrag.chat.users")

USER_ROLES = ("admin", "user", "pending")


class EmailTaken(ValueError):
    """Raised by ``create`` when the address already has an account.

    Callers facing the network must not translate this into a distinct response:
    "this email exists" is exactly what an enumeration attack wants to hear.
    """


def normalize_email(email: str) -> str:
    """Case-fold and trim, so ``Alice@Example.com`` and ``alice@example.com`` are one
    account rather than two."""
    return (email or "").strip().lower()


@dataclass
class AppUser:
    """A profile. Note what is absent: any credential."""

    id: str
    email: str
    name: str
    role: str = "pending"
    created_at: int = field(default_factory=now_ts)
    updated_at: int = field(default_factory=now_ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class AuthRecord:
    """What a login needs and nothing more. Never serialised to a client."""

    user_id: str
    password_hash: str
    active: bool
    role: str


def _check_role(role: str) -> str:
    if role not in USER_ROLES:
        raise ValueError(f"role must be one of {USER_ROLES}, got {role!r}")
    return role


class BaseUserStore:
    """Interface shared by the Postgres and in-memory stores."""

    async def initialize(self) -> None:  # pragma: no cover - trivial
        return None

    async def close(self) -> None:  # pragma: no cover - trivial
        return None

    async def count(self) -> int:
        raise NotImplementedError

    async def create(self, email: str, name: str, password_hash: str) -> AppUser:
        """Create an account. The first one is ``admin``; every later one is
        ``pending`` until an admin approves it. Raises :class:`EmailTaken`."""
        raise NotImplementedError

    async def get(self, user_id: str) -> AppUser | None:
        raise NotImplementedError

    async def get_by_email(self, email: str) -> AppUser | None:
        raise NotImplementedError

    async def get_auth(self, email: str) -> AuthRecord | None:
        raise NotImplementedError

    async def list_users(self) -> list[AppUser]:
        raise NotImplementedError

    async def set_role(self, user_id: str, role: str) -> AppUser | None:
        raise NotImplementedError

    async def set_active(self, user_id: str, active: bool) -> AppUser | None:
        raise NotImplementedError

    async def set_password(self, user_id: str, password_hash: str) -> bool:
        raise NotImplementedError


class InMemoryUserStore(BaseUserStore):
    """Process-local store used by tests. Nothing survives a restart."""

    def __init__(self) -> None:
        self._users: dict[str, AppUser] = {}
        self._auth: dict[str, tuple[str, bool]] = {}  # user_id -> (hash, active)

    async def count(self) -> int:
        return len(self._users)

    async def create(self, email: str, name: str, password_hash: str) -> AppUser:
        address = normalize_email(email)
        # No await between the check and the insert: asyncio cannot interleave
        # another create here, which is the in-memory equivalent of the table lock.
        if any(u.email == address for u in self._users.values()):
            raise EmailTaken(address)
        role = "admin" if not self._users else "pending"
        user = AppUser(id=new_id(), email=address, name=name.strip() or address, role=role)
        self._users[user.id] = user
        self._auth[user.id] = (password_hash, True)
        return user

    async def get(self, user_id: str) -> AppUser | None:
        return self._users.get(user_id)

    async def get_by_email(self, email: str) -> AppUser | None:
        address = normalize_email(email)
        return next((u for u in self._users.values() if u.email == address), None)

    async def get_auth(self, email: str) -> AuthRecord | None:
        user = await self.get_by_email(email)
        if user is None:
            return None
        password_hash, active = self._auth[user.id]
        return AuthRecord(
            user_id=user.id, password_hash=password_hash, active=active, role=user.role
        )

    async def list_users(self) -> list[AppUser]:
        return sorted(self._users.values(), key=lambda u: (u.created_at, u.id))

    async def set_role(self, user_id: str, role: str) -> AppUser | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        user.role = _check_role(role)
        user.updated_at = now_ts()
        return user

    async def set_active(self, user_id: str, active: bool) -> AppUser | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        password_hash, _ = self._auth[user_id]
        self._auth[user_id] = (password_hash, bool(active))
        user.updated_at = now_ts()
        return user

    async def set_password(self, user_id: str, password_hash: str) -> bool:
        if user_id not in self._auth:
            return False
        _, active = self._auth[user_id]
        self._auth[user_id] = (password_hash, active)
        return True

    def is_active(self, user_id: str) -> bool:
        """Test helper; the network path reads ``active`` through ``get_auth``."""
        return self._auth.get(user_id, ("", False))[1]


class PostgresUserStore(BaseUserStore):
    """asyncpg-backed store against the dedicated application database.

    Owns a small pool of its own rather than borrowing the chat store's, so the two
    stores can be initialised and closed independently. Both apply ``schema.sql``,
    which is idempotent, so neither has to run first.
    """

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 4) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any | None = None

    async def initialize(self) -> None:
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - asyncpg ships with [api]
            raise ChatStoreUnavailable("asyncpg is required; install memgraphrag[api]") from exc
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=self._min_size, max_size=self._max_size
        )
        async with self._pool.acquire() as conn:
            await conn.execute(load_schema_sql())
        logger.info("User store ready (dedicated application database)")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise ChatStoreUnavailable("User store is not initialized")
        return self._pool

    @staticmethod
    def _user_from_row(row: Any) -> AppUser:
        return AppUser(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            role=row["role"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    _USER_COLUMNS = "id, email, name, role, created_at, updated_at"

    async def count(self) -> int:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            return int(await conn.fetchval("SELECT COUNT(*) FROM app_user") or 0)

    async def create(self, email: str, name: str, password_hash: str) -> AppUser:
        address = normalize_email(email)
        user = AppUser(id=new_id(), email=address, name=name.strip() or address)
        pool = self._require_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # SHARE ROW EXCLUSIVE conflicts with itself, so a second concurrent
                # create blocks here until this one commits — and then sees the
                # first admin already in place. Without the lock, two first sign-ups
                # each count zero rows and both become admin.
                await conn.execute("LOCK TABLE app_user IN SHARE ROW EXCLUSIVE MODE")
                taken = await conn.fetchval("SELECT 1 FROM app_user WHERE email = $1", address)
                if taken:
                    raise EmailTaken(address)
                total = await conn.fetchval("SELECT COUNT(*) FROM app_user")
                user.role = "admin" if int(total or 0) == 0 else "pending"
                await conn.execute(
                    f"""
                    INSERT INTO app_user ({self._USER_COLUMNS})
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    user.id,
                    user.email,
                    user.name,
                    user.role,
                    user.created_at,
                    user.updated_at,
                )
                await conn.execute(
                    "INSERT INTO app_auth (user_id, password_hash, active) VALUES ($1, $2, TRUE)",
                    user.id,
                    password_hash,
                )
        return user

    async def get(self, user_id: str) -> AppUser | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {self._USER_COLUMNS} FROM app_user WHERE id = $1", user_id
            )
        return None if row is None else self._user_from_row(row)

    async def get_by_email(self, email: str) -> AppUser | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {self._USER_COLUMNS} FROM app_user WHERE email = $1",
                normalize_email(email),
            )
        return None if row is None else self._user_from_row(row)

    async def get_auth(self, email: str) -> AuthRecord | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT u.id, u.role, a.password_hash, a.active
                FROM app_user u JOIN app_auth a ON a.user_id = u.id
                WHERE u.email = $1
                """,
                normalize_email(email),
            )
        if row is None:
            return None
        return AuthRecord(
            user_id=row["id"],
            password_hash=row["password_hash"],
            active=bool(row["active"]),
            role=row["role"],
        )

    async def list_users(self) -> list[AppUser]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {self._USER_COLUMNS} FROM app_user ORDER BY created_at, id"
            )
        return [self._user_from_row(r) for r in rows]

    async def set_role(self, user_id: str, role: str) -> AppUser | None:
        _check_role(role)
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE app_user SET role = $2, updated_at = $3
                WHERE id = $1
                RETURNING {self._USER_COLUMNS}
                """,
                user_id,
                role,
                now_ts(),
            )
        return None if row is None else self._user_from_row(row)

    async def set_active(self, user_id: str, active: bool) -> AppUser | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                changed = await conn.execute(
                    "UPDATE app_auth SET active = $2 WHERE user_id = $1", user_id, bool(active)
                )
                if changed == "UPDATE 0":
                    return None
                row = await conn.fetchrow(
                    f"""
                    UPDATE app_user SET updated_at = $2 WHERE id = $1
                    RETURNING {self._USER_COLUMNS}
                    """,
                    user_id,
                    now_ts(),
                )
        return None if row is None else self._user_from_row(row)

    async def set_password(self, user_id: str, password_hash: str) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            changed = await conn.execute(
                "UPDATE app_auth SET password_hash = $2 WHERE user_id = $1",
                user_id,
                password_hash,
            )
        return changed != "UPDATE 0"


def create_user_store(dsn: str | None) -> BaseUserStore | None:
    """Return a Postgres store for ``dsn``, or ``None`` when it is unset."""
    if not dsn:
        return None
    return PostgresUserStore(dsn)
