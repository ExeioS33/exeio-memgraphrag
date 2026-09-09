"""Accounts: sign-up, identity, and the admin's approval queue.

Sign-up is the one route here that anyone can call, which makes it the most
attackable endpoint on the server after ``/login``. It borrows ``/login``'s per-IP
limiter, and it answers the same way whether or not the address already has an
account: "this email exists" is precisely what an enumeration attack is after.

Approval is the whole point of the model. Every account after the first is born
``pending`` and can do nothing until an admin promotes it, so signing up grants no
access by itself — an instance left reachable is not thereby left open.
"""

from __future__ import annotations

import logging
from typing import Any

from memgraphrag.api.auth import hash_password
from memgraphrag.chat.users import EmailTaken, normalize_email
from memgraphrag.utils.step_log import main_step

logger = logging.getLogger("memgraphrag.api.auth_router")

try:
    from fastapi import APIRouter, Depends, HTTPException, Request, status
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    status = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]  # noqa: E731

MIN_PASSWORD_LENGTH = 8
STORE_UNCONFIGURED = (
    "Accounts are not enabled on this server. Set AUTH_SIGNUP_ENABLED=true and "
    "APP_DATABASE_URL to turn them on."
)
DEFAULT_SECRET_REFUSAL = (
    "Refusing to create accounts while TOKEN_SECRET is unset: tokens would be signed "
    "with a secret published in the repository and could be forged by anyone."
)


class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    name: str = Field(default="", max_length=120)
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=1024)


class ResetPasswordRequest(BaseModel):
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=1024)


def _require_store(request: Any) -> Any:
    store = getattr(request.app.state, "user_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=STORE_UNCONFIGURED
        )
    return store


def create_auth_router() -> Any:
    if APIRouter is None:  # pragma: no cover
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import current_user, require_admin
    from memgraphrag.api.rate_limit import client_key

    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/signup", status_code=status.HTTP_202_ACCEPTED)
    async def signup(request: Request, body: SignupRequest):
        store = _require_store(request)
        handler = request.app.state.auth_handler
        if getattr(handler, "uses_default_secret", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=DEFAULT_SECRET_REFUSAL
            )

        limiter = request.app.state.login_limiter
        key = "signup:" + client_key(request)
        retry_after = limiter.check(key)
        if retry_after is not None:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many sign-up attempts. Try again later.",
                headers={"Retry-After": str(max(1, int(retry_after) + 1))},
            )

        email = normalize_email(body.email)
        if "@" not in email:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid email"
            )

        # One response for both outcomes. An existing address gets exactly what a
        # new one gets, and the admin queue is where the difference is sorted out.
        pending_reply = {
            "status": "pending",
            "message": "Account created. An administrator has to approve it before you can sign in.",
        }
        try:
            user = await store.create(email, body.name, hash_password(body.password))
        except EmailTaken:
            return pending_reply

        main_step(logger, "api.auth.signup", user=user.id, role=user.role)
        if user.role != "admin":
            return pending_reply

        # The first account. It inherits every thread conversed before accounts
        # existed; those belonged to `guest` and would otherwise be nobody's.
        chat_store = getattr(request.app.state, "chat_store", None)
        moved = 0
        if chat_store is not None and hasattr(chat_store, "reassign_owner"):
            moved = await chat_store.reassign_owner("guest", user.id)
        main_step(logger, "api.auth.first_admin", user=user.id, threads_adopted=moved)
        return {
            "status": "admin",
            "message": "First account created as administrator. You can sign in now.",
            "threads_adopted": moved,
        }

    @router.get("/me")
    async def me(info: dict[str, Any] = Depends(current_user)):
        return {
            "id": info.get("sub"),
            "name": info.get("name") or info.get("username"),
            "email": info.get("email"),
            "role": info.get("role"),
            "auth_source": (info.get("metadata") or {}).get("auth_source", "env"),
        }

    @router.get("/users")
    async def list_users(request: Request, _admin: dict[str, Any] = Depends(require_admin)):
        store = _require_store(request)
        users = await store.list_users()
        # `active` lives in the auth table by design; fetch it per user rather than
        # widening AppUser to carry it, so the profile object stays credential-free.
        payload = []
        for user in users:
            record = await store.get_auth(user.email)
            entry = user.to_dict()
            entry["active"] = bool(record.active) if record else False
            payload.append(entry)
        return {"users": payload, "total": len(payload)}

    @router.post("/users/{user_id}/approve")
    async def approve(
        request: Request, user_id: str, _admin: dict[str, Any] = Depends(require_admin)
    ):
        store = _require_store(request)
        user = await store.set_role(user_id, "user")
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such user")
        main_step(logger, "api.auth.approve", user=user_id, by=_admin.get("sub"))
        return user.to_dict()

    @router.post("/users/{user_id}/deactivate")
    async def deactivate(
        request: Request, user_id: str, admin: dict[str, Any] = Depends(require_admin)
    ):
        if user_id == admin.get("sub"):
            # The last admin locking themselves out is not a state anyone can recover
            # from through the UI.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot deactivate yourself"
            )
        store = _require_store(request)
        user = await store.set_active(user_id, False)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such user")
        main_step(logger, "api.auth.deactivate", user=user_id, by=admin.get("sub"))
        return user.to_dict()

    @router.post("/users/{user_id}/reset-password")
    async def reset_password(
        request: Request,
        user_id: str,
        body: ResetPasswordRequest,
        admin: dict[str, Any] = Depends(require_admin),
    ):
        store = _require_store(request)
        changed = await store.set_password(user_id, hash_password(body.password))
        if not changed:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such user")
        main_step(logger, "api.auth.reset_password", user=user_id, by=admin.get("sub"))
        return {"status": "ok", "user_id": user_id}

    return router
