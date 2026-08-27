"""JWT / password authentication for MemGraphRAG API.

Adapted from LightRAG ``lightrag/api/auth.py`` — slim AuthHandler using python-jose
JWT and optional bcrypt with plaintext fallback for POC.

Importing this module reads no ``.env``; see ``config.load_env_file``.
"""

from __future__ import annotations

import hmac
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from memgraphrag.api.config import DEFAULT_TOKEN_SECRET, global_args

logger = logging.getLogger("memgraphrag.api.auth")

MAX_TOKEN_SUBJECT_LENGTH = 256
BCRYPT_PASSWORD_PREFIX = "{bcrypt}"

try:
    from jose import JWTError, jwt
except ImportError:  # pragma: no cover - optional until [api] extras installed
    JWTError = Exception  # type: ignore[misc, assignment]
    jwt = None  # type: ignore[assignment]

try:
    import bcrypt as _bcrypt
except ImportError:  # pragma: no cover
    _bcrypt = None


def _verify_password(plain: str, stored: str) -> bool:
    """Verify plaintext against stored password (bcrypt prefix or plain)."""
    if stored.startswith(BCRYPT_PASSWORD_PREFIX):
        hashed = stored[len(BCRYPT_PASSWORD_PREFIX) :].encode("utf-8")
        if _bcrypt is None:
            logger.warning("bcrypt not installed; cannot verify {bcrypt} password")
            return False
        try:
            return bool(_bcrypt.checkpw(plain.encode("utf-8"), hashed))
        except Exception as exc:
            logger.warning("bcrypt verification failed: %s", exc)
            return False
    # Constant-time comparison: a plain `==` on secrets leaks their prefix length
    # through response timing.
    return hmac.compare_digest(plain, stored)


class AuthHandler:
    """Create and validate JWT tokens; verify AUTH_ACCOUNTS passwords."""

    def __init__(self, args: Any | None = None) -> None:
        cfg = args or global_args
        auth_accounts = getattr(cfg, "auth_accounts", "") or ""
        api_key = os.getenv("MEMGRAPHRAG_API_KEY") or getattr(cfg, "key", None) or ""
        require_auth = bool(getattr(cfg, "require_auth", False))
        self.secret = getattr(cfg, "token_secret", None) or ""
        if not self.secret:
            # DEFAULT_TOKEN_SECRET is published in this repository, so tokens signed
            # with it can be forged by anyone.
            if auth_accounts:
                raise ValueError(
                    "TOKEN_SECRET must be explicitly set when AUTH_ACCOUNTS is configured."
                )
            self.secret = DEFAULT_TOKEN_SECRET
            if api_key or require_auth:
                # API-key-only mode stays usable: a forged token can only carry
                # role="guest", and guest tokens do not bypass the key check
                # (see dependencies.combined_dependency). Still worth flagging.
                logger.warning(
                    "TOKEN_SECRET not set; JWTs are signed with the public default "
                    "secret. Access is still gated by the API key, but /login tokens "
                    "are forgeable. Set TOKEN_SECRET."
                )
            else:
                logger.warning(
                    "TOKEN_SECRET not set and no authentication configured; using the "
                    "public default JWT secret. Do not expose this server."
                )
        algorithm = getattr(cfg, "jwt_algorithm", None) or "HS256"
        if algorithm.lower() == "none":
            raise ValueError("JWT_ALGORITHM 'none' is not permitted.")
        self.algorithm = algorithm
        self.expire_hours = float(getattr(cfg, "token_expire_hours", 48) or 48)
        self.guest_expire_hours = float(getattr(cfg, "guest_token_expire_hours", 24) or 24)
        self.accounts: dict[str, str] = {}
        if auth_accounts:
            for account in auth_accounts.split(","):
                account = account.strip()
                if not account:
                    continue
                try:
                    username, password = account.split(":", 1)
                except ValueError as exc:
                    raise ValueError(
                        "AUTH_ACCOUNTS must use comma-separated user:password pairs."
                    ) from exc
                if not username or not password:
                    raise ValueError("AUTH_ACCOUNTS must use comma-separated user:password pairs.")
                if len(username) > MAX_TOKEN_SUBJECT_LENGTH:
                    raise ValueError(
                        f"AUTH_ACCOUNTS usernames must be at most "
                        f"{MAX_TOKEN_SUBJECT_LENGTH} characters."
                    )
                self.accounts[username] = password

    def verify_password(self, username: str, plain_password: str) -> bool:
        stored = self.accounts.get(username)
        if stored is None:
            return False
        return _verify_password(plain_password, stored)

    def create_token(
        self,
        username: str,
        role: str = "user",
        custom_expire_hours: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if jwt is None:
            raise RuntimeError("python-jose is required for JWT auth; install memgraphrag[api]")
        if custom_expire_hours is None:
            expire_hours = self.guest_expire_hours if role == "guest" else self.expire_hours
        else:
            expire_hours = custom_expire_hours
        expire = datetime.now(timezone.utc) + timedelta(hours=expire_hours)
        payload = {
            "sub": username,
            "exp": expire,
            "role": role,
            "metadata": metadata or {},
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict[str, Any]:
        """Validate JWT; raise HTTPException-like dict errors via fastapi if available."""
        try:
            from fastapi import HTTPException, status
        except ImportError:  # pragma: no cover
            HTTPException = None  # type: ignore[misc, assignment]
            status = None  # type: ignore[assignment]

        def _unauthorized(detail: str) -> None:
            if HTTPException is not None and status is not None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
            raise ValueError(detail)

        if jwt is None:
            _unauthorized("JWT support not installed")

        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            username = payload.get("sub")
            if not isinstance(username, str) or not username:
                _unauthorized("Invalid token")
            if len(username) > MAX_TOKEN_SUBJECT_LENGTH:
                _unauthorized("Invalid token")
            expire_timestamp = payload.get("exp")
            if expire_timestamp is None:
                _unauthorized("Invalid token")
            try:
                expire_time = datetime.fromtimestamp(float(expire_timestamp), timezone.utc)
            except (OverflowError, OSError, ValueError, TypeError):
                _unauthorized("Invalid token")
            if datetime.now(timezone.utc) > expire_time:
                _unauthorized("Token expired")
            return {
                "username": username,
                "role": payload.get("role", "user"),
                "metadata": payload.get("metadata", {}),
                "exp": expire_time,
            }
        except JWTError:
            _unauthorized("Invalid token")
            raise  # pragma: no cover — unreachable


auth_handler = AuthHandler()
