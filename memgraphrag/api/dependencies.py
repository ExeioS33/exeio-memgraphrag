"""FastAPI auth dependencies for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/utils_api.py`` (``get_combined_auth_dependency``,
``WHITELIST_PATHS``). Accepts JWT Bearer OR ``X-API-Key``.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any, Callable, Optional

from memgraphrag.api.auth import auth_handler
from memgraphrag.api.config import global_args

logger = logging.getLogger("memgraphrag.api.dependencies")

try:
    from fastapi import HTTPException, Request, Security, status
    from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
except ImportError:  # pragma: no cover
    HTTPException = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    Security = None  # type: ignore[misc, assignment]
    status = None  # type: ignore[assignment]
    APIKeyHeader = None  # type: ignore[misc, assignment]
    OAuth2PasswordBearer = None  # type: ignore[misc, assignment]

auth_configured = bool(auth_handler.accounts)


def compile_whitelist(paths_csv: str) -> list[tuple[str, bool]]:
    """Compile WHITELIST_PATHS into (pattern, is_prefix) pairs."""
    patterns: list[tuple[str, bool]] = []
    for raw in (paths_csv or "").split(","):
        path = raw.strip()
        if not path:
            continue
        if path.endswith("/*"):
            patterns.append((path[:-2] or "/", True))
        else:
            patterns.append((path, False))
    return patterns


whitelist_patterns = compile_whitelist(
    getattr(global_args, "whitelist_paths", "/health,/docs,/openapi.json")
)


def _match_whitelist(path: str, patterns: list[tuple[str, bool]]) -> bool:
    for pattern, is_prefix in patterns:
        if is_prefix:
            if path == pattern or path.startswith(pattern + "/"):
                return True
        elif path == pattern:
            return True
    return False


def path_is_whitelisted(path: str) -> bool:
    """Return True if ``path`` matches the module-level WHITELIST_PATHS.

    Kept for backwards compatibility. Request handling resolves the whitelist from
    ``request.app.state`` instead, so that ``create_app(args)`` is honoured.
    """
    return _match_whitelist(path, whitelist_patterns)


def resolve_auth_context(request: Any) -> tuple[Any, list[tuple[str, bool]], bool]:
    """Return ``(handler, whitelist_patterns, auth_configured)`` for this request.

    The module-level ``auth_handler`` / ``whitelist_patterns`` are bound at import
    time from ``global_args``, so any configuration passed to ``create_app(args)``
    would otherwise be silently ignored — which used to let ``/login`` mint a valid
    guest token for any password. ``create_app`` now stores the per-app objects on
    ``app.state``; we prefer those and fall back to the module-level ones.
    """
    state = getattr(getattr(request, "app", None), "state", None)
    handler = getattr(state, "auth_handler", None) or auth_handler
    patterns = getattr(state, "whitelist_patterns", None)
    if patterns is None:
        patterns = whitelist_patterns
    return handler, patterns, bool(getattr(handler, "accounts", None))


def get_combined_auth_dependency(
    api_key: Optional[str] = None,
    api_key_header_name: str = "X-API-Key",
) -> Callable[..., Any]:
    """Build a FastAPI dependency: whitelist OR valid JWT Bearer OR matching API key.

    When neither AUTH_ACCOUNTS nor an API key is configured, all requests pass.
    """
    if Request is None or OAuth2PasswordBearer is None:
        raise RuntimeError(
            "fastapi is required for auth dependencies; install memgraphrag[api]"
        )

    api_key_configured = bool(api_key)
    oauth2_scheme = OAuth2PasswordBearer(
        tokenUrl="login", auto_error=False, description="OAuth2 Password Authentication"
    )
    api_key_header = None
    if api_key_configured:
        api_key_header = APIKeyHeader(
            name=api_key_header_name,
            auto_error=False,
            description="API Key Authentication",
        )

    async def combined_dependency(
        request: Request,
        token: Optional[str] = Security(oauth2_scheme),
        api_key_header_value: Optional[str] = (
            None if api_key_header is None else Security(api_key_header)
        ),
    ):
        handler, patterns, request_auth_configured = resolve_auth_context(request)

        path = request.url.path or "/"
        if _match_whitelist(path, patterns):
            return

        if token:
            try:
                token_info = handler.validate_token(token)
                if not request_auth_configured and token_info.get("role") == "guest":
                    if not api_key_configured:
                        return
                    # API-key-only: guest JWT must not bypass the key check
                elif request_auth_configured and token_info.get("role") != "guest":
                    return
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token. Please login again.",
                    )
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token. Please login again.",
                )

        if not request_auth_configured and not api_key_configured:
            # Fail-closed escape hatch: REQUIRE_AUTH=true means "never serve an
            # unauthenticated request", so a .env that failed to load (wrong working
            # directory) degrades into 403 rather than into an open server.
            state = getattr(getattr(request, "app", None), "state", None)
            if not getattr(getattr(state, "args", None), "require_auth", False):
                return
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authenticated",
            )

        if (
            api_key_configured
            and api_key_header_value
            # Constant-time comparison; `==` on the API key leaks its prefix length.
            and hmac.compare_digest(api_key_header_value, api_key or "")
        ):
            return

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
        )

    return combined_dependency
