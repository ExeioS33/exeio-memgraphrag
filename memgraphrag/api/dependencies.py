"""FastAPI auth dependencies for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/utils_api.py`` (``get_combined_auth_dependency``,
``WHITELIST_PATHS``). Accepts JWT Bearer OR ``X-API-Key``.
"""

from __future__ import annotations

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


def _compile_whitelist(paths_csv: str) -> list[tuple[str, bool]]:
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


whitelist_patterns = _compile_whitelist(
    getattr(global_args, "whitelist_paths", "/health,/docs,/openapi.json,/api/*")
)


def path_is_whitelisted(path: str) -> bool:
    """Return True if ``path`` matches WHITELIST_PATHS."""
    for pattern, is_prefix in whitelist_patterns:
        if is_prefix:
            if path == pattern or path.startswith(pattern + "/"):
                return True
        elif path == pattern:
            return True
    return False


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
        path = request.url.path or "/"
        if path_is_whitelisted(path):
            return

        if token:
            try:
                token_info = auth_handler.validate_token(token)
                if not auth_configured and token_info.get("role") == "guest":
                    if not api_key_configured:
                        return
                    # API-key-only: guest JWT must not bypass the key check
                elif auth_configured and token_info.get("role") != "guest":
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

        if not auth_configured and not api_key_configured:
            return

        if (
            api_key_configured
            and api_key_header_value
            and api_key_header_value == api_key
        ):
            return

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
        )

    return combined_dependency
