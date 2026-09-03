"""Bearer verification for the MCP surface, on top of the API's own auth.

The MCP SDK asks for a ``TokenVerifier``: one method, ``verify_token(token) ->
AccessToken | None``. Implementing it over ``AuthHandler`` and the configured API
key — rather than beside them — is the whole point. A second identity system would
mean a second place to revoke a credential, and the one nobody remembers is the one
that stays open.

Returning ``None`` is how the protocol says "no". The SDK turns that into a 401
carrying the ``WWW-Authenticate`` header a compliant client needs.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ApiTokenVerifier:
    """Accepts the same credentials the HTTP routes accept: a JWT, or the API key."""

    def __init__(self, auth_handler: Any | None = None, api_key: str | None = None) -> None:
        self._auth_handler = auth_handler
        self._api_key = (api_key or "").strip() or None

    async def verify_token(self, token: str) -> Any | None:
        from mcp.server.auth.provider import AccessToken

        candidate = (token or "").strip()
        if not candidate:
            return None

        if self._api_key and hmac.compare_digest(candidate, self._api_key):
            # Constant-time, like the HTTP path: a length-sensitive comparison on a
            # shared secret leaks it a byte at a time.
            return AccessToken(token=candidate, client_id="api-key", scopes=["read"])

        handler = self._auth_handler
        if handler is not None:
            try:
                payload = handler.validate_token(candidate)
            except Exception:
                # `validate_token` raises an HTTPException on a bad token; over MCP
                # that has to become a plain "no", not a leaked FastAPI error.
                return None
            subject = str((payload or {}).get("sub") or "user")
            return AccessToken(token=candidate, client_id=subject, scopes=["read"])

        return None
