"""In-process shared state for retrieval refresh signaling.

Simplified from LightRAG ``lightrag/kg/shared_storage.py``: a single-process
dict + ``asyncio.Lock`` used by gunicorn workers within one process (or as
a best-effort in-memory signal before a multi-process bus is added).
"""

from __future__ import annotations

import asyncio
from typing import Dict

_lock = asyncio.Lock()
_refresh_flags: Dict[str, bool] = {}
_retrieval_versions: Dict[str, int] = {}


async def set_refresh_flag(workspace: str) -> None:
    """Mark ``workspace`` as needing a retrieval-state refresh."""
    key = workspace or ""
    async with _lock:
        _refresh_flags[key] = True


async def consume_refresh_flag(workspace: str) -> bool:
    """Return and clear the refresh flag for ``workspace``.

    Returns:
        ``True`` if a refresh was pending, else ``False``.
    """
    key = workspace or ""
    async with _lock:
        pending = _refresh_flags.get(key, False)
        _refresh_flags[key] = False
        return pending


async def get_retrieval_version(workspace: str) -> int:
    """Return the current retrieval-state version for ``workspace``."""
    key = workspace or ""
    async with _lock:
        return _retrieval_versions.get(key, 0)


async def bump_retrieval_version(workspace: str) -> int:
    """Increment and return the retrieval-state version for ``workspace``."""
    key = workspace or ""
    async with _lock:
        version = _retrieval_versions.get(key, 0) + 1
        _retrieval_versions[key] = version
        return version
