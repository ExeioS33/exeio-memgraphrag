"""JSON-file-backed key-value storage.

Adapted from LightRAG ``lightrag/kg/json_kv_impl.py`` — simplified to a
single-process in-memory dict flushed to
``working_dir/workspace/namespace.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from memgraphrag.base import BaseKVStorage
from memgraphrag.exceptions import CorruptKVFileError
from memgraphrag.utils.env import get_env_value

logger = logging.getLogger(__name__)

# Re-exported: the exception is raised here but lives next to the other storage
# errors, so a caller can catch it without importing a backend module.
__all__ = ["CorruptKVFileError", "JsonKVStorage"]


def _workspace_dir(working_dir: str, workspace: str) -> str:
    if workspace:
        return os.path.join(working_dir, workspace)
    return working_dir


def _load_json(path: str) -> dict[str, Any]:
    """Load a KV file, distinguishing "absent" (``{}``) from "unreadable" (raises).

    Swallowing a decode error and returning ``{}`` was silently destructive: the very
    next ``upsert`` rewrote the file with only the new record, so a truncated
    ``doc_status.json`` cost the whole document index after a mere warning.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise CorruptKVFileError(f"Cannot read KV file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CorruptKVFileError(
            f"KV file {path} holds {type(data).__name__}, expected a JSON object"
        )
    return data


def _quarantine(path: str) -> str:
    """Move an unreadable KV file aside so a fresh start never overwrites it."""
    target = f"{path}.corrupt-{int(time.time())}"
    suffix = 1
    while os.path.exists(target):
        target = f"{path}.corrupt-{int(time.time())}-{suffix}"
        suffix += 1
    os.replace(path, target)
    return target


def _write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass
class JsonKVStorage(BaseKVStorage):
    """File-backed KV store under ``working_dir/workspace/namespace.json``."""

    _data: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _file_name: str = field(default="", init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _dirty: bool = field(default=False, init=False, repr=False)
    _batch_depth: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        working_dir = self.global_config.get("working_dir", "./data/rag_storage")
        workspace_dir = _workspace_dir(working_dir, self.workspace or "")
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"{self.namespace}.json")
        self._data = {}
        self._dirty = False

    async def initialize(self) -> None:
        async with self._lock:
            try:
                self._data = _load_json(self._file_name)
            except CorruptKVFileError as exc:
                # Refusing to start beats losing the index: an empty in-memory dict is
                # flushed over the damaged file on the first upsert.
                if not get_env_value("MEMGRAPHRAG_KV_QUARANTINE_CORRUPT", False, bool):
                    raise
                moved = _quarantine(self._file_name)
                logger.warning(
                    "%s is unreadable (%s); moved to %s and starting empty",
                    self._file_name,
                    exc,
                    moved,
                )
                self._data = {}
            self._dirty = False
            logger.debug(
                "JsonKVStorage loaded %s (%d keys)",
                self._file_name,
                len(self._data),
            )

    async def finalize(self) -> None:
        self._batch_depth = 0
        await self._flush(force=True)

    @asynccontextmanager
    async def batch(self):
        """Defer the JSON rewrite until the outermost batch exits.

        Every ``upsert`` rewrites the whole file, so ingesting N chunks one call at a
        time costs O(N^2) bytes written on the event loop. Callers writing many
        records should wrap the loop::

            async with storage.batch():
                ...  # many upserts, one write at the end
        """
        self._batch_depth += 1
        try:
            yield self
        finally:
            self._batch_depth -= 1
            if self._batch_depth <= 0:
                self._batch_depth = 0
                await self._flush()

    async def _flush(self, *, force: bool = False) -> None:
        # Inside a batch, only the dirty flag matters; the outermost exit writes once.
        if self._batch_depth > 0 and not force:
            return
        async with self._lock:
            if not self._dirty:
                return
            # Serialising and writing is blocking IO; keep it off the event loop. The
            # lock is held, so no coroutine can mutate _data while it is dumped.
            await asyncio.to_thread(_write_json, self._file_name, self._data)
            self._dirty = False

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._lock:
            value = self._data.get(id)
            return dict(value) if value is not None else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._lock:
            results: list[dict[str, Any]] = []
            for doc_id in ids:
                value = self._data.get(doc_id)
                results.append(dict(value) if value is not None else None)  # type: ignore[arg-type]
            return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._lock:
            return set(keys) - set(self._data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        async with self._lock:
            for key, value in data.items():
                record = dict(value)
                record["_id"] = key
                self._data[key] = record
            self._dirty = True
        await self._flush()

    async def delete(self, ids: list[str]) -> None:
        async with self._lock:
            deleted = False
            for doc_id in ids:
                if self._data.pop(doc_id, None) is not None:
                    deleted = True
            if deleted:
                self._dirty = True
        await self._flush()

    async def get_all(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    async def drop(self) -> None:
        async with self._lock:
            self._data = {}
            self._dirty = True
        await self._flush()
