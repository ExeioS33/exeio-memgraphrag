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
from dataclasses import dataclass, field
from typing import Any

from memgraphrag.base import BaseKVStorage

logger = logging.getLogger(__name__)


def _workspace_dir(working_dir: str, workspace: str) -> str:
    if workspace:
        return os.path.join(working_dir, workspace)
    return working_dir


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load JSON from %s: %s", path, exc)
        return {}


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

    def __post_init__(self) -> None:
        working_dir = self.global_config.get("working_dir", "./data/rag_storage")
        workspace_dir = _workspace_dir(working_dir, self.workspace or "")
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"{self.namespace}.json")
        self._data = {}
        self._dirty = False

    async def initialize(self) -> None:
        async with self._lock:
            self._data = _load_json(self._file_name)
            self._dirty = False
            logger.debug(
                "JsonKVStorage loaded %s (%d keys)",
                self._file_name,
                len(self._data),
            )

    async def finalize(self) -> None:
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._dirty:
                return
            _write_json(self._file_name, self._data)
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
