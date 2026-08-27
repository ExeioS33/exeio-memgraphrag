"""JSON-file-backed document-status storage.

Adapted from LightRAG ``lightrag/kg/json_doc_status_impl.py`` — extends
``JsonKVStorage`` with ``get_docs_by_statuses`` (MemGraphRAG naming).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memgraphrag.base import DocStatus, DocStatusStorage
from memgraphrag.storage.json_kv_impl import JsonKVStorage


@dataclass
class JsonDocStatusStorage(JsonKVStorage, DocStatusStorage):
    """Document-status KV store with status filtering."""

    async def get_docs_by_statuses(self, statuses: list[DocStatus]) -> dict[str, dict[str, Any]]:
        """Return documents whose ``status`` field is in ``statuses``."""
        wanted = {s.value if isinstance(s, DocStatus) else str(s) for s in statuses}
        async with self._lock:
            return {
                doc_id: dict(record)
                for doc_id, record in self._data.items()
                if str(record.get("status", "")) in wanted
            }
