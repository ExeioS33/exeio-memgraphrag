"""NanoVectorDB file-backed vector storage.

Adapted from LightRAG ``lightrag/kg/nano_vector_db_impl.py`` — simplified
to match MemGraphRAG ``BaseVectorStorage`` (caller supplies embeddings;
``query(query_embedding, top_k)``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from nano_vectordb import NanoVectorDB

from memgraphrag.base import BaseVectorStorage
from memgraphrag.constants import EMBEDDING_DIM

logger = logging.getLogger(__name__)


def _workspace_dir(working_dir: str, workspace: str) -> str:
    if workspace:
        return os.path.join(working_dir, workspace)
    return working_dir


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    """Vector store for chunks / entities / facts via ``nano-vectordb``."""

    _client: Any = field(default=None, init=False, repr=False)
    _file_name: str = field(default="", init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    cosine_better_than_threshold: float = field(default=0.0, init=False, repr=False)
    _embedding_dim: int = field(default=EMBEDDING_DIM, init=False, repr=False)

    def __post_init__(self) -> None:
        working_dir = self.global_config.get("working_dir", "./data/rag_storage")
        workspace_dir = _workspace_dir(working_dir, self.workspace or "")
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"vdb_{self.namespace}.json")

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {}) or {}
        self.cosine_better_than_threshold = float(
            kwargs.get("cosine_better_than_threshold", 0.0)
        )

        emb_dim = None
        if self.embedding_func is not None:
            emb_dim = getattr(self.embedding_func, "embedding_dim", None)
        if emb_dim is None:
            emb_dim = self.global_config.get("embedding_dim", EMBEDDING_DIM)
        self._embedding_dim = int(emb_dim)

        self._client = NanoVectorDB(
            self._embedding_dim,
            storage_file=self._file_name,
        )

    async def initialize(self) -> None:
        # NanoVectorDB loads from disk in __init__; nothing else required.
        logger.debug(
            "NanoVectorDBStorage ready namespace=%s file=%s",
            self.namespace,
            self._file_name,
        )

    async def finalize(self) -> None:
        async with self._lock:
            self._client.save()

    async def query(
        self, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        embedding = np.asarray(query_embedding, dtype=np.float32)
        async with self._lock:
            results = self._client.query(
                query=embedding,
                top_k=top_k,
                better_than_threshold=self.cosine_better_than_threshold,
            )
        # `__metrics__` is already a cosine similarity. Publish it as `score` per the
        # BaseVectorStorage contract; `distance` stays as a deprecated alias.
        return [
            {
                **{k: v for k, v in dp.items() if k not in ("__vector__", "vector")},
                "id": dp["__id__"],
                "score": float(dp.get("__metrics__") or 0.0),
                "distance": float(dp.get("__metrics__") or 0.0),
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert vectors; each value must include ``content`` and ``embedding``."""
        if not data:
            return
        now = int(time.time())
        list_data = []
        for doc_id, record in data.items():
            if "embedding" not in record:
                raise ValueError(
                    f"NanoVectorDBStorage.upsert requires 'embedding' for id={doc_id}"
                )
            vector = np.asarray(record["embedding"], dtype=np.float32)
            meta = {
                k: v
                for k, v in record.items()
                if k not in ("embedding", "__vector__", "vector")
            }
            list_data.append(
                {
                    "__id__": doc_id,
                    "__created_at__": now,
                    "__vector__": vector,
                    **meta,
                }
            )
        async with self._lock:
            self._client.upsert(datas=list_data)
            self._client.save()

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        async with self._lock:
            self._client.delete(ids)
            self._client.save()

    async def drop(self) -> None:
        async with self._lock:
            if os.path.exists(self._file_name):
                try:
                    os.remove(self._file_name)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove vector file %s: %s", self._file_name, exc
                    )
            self._client = NanoVectorDB(
                self._embedding_dim,
                storage_file=self._file_name,
            )
