"""Storage ABCs and query parameters for MemGraphRAG.

Adapted from LightRAG ``lightrag/base.py`` (StorageNameSpace, BaseKVStorage,
BaseVectorStorage, BaseGraphStorage, DocStatus, DocStatusStorage, QueryParam).
Query fields and modes are MemGraphRAG-native (PPR / naive / context / bypass).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .constants import (
    DAMPING,
    FACT_SIMILARITY_THRESHOLD,
    LINKING_TOP_K,
    PASSAGE_NODE_WEIGHT,
    SCHEMA_NODE_WEIGHT,
    SCHEMA_TOP_K,
    SKIP_FACT_RERANK,
    TOP_K,
)
from .utils.env import get_env_value


@dataclass
class QueryParam:
    """Configuration for a MemGraphRAG query."""

    mode: Literal["ppr", "naive", "context", "bypass"] = "ppr"
    """Retrieval mode: PPR+QA, dense passages, context-only, or direct LLM."""

    only_need_context: bool = False
    """If True, return retrieved context without generating an answer."""

    structured_output: bool = True
    """If True, ask the LLM for JSON (answer/thought/citations/confidence)."""

    stream: bool = False
    """If True, stream the LLM response."""

    top_k: int = field(default_factory=lambda: get_env_value("TOP_K", TOP_K, int))
    """Number of passages / results to return."""

    linking_top_k: int = field(
        default_factory=lambda: get_env_value("LINKING_TOP_K", LINKING_TOP_K, int)
    )
    """Number of linked nodes at each retrieval step."""

    passage_node_weight: float = field(
        default_factory=lambda: get_env_value(
            "PASSAGE_NODE_WEIGHT", PASSAGE_NODE_WEIGHT, float
        )
    )
    """Multiplicative weight for passage nodes in PPR."""

    damping: float = field(
        default_factory=lambda: get_env_value("DAMPING", DAMPING, float)
    )
    """Damping factor for Personalized PageRank."""

    fact_similarity_threshold: float = field(
        default_factory=lambda: get_env_value(
            "FACT_SIMILARITY_THRESHOLD", FACT_SIMILARITY_THRESHOLD, float
        )
    )
    """Minimum fact similarity when skip_fact_rerank is enabled."""

    skip_fact_rerank: bool = field(
        default_factory=lambda: get_env_value(
            "SKIP_FACT_RERANK", SKIP_FACT_RERANK, bool
        )
    )
    """If True, skip fact reranking and filter by similarity threshold."""

    schema_top_k: int = field(
        default_factory=lambda: get_env_value("SCHEMA_TOP_K", SCHEMA_TOP_K, int)
    )
    """Number of ontology schemas to link from the query embedding."""

    schema_node_weight: float = field(
        default_factory=lambda: get_env_value(
            "SCHEMA_NODE_WEIGHT", SCHEMA_NODE_WEIGHT, float
        )
    )
    """Multiplicative weight for schema-expanded seeds in PPR."""

    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Past turns: [{"role": "user"|"assistant", "content": "..."}]."""

    user_prompt: str | None = None
    """Optional extra instruction injected into the QA user prompt."""


@dataclass
class StorageNameSpace(ABC):
    """Base namespace-scoped storage handle."""

    workspace: str
    namespace: str
    global_config: dict[str, Any]
    embedding_func: Any | None = None

    @abstractmethod
    async def initialize(self) -> None:
        """Open connections / load on-disk state."""

    @abstractmethod
    async def finalize(self) -> None:
        """Flush and release resources."""


@dataclass
class BaseKVStorage(StorageNameSpace, ABC):
    """Key-value storage ABC."""

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Return the record for ``id``, or ``None`` if missing."""

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Return records for the given ids (missing entries omitted or None)."""

    @abstractmethod
    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return the subset of ``keys`` that do not yet exist."""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update records keyed by id."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete records by id."""

    async def get_all(self) -> dict[str, dict[str, Any]]:
        """Optional: return all records. Backends may leave this unimplemented."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_all()"
        )

    async def drop(self) -> None:
        """Optional: wipe all records in this namespace."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement drop()"
        )


@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    """Vector similarity storage ABC."""

    @abstractmethod
    async def query(
        self, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` nearest neighbours for ``query_embedding``."""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert vectors.

        Each value must include at least ``content`` and ``embedding``.
        """

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by id."""

    async def drop(self) -> None:
        """Optional: wipe all vectors in this namespace."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement drop()"
        )


@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    """Typed memory-graph storage ABC."""

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Return whether ``node_id`` exists."""

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Return whether an edge exists between the two nodes."""

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        """Insert or update a node and its properties."""

    @abstractmethod
    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict[str, Any],
    ) -> None:
        """Insert or update an edge and its properties."""

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return node properties, or ``None`` if missing."""

    @abstractmethod
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        """Return edge properties, or ``None`` if missing."""

    @abstractmethod
    async def get_all_nodes(self) -> list[dict[str, Any]]:
        """Return all nodes (each as a property dict)."""

    @abstractmethod
    async def get_all_edges(self) -> list[dict[str, Any]]:
        """Return all edges (each as a property dict)."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all nodes and edges."""

    async def drop(self) -> None:
        """Wipe the graph namespace (default: :meth:`clear`)."""
        await self.clear()

    async def node_degree(self, node_id: str) -> int:
        """Optional: return the degree of ``node_id``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement node_degree()"
        )


class DocStatus(str, Enum):
    """Document processing lifecycle.

    Pipeline order: PENDING → PARSING → PROCESSING → PROCESSED | FAILED.
    """

    PENDING = "pending"
    PARSING = "parsing"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """Document-status KV storage with status filtering."""

    @abstractmethod
    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, dict[str, Any]]:
        """Return documents whose status is in ``statuses``, keyed by doc id."""
