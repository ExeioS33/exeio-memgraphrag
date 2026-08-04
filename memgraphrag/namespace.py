"""Storage namespace constants for MemGraphRAG.

Adapted from LightRAG ``lightrag/namespace.py`` with MemGraphRAG-native
namespaces (KV_MEMORY, KV_OPENIE, VECTOR_FACTS, GRAPH_MEMORY). The
``make_namespace`` helper follows the workspace:base pattern from
LightRAG ``lightrag/kg/shared_storage.py``.
"""

from __future__ import annotations


class NameSpace:
    """Logical storage namespace identifiers (never change after data is written)."""

    KV_MEMORY = "memory"
    KV_LLM_CACHE = "llm_response_cache"
    KV_OPENIE = "openie"
    KV_TEXT_CHUNKS = "text_chunks"
    KV_FULL_DOCS = "full_docs"

    VECTOR_CHUNKS = "chunks"
    VECTOR_ENTITIES = "entities"
    VECTOR_FACTS = "facts"
    VECTOR_SCHEMAS = "schemas"

    GRAPH_MEMORY = "memory_graph"

    DOC_STATUS = "doc_status"


def make_namespace(workspace: str, base: str) -> str:
    """Compose a workspace-qualified namespace string.

    Empty workspace returns ``base`` unchanged; otherwise ``"{workspace}:{base}"``.
    """
    if workspace:
        return f"{workspace}:{base}"
    return base
