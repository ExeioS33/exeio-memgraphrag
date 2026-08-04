"""MemGraphRAG API routers package."""

from __future__ import annotations

__all__ = [
    "create_documents_router",
    "create_graphs_router",
    "create_ollama_router",
    "create_query_router",
]


def __getattr__(name: str):
    if name == "create_documents_router":
        from memgraphrag.api.routers.documents import create_documents_router

        return create_documents_router
    if name == "create_query_router":
        from memgraphrag.api.routers.query import create_query_router

        return create_query_router
    if name == "create_graphs_router":
        from memgraphrag.api.routers.graphs import create_graphs_router

        return create_graphs_router
    if name == "create_ollama_router":
        from memgraphrag.api.routers.ollama import create_ollama_router

        return create_ollama_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
