"""Storage registry for MemGraphRAG.

Adapted from LightRAG ``lightrag/kg/__init__.py`` with MemGraphRAG-native
backends (``IgraphStorage`` instead of NetworkX; doc-status method
``get_docs_by_statuses``).
"""

from __future__ import annotations

STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": [
            "JsonKVStorage",
            "PGKVStorage",
        ],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": [
            "IgraphStorage",
            "Neo4JStorage",
        ],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    "VECTOR_STORAGE": {
        "implementations": [
            "NanoVectorDBStorage",
            "PGVectorStorage",
        ],
        "required_methods": ["query", "upsert"],
    },
    "DOC_STATUS_STORAGE": {
        "implementations": [
            "JsonDocStatusStorage",
            "PGDocStatusStorage",
        ],
        "required_methods": ["get_docs_by_statuses"],
    },
}

STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    "JsonKVStorage": [],
    "PGKVStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "IgraphStorage": [],
    "Neo4JStorage": ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"],
    "NanoVectorDBStorage": [],
    "PGVectorStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "JsonDocStatusStorage": [],
    "PGDocStatusStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
}

STORAGES = {
    "JsonKVStorage": ".storage.json_kv_impl",
    "JsonDocStatusStorage": ".storage.json_doc_status_impl",
    "NanoVectorDBStorage": ".storage.nano_vector_db_impl",
    "IgraphStorage": ".storage.igraph_impl",
    "PGKVStorage": ".storage.postgres_impl",
    "PGVectorStorage": ".storage.postgres_impl",
    "PGDocStatusStorage": ".storage.postgres_impl",
    "Neo4JStorage": ".storage.neo4j_impl",
}


def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify that ``storage_name`` is registered for ``storage_type``.

    Args:
        storage_type: One of KV_STORAGE, GRAPH_STORAGE, VECTOR_STORAGE,
            DOC_STATUS_STORAGE.
        storage_name: Concrete class name (e.g. ``JsonKVStorage``).

    Raises:
        ValueError: If the type is unknown or the name is incompatible.
    """
    if storage_type not in STORAGE_IMPLEMENTATIONS:
        raise ValueError(f"Unknown storage type: {storage_type}")

    storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
    if storage_name not in storage_info["implementations"]:
        raise ValueError(
            f"Storage implementation '{storage_name}' is not compatible with "
            f"{storage_type}. Compatible implementations are: "
            f"{', '.join(storage_info['implementations'])}"
        )
