"""API configuration for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/config.py`` — slim argparse + env loading into
``global_args``. Project-prefixed storage/API-key vars use ``MEMGRAPHRAG_``.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from memgraphrag.constants import (
    DAMPING,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    EMBEDDING_DIM,
    FACT_SIMILARITY_THRESHOLD,
    HOST,
    INPUT_DIR,
    LINKING_TOP_K,
    LOGIN_MAX_ATTEMPTS,
    LOGIN_WINDOW_SECONDS,
    MAX_ASYNC_LLM,
    MAX_UPLOAD_SIZE,
    PASSAGE_NODE_WEIGHT,
    PORT,
    PPR_ENGINE,
    SKIP_FACT_RERANK,
    TOP_K,
    WORKING_DIR,
)
from memgraphrag.utils.env import get_env_value

load_dotenv(dotenv_path=".env", override=False)

DEFAULT_TOKEN_SECRET = "memgraphrag-jwt-default-secret-key!"
DEFAULT_WORKERS = 1


class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "IgraphStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args, overlaying environment defaults."""
    parser = argparse.ArgumentParser(description="MemGraphRAG API Server")

    parser.add_argument(
        "--host",
        type=str,
        default=get_env_value("HOST", HOST),
        help="Server bind host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_value("PORT", PORT, int),
        help="Server bind port",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=get_env_value("WORKERS", DEFAULT_WORKERS, int),
        help="Number of worker processes (gunicorn)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=get_env_value("WORKING_DIR", WORKING_DIR),
        help="Working directory for RAG storage",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=get_env_value("INPUT_DIR", INPUT_DIR),
        help="Directory for uploaded / scanned documents",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=get_env_value("WORKSPACE", ""),
        help="Optional workspace namespace for storage isolation",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=get_env_value("MEMGRAPHRAG_API_KEY", None),
        help="API key (also via MEMGRAPHRAG_API_KEY)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=get_env_value("LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=get_env_value("SSL", False, bool),
        help="Enable SSL",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=get_env_value("SSL_CERTFILE", None),
        help="SSL certificate file",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=get_env_value("SSL_KEYFILE", None),
        help="SSL key file",
    )

    args = parser.parse_args(argv)

    # Storage backends (MEMGRAPHRAG_* preferred, unprefixed fallback)
    args.kv_storage = get_env_value(
        "MEMGRAPHRAG_KV_STORAGE",
        get_env_value("KV_STORAGE", DefaultRAGStorageConfig.KV_STORAGE),
    )
    args.vector_storage = get_env_value(
        "MEMGRAPHRAG_VECTOR_STORAGE",
        get_env_value("VECTOR_STORAGE", DefaultRAGStorageConfig.VECTOR_STORAGE),
    )
    args.graph_storage = get_env_value(
        "MEMGRAPHRAG_GRAPH_STORAGE",
        get_env_value("GRAPH_STORAGE", DefaultRAGStorageConfig.GRAPH_STORAGE),
    )
    args.doc_status_storage = get_env_value(
        "MEMGRAPHRAG_DOC_STATUS_STORAGE",
        get_env_value("DOC_STATUS_STORAGE", DefaultRAGStorageConfig.DOC_STATUS_STORAGE),
    )

    # LLM / embedding bindings
    args.llm_binding = get_env_value("LLM_BINDING", "openai")
    args.llm_binding_host = get_env_value("LLM_BINDING_HOST", None)
    args.llm_binding_api_key = get_env_value("LLM_BINDING_API_KEY", None)
    args.llm_model = get_env_value("LLM_MODEL", "gpt-4o-mini")
    args.embedding_binding = get_env_value("EMBEDDING_BINDING", "openai")
    args.embedding_binding_host = get_env_value("EMBEDDING_BINDING_HOST", None)
    args.embedding_binding_api_key = get_env_value("EMBEDDING_BINDING_API_KEY", "")
    args.embedding_model = get_env_value("EMBEDDING_MODEL", "text-embedding-3-small")
    args.embedding_dim = get_env_value("EMBEDDING_DIM", EMBEDDING_DIM, int)
    args.max_async_llm = get_env_value("MAX_ASYNC_LLM", MAX_ASYNC_LLM, int)

    # Query / PPR knobs
    args.top_k = get_env_value("TOP_K", TOP_K, int)
    args.linking_top_k = get_env_value("LINKING_TOP_K", LINKING_TOP_K, int)
    args.passage_node_weight = get_env_value(
        "PASSAGE_NODE_WEIGHT", PASSAGE_NODE_WEIGHT, float
    )
    args.damping = get_env_value("DAMPING", DAMPING, float)
    args.fact_similarity_threshold = get_env_value(
        "FACT_SIMILARITY_THRESHOLD", FACT_SIMILARITY_THRESHOLD, float
    )
    args.skip_fact_rerank = get_env_value("SKIP_FACT_RERANK", SKIP_FACT_RERANK, bool)
    args.ppr_engine = get_env_value("PPR_ENGINE", PPR_ENGINE)

    # Auth
    args.auth_accounts = get_env_value("AUTH_ACCOUNTS", "")
    args.token_secret = get_env_value("TOKEN_SECRET", None)
    args.token_expire_hours = get_env_value("TOKEN_EXPIRE_HOURS", 48, float)
    args.guest_token_expire_hours = get_env_value("GUEST_TOKEN_EXPIRE_HOURS", 24, float)
    args.jwt_algorithm = get_env_value("JWT_ALGORITHM", "HS256")
    # Fail-closed switch: when true, refuse to serve unauthenticated requests even if
    # neither AUTH_ACCOUNTS nor MEMGRAPHRAG_API_KEY resolved (e.g. a .env not found
    # because the process was started from another working directory).
    args.require_auth = get_env_value("REQUIRE_AUTH", False, bool)
    args.login_max_attempts = get_env_value("LOGIN_MAX_ATTEMPTS", LOGIN_MAX_ATTEMPTS, int)
    args.login_window_seconds = get_env_value(
        "LOGIN_WINDOW_SECONDS", LOGIN_WINDOW_SECONDS, float
    )

    # Upload limits
    args.max_upload_size = get_env_value("MAX_UPLOAD_SIZE", MAX_UPLOAD_SIZE, int)

    # CORS / whitelist / Ollama emulation
    args.cors_origins = get_env_value("CORS_ORIGINS", "*")
    # NOTE: "/api/*" is deliberately NOT whitelisted by default. The Ollama emulation
    # router is mounted on /api and its /api/chat and /api/generate routes invoke the
    # billed LLM (including the /bypass mode, which skips retrieval entirely).
    args.whitelist_paths = get_env_value(
        "WHITELIST_PATHS", "/health,/docs,/openapi.json"
    )
    args.ollama_model_name = get_env_value(
        "OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME
    )
    args.ollama_model_tag = get_env_value(
        "OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG
    )

    # Prefer explicit CLI --key over env when both set (argparse already applied env default)
    if not args.key:
        args.key = get_env_value("MEMGRAPHRAG_API_KEY", None)

    return args


def namespace_from_dict(data: dict[str, Any] | None = None, **kwargs: Any) -> SimpleNamespace:
    """Build a config namespace for tests / programmatic create_app."""
    base = vars(parse_args([]))
    if data:
        base.update(data)
    base.update(kwargs)
    return SimpleNamespace(**base)


# Module-level singleton used by auth / dependencies
global_args: argparse.Namespace = parse_args([])
