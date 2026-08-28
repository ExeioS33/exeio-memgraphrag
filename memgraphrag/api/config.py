"""API configuration for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/config.py`` — slim argparse + env loading into
``global_args``. Project-prefixed storage/API-key vars use ``MEMGRAPHRAG_``.

Unlike upstream, importing this module does not read ``.env``: entry points call
``load_env_file()`` explicitly so that importing the API package never mutates the
environment of an unrelated process (notably pytest).
"""

from __future__ import annotations

import argparse
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

DEFAULT_TOKEN_SECRET = "memgraphrag-jwt-default-secret-key!"
DEFAULT_WORKERS = 1
DEFAULT_DOTENV_PATH = ".env"


class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "IgraphStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"


#: Backends that keep their whole state in files under ``WORKING_DIR``. Their write
#: locks are ``asyncio`` locks living inside one interpreter, so two OS processes
#: pointed at the same directory interleave rewrites of the JSON / GraphML files.
FILE_BACKED_STORAGES = frozenset(
    {
        "JsonKVStorage",
        "NanoVectorDBStorage",
        "IgraphStorage",
        "JsonDocStatusStorage",
    }
)

#: ``args`` attribute name -> (project-prefixed env var, unprefixed fallback, default).
STORAGE_ENV_VARS: dict[str, tuple[str, str, str]] = {
    "kv_storage": (
        "MEMGRAPHRAG_KV_STORAGE",
        "KV_STORAGE",
        DefaultRAGStorageConfig.KV_STORAGE,
    ),
    "vector_storage": (
        "MEMGRAPHRAG_VECTOR_STORAGE",
        "VECTOR_STORAGE",
        DefaultRAGStorageConfig.VECTOR_STORAGE,
    ),
    "graph_storage": (
        "MEMGRAPHRAG_GRAPH_STORAGE",
        "GRAPH_STORAGE",
        DefaultRAGStorageConfig.GRAPH_STORAGE,
    ),
    "doc_status_storage": (
        "MEMGRAPHRAG_DOC_STATUS_STORAGE",
        "DOC_STATUS_STORAGE",
        DefaultRAGStorageConfig.DOC_STATUS_STORAGE,
    ),
}


def load_env_file(dotenv_path: str = DEFAULT_DOTENV_PATH, *, override: bool = False) -> bool:
    """Load ``dotenv_path`` into ``os.environ`` and rebuild :data:`global_args`.

    Importing this module used to call ``load_dotenv`` as a side effect, so merely
    importing anything under ``memgraphrag.api`` injected the developer's ``.env``
    — real provider API keys included — into the whole process. Under pytest that
    turned ``--run-integration`` into a false green: tests that should have skipped
    for lack of credentials found them and hit live endpoints. Entry points now ask
    for the file explicitly.

    Returns:
        ``True`` when python-dotenv found and read the file.
    """
    global global_args
    loaded = load_dotenv(dotenv_path=dotenv_path, override=override)
    global_args = parse_args([])
    return loaded


def resolve_storage_backends() -> dict[str, str]:
    """Resolve the four storage backend names from the environment."""
    return {
        arg: get_env_value(prefixed, get_env_value(fallback, default))
        for arg, (prefixed, fallback, default) in STORAGE_ENV_VARS.items()
    }


def file_backed_storages(args: Any | None = None) -> list[str]:
    """Return the selected backends that store their state as plain files."""
    backends = (
        {arg: getattr(args, arg, "") or "" for arg in STORAGE_ENV_VARS}
        if args is not None
        else resolve_storage_backends()
    )
    return sorted({name for name in backends.values() if name in FILE_BACKED_STORAGES})


def validate_worker_count(workers: int, args: Any | None = None) -> None:
    """Refuse ``workers > 1`` while a file-backed storage backend is selected.

    Nothing in the request path is multi-process safe: the ingest ``pipeline_lock``
    is an ``asyncio.Lock`` and ``memgraphrag.storage.shared`` is a plain in-process
    dict, so a second worker neither waits for the first nor sees its refresh
    signals. With file backends that means two processes rewriting the same
    ``kv_store_*.json`` / ``graph_*.graphml`` and losing each other's writes.

    Raises:
        ValueError: when the combination cannot be served safely.
    """
    if workers <= 1:
        return
    selected = file_backed_storages(args)
    if not selected:
        return
    raise ValueError(
        f"WORKERS={workers} is not supported with the file-backed storage "
        f"backend(s) {', '.join(selected)}. Their write locks are asyncio locks "
        "held inside a single process, so two workers sharing WORKING_DIR "
        "interleave writes and corrupt the JSON / GraphML files. Either run with "
        "WORKERS=1, or move to the shared-database backends "
        "(MEMGRAPHRAG_KV_STORAGE=PGKVStorage, "
        "MEMGRAPHRAG_VECTOR_STORAGE=PGVectorStorage, "
        "MEMGRAPHRAG_DOC_STATUS_STORAGE=PGDocStatusStorage, "
        "MEMGRAPHRAG_GRAPH_STORAGE=Neo4JStorage)."
    )


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
        help=(
            "Number of gunicorn worker processes. Must stay 1 unless every storage "
            "backend is a shared database (see validate_worker_count)"
        ),
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
    for arg_name, backend in resolve_storage_backends().items():
        setattr(args, arg_name, backend)

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
    args.passage_node_weight = get_env_value("PASSAGE_NODE_WEIGHT", PASSAGE_NODE_WEIGHT, float)
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
    args.login_window_seconds = get_env_value("LOGIN_WINDOW_SECONDS", LOGIN_WINDOW_SECONDS, float)

    # Upload limits
    args.max_upload_size = get_env_value("MAX_UPLOAD_SIZE", MAX_UPLOAD_SIZE, int)

    # CORS / whitelist / Ollama emulation
    args.cors_origins = get_env_value("CORS_ORIGINS", "*")
    # NOTE: "/api/*" is deliberately NOT whitelisted by default. The Ollama emulation
    # router is mounted on /api and its /api/chat and /api/generate routes invoke the
    # billed LLM (including the /bypass mode, which skips retrieval entirely).
    args.whitelist_paths = get_env_value("WHITELIST_PATHS", "/health,/docs,/openapi.json")
    args.ollama_model_name = get_env_value("OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME)
    args.ollama_model_tag = get_env_value("OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG)

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
