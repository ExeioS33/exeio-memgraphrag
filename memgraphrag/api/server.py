"""MemGraphRAG FastAPI server.

Adapted from LightRAG ``lightrag/api/lightrag_server.py`` — slim create_app + main.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Optional

from dotenv import load_dotenv

from memgraphrag import __version__ as core_version
from memgraphrag.api import __api_version__
from memgraphrag.api.config import global_args, parse_args
from memgraphrag.constants import DEFAULT_OLLAMA_MODEL_NAME, DEFAULT_OLLAMA_MODEL_TAG
from memgraphrag.core import MemGraphRAG
from memgraphrag.llm.openai_compatible import openai_complete, openai_embed

# Prefer the mounted/project .env over stale process env after compose recreates.
# Do not override process env (Compose / k8s inject storage bindings).
load_dotenv(dotenv_path=".env", override=False)

logger = logging.getLogger("memgraphrag.api.server")

try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import OAuth2PasswordRequestForm
except ImportError:  # pragma: no cover
    uvicorn = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    status = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[misc, assignment]
    OAuth2PasswordRequestForm = None  # type: ignore[misc, assignment]


def _build_rag(args: Any) -> MemGraphRAG:
    """Construct MemGraphRAG with openai-compatible LLM/embed bindings."""

    async def llm_model_func(prompt: str, **kwargs: Any) -> str:
        result = await openai_complete(prompt, model=getattr(args, "llm_model", None), **kwargs)
        return str(result)

    async def embedding_func(texts, **kwargs: Any):
        return await openai_embed(
            texts,
            model=getattr(args, "embedding_model", None),
            embedding_dim=getattr(args, "embedding_dim", None),
            **kwargs,
        )

    return MemGraphRAG(
        working_dir=getattr(args, "working_dir", "./data/rag_storage"),
        workspace=getattr(args, "workspace", "") or "",
        kv_storage=getattr(args, "kv_storage", "JsonKVStorage"),
        vector_storage=getattr(args, "vector_storage", "NanoVectorDBStorage"),
        graph_storage=getattr(args, "graph_storage", "IgraphStorage"),
        doc_status_storage=getattr(args, "doc_status_storage", "JsonDocStatusStorage"),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        embedding_dim=getattr(args, "embedding_dim", None),
        ppr_engine=getattr(args, "ppr_engine", "igraph"),
        top_k=getattr(args, "top_k", 10),
        linking_top_k=getattr(args, "linking_top_k", 50),
        passage_node_weight=getattr(args, "passage_node_weight", 0.05),
        damping=getattr(args, "damping", 0.5),
        fact_similarity_threshold=getattr(args, "fact_similarity_threshold", 0.6),
        skip_fact_rerank=getattr(args, "skip_fact_rerank", True),
        max_async_llm=getattr(args, "max_async_llm", 4),
    )


def create_app(
    args: Any | None = None,
    *,
    testing: bool = False,
    rag: Any | None = None,
) -> Any:
    """Create the FastAPI application.

    Args:
        args: Config namespace (defaults to ``global_args``).
        testing: If True, skip storage initialize / retrieval warm-up in lifespan.
        rag: Optional pre-built MemGraphRAG (or mock) for tests.
    """
    if FastAPI is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.auth import auth_handler
    from memgraphrag.api.dependencies import get_combined_auth_dependency
    from memgraphrag.api.routers.documents import create_documents_router
    from memgraphrag.api.routers.graphs import create_graphs_router
    from memgraphrag.api.routers.ollama import create_ollama_router
    from memgraphrag.api.routers.query import create_query_router

    cfg = args or global_args
    api_key = (
        os.getenv("MEMGRAPHRAG_API_KEY")
        or getattr(cfg, "key", None)
        or None
    )
    combined_auth = get_combined_auth_dependency(
        api_key, api_key_header_name="X-API-Key"
    )

    engine = rag if rag is not None else _build_rag(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.background_tasks = set()
        try:
            if not testing:
                await app.state.rag.initialize_storages()
                try:
                    await app.state.rag.prepare_retrieval()
                except Exception as exc:
                    logger.warning("Retrieval warm-up skipped: %s", exc)
                logger.info("MemGraphRAG server ready")
            yield
        finally:
            if not testing and hasattr(app.state.rag, "finalize_storages"):
                try:
                    await app.state.rag.finalize_storages()
                except Exception as exc:
                    logger.warning("finalize_storages: %s", exc)

    app = FastAPI(
        title="MemGraphRAG API",
        description="Memory-based GraphRAG API (Ollama-compatible /api)",
        version=__api_version__,
        lifespan=lifespan,
    )

    cors_origins = getattr(cfg, "cors_origins", "*") or "*"
    origins = (
        ["*"]
        if cors_origins.strip() == "*"
        else [o.strip() for o in cors_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.rag = engine
    app.state.args = cfg
    app.state.input_dir = getattr(cfg, "input_dir", "./data/inputs")
    app.state.testing = testing
    app.state.pipeline_lock = asyncio.Lock()
    app.state.pipeline_busy = False
    os.makedirs(app.state.input_dir, exist_ok=True)

    app.include_router(create_documents_router(api_key))
    app.include_router(create_query_router(api_key))
    app.include_router(create_graphs_router(api_key))
    app.include_router(
        create_ollama_router(
            api_key=api_key,
            model_name=getattr(cfg, "ollama_model_name", DEFAULT_OLLAMA_MODEL_NAME),
            model_tag=getattr(cfg, "ollama_model_tag", DEFAULT_OLLAMA_MODEL_TAG),
        )
    )

    @app.get("/health", dependencies=[Depends(combined_auth)])
    async def health(request: Request):
        return {
            "status": "healthy",
            "core_version": core_version,
            "api_version": __api_version__,
            "auth_mode": "enabled" if auth_handler.accounts else "disabled",
            "pipeline_busy": bool(getattr(app.state, "pipeline_busy", False)),
            "working_dir": getattr(request.app.state.rag, "working_dir", None),
            "workspace": getattr(request.app.state.rag, "workspace", ""),
        }

    @app.post("/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        if not auth_handler.accounts:
            guest_token = auth_handler.create_token(
                username="guest",
                role="guest",
                metadata={"auth_mode": "disabled"},
            )
            return {
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": __api_version__,
            }
        if not auth_handler.verify_password(form_data.username, form_data.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect credentials",
            )
        user_token = auth_handler.create_token(
            username=form_data.username,
            role="user",
            metadata={"auth_mode": "enabled"},
        )
        return {
            "access_token": user_token,
            "token_type": "bearer",
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": __api_version__,
        }

    return app


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry: ``memgraphrag-server`` / ``python -m memgraphrag.api.server``."""
    if uvicorn is None:
        raise RuntimeError("uvicorn is required; install memgraphrag[api]")

    args = parse_args(argv)
    # Refresh module-level global_args for auth/dependencies
    import memgraphrag.api.config as config_mod

    config_mod.global_args = args

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    app = create_app(args)

    uvicorn_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": int(args.port),
        "log_level": str(args.log_level).lower(),
    }
    if getattr(args, "ssl", False):
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile

    uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
