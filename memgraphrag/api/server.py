"""MemGraphRAG FastAPI server.

Adapted from LightRAG ``lightrag/api/lightrag_server.py`` — slim create_app + main.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from memgraphrag import __version__ as core_version
from memgraphrag.api import __api_version__
from memgraphrag.api.config import global_args, parse_args
from memgraphrag.api.middleware import (
    MetricsMiddleware,
    MetricsRegistry,
    RequestContextMiddleware,
    RequestIdLogFilter,
    render_prometheus,
)
from memgraphrag.api.rate_limit import FixedWindowRateLimiter, client_key
from memgraphrag.constants import (
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    LOGIN_MAX_ATTEMPTS,
    LOGIN_WINDOW_SECONDS,
)
from memgraphrag.core import MemGraphRAG
from memgraphrag.llm.openai_compatible import openai_complete, openai_embed
from memgraphrag.pipeline import reset_interrupted_documents

logger = logging.getLogger("memgraphrag.api.server")

# Seconds allowed for in-flight indexing to finish on shutdown before it is cancelled.
# Bounded on purpose: an unbounded wait turns a rolling restart into a hang, and
# container runtimes SIGKILL after their own grace period anyway.
DEFAULT_SHUTDOWN_DRAIN_TIMEOUT = 30.0

# Include wall-clock time on every app + uvicorn line (default uvicorn fmt has none).
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
# Carries the X-Request-ID of the request being served: on a single asyncio loop the
# [STAGE]/[LLM] lines of concurrent requests interleave and are otherwise impossible to
# attribute. Only usable on handlers carrying RequestIdLogFilter, which supplies the
# `request_id` attribute the format references.
_LOG_FMT = "%(asctime)s %(levelname)s [%(name)s] [req=%(request_id)s] %(message)s"
_UVICORN_DEFAULT_FMT = "%(asctime)s %(levelprefix)s %(message)s"
_UVICORN_ACCESS_FMT = (
    '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
)


def _logging_config(level: str) -> dict[str, Any]:
    """Uvicorn dictConfig with timestamps for default + access loggers."""
    lvl = str(level).upper()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": _UVICORN_DEFAULT_FMT,
                "datefmt": _LOG_DATEFMT,
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": _UVICORN_ACCESS_FMT,
                "datefmt": _LOG_DATEFMT,
            },
            "app": {
                "format": _LOG_FMT,
                "datefmt": _LOG_DATEFMT,
            },
        },
        "filters": {
            "request_id": {
                "()": "memgraphrag.api.middleware.RequestIdLogFilter",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "app": {
                "formatter": "app",
                "filters": ["request_id"],
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": lvl, "propagate": False},
            "uvicorn.error": {"level": lvl},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": lvl,
                "propagate": False,
            },
            "memgraphrag": {
                "handlers": ["app"],
                "level": lvl,
                "propagate": False,
            },
        },
        "root": {"handlers": ["app"], "level": lvl},
    }


try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse
    from fastapi.security import OAuth2PasswordRequestForm
except ImportError:  # pragma: no cover
    uvicorn = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    Response = None  # type: ignore[misc, assignment]
    status = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[misc, assignment]
    PlainTextResponse = None  # type: ignore[misc, assignment]
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


def shutdown_drain_timeout(args: Any) -> float:
    """Seconds to wait for background indexing at shutdown."""
    raw = os.getenv("MEMGRAPHRAG_SHUTDOWN_DRAIN_TIMEOUT") or getattr(
        args, "shutdown_drain_timeout", None
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_SHUTDOWN_DRAIN_TIMEOUT


async def recover_interrupted_documents(app: Any) -> list[str]:
    """Requeue documents a previous process left mid-pipeline. Never raises."""
    doc_status = getattr(getattr(app.state, "rag", None), "doc_status", None)
    if doc_status is None:
        return []
    try:
        recovered = await reset_interrupted_documents(doc_status)
    except Exception as exc:  # noqa: BLE001
        # Recovery is best-effort: a doc-status backend that is briefly unreachable
        # must not stop the server from coming up.
        logger.warning("Interrupted-document recovery skipped: %s", exc)
        return []
    if recovered:
        logger.info(
            "Recovered %d document(s) stuck in parsing/processing; requeued as pending",
            len(recovered),
        )
    return recovered


async def drain_background_tasks(app: Any, timeout: float) -> int:
    """Await tracked background indexing tasks; cancel whatever exceeds ``timeout``.

    Returns the number of tasks that had to be cancelled. Without this, a SIGTERM
    kills an in-flight indexing run at whatever point it had reached.
    """
    tasks = {t for t in (getattr(app.state, "background_tasks", None) or ()) if not t.done()}
    if not tasks:
        return 0
    logger.info("Waiting up to %.1fs for %d background indexing task(s)", timeout, len(tasks))
    _done, unfinished = await asyncio.wait(tasks, timeout=timeout)
    for task in unfinished:
        task.cancel()
    if unfinished:
        # Give the cancellations a chance to unwind their `finally` blocks (the
        # pipeline lock is released there) before the loop goes away.
        await asyncio.gather(*unfinished, return_exceptions=True)
        logger.warning(
            "Cancelled %d background indexing task(s) still running after %.1fs",
            len(unfinished),
            timeout,
        )
    return len(unfinished)


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

    from memgraphrag.api.auth import AuthHandler
    from memgraphrag.api.dependencies import (
        compile_whitelist,
        get_combined_auth_dependency,
    )
    from memgraphrag.api.routers.documents import create_documents_router
    from memgraphrag.api.routers.graphs import create_graphs_router
    from memgraphrag.api.routers.ollama import create_ollama_router
    from memgraphrag.api.routers.query import create_query_router

    cfg = args or global_args
    api_key = os.getenv("MEMGRAPHRAG_API_KEY") or getattr(cfg, "key", None) or None
    # Built from `cfg`, not from the import-time `global_args` singleton: otherwise a
    # programmatic create_app(args) silently runs with no accounts and no whitelist.
    auth_handler = AuthHandler(cfg)
    whitelist_patterns = compile_whitelist(
        getattr(cfg, "whitelist_paths", "/health,/docs,/openapi.json")
    )
    combined_auth = get_combined_auth_dependency(api_key, api_key_header_name="X-API-Key")

    engine = rag if rag is not None else _build_rag(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.background_tasks = set()
        try:
            if not testing:
                await app.state.rag.initialize_storages()
                # Before serving: PARSING/PROCESSING can only be leftovers from a
                # worker that died mid-flight, and process_pending never looks at
                # them, so without this sweep those documents wedge forever.
                await recover_interrupted_documents(app)
                try:
                    await app.state.rag.prepare_retrieval()
                    app.state.retrieval_ready = True
                    app.state.retrieval_error = None
                except Exception as exc:
                    # Swallowed on purpose (an empty corpus must still boot), but the
                    # failure is now visible on /health instead of only in the log.
                    app.state.retrieval_ready = False
                    app.state.retrieval_error = str(exc)
                    logger.warning("Retrieval warm-up skipped: %s", exc)
                logger.info("MemGraphRAG server ready")
            yield
        finally:
            await drain_background_tasks(app, shutdown_drain_timeout(cfg))
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
    allow_any_origin = cors_origins.strip() == "*"
    origins = (
        ["*"] if allow_any_origin else [o.strip() for o in cors_origins.split(",") if o.strip()]
    )
    # Starlette reflects the caller's Origin when allow_origins=["*"], so pairing it
    # with allow_credentials=True lets any third-party page drive this API from an
    # authenticated user's browser — including DELETE /documents/. Credentials are
    # only enabled once CORS_ORIGINS names explicit origins.
    if allow_any_origin:
        logger.warning(
            "CORS_ORIGINS='*': cookie/credential-bearing cross-origin requests are "
            "disabled. Set CORS_ORIGINS to explicit origins to allow them."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not allow_any_origin,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    metrics_registry = MetricsRegistry()
    app.add_middleware(MetricsMiddleware, registry=metrics_registry)
    # Added last, so it is the outermost layer: every response — including CORS
    # preflights and error responses raised inside other middleware — carries the id.
    app.add_middleware(RequestContextMiddleware)

    app.state.metrics = metrics_registry
    app.state.rag = engine
    app.state.args = cfg
    app.state.auth_handler = auth_handler
    app.state.whitelist_patterns = whitelist_patterns
    app.state.login_limiter = FixedWindowRateLimiter(
        max_attempts=int(getattr(cfg, "login_max_attempts", LOGIN_MAX_ATTEMPTS)),
        window_seconds=float(getattr(cfg, "login_window_seconds", LOGIN_WINDOW_SECONDS)),
    )
    app.state.input_dir = getattr(cfg, "input_dir", "./data/inputs")
    app.state.testing = testing
    app.state.pipeline_lock = asyncio.Lock()
    app.state.pipeline_busy = False
    app.state.background_tasks = set()
    # `testing=True` skips the warm-up entirely, so there is nothing to be un-ready
    # about; a real boot starts not-ready and flips once prepare_retrieval succeeds.
    app.state.retrieval_ready = bool(testing)
    app.state.retrieval_error = None
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

    def _retrieval_state() -> tuple[bool, str, str | None]:
        """(ready, state, error) for the retrieval engine."""
        error = getattr(app.state, "retrieval_error", None)
        if error:
            return False, "error", str(error)
        ready = bool(getattr(app.state, "retrieval_ready", False))
        return ready, "ready" if ready else "not_ready", None

    @app.get("/health", dependencies=[Depends(combined_auth)])
    async def health():
        # Liveness: 200 while the process answers. `ready` is the honest readiness
        # bit — this endpoint used to report "healthy" even when prepare_retrieval had
        # failed and every query was going to 500.
        ready, retrieval_status, retrieval_error = _retrieval_state()
        # working_dir / workspace are deliberately absent: /health sits in
        # WHITELIST_PATHS, so they were server filesystem layout handed to anyone.
        return {
            "status": "healthy",
            "core_version": core_version,
            "api_version": __api_version__,
            "auth_mode": "enabled" if auth_handler.accounts else "disabled",
            "pipeline_busy": bool(getattr(app.state, "pipeline_busy", False)),
            "ready": ready,
            "retrieval_status": retrieval_status,
            "retrieval_error": retrieval_error,
        }

    @app.get("/health/ready")
    async def health_ready(response: Response):
        # Readiness probe: 503 until the engine can actually serve a query, so an
        # orchestrator keeps the instance out of rotation instead of sending it
        # traffic it will fail. Unauthenticated but leaks nothing beyond the bit
        # itself, matching the whitelisted /health.
        ready, retrieval_status, retrieval_error = _retrieval_state()
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "ready": ready,
            "retrieval_status": retrieval_status,
            "retrieval_error": retrieval_error,
        }

    @app.get("/metrics", dependencies=[Depends(combined_auth)])
    async def metrics():
        # Behind the same auth as the rest of the API: route names, latencies and
        # traffic volume are operational intelligence, not public data.
        ready, _state, _error = _retrieval_state()
        body = render_prometheus(
            metrics_registry,
            gauges={
                "memgraphrag_pipeline_busy": (
                    float(bool(getattr(app.state, "pipeline_busy", False))),
                    "1 while the ingestion pipeline lock is held.",
                ),
                "memgraphrag_retrieval_ready": (
                    float(ready),
                    "1 once the retrieval engine has been prepared.",
                ),
                "memgraphrag_background_tasks": (
                    float(
                        len(
                            [
                                t
                                for t in (getattr(app.state, "background_tasks", None) or ())
                                if not t.done()
                            ]
                        )
                    ),
                    "Background indexing tasks currently running.",
                ),
            },
        )
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.post("/login")
    async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
        # /login is unauthenticated by nature, so it is the one endpoint an attacker
        # can hammer for free. Throttle per client before touching the password.
        limiter = app.state.login_limiter
        key = client_key(request)
        retry_after = limiter.check(key)
        if retry_after is not None:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Try again later.",
                headers={"Retry-After": str(max(1, int(retry_after) + 1))},
            )

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
        # Successful login clears the budget so a legitimate user is never locked out
        # by their own earlier typos.
        limiter.reset(key)
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

    # Entry points ask for .env explicitly. Importing this module used to call
    # load_dotenv() as a side effect, so `import memgraphrag.api.server` from a
    # directory holding a .env injected it — provider keys included — into the
    # process, which made the test suite non-hermetic and --run-integration a
    # false green. Loaded before parse_args so CLI flags still win over the file.
    import memgraphrag.api.config as config_mod

    config_mod.load_env_file()
    args = parse_args(argv)
    # Refresh module-level global_args for auth/dependencies
    config_mod.global_args = args

    log_level = str(args.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=_LOG_FMT,
        datefmt=_LOG_DATEFMT,
        force=True,
    )
    # basicConfig takes no filters, so attach it by hand: the format above references
    # %(request_id)s, which only this filter supplies.
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIdLogFilter())
    logger.info("MemGraphRAG API logging ready (timestamps + request ids enabled)")
    app = create_app(args)

    uvicorn_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": int(args.port),
        "log_level": str(args.log_level).lower(),
        "log_config": _logging_config(args.log_level),
    }
    if getattr(args, "ssl", False):
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile

    uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
