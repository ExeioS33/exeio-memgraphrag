"""Thin Gunicorn launcher for MemGraphRAG API.

Adapted from LightRAG ``lightrag/api/run_with_gunicorn.py``.

This is an entry point, so it loads ``.env`` explicitly (importing the API
package no longer does it as a side effect) and refuses worker counts the
storage layer cannot survive.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("memgraphrag.api.gunicorn_runner")


def main() -> None:
    """Start MemGraphRAG under Gunicorn with Uvicorn workers."""
    try:
        from gunicorn.app.base import BaseApplication
    except ImportError as exc:
        raise SystemExit("gunicorn is required; install memgraphrag[api]") from exc

    import memgraphrag.api.config as config_mod
    from memgraphrag.api.config import load_env_file, parse_args, validate_worker_count

    load_env_file()
    args = parse_args()
    config_mod.global_args = args

    bind = f"{args.host}:{args.port}"
    workers = max(1, int(args.workers or 1))
    loglevel = str(args.log_level or "info").lower()

    try:
        validate_worker_count(workers, args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if workers > 1:
        # Even on shared databases nothing coordinates the workers: the ingest
        # `pipeline_lock` is per-process, so "409 while busy" only holds within
        # the worker that owns the running ingest.
        logger.warning(
            "WORKERS=%d: the ingest pipeline lock is per-process, so concurrent "
            "ingests in different workers are not serialized.",
            workers,
        )

    # Export for gunicorn_config.py
    os.environ.setdefault("MEMGRAPHRAG_GUNICORN_BIND", bind)
    os.environ.setdefault("MEMGRAPHRAG_GUNICORN_WORKERS", str(workers))
    os.environ.setdefault("MEMGRAPHRAG_GUNICORN_LOGLEVEL", loglevel)
    if getattr(args, "ssl", False) and args.ssl_certfile and args.ssl_keyfile:
        os.environ["MEMGRAPHRAG_GUNICORN_CERTFILE"] = args.ssl_certfile
        os.environ["MEMGRAPHRAG_GUNICORN_KEYFILE"] = args.ssl_keyfile

    class MemGraphRAGApplication(BaseApplication):
        def __init__(self, app_uri: str, options: dict | None = None):
            self.app_uri = app_uri
            self.options = options or {}
            super().__init__()

        def load_config(self) -> None:
            config_path = os.path.join(os.path.dirname(__file__), "gunicorn_config.py")
            self.cfg.set("config", config_path)
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key, value)

        def load(self):
            from memgraphrag.api.server import create_app

            return create_app(args)

    options = {
        "bind": bind,
        "workers": workers,
        "loglevel": loglevel,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "preload_app": True,
    }
    if getattr(args, "ssl", False):
        options["certfile"] = args.ssl_certfile
        options["keyfile"] = args.ssl_keyfile

    print(f"Starting MemGraphRAG (gunicorn) on {bind} with {workers} worker(s)")
    MemGraphRAGApplication("memgraphrag.api.server:create_app", options).run()


if __name__ == "__main__":
    main()
