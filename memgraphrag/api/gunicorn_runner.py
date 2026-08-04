"""Thin Gunicorn launcher for MemGraphRAG API.

Adapted from LightRAG ``lightrag/api/run_with_gunicorn.py``.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    """Start MemGraphRAG under Gunicorn with Uvicorn workers."""
    try:
        from gunicorn.app.base import BaseApplication
    except ImportError as exc:
        raise SystemExit(
            "gunicorn is required; install memgraphrag[api]"
        ) from exc

    from memgraphrag.api.config import parse_args
    import memgraphrag.api.config as config_mod

    args = parse_args()
    config_mod.global_args = args

    bind = f"{args.host}:{args.port}"
    workers = max(1, int(args.workers or 1))
    loglevel = str(args.log_level or "info").lower()

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
            config_path = os.path.join(
                os.path.dirname(__file__), "gunicorn_config.py"
            )
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
