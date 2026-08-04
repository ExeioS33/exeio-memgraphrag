"""Gunicorn configuration for MemGraphRAG API.

Adapted from LightRAG ``lightrag/api/gunicorn_config.py`` (thin wrapper).
Variables may be overridden by ``gunicorn_runner.py`` via ``cfg.set``.
"""

from __future__ import annotations

import os

bind = os.getenv("MEMGRAPHRAG_GUNICORN_BIND", "0.0.0.0:9621")
workers = int(os.getenv("MEMGRAPHRAG_GUNICORN_WORKERS", "1"))
loglevel = os.getenv("MEMGRAPHRAG_GUNICORN_LOGLEVEL", "info")
certfile = os.getenv("MEMGRAPHRAG_GUNICORN_CERTFILE") or None
keyfile = os.getenv("MEMGRAPHRAG_GUNICORN_KEYFILE") or None

preload_app = True
worker_class = "uvicorn.workers.UvicornWorker"

accesslog = os.getenv("ACCESS_LOG", "-")
errorlog = os.getenv("ERROR_LOG", "-")


def on_starting(server):  # noqa: ANN001
    print("=" * 60)
    print(f"MemGraphRAG Gunicorn master starting ({workers} worker(s))")
    print(f"bind={bind} pid={os.getpid()}")
    print("=" * 60)


def on_exit(server):  # noqa: ANN001
    print("MemGraphRAG Gunicorn master shutting down")
