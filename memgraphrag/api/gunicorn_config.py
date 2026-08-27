"""Gunicorn configuration for MemGraphRAG API.

Adapted from LightRAG ``lightrag/api/gunicorn_config.py`` (thin wrapper).
Variables may be overridden by ``gunicorn_runner.py`` via ``cfg.set``.

Gunicorn loads this file as the master's configuration, so it doubles as an entry
point: ``gunicorn -c memgraphrag/api/gunicorn_config.py`` bypasses
``gunicorn_runner`` entirely and must therefore load ``.env`` and re-run the
worker-count check itself.
"""

from __future__ import annotations

import os

from memgraphrag.api.config import load_env_file, validate_worker_count
from memgraphrag.api.server import _logging_config

load_env_file()

bind = os.getenv("MEMGRAPHRAG_GUNICORN_BIND", "0.0.0.0:9621")
workers = int(os.getenv("MEMGRAPHRAG_GUNICORN_WORKERS", "1"))
loglevel = os.getenv("MEMGRAPHRAG_GUNICORN_LOGLEVEL", "info")
certfile = os.getenv("MEMGRAPHRAG_GUNICORN_CERTFILE") or None
keyfile = os.getenv("MEMGRAPHRAG_GUNICORN_KEYFILE") or None

# Abort the master before any worker forks: that is the last moment at which a
# shared file-backed WORKING_DIR is still intact.
try:
    validate_worker_count(workers)
except ValueError as exc:
    raise SystemExit(str(exc)) from exc

preload_app = True
worker_class = "uvicorn.workers.UvicornWorker"

accesslog = os.getenv("ACCESS_LOG", "-")
errorlog = os.getenv("ERROR_LOG", "-")

# Gunicorn installs its own logging, so the request-id filter that
# ``server._logging_config`` wires into uvicorn never reached this entry point and
# every line came out without its [req=...] tag — exactly where correlation matters
# most, since several workers interleave on one stream. Reuse the same dictConfig
# rather than a second copy that can drift from it.
logconfig_dict = _logging_config(loglevel)


def on_starting(server):  # noqa: ANN001
    print("=" * 60)
    print(f"MemGraphRAG Gunicorn master starting ({workers} worker(s))")
    print(f"bind={bind} pid={os.getpid()}")
    print("=" * 60)


def on_exit(server):  # noqa: ANN001
    print("MemGraphRAG Gunicorn master shutting down")
