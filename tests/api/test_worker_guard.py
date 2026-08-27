"""Tests for the WORKERS>1 refusal on file-backed storage backends."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memgraphrag.api.config import (
    FILE_BACKED_STORAGES,
    file_backed_storages,
    validate_worker_count,
)

DATABASE_BACKENDS = SimpleNamespace(
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="Neo4JStorage",
    doc_status_storage="PGDocStatusStorage",
)

DEFAULT_BACKENDS = SimpleNamespace(
    kv_storage="JsonKVStorage",
    vector_storage="NanoVectorDBStorage",
    graph_storage="IgraphStorage",
    doc_status_storage="JsonDocStatusStorage",
)


@pytest.mark.offline
def test_single_worker_is_always_accepted() -> None:
    validate_worker_count(1, DEFAULT_BACKENDS)
    validate_worker_count(0, DEFAULT_BACKENDS)


@pytest.mark.offline
def test_multiple_workers_rejected_on_file_backed_defaults() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_worker_count(4, DEFAULT_BACKENDS)

    message = str(excinfo.value)
    # The operator has to learn which backend is the problem and what to do next.
    assert "WORKERS=4" in message
    for backend in sorted(FILE_BACKED_STORAGES):
        assert backend in message
    assert "PGKVStorage" in message and "Neo4JStorage" in message


@pytest.mark.offline
def test_multiple_workers_rejected_when_only_one_backend_is_file_backed() -> None:
    mixed = SimpleNamespace(**vars(DATABASE_BACKENDS))
    mixed.graph_storage = "IgraphStorage"

    assert file_backed_storages(mixed) == ["IgraphStorage"]
    with pytest.raises(ValueError, match="IgraphStorage"):
        validate_worker_count(2, mixed)


@pytest.mark.offline
def test_multiple_workers_accepted_on_shared_databases() -> None:
    validate_worker_count(8, DATABASE_BACKENDS)


@pytest.mark.offline
def test_backends_resolved_from_environment_when_args_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "MEMGRAPHRAG_KV_STORAGE",
        "MEMGRAPHRAG_VECTOR_STORAGE",
        "MEMGRAPHRAG_GRAPH_STORAGE",
        "MEMGRAPHRAG_DOC_STATUS_STORAGE",
        "KV_STORAGE",
        "VECTOR_STORAGE",
        "GRAPH_STORAGE",
        "DOC_STATUS_STORAGE",
    ):
        monkeypatch.delenv(name, raising=False)

    # gunicorn_config.py has no argparse namespace to hand over, so the check must
    # be able to read the selection straight from the environment.
    assert file_backed_storages() == sorted(FILE_BACKED_STORAGES)
    with pytest.raises(ValueError):
        validate_worker_count(2)

    monkeypatch.setenv("MEMGRAPHRAG_KV_STORAGE", "PGKVStorage")
    monkeypatch.setenv("MEMGRAPHRAG_VECTOR_STORAGE", "PGVectorStorage")
    monkeypatch.setenv("MEMGRAPHRAG_GRAPH_STORAGE", "Neo4JStorage")
    monkeypatch.setenv("MEMGRAPHRAG_DOC_STATUS_STORAGE", "PGDocStatusStorage")
    assert file_backed_storages() == []
    validate_worker_count(2)


@pytest.mark.offline
def test_gunicorn_runner_turns_the_refusal_into_a_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("gunicorn")

    from memgraphrag.api import gunicorn_runner

    monkeypatch.setenv("WORKERS", "3")
    monkeypatch.setenv("MEMGRAPHRAG_KV_STORAGE", "JsonKVStorage")
    monkeypatch.setenv("MEMGRAPHRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
    monkeypatch.setenv("MEMGRAPHRAG_GRAPH_STORAGE", "IgraphStorage")
    monkeypatch.setenv("MEMGRAPHRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")
    monkeypatch.setattr("sys.argv", ["memgraphrag-gunicorn"])
    # Do not let the entry point read the developer's .env during the test.
    monkeypatch.setattr("memgraphrag.api.config.load_env_file", lambda *a, **k: False, raising=True)

    with pytest.raises(SystemExit) as excinfo:
        gunicorn_runner.main()

    assert "WORKERS=3" in str(excinfo.value)
