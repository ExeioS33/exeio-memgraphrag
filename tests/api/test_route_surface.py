"""The documented API surface must match the app the server actually builds.

Three agents added routes in parallel (`/health/ready`, `/metrics`) while the
endpoint table in `docs/MemGraphRAG-API-Server.md` and the parity claim in
AGENTS.md still described the older surface. Documentation that drifts from the
router is what turned "full API parity with LightRAG" into a false claim in the
first place, so the count and the paths are pinned here rather than trusted.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")

from memgraphrag.api.server import create_app

pytestmark = pytest.mark.offline

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DOC = REPO_ROOT / "docs" / "MemGraphRAG-API-Server.md"
AGENTS_DOC = REPO_ROOT / "AGENTS.md"


def _mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.working_dir = "/tmp/memgraphrag-test"
    rag.workspace = ""
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    return rag


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        host="127.0.0.1",
        port=9621,
        workers=1,
        working_dir="/tmp/memgraphrag-test",
        input_dir="/tmp/memgraphrag-inputs",
        workspace="",
        key=None,
        log_level="INFO",
        ssl=False,
        ssl_certfile=None,
        ssl_keyfile=None,
        auth_accounts="",
        token_secret=None,
        cors_origins="*",
        whitelist_paths="/health,/docs,/openapi.json",
        ollama_model_name="memgraphrag",
        ollama_model_tag="latest",
    )


def _operations() -> set[tuple[str, str]]:
    app = create_app(_args(), testing=True, rag=_mock_rag())
    spec = app.openapi()
    return {
        (method.upper(), path)
        for path, ops in spec["paths"].items()
        for method in ops
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    }


def test_documented_operation_count_matches_the_app() -> None:
    doc = API_DOC.read_text(encoding="utf-8")
    match = re.search(r"That is the whole surface: (\d+) operations in total", doc)
    assert match, "the endpoint section no longer states an operation count"
    assert int(match.group(1)) == len(_operations())


def test_agents_parity_claim_uses_the_same_number() -> None:
    # The parity paragraph is the one an integrator quotes to a customer.
    text = AGENTS_DOC.read_text(encoding="utf-8")
    match = re.search(r"with (\d+) operations against LightRAG", text)
    assert match, "AGENTS.md no longer states the operation count"
    assert int(match.group(1)) == len(_operations())


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/health"),
        ("GET", "/health/ready"),
        ("GET", "/metrics"),
        ("GET", "/documents/"),
        ("GET", "/graphs"),
    ],
)
def test_endpoint_table_lists_the_operational_routes(method: str, path: str) -> None:
    doc = API_DOC.read_text(encoding="utf-8")
    # The table writes paths inside backticks; /documents/ appears as `/documents/`.
    assert f"`{path}`" in doc, f"{method} {path} is served but absent from the endpoint table"
    assert (method, path) in _operations()
