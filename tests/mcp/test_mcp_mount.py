"""Mounting the MCP server inside the API app.

The tests that matter here all exercise a *request*, not an import. Every one of
the three documented traps looks fine at start-up and fails on the first call:
a missing `session_manager.run()` raises "Task group is not initialized", an
unlisted host answers 421, and a missing token answers 401.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from memgraphrag.api.config import namespace_from_dict
from memgraphrag.api.server import create_app
from memgraphrag.mcp.server import allowed_hosts, mcp_enabled
from memgraphrag.utils.misc import QuerySolution

pytestmark = pytest.mark.offline

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.prepare_retrieval = AsyncMock()
    rag.aretrieve = AsyncMock(
        return_value=[QuerySolution(question="q", docs=["p"], sources=["a.pdf"])]
    )
    return rag


def _client(**overrides) -> TestClient:
    args = namespace_from_dict({"mcp_enabled": True, **overrides})
    return TestClient(create_app(args, testing=True, rag=_mock_rag()))


def test_off_by_default() -> None:
    """Mounting a second way into the corpus is a deployment decision."""
    assert mcp_enabled(namespace_from_dict({})) is False
    assert mcp_enabled(namespace_from_dict({"mcp_enabled": True})) is True


def test_allowed_hosts_are_parsed_as_an_exact_list() -> None:
    args = namespace_from_dict({"mcp_allowed_hosts": "rag.example.com, rag.example.com:9621"})
    assert allowed_hosts(args) == ["rag.example.com", "rag.example.com:9621"]
    assert allowed_hosts(namespace_from_dict({})) == []


def test_the_first_request_succeeds() -> None:
    """The regression this guards is not a start-up failure.

    `session_manager.run()` lives in the *host* app's lifespan because Starlette
    never runs a mounted sub-app's. Without it the server boots cleanly and the
    first call dies on "Task group is not initialized", so asserting that the app
    imports would prove nothing.
    """
    with _client(mcp_allowed_hosts="testserver") as client:
        response = client.post("/mcp/", json=INIT, headers=HEADERS)
    assert response.status_code != 500, response.text
    assert "Task group is not initialized" not in response.text


def test_an_unlisted_host_is_refused_by_the_transport() -> None:
    """DNS-rebinding protection allows localhost only until told otherwise, which is
    exactly why a first remote deployment answers 421 and looks like a routing bug.
    """
    with _client(mcp_allowed_hosts="rag.example.com") as client:
        response = client.post("/mcp/", json=INIT, headers=HEADERS)
    assert response.status_code == 421


def test_a_listed_host_passes() -> None:
    with _client(mcp_allowed_hosts="testserver,testserver:80") as client:
        response = client.post("/mcp/", json=INIT, headers=HEADERS)
    assert response.status_code != 421


def test_a_missing_token_is_refused_when_the_api_has_credentials() -> None:
    with _client(mcp_allowed_hosts="testserver", key="secret-key") as client:
        response = client.post("/mcp/", json=INIT, headers=HEADERS)
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_the_api_key_is_accepted_as_a_bearer_token() -> None:
    """One identity system, not two: the MCP verifier sits on the API's own
    credentials so there is only one place to revoke one."""
    with _client(mcp_allowed_hosts="testserver", key="secret-key") as client:
        response = client.post(
            "/mcp/", json=INIT, headers={**HEADERS, "Authorization": "Bearer secret-key"}
        )
    assert response.status_code != 401


def test_a_wrong_token_is_refused() -> None:
    with _client(mcp_allowed_hosts="testserver", key="secret-key") as client:
        response = client.post(
            "/mcp/", json=INIT, headers={**HEADERS, "Authorization": "Bearer nope"}
        )
    assert response.status_code == 401


def test_mounting_mcp_adds_no_openapi_operation() -> None:
    """A Mount is not a route. The two operation-count guards in
    tests/api/test_route_surface.py must stay green whether MCP is on or off."""
    with _client(mcp_allowed_hosts="testserver") as on:
        with_mcp = sum(len(v) for v in on.app.openapi()["paths"].values())
    args = namespace_from_dict({})
    with TestClient(create_app(args, testing=True, rag=_mock_rag())) as off:
        without = sum(len(v) for v in off.app.openapi()["paths"].values())
    assert with_mcp == without
