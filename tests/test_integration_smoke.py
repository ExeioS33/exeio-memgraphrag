"""Gated integration smoke tests (require --run-integration)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.requires_db
def test_postgres_env_documented():
    """Placeholder: real DB checks run only when compose is up."""
    if not os.getenv("POSTGRES_HOST"):
        pytest.skip("POSTGRES_HOST not set")
    assert os.getenv("POSTGRES_DATABASE")


@pytest.mark.requires_db
def test_neo4j_env_documented():
    if not os.getenv("NEO4J_URI"):
        pytest.skip("NEO4J_URI not set")
    assert os.getenv("NEO4J_USERNAME")
