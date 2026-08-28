"""Pytest configuration for MemGraphRAG.

Marker policy — the suite declares, per test, whether it needs a live backend:

* ``offline``: needs no external service. Every unit test carries it, either as a
  module-level ``pytestmark`` or as a per-test decorator.
* ``integration`` (plus ``requires_db`` / ``requires_api``): needs the Docker
  Compose backends. Skipped unless ``--run-integration`` is passed.

CI deliberately runs the **whole** suite (``./scripts/test.sh tests``) instead of
``pytest -m offline``. A ``-m offline`` gate reports green while silently
deselecting every test whose marker someone forgot, and that is exactly how half
of this suite stopped being checked. Running everything means a missing marker
costs nothing; ``tests/test_suite_hygiene.py`` then keeps the markers themselves
honest so ``-m offline`` stays a usable local shortcut.
"""

from __future__ import annotations

import pytest

#: Markers that declare a test needs something this process cannot provide.
INTEGRATION_MARKERS = ("integration", "requires_db", "requires_api")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that need Docker Compose backends",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="need --run-integration")
    for item in items:
        if any(marker in item.keywords for marker in INTEGRATION_MARKERS):
            item.add_marker(skip_integration)
