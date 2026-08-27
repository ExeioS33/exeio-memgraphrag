"""Guards on the test suite's own invariants.

Two silent-green failures motivated this module:

1. Only 41 of 86 tests carried ``offline``, so ``pytest -m offline`` — the
   pattern a CI would naturally copy — checked half the suite and still passed.
2. ``tests/test_integration_smoke.py`` asserted that environment variables were
   non-empty rather than that a backend answered, so ``--run-integration``
   reported success against backends that were not even running.

Both regressions are invisible in a normal run: nothing fails, there is just
less coverage than the report suggests. These tests make them fail loudly.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

TESTS_DIR = Path(__file__).resolve().parent

#: A test declares itself either runnable anywhere, or needing a live backend.
OFFLINE_MARKER = "offline"
BACKEND_MARKERS = frozenset({"integration", "requires_db", "requires_api"})


def _marker_name(node: ast.expr) -> str | None:
    """Return ``x`` for a ``pytest.mark.x`` expression, with or without a call."""
    if isinstance(node, ast.Call):
        node = node.func
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
    ):
        return node.attr
    return None


def _marker_names(nodes: list[ast.expr]) -> set[str]:
    return {name for node in nodes if (name := _marker_name(node)) is not None}


def _module_markers(tree: ast.Module) -> set[str]:
    """Markers from a module-level ``pytestmark = ...`` assignment."""
    markers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            continue
        value = node.value
        elements = value.elts if isinstance(value, ast.List | ast.Tuple) else [value]
        markers |= _marker_names(list(elements))
    return markers


def _undeclared_tests(path: Path) -> list[str]:
    """Names of test functions in ``path`` carrying neither offline nor a backend marker."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    undeclared: list[str] = []

    def visit(body: list[ast.stmt], inherited: set[str], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(
                    node.body,
                    inherited | _marker_names(node.decorator_list),
                    f"{prefix}{node.name}::",
                )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if not node.name.startswith("test_"):
                    continue
                markers = inherited | _marker_names(node.decorator_list)
                if OFFLINE_MARKER not in markers and not (markers & BACKEND_MARKERS):
                    undeclared.append(f"{prefix}{node.name}")

    visit(tree.body, _module_markers(tree), "")
    return undeclared


def test_every_test_declares_offline_or_backend_need() -> None:
    """No test may stay unmarked: `-m offline` must not quietly drop half the suite.

    Read with ``ast`` rather than through pytest's own collection on purpose —
    a guard implemented as a collection hook is itself deselectable by the very
    ``-m`` filter it exists to protect, and would inspect only the subset the
    caller already chose to run.
    """
    offenders = {
        str(path.relative_to(TESTS_DIR)): undeclared
        for path in sorted(TESTS_DIR.rglob("test_*.py"))
        if (undeclared := _undeclared_tests(path))
    }
    assert not offenders, (
        "these tests carry neither @pytest.mark.offline nor a backend marker "
        f"({', '.join(sorted(BACKEND_MARKERS))}), so `pytest -m offline` skips them "
        f"without saying so: {offenders}"
    )


def test_integration_gate_fails_against_a_dead_backend() -> None:
    """`--run-integration` must dial the backend, not merely read the environment.

    The previous smoke tests passed whenever a developer ``.env`` defined
    ``POSTGRES_*``; this reruns them in a subprocess pointed at a closed port and
    fails if they still report success.
    """
    pytest.importorskip("asyncpg", reason="asyncpg not installed (extra: api)")

    env = dict(os.environ)
    env.update(
        {
            "POSTGRES_HOST": "127.0.0.1",
            # Port 1 is reserved and never bound: connect() is refused immediately.
            "POSTGRES_PORT": "1",
            "POSTGRES_USER": "nobody",
            "POSTGRES_PASSWORD": "nobody",
            "POSTGRES_DATABASE": "nowhere",
            # Leave Neo4j unconfigured so only the PostgreSQL dial is exercised.
            "NEO4J_URI": "",
            "NEO4J_USERNAME": "",
            "NEO4J_PASSWORD": "",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(TESTS_DIR / "test_integration_smoke.py"),
            "--run-integration",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=TESTS_DIR.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode != 0, (
        "the integration smoke tests passed against a backend that is not listening: "
        f"the --run-integration gate is decorative\n{completed.stdout}\n{completed.stderr}"
    )
