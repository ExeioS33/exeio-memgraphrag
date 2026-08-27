"""Tests that importing the API package never reads a ``.env`` by itself.

Importing ``memgraphrag.api.config`` (or ``.auth``, or anything pulling them in)
used to call ``load_dotenv`` at module level. Any process whose working directory
held a developer ``.env`` therefore inherited it — provider API keys included —
which is how ``pytest --run-integration`` came back green instead of skipping.
These tests run a fresh interpreter in a throw-away directory so the assertion is
about the import itself, not about the state of the current process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "MEMGRAPHRAG_TEST_DOTENV_MARKER"


def _run_in_dir_with_dotenv(tmp_path: Path, snippet: str) -> str:
    """Run ``snippet`` in a fresh interpreter whose cwd holds a poisoned .env."""
    (tmp_path / ".env").write_text(f"{MARKER}=leaked\n", encoding="utf-8")
    env = dict(os.environ)
    env.pop(MARKER, None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.offline
def test_importing_config_does_not_load_dotenv(tmp_path: Path) -> None:
    output = _run_in_dir_with_dotenv(
        tmp_path,
        f"import os\nimport memgraphrag.api.config  # noqa: F401\nprint(os.getenv('{MARKER}'))\n",
    )
    assert output == "None"


@pytest.mark.offline
def test_importing_auth_does_not_load_dotenv(tmp_path: Path) -> None:
    output = _run_in_dir_with_dotenv(
        tmp_path,
        f"import os\nimport memgraphrag.api.auth  # noqa: F401\nprint(os.getenv('{MARKER}'))\n",
    )
    assert output == "None"


@pytest.mark.offline
def test_importing_the_server_module_does_not_load_dotenv(tmp_path: Path) -> None:
    # server.py kept its own module-level load_dotenv after config/auth lost theirs,
    # so importing the app (which every API test does) still leaked the developer
    # .env into the interpreter. The file is now read by main() instead.
    output = _run_in_dir_with_dotenv(
        tmp_path,
        f"import os\nimport memgraphrag.api.server  # noqa: F401\nprint(os.getenv('{MARKER}'))\n",
    )
    assert output == "None"


@pytest.mark.offline
def test_load_env_file_still_loads_dotenv_for_entry_points(tmp_path: Path) -> None:
    # The real server must keep working: an explicit call reads the same file.
    output = _run_in_dir_with_dotenv(
        tmp_path,
        "import os\n"
        "from memgraphrag.api.config import load_env_file\n"
        "loaded = load_env_file()\n"
        f"print(loaded, os.getenv('{MARKER}'))\n",
    )
    assert output == "True leaked"


@pytest.mark.offline
def test_load_env_file_refreshes_global_args(tmp_path: Path) -> None:
    # global_args is built at import time, i.e. before .env exists in the
    # environment; entry points rely on load_env_file to rebuild it.
    (tmp_path / ".env").write_text("TOP_K=77\n", encoding="utf-8")
    env = dict(os.environ)
    env.pop("TOP_K", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import memgraphrag.api.config as c\n"
                "before = c.global_args.top_k\n"
                "c.load_env_file()\n"
                "print(before, c.global_args.top_k)\n"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    before, after = result.stdout.split()
    assert before != "77"
    assert after == "77"
