"""Load project ``.env`` for CLI / Streamlit so SSL_* and API settings apply."""

from __future__ import annotations

from pathlib import Path


def load_client_env() -> list[str]:
    """Load nearest ``.env`` files (cwd then package repo). Return paths loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return []

    loaded: list[str] = []
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        load_dotenv(resolved, override=False)
        loaded.append(str(resolved))
    return loaded
