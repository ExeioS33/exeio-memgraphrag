# UV Lock Guide

This project uses [uv](https://github.com/astral-sh/uv) for installs and locking.

## Pinning policy

- **Direct dependencies** in `pyproject.toml` use exact pins (`package==X.Y.Z`) taken from the current `uv.lock` / `uv export` resolution.
- **Transitive dependencies** stay locked only in `uv.lock` (do not duplicate them in `pyproject.toml`).
- **NumPy** is pinned per Python minor with environment markers (lock may resolve different wheels for 3.10 / 3.11 / 3.12+).
- Docker builds use `uv sync --frozen --extra api` so the image matches the committed lockfile.

```bash
# Install API + test extras
uv sync --extra api --extra pytest

# After changing pyproject.toml dependencies:
uv lock
uv sync --extra api --extra pytest

# Inspect the resolved set used for pins
uv export --extra api --extra pytest --no-hashes

# Run tests
./scripts/test.sh tests
# or
uv run pytest tests
```

Commit `uv.lock` whenever dependency resolution changes so Docker `uv sync --frozen` stays reproducible.
