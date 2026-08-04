# UV Lock Guide

This project uses [uv](https://github.com/astral-sh/uv) for installs and locking.

```bash
# Install API + test extras
uv sync --extra api --extra pytest

# After changing pyproject.toml dependencies:
uv lock
uv sync --extra api --extra pytest

# Run tests
./scripts/test.sh tests
# or
uv run pytest tests
```

Commit `uv.lock` whenever dependency resolution changes so Docker `uv sync --frozen` stays reproducible.
