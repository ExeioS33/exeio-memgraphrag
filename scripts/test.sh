#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -eq 0 ]; then
    set -- tests
fi

declare -a TRIED=()

run_python() {
    local candidate="$1"
    local label="$2"
    shift 2

    TRIED+=("$label: $candidate")

    if [[ "$candidate" == */* ]]; then
        if [ ! -x "$candidate" ]; then
            return 1
        fi
        if "$candidate" -c "import pytest" >/dev/null 2>&1; then
            printf "Using %s: %s\n" "$label" "$candidate"
            exec "$candidate" -m pytest "$@"
        fi
        return 1
    fi

    local resolved
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    if [ -z "$resolved" ]; then
        return 1
    fi
    if "$resolved" -c "import pytest" >/dev/null 2>&1; then
        printf "Using %s: %s\n" "$label" "$resolved"
        exec "$resolved" -m pytest "$@"
    fi
    return 1
}

run_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        return 1
    fi
    TRIED+=("uv-managed environment: uv run python -m pytest")
    if uv run python -c "import pytest" >/dev/null 2>&1; then
        printf "Using uv-managed environment\n"
        exec uv run python -m pytest "$@"
    fi
    return 1
}

if [ -n "${PYTHON:-}" ]; then
    if run_python "$PYTHON" "PYTHON override" "$@"; then
        exit 0
    fi
    printf "Configured PYTHON does not provide pytest: %s\n" "$PYTHON" >&2
    exit 1
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
    run_python "$VIRTUAL_ENV/bin/python" "active virtualenv" "$@" || true
fi

run_uv "$@" || true
run_python "$ROOT_DIR/.venv/bin/python" "repo .venv" "$@" || true
run_python python3 "PATH python3" "$@" || true

printf "Unable to find a Python environment with pytest available.\n" >&2
printf "Tried:\n" >&2
for entry in "${TRIED[@]}"; do
    printf "  - %s\n" "$entry" >&2
done
exit 1
