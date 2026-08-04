# syntax=docker/dockerfile:1
# Built/tagged by Compose as exeio-memgraphrag:<MEMGRAPHRAG_VERSION>

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY memgraphrag/ ./memgraphrag/
COPY README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api --no-editable

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG MEMGRAPHRAG_VERSION=0.1.0

LABEL org.opencontainers.image.title="exeio-memgraphrag" \
      org.opencontainers.image.description="EXEIO MemGraphRAG API server" \
      org.opencontainers.image.vendor="EXEIO" \
      org.opencontainers.image.source="https://github.com/ExeioS33/exeio-memgraphrag" \
      org.opencontainers.image.version="${MEMGRAPHRAG_VERSION}" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gosu curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash memgraphrag

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/memgraphrag /app/memgraphrag
COPY --from=builder /app/pyproject.toml /app/README.md /app/LICENSE /app/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /app/data/rag_storage /app/data/inputs /app/certs \
    && chown -R memgraphrag:memgraphrag /app/data

ENV PATH="/app/.venv/bin:${PATH}"
ENV HOST=0.0.0.0
ENV PORT=9621
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV PYTHONUNBUFFERED=1
ENV MEMGRAPHRAG_VERSION=${MEMGRAPHRAG_VERSION}

EXPOSE 9621

# Entrypoint starts as root to fix bind-mount ownership, then drops to UID 1000.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "memgraphrag.api.server"]
