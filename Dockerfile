# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY memgraphrag/ ./memgraphrag/
COPY README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api --no-editable

FROM python:3.12-slim-bookworm

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
    && mkdir -p /app/data/rag_storage /app/data/inputs \
    && chown -R memgraphrag:memgraphrag /app/data

ENV PATH="/app/.venv/bin:${PATH}"
ENV HOST=0.0.0.0
ENV PORT=9621
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV PYTHONUNBUFFERED=1

EXPOSE 9621

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "memgraphrag.api.server"]
