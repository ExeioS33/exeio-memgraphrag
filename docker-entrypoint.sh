#!/bin/sh
set -e

# Drop to non-root user after fixing bind-mount ownership (LightRAG pattern).
if [ "${1#-}" != "$1" ]; then
    set -- python -m memgraphrag.api.server "$@"
fi

if [ "$(id -u)" = "0" ]; then
    for _d in /app/data "$WORKING_DIR" "$INPUT_DIR" "$TIKTOKEN_CACHE_DIR"; do
        case "$_d" in
            ""|"/"|"/bin"|"/usr"|"/etc"|"/var"|"/tmp") continue ;;
        esac
        mkdir -p "$_d" 2>/dev/null || true
        chown -R memgraphrag:memgraphrag "$_d" 2>/dev/null || true
    done
    exec gosu memgraphrag "$@"
fi

exec "$@"
