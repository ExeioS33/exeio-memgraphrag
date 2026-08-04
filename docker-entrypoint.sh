#!/bin/sh
set -e

# Merge a corporate inspection CA (Fortinet/Zscaler/…) into the trust store.
# httpx/openai honor SSL_CERT_FILE when our Python clients pass verify= that path.
_build_ca_bundle() {
    _corp="${MEMGRAPHRAG_CORP_CA_FILE:-/app/certs/corporate-ca.crt}"
    _out="${MEMGRAPHRAG_CA_BUNDLE_OUT:-/tmp/memgraphrag-ca-bundle.crt}"
    _sys="/etc/ssl/certs/ca-certificates.crt"
    if [ -f "$_corp" ] && [ -f "$_sys" ]; then
        cat "$_sys" "$_corp" > "$_out"
        export SSL_CERT_FILE="$_out"
        export REQUESTS_CA_BUNDLE="$_out"
        export CURL_CA_BUNDLE="$_out"
        echo "memgraphrag: merged corporate CA into $_out"
    elif [ -f "$_corp" ]; then
        export SSL_CERT_FILE="$_corp"
        export REQUESTS_CA_BUNDLE="$_corp"
        export CURL_CA_BUNDLE="$_corp"
        echo "memgraphrag: using corporate CA $_corp"
    fi
}

# Drop to non-root user after fixing bind-mount ownership (LightRAG pattern).
if [ "${1#-}" != "$1" ]; then
    set -- python -m memgraphrag.api.server "$@"
fi

_build_ca_bundle

if [ "$(id -u)" = "0" ]; then
    for _d in /app/data "$WORKING_DIR" "$INPUT_DIR" "$TIKTOKEN_CACHE_DIR"; do
        case "$_d" in
            ""|"/"|"/bin"|"/usr"|"/etc"|"/var"|"/tmp") continue ;;
        esac
        mkdir -p "$_d" 2>/dev/null || true
        chown -R memgraphrag:memgraphrag "$_d" 2>/dev/null || true
    done
    # Re-export CA env through gosu so the app process sees the merged bundle.
    exec gosu memgraphrag env \
        SSL_CERT_FILE="${SSL_CERT_FILE:-}" \
        REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-}" \
        CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-}" \
        "$@"
fi

exec "$@"
