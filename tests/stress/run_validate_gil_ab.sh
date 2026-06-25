#!/bin/bash
# GIL on/off A/B sweep for the auth-server /validate hot path (issue #1316).
#
# Runs measure_validate_latency.py as a concurrency sweep twice against the SAME
# free-threaded image: once with the GIL re-enabled (PYTHON_GIL=1) and once with
# it disabled (free-threaded). No rebuild is needed -- the GIL is toggled via the
# auth-server extra_env file, which only requires recreating the auth-server
# container between the two halves.
#
# Usage:
#   bash tests/stress/run_validate_gil_ab.sh [flags]
#
# Optional flags (override the defaults below):
#   --concurrency LIST   Comma-separated levels. Default: 50,75,100,125,150,175,200
#   --repeats N          Repeats per level (median-aggregated). Default: 3
#   --requests N         Requests per path per run. Default: 3000
#   --token-file PATH    JWT token file. Default: .token
#   --scrape-wait N      Seconds to wait for Prometheus scrape. Default: 18
#
# Prerequisites:
#   - The stack is up (docker compose up -d) on the FREE-THREADED auth image.
#   - Prometheus reachable on localhost:9090, auth-server on localhost:8888.
#   - A valid JWT in the token file.
#
# Output:
#   tests/stress/results/validate/gil-on-sweep.json
#   tests/stress/results/validate/gil-off-sweep.json
#   plus a printed per-concurrency comparison table at the end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
CONCURRENCY="50,75,100,125,150,175,200"
REPEATS=3
REQUESTS=3000
TOKEN_FILE=".token"
SCRAPE_WAIT=18
EXTRA_ENV_DIR="${MCP_EXTRA_ENV_DIR:-./extra_env}"
RESULTS_DIR="tests/stress/results/validate"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --requests) REQUESTS="$2"; shift 2 ;;
        --token-file) TOKEN_FILE="$2"; shift 2 ;;
        --scrape-wait) SCRAPE_WAIT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

cd "$REPO_ROOT"
mkdir -p "$RESULTS_DIR" "$EXTRA_ENV_DIR"

GIL_ENV_FILE="$EXTRA_ENV_DIR/auth-server.env"

# Recreate auth-server and wait for it to report healthy, then echo its GIL state.
_recreate_auth() {
    docker compose up -d --no-deps auth-server >/dev/null 2>&1
    # Give uvicorn time to boot and emit the free-threading status line.
    sleep 9
    docker logs mcp-gateway-registry-auth-server-1 2>&1 \
        | grep -E "GIL enabled at runtime" | tail -1
}

_run_sweep() {
    local label="$1"
    local out="$2"
    shift 2
    uv run python -m tests.stress.measure_validate_latency \
        --label "$label" \
        --sweep "$CONCURRENCY" \
        --repeats "$REPEATS" \
        --requests "$REQUESTS" \
        --token-file "$TOKEN_FILE" \
        --scrape-wait "$SCRAPE_WAIT" \
        --out "$out" \
        "$@"
}

echo "=========================================================="
echo "GIL A/B sweep: concurrency=[$CONCURRENCY] repeats=$REPEATS requests=$REQUESTS"
echo "=========================================================="

# --- Phase 1: GIL ON (baseline) ---
echo ""
echo ">>> Phase 1/2: GIL ON (PYTHON_GIL=1)"
printf 'PYTHON_GIL=1\n' > "$GIL_ENV_FILE"
_recreate_auth
_run_sweep "gil-on" "$RESULTS_DIR/gil-on-sweep.json"

# --- Phase 2: GIL OFF (free-threaded), compare to baseline ---
echo ""
echo ">>> Phase 2/2: GIL OFF (free-threaded)"
rm -f "$GIL_ENV_FILE"
_recreate_auth
_run_sweep "gil-off" "$RESULTS_DIR/gil-off-sweep.json" \
    --free-threaded \
    --compare-to "$RESULTS_DIR/gil-on-sweep.json"

echo ""
echo "=========================================================="
echo "Done. Reports:"
echo "  $RESULTS_DIR/gil-on-sweep.json"
echo "  $RESULTS_DIR/gil-off-sweep.json"
echo "auth-server is left in the default (GIL-off / free-threaded) state."
echo "=========================================================="
