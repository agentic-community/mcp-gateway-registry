#!/bin/bash
# Phase 1 stress-test runner: generate payloads + register against a running registry.
#
# API performance, UI performance, and report builder are not in this script yet
# (Phases 2-4 of the lld-stress-test.md plan).
#
# Usage:
#   bash tests/stress/run_stress_test.sh <backend> <size> [entity-type]
#
# entity-type defaults to "all". Pass `servers`, `agents`, or `skills` to scope
# the run to a single type. Only that type's generator and registration step
# run, which is the smallest-possible self-contained demo on a local stack
# (`skills` is the safe bet -- the other two reliably crash `mongodb-ce` under
# Docker; see .scratchpad/registry-bottleneck-findings.md, Findings 1-5).
#
# Env vars consumed (all optional):
#   STRESS_BASE_URL   - registry URL (default: http://localhost)
#   STRESS_TOKEN_FILE - JWT token file. When unset, the script auto-detects an
#                       existing file under .oauth-tokens/ and regenerates one
#                       via keycloak/setup/generate-agent-token.sh if none is
#                       found or all candidates are expired.
#   STRESS_SKIP_TOKEN_REFRESH - set to any non-empty value to disable the
#                       auto-regeneration step (use the detected file as-is).
#   ANS_API_KEY / ANS_API_SECRET - required for the agents generator
#   GITHUB_TOKEN      - optional; raises GitHub API rate limit for the skills generator

set -euo pipefail

BACKEND="${1:?must pass backend (mongodb-ce|documentdb)}"
SIZE="${2:?must pass size (100|500|1000)}"
ENTITY_TYPE="${3:-all}"

case "$BACKEND" in
  mongodb-ce|documentdb) ;;
  *) echo "Unknown backend: $BACKEND (must be mongodb-ce or documentdb)" >&2; exit 1 ;;
esac

case "$SIZE" in
  100|500|1000) ;;
  *) echo "Unknown size: $SIZE (must be 100, 500, or 1000)" >&2; exit 1 ;;
esac

case "$ENTITY_TYPE" in
  servers|agents|skills|all) ;;
  *) echo "Unknown entity-type: $ENTITY_TYPE (must be servers|agents|skills|all)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

BASE_URL="${STRESS_BASE_URL:-http://localhost}"

# ---------------------------------------------------------------------------
# Token resolution: pick an existing JWT file or generate one.
# ---------------------------------------------------------------------------

# Returns 0 if the JSON token file's "expires_at" is in the future (or absent).
_token_file_valid() {
  local file="$1"
  [ -f "$file" ] || return 1
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$file" <<'PY' || return 1
import json, sys
from datetime import datetime, timezone
try:
    data = json.loads(open(sys.argv[1]).read())
except Exception:
    sys.exit(0)  # opaque file: assume valid, let the loader's 401 path handle it
expires_at = data.get("expires_at")
if not expires_at:
    sys.exit(0)
try:
    exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
except Exception:
    sys.exit(0)
sys.exit(0 if exp > datetime.now(timezone.utc) else 1)
PY
  fi
  return 0
}

_resolve_token_file() {
  if [ -n "${STRESS_TOKEN_FILE:-}" ]; then
    if _token_file_valid "$STRESS_TOKEN_FILE"; then
      echo "$STRESS_TOKEN_FILE"
      return 0
    fi
    echo "STRESS_TOKEN_FILE=$STRESS_TOKEN_FILE is missing or expired." >&2
  fi

  local candidates=(
    ".oauth-tokens/ingress.json"
    ".oauth-tokens/mcp-gateway-m2m-token.json"
  )
  for candidate in "${candidates[@]}"; do
    if _token_file_valid "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done

  if [ -n "${STRESS_SKIP_TOKEN_REFRESH:-}" ]; then
    echo "No valid token file found and STRESS_SKIP_TOKEN_REFRESH is set." >&2
    return 1
  fi

  local generator="keycloak/setup/generate-agent-token.sh"
  if [ ! -x "$generator" ] && [ ! -f "$generator" ]; then
    echo "No valid token file and $generator is not available." >&2
    return 1
  fi

  echo "No valid JWT token found; regenerating via $generator..." >&2
  bash "$generator" >&2
  local generated=".oauth-tokens/mcp-gateway-m2m-token.json"
  if _token_file_valid "$generated"; then
    echo "$generated"
    return 0
  fi
  echo "Token regeneration did not produce a valid file at $generated." >&2
  return 1
}

TOKEN_FILE="$(_resolve_token_file)"
echo "Using JWT token file: $TOKEN_FILE"

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

echo "[1/3] Generating data (size=$SIZE, entity-type=$ENTITY_TYPE)..."
if [ "$ENTITY_TYPE" = "all" ] || [ "$ENTITY_TYPE" = "servers" ]; then
  uv run python -m tests.stress.generators.generate_servers --count "$SIZE"
fi
if [ "$ENTITY_TYPE" = "all" ] || [ "$ENTITY_TYPE" = "agents" ]; then
  uv run python -m tests.stress.generators.generate_agents --count "$SIZE"
fi
if [ "$ENTITY_TYPE" = "all" ] || [ "$ENTITY_TYPE" = "skills" ]; then
  uv run python -m tests.stress.generators.generate_skills --count "$SIZE"
fi

echo "[2/3] Registering entities against backend=$BACKEND base_url=$BASE_URL..."
uv run python -m tests.stress.register_entities \
    --entity-type "$ENTITY_TYPE" \
    --count "$SIZE" \
    --backend "$BACKEND" \
    --base-url "$BASE_URL" \
    --token-file "$TOKEN_FILE"

echo "[3/3] Phase 1 complete. Results at tests/stress/results/$BACKEND/size-$SIZE/registration.json"
echo "Note: API/UI performance measurement and report builder are not yet implemented (Phases 2-4)."
