#!/bin/bash
# Phase 1 stress-test runner: generate payloads + register against a running registry.
#
# API performance, UI performance, and report builder are not in this script yet
# (Phases 2-4 of the lld-stress-test.md plan).
#
# Usage:
#   bash tests/stress/run_stress_test.sh <size> [entity-type] [flags]
#
# Positional args:
#   size         100 | 500 | 1000
#   entity-type  servers | agents | skills | all   (default: all)
#
# Optional flags (override env-var defaults; flag wins over env):
#   --token-file PATH    JWT token file. Default: $STRESS_TOKEN_FILE or .token.
#   --base-url URL       Registry base URL. Default: $STRESS_BASE_URL or http://localhost.
#   --skip-generate      Skip payload generation (step 1/3), go straight to registration.
#                        Use when the data files already exist from a previous run.
#
# The storage backend (mongodb-ce, documentdb, etc.) is auto-detected from the
# registry's GET /api/stats endpoint. No need to specify it manually.
#
# `skills` is the safe single-type demo on a local stack. `servers` and `agents`
# reliably crash `mongodb-ce` under Docker; see
# .scratchpad/registry-bottleneck-findings.md, Findings 1-5.
#
# Env vars consumed (all optional):
#   STRESS_BASE_URL   - registry URL (default: http://localhost)
#   STRESS_TOKEN_FILE - JWT token file (default: .token in the repo root).
#                       Accepts two formats:
#                         (a) the nested JSON shape produced by the registry UI's
#                             "Get JWT Token" button, one of:
#                                 {"access_token": "..."}
#                                 {"tokens": {"access_token": "..."}}
#                                 {"token_data": {"access_token": "..."}}
#                         (b) plain-text: the file contains nothing but the
#                             raw JWT string.
#                       This script does NOT auto-generate tokens. Get one from
#                       the registry UI, save it to .token (or any path you set
#                       via STRESS_TOKEN_FILE), and re-run.
#   STRESS_MEASURE_API - set to any non-empty value to chain Phase 2
#                       (API performance measurement) after Phase 1.
#                       Default: skipped.
#   STRESS_MEASURE_ITERATIONS - iterations per API operation when
#                       STRESS_MEASURE_API is set. Default: 50.
#   ANS_API_KEY / ANS_API_SECRET - required for the agents generator
#   GITHUB_TOKEN / GITHUB_PAT - optional; raises GitHub API rate limit for the
#                       skills generator. Either name works.

set -euo pipefail

SIZE="${1:?must pass size (100|500|1000)}"
ENTITY_TYPE="all"

# Optional 2nd positional: entity-type (servers|agents|skills|all).
# We only consume it as positional when it doesn't start with `--`, otherwise
# treat the rest of argv as flags.
shift 1
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  ENTITY_TYPE="$1"
  shift
fi

# Flag overrides for token file and base URL. CLI takes precedence over env.
TOKEN_FILE_FLAG=""
BASE_URL_FLAG=""
SKIP_GENERATE=false
while [ $# -gt 0 ]; do
  case "$1" in
    --token-file)
      TOKEN_FILE_FLAG="${2:?--token-file requires a path}"
      shift 2
      ;;
    --token-file=*)
      TOKEN_FILE_FLAG="${1#--token-file=}"
      shift
      ;;
    --base-url)
      BASE_URL_FLAG="${2:?--base-url requires a URL}"
      shift 2
      ;;
    --base-url=*)
      BASE_URL_FLAG="${1#--base-url=}"
      shift
      ;;
    --skip-generate)
      SKIP_GENERATE=true
      shift
      ;;
    *)
      echo "Unknown flag: $1" >&2
      echo "Supported flags: --token-file PATH, --base-url URL" >&2
      exit 1
      ;;
  esac
done

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

# Source .env so generators pick up ANS_API_KEY, GITHUB_PAT, etc.
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Precedence: --base-url flag > $STRESS_BASE_URL > default.
BASE_URL="${BASE_URL_FLAG:-${STRESS_BASE_URL:-http://localhost}}"

# ---------------------------------------------------------------------------
# Token resolution: confirm the user-supplied (or default) token file exists
# and parses. We do NOT auto-generate tokens here -- the user is expected to
# grab one from the registry UI's "Get JWT Token" button.
# ---------------------------------------------------------------------------

# Precedence: --token-file flag > $STRESS_TOKEN_FILE > default (.token).
TOKEN_FILE="${TOKEN_FILE_FLAG:-${STRESS_TOKEN_FILE:-.token}}"

_token_resolution_help() {
  cat >&2 <<EOF
Get a JWT for the stress test by either:
  1. Open the registry UI, click "Get JWT Token", and save the downloaded
     JSON file as .token in the repo root, OR
  2. Save the raw JWT string (just the eyJ... token, nothing else) to .token.

Both formats are accepted. Override the path via STRESS_TOKEN_FILE if you
want to keep the file elsewhere.
EOF
}

if [ ! -f "$TOKEN_FILE" ]; then
  echo "Token file not found: $TOKEN_FILE" >&2
  _token_resolution_help
  exit 1
fi

# Validate that the file contains either a parseable JSON object with an
# access_token, OR a non-empty plain-text token. This mirrors the loader's
# accepted formats so we fail fast with a clear message instead of waiting
# for a 401 from the registry.
if ! python3 - "$TOKEN_FILE" <<'PY'
import json, sys
path = sys.argv[1]
raw = open(path).read()
try:
    data = json.loads(raw)
except Exception:
    if raw.strip():
        sys.exit(0)
    print(f"Token file is empty: {path}", file=sys.stderr)
    sys.exit(1)
token = (
    data.get("access_token")
    or data.get("tokens", {}).get("access_token")
    or data.get("token_data", {}).get("access_token")
)
if not token:
    print(
        f"Token file {path} is JSON but has no access_token field "
        "(checked: access_token, tokens.access_token, token_data.access_token)",
        file=sys.stderr,
    )
    sys.exit(1)
PY
then
  _token_resolution_help
  exit 1
fi

echo "Using JWT token file: $TOKEN_FILE"

# ---------------------------------------------------------------------------
# Auto-detect backend from the registry's GET /api/stats endpoint.
# ---------------------------------------------------------------------------

# Extract the raw JWT string from the token file (handles both JSON and plain text).
JWT_TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
import json, sys
path = sys.argv[1]
raw = open(path).read()
try:
    data = json.loads(raw)
except Exception:
    print(raw.strip())
    sys.exit(0)
token = (
    data.get("access_token")
    or data.get("tokens", {}).get("access_token")
    or data.get("token_data", {}).get("access_token")
)
print(token)
PY
)

echo "Detecting storage backend from $BASE_URL/api/stats..."
STATS_RESPONSE=$(curl -sf -H "Authorization: Bearer $JWT_TOKEN" "$BASE_URL/api/stats" 2>&1) || {
  echo "Failed to reach $BASE_URL/api/stats. Is the registry running?" >&2
  echo "Response: $STATS_RESPONSE" >&2
  exit 1
}

BACKEND=$(echo "$STATS_RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['database_status']['backend'])")
if [ -z "$BACKEND" ]; then
  echo "Could not detect backend from /api/stats response." >&2
  exit 1
fi
echo "Detected backend: $BACKEND"

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------

if [ "$SKIP_GENERATE" = true ]; then
  echo "[1/3] Skipping generation (--skip-generate). Using existing data files."
else
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
fi

echo "[2/3] Registering entities against backend=$BACKEND base_url=$BASE_URL..."
uv run python -m tests.stress.register_entities \
    --entity-type "$ENTITY_TYPE" \
    --count "$SIZE" \
    --backend "$BACKEND" \
    --base-url "$BASE_URL" \
    --token-file "$TOKEN_FILE"

echo "[3/3] Phase 1 complete. Results at tests/stress/results/$BACKEND/size-$SIZE/registration.json"

if [ -n "${STRESS_MEASURE_API:-}" ]; then
  ITERATIONS="${STRESS_MEASURE_ITERATIONS:-50}"
  echo
  echo "[Phase 2] Measuring API performance (iterations=$ITERATIONS)..."
  uv run python -m tests.stress.measure_api_performance \
      --backend "$BACKEND" \
      --size "$SIZE" \
      --base-url "$BASE_URL" \
      --iterations "$ITERATIONS" \
      --token-file "$TOKEN_FILE"
  echo "[Phase 2] Complete. Results at tests/stress/results/$BACKEND/size-$SIZE/api_perf.{json,md}"
else
  echo "Note: set STRESS_MEASURE_API=1 to also run Phase 2 (API performance) after registration."
  echo "      UI performance measurement and report builder are not yet implemented (Phases 3-4)."
fi
