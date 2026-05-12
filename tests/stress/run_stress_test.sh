#!/bin/bash
# Phase 1 stress-test runner: generate payloads + register against a running registry.
#
# API performance, UI performance, and report builder are not in this script yet
# (Phases 2-4 of the lld-stress-test.md plan).
#
# Usage:
#   bash tests/stress/run_stress_test.sh <backend> <size>
#
# Env vars consumed:
#   STRESS_BASE_URL   - registry URL (default: http://localhost)
#   STRESS_TOKEN_FILE - JWT token file (default: .oauth-tokens/ingress.json)
#   ANS_API_KEY / ANS_API_SECRET - required for the agents generator
#   GITHUB_TOKEN      - optional; raises GitHub API rate limit for the skills generator

set -euo pipefail

BACKEND="${1:?must pass backend (mongodb-ce|documentdb)}"
SIZE="${2:?must pass size (100|500|1000)}"

case "$BACKEND" in
  mongodb-ce|documentdb) ;;
  *) echo "Unknown backend: $BACKEND (must be mongodb-ce or documentdb)" >&2; exit 1 ;;
esac

case "$SIZE" in
  100|500|1000) ;;
  *) echo "Unknown size: $SIZE (must be 100, 500, or 1000)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

BASE_URL="${STRESS_BASE_URL:-http://localhost}"

echo "[1/3] Generating data (size=$SIZE)..."
uv run python -m tests.stress.generators.generate_servers --count "$SIZE"
uv run python -m tests.stress.generators.generate_agents --count "$SIZE"
uv run python -m tests.stress.generators.generate_skills --count "$SIZE"

echo "[2/3] Registering entities against backend=$BACKEND base_url=$BASE_URL..."
uv run python -m tests.stress.register_entities \
    --entity-type all \
    --count "$SIZE" \
    --backend "$BACKEND" \
    --base-url "$BASE_URL"

echo "[3/3] Phase 1 complete. Results at tests/stress/results/$BACKEND/size-$SIZE/registration.json"
echo "Note: API/UI performance measurement and report builder are not yet implemented (Phases 2-4)."
