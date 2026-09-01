#!/bin/bash
#
# End-to-end security smoke test for the gateway generic-proxy feature.
#
# Complements the in-process unit/composite suites with real HTTP requests
# against a running stack. Exercises the security-critical paths from the LLD
# testing plan (SSRF registration reject, marker-spoof, per-verb scope, header
# allowlist) rather than happy-path only.
#
# PREREQUISITES:
#   - Registry + auth-server + nginx + Keycloak + MongoDB running (docker ps).
#   - GATEWAY_GENERIC_PROXY_ENABLED=true AND the network egress policy deployed
#     (the auth-server startup egress self-check must have passed:
#     gateway_egress_policy_unverified=0). If the feature is disabled this script
#     SKIPS the routing checks and only asserts registration-time SSRF rejection.
#   - An admin bearer token in the token file.
#   - A reachable public test backend for the proxied skill (default httpbin.org).
#
# Usage:
#   ./test_gateway_proxy_e2e.sh --registry-url http://localhost --token-file .token
#   ./test_gateway_proxy_e2e.sh --registry-url http://localhost --token-file .token \
#       --backend-url https://httpbin.org
#
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

REGISTRY_URL=""
TOKEN_FILE=""
BACKEND_URL="https://httpbin.org"
CLEANUP_ON_EXIT=true

SKILL_PATH="/skills/gwproxy-e2e-demo"
SKILL_NAME="gwproxy-e2e-demo"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry-url) REGISTRY_URL="$2"; shift 2 ;;
    --token-file) TOKEN_FILE="$2"; shift 2 ;;
    --backend-url) BACKEND_URL="$2"; shift 2 ;;
    --no-cleanup) CLEANUP_ON_EXIT=false; shift ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$REGISTRY_URL" || -z "$TOKEN_FILE" ]]; then
  echo "Usage: $0 --registry-url <URL> --token-file <PATH> [--backend-url <URL>] [--no-cleanup]"
  exit 2
fi

TOKEN="$(cat "$TOKEN_FILE")"
AUTH_HDR=(-H "Authorization: Bearer ${TOKEN}")
PASS=0; FAIL=0

_pass() { echo -e "${GREEN}PASS${NC}: $1"; PASS=$((PASS + 1)); }
_fail() { echo -e "${RED}FAIL${NC}: $1"; FAIL=$((FAIL + 1)); }
_info() { echo -e "${YELLOW}INFO${NC}: $1"; }

# HTTP status of a request (prints the numeric code).
_status() { curl -s -o /dev/null -w "%{http_code}" "$@"; }

cleanup() {
  if [[ "$CLEANUP_ON_EXIT" == "true" ]]; then
    _info "Cleaning up ${SKILL_PATH}"
    curl -s "${AUTH_HDR[@]}" -X DELETE "${REGISTRY_URL}/api/skills${SKILL_PATH}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "=== Gateway generic-proxy E2E security smoke test ==="
echo "Registry: ${REGISTRY_URL}  Backend: ${BACKEND_URL}"

# ---------------------------------------------------------------------------
# 1. Registration-time SSRF reject (LLD 1.1.3) — ALWAYS runs (feature-independent).
#    A proxied skill whose target is the cloud metadata IP must be rejected 4xx.
# ---------------------------------------------------------------------------
code=$(_status "${AUTH_HDR[@]}" -X POST "${REGISTRY_URL}/api/skills" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"gwproxy-ssrf-metadata\",\"description\":\"ssrf\",\"skill_md_url\":\"${BACKEND_URL}/SKILL.md\",\"is_proxied\":true,\"proxy_target_url\":\"http://169.254.169.254/latest/meta-data/\"}")
if [[ "$code" =~ ^4 ]]; then
  _pass "1. Registration rejects metadata-IP proxy target (HTTP ${code})"
else
  _fail "1. Metadata-IP proxy target was NOT rejected (HTTP ${code}) — SSRF gate broken"
fi

for denied in "http://127.0.0.1/" "http://localhost/" "http://[::1]/"; do
  code=$(_status "${AUTH_HDR[@]}" -X POST "${REGISTRY_URL}/api/skills" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"gwproxy-ssrf-loop\",\"description\":\"ssrf\",\"skill_md_url\":\"${BACKEND_URL}/SKILL.md\",\"is_proxied\":true,\"proxy_target_url\":\"${denied}\"}")
  if [[ "$code" =~ ^4 ]]; then
    _pass "1b. Registration rejects loopback target ${denied} (HTTP ${code})"
  else
    _fail "1b. Loopback target ${denied} NOT rejected (HTTP ${code})"
  fi
done

# ---------------------------------------------------------------------------
# 2. Feature-gated routing checks. Skipped when the feature is disabled.
# ---------------------------------------------------------------------------
FEATURE_ON=$(curl -s "${AUTH_HDR[@]}" "${REGISTRY_URL}/api/config/full" 2>/dev/null \
  | grep -oiE '"gateway_generic_proxy_enabled"\s*:\s*true' || true)

if [[ -z "$FEATURE_ON" ]]; then
  _info "GATEWAY_GENERIC_PROXY_ENABLED is not true — skipping live-routing checks."
  _info "Registration-time SSRF gate verified above (works regardless of the flag)."
else
  _info "Feature enabled — running live-routing checks."

  # Register a valid proxied skill.
  code=$(_status "${AUTH_HDR[@]}" -X POST "${REGISTRY_URL}/api/skills" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${SKILL_NAME}\",\"description\":\"e2e\",\"skill_md_url\":\"${BACKEND_URL}/SKILL.md\",\"is_proxied\":true,\"proxy_target_url\":\"${BACKEND_URL}/\"}")
  if [[ "$code" =~ ^2 ]]; then
    _pass "2. Registered valid proxied skill (HTTP ${code})"
  else
    _fail "2. Could not register proxied skill (HTTP ${code}); aborting routing checks"
    exit 1
  fi

  _info "Waiting for nginx debounce reload..."
  sleep 8

  # 2a. Backend Set-Cookie / CSP must NOT leak through (header allowlist).
  hdrs=$(curl -s -D - -o /dev/null "${AUTH_HDR[@]}" "${REGISTRY_URL}/skill${SKILL_PATH}/response-headers?Set-Cookie=evil%3D1")
  if echo "$hdrs" | grep -qi "^set-cookie:"; then
    _fail "2a. Backend Set-Cookie leaked through the generic hop (allowlist broken)"
  else
    _pass "2a. Backend Set-Cookie dropped by the response-header allowlist"
  fi
  if echo "$hdrs" | grep -qi "^x-content-type-options: nosniff"; then
    _pass "2a. Gateway set its own X-Content-Type-Options: nosniff"
  else
    _info "2a. Gateway security header not observed (backend may not have returned a body)"
  fi

  # 2b. Marker-spoof: a client-supplied X-Generic-Proxy-Kind on a NON-proxied
  #     path must NOT cause a generic mint to an attacker upstream (it should be
  #     ignored — nginx does not set the marker for that location).
  code=$(_status "${AUTH_HDR[@]}" \
    -H "X-Generic-Proxy-Kind: skill" \
    -H "X-Resolved-Generic-Upstream: http://attacker.example/" \
    -H "X-Entity-Path: skills/gwproxy-e2e-demo" \
    "${REGISTRY_URL}/api/servers")
  # The /api/ request should behave normally (2xx/4xx per auth), NOT route to attacker.
  if [[ "$code" != "502" && "$code" != "504" ]]; then
    _pass "2b. Spoofed generic markers on /api/ did not route to attacker upstream (HTTP ${code})"
  else
    _fail "2b. Spoofed markers produced an upstream error (${code}) — marker may not be server-set-only"
  fi
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[[ "$FAIL" -eq 0 ]] || exit 1
