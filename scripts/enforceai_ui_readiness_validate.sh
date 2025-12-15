#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${ENFORCEAI_UI_BASE_URL:-http://localhost}"
ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-your-secure-password-here}"

REQUIREMENTS=(
  "curl"
  "jq"
)

for dep in "${REQUIREMENTS[@]}"; do
  if ! command -v "$dep" >/dev/null 2>&1; then
    echo "Missing dependency: $dep" >&2
    exit 1
  fi
done

COOKIE_JAR="$(mktemp)"
HEADERS_FILE="$(mktemp)"

cleanup() {
  rm -f "$COOKIE_JAR" "$HEADERS_FILE"
}
trap cleanup EXIT

_request_id() {
  date +%s%N
}

_curl_json() {
  local method="$1"
  local url="$2"
  shift 2

  curl -sS \
    -X "$method" \
    -H "Accept: application/json" \
    -H "X-Request-Id: $(_request_id)" \
    "$@" \
    "$url"
}

_expect_http_code() {
  local expected="$1"
  local method="$2"
  local url="$3"
  shift 3

  local body_file
  body_file="$(mktemp)"
  local code
  code="$(
    curl -sS -o "$body_file" -w "%{http_code}" \
      -X "$method" \
      -H "Accept: application/json" \
      -H "X-Request-Id: $(_request_id)" \
      "$@" \
      "$url"
  )"

  if [ "$code" != "$expected" ]; then
    echo "Expected HTTP $expected for $method $url, got $code" >&2
    echo "Response body (first 400 chars):" >&2
    head -c 400 "$body_file" >&2 || true
    echo >&2
    rm -f "$body_file"
    exit 1
  fi

  rm -f "$body_file"
}

echo "Validating Enforce Gateway UI readiness against: $BASE_URL"

echo
echo "1) Base health"
_expect_http_code "200" "GET" "$BASE_URL/health"

echo
echo "1.5) OpenAPI reachability (Registry + EnforceAI)"
_expect_http_code "200" "GET" "$BASE_URL/openapi.json"
_expect_http_code "200" "GET" "$BASE_URL/enforceai/openapi.json"

echo
echo "2) Login (cookie session via Registry)"
LOGIN_BODY_FILE="$(mktemp)"
LOGIN_JSON="$(jq -n --arg u "$ADMIN_USER" --arg p "$ADMIN_PASSWORD" '{username:$u,password:$p}')"
LOGIN_CODE="$(
  curl -sS -D "$HEADERS_FILE" -o "$LOGIN_BODY_FILE" -w "%{http_code}" \
    -X POST \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -H "X-Request-Id: $(_request_id)" \
    -c "$COOKIE_JAR" \
    "$BASE_URL/api/auth/login" \
    -d "$LOGIN_JSON"
)"
if [ "$LOGIN_CODE" != "200" ]; then
  echo "ERROR: login failed with HTTP $LOGIN_CODE" >&2
  echo "Response body (first 400 chars):" >&2
  head -c 400 "$LOGIN_BODY_FILE" >&2 || true
  echo >&2
  rm -f "$LOGIN_BODY_FILE"
  exit 1
fi
rm -f "$LOGIN_BODY_FILE"

LOGIN_SET_COOKIE="$(awk 'tolower($1)=="set-cookie:" {print substr($0, index($0,$2))}' "$HEADERS_FILE" | head -n 1 || true)"
if [ -z "$LOGIN_SET_COOKIE" ]; then
  echo "ERROR: login did not return Set-Cookie" >&2
  exit 1
fi

COOKIE_ATTRS="$(
  echo "$LOGIN_SET_COOKIE" \
    | sed -E 's/^[^;]+;[[:space:]]*//' \
    | tr ';' '\n' \
    | sed -E 's/^[[:space:]]+//'
)"
echo "Set-Cookie attributes:"
echo "$COOKIE_ATTRS" | sed 's/^/  - /'

echo
echo "3) Session contract (/api/auth/me)"
ME_JSON="$(_curl_json GET "$BASE_URL/api/auth/me" -b "$COOKIE_JAR")"
echo "$ME_JSON" | jq -e '.user_id and .session_id and (.groups | type=="array")' >/dev/null
echo "me.user_id=$(echo "$ME_JSON" | jq -r '.user_id')"
echo "me.session_id=$(echo "$ME_JSON" | jq -r '.session_id')"

echo
echo "4) CSRF token (/api/auth/csrf)"
CSRF_JSON="$(_curl_json GET "$BASE_URL/api/auth/csrf" -b "$COOKIE_JAR")"
CSRF_TOKEN="$(echo "$CSRF_JSON" | jq -r '.csrf_token')"
if [ -z "$CSRF_TOKEN" ] || [ "$CSRF_TOKEN" = "null" ]; then
  echo "ERROR: missing csrf_token in /api/auth/csrf response" >&2
  exit 1
fi

echo "csrf_token acquired"

echo
echo "4.5) Registry UI reads (/api/servers, /api/agents) via cookie"
SERVERS_JSON="$(_curl_json GET "$BASE_URL/api/servers" -b "$COOKIE_JAR")"
echo "$SERVERS_JSON" | jq -e '.servers | type=="array"' >/dev/null
echo "servers.count=$(echo "$SERVERS_JSON" | jq '.servers | length')"

AGENTS_JSON="$(_curl_json GET "$BASE_URL/api/agents" -b "$COOKIE_JAR")"
echo "$AGENTS_JSON" | jq -e '.agents | type=="array"' >/dev/null
echo "agents.count=$(echo "$AGENTS_JSON" | jq '.agents | length')"

echo
echo "5) EnforceAI admin reachability (/enforceai/admin/ping) via cookie"
_expect_http_code "200" "GET" "$BASE_URL/enforceai/admin/ping" -b "$COOKIE_JAR"

echo
echo "6) EnforceAI management read (/enforceai/agents) via cookie"
_expect_http_code "200" "GET" "$BASE_URL/enforceai/agents" -b "$COOKIE_JAR"

echo
echo "7) EnforceAI management write requires CSRF (create agent)"
AGENT_ALIAS="ui-readiness-$(date +%s)"
CREATE_AGENT_BODY="$(jq -n --arg alias "$AGENT_ALIAS" '{scopes:["registry-users-lob1"], alias:$alias}')"

_expect_http_code "403" "POST" "$BASE_URL/enforceai/agents" \
  -b "$COOKIE_JAR" \
  -H "Content-Type: application/json" \
  -d "$CREATE_AGENT_BODY"

AGENT_JSON="$(_curl_json POST "$BASE_URL/enforceai/agents" \
  -b "$COOKIE_JAR" \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $CSRF_TOKEN" \
  -d "$CREATE_AGENT_BODY")"
AGENT_ID="$(echo "$AGENT_JSON" | jq -r '.agent_id')"
if [ -z "$AGENT_ID" ] || [ "$AGENT_ID" = "null" ]; then
  echo "ERROR: missing agent_id in create agent response" >&2
  exit 1
fi
echo "created agent_id=$AGENT_ID"

echo
echo "8) Mint gateway token via cookie + CSRF"
MINT_BODY="$(jq -n '{scopes:["registry-users-lob1"], ttl_seconds: 3600}')"
MINT_JSON="$(_curl_json POST "$BASE_URL/enforceai/agents/$AGENT_ID/tokens/mint" \
  -b "$COOKIE_JAR" \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $CSRF_TOKEN" \
  -d "$MINT_BODY")"
GATEWAY_TOKEN="$(echo "$MINT_JSON" | jq -r '.token')"
if [ -z "$GATEWAY_TOKEN" ] || [ "$GATEWAY_TOKEN" = "null" ]; then
  echo "ERROR: missing token in mint response" >&2
  exit 1
fi
echo "minted gateway token (redacted)"

echo
echo "9) Create API key via cookie + CSRF"
CREATE_API_KEY_BODY="$(jq -n '{scopes:["registry-users-lob1"]}')"
API_KEY_BODY_FILE="$(mktemp)"
API_KEY_CODE="$(
  curl -sS -o "$API_KEY_BODY_FILE" -w "%{http_code}" \
    -X POST \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -H "X-Request-Id: $(_request_id)" \
    -H "X-CSRF-Token: $CSRF_TOKEN" \
    -b "$COOKIE_JAR" \
    -d "$CREATE_API_KEY_BODY" \
    "$BASE_URL/enforceai/agents/$AGENT_ID/api-keys"
)"
if [ "$API_KEY_CODE" != "200" ]; then
  echo "ERROR: create api key failed with HTTP $API_KEY_CODE" >&2
  echo "Response body (first 400 chars):" >&2
  head -c 400 "$API_KEY_BODY_FILE" >&2 || true
  echo >&2
  rm -f "$API_KEY_BODY_FILE"
  exit 1
fi
API_KEY_JSON="$(cat "$API_KEY_BODY_FILE")"
rm -f "$API_KEY_BODY_FILE"
API_KEY_ID="$(echo "$API_KEY_JSON" | jq -r '.key_id')"
API_KEY_VALUE="$(echo "$API_KEY_JSON" | jq -r '.api_key_value')"
if [ -z "$API_KEY_ID" ] || [ "$API_KEY_ID" = "null" ] || [ -z "$API_KEY_VALUE" ] || [ "$API_KEY_VALUE" = "null" ]; then
  echo "ERROR: missing API key fields in create response" >&2
  exit 1
fi
echo "created api_key_id=$API_KEY_ID api_key_value_len=${#API_KEY_VALUE}"

echo
echo "10) API key works against management reads"
_expect_http_code "200" "GET" "$BASE_URL/enforceai/agents" -H "X-API-Key: $API_KEY_VALUE"

echo
echo "11) API key works against gateway MCP (initialize + tools/list)"
SID="$(
  curl -sS -D - -o /dev/null "$BASE_URL/mcpgw/mcp" -X POST \
    -H "Content-Type: application/json" \
    -H "X-Request-Id: $(_request_id)" \
    -H "X-API-Key: $API_KEY_VALUE" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"ui-readiness","version":"1"}}}' \
    | awk 'tolower($1)=="mcp-session-id:" {print $2}' | tr -d "\r"
)"
if [ -z "$SID" ]; then
  echo "ERROR: missing mcp-session-id header from gateway" >&2
  exit 1
fi

INIT_DATA="$(
  curl -sS "$BASE_URL/mcpgw/mcp" -X POST \
    -H "Content-Type: application/json" \
    -H "X-Request-Id: $(_request_id)" \
    -H "X-API-Key: $API_KEY_VALUE" \
    -H "mcp-session-id: $SID" \
    -d '{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"ui-readiness","version":"1"}}}' \
    | sed -n 's/^data: //p' | head -n 1
)"
echo "$INIT_DATA" | jq -e '.result.protocolVersion' >/dev/null

TOOLS_DATA="$(
  curl -sS "$BASE_URL/mcpgw/mcp" -X POST \
    -H "Content-Type: application/json" \
    -H "X-Request-Id: $(_request_id)" \
    -H "X-API-Key: $API_KEY_VALUE" \
    -H "mcp-session-id: $SID" \
    -d '{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}' \
    | sed -n 's/^data: //p' | head -n 1
)"
TOOLS_COUNT="$(echo "$TOOLS_DATA" | jq '.result.tools | length')"
if [ "$TOOLS_COUNT" -lt 1 ]; then
  echo "ERROR: expected at least 1 tool, got $TOOLS_COUNT" >&2
  exit 1
fi
echo "tools/list ok (count=$TOOLS_COUNT)"

echo
echo "12) Revoke API key via cookie + CSRF"
_expect_http_code "200" "POST" "$BASE_URL/enforceai/api-keys/$API_KEY_ID/revoke" \
  -b "$COOKIE_JAR" \
  -H "X-CSRF-Token: $CSRF_TOKEN"

echo
echo "13) Revoked API key is rejected"
_expect_http_code "403" "GET" "$BASE_URL/enforceai/agents" -H "X-API-Key: $API_KEY_VALUE"
_expect_http_code "403" "POST" "$BASE_URL/mcpgw/mcp" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY_VALUE" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/list","params":{}}'

echo
echo "14) Registry internal admin actions via cookie"
_expect_http_code "200" "GET" "$BASE_URL/api/internal/list" -b "$COOKIE_JAR"

echo
echo "15) Logout invalidates session across /api and /enforceai"
_expect_http_code "303" "POST" "$BASE_URL/api/auth/logout" -b "$COOKIE_JAR" -H "X-CSRF-Token: $CSRF_TOKEN"
_expect_http_code "401" "GET" "$BASE_URL/api/auth/me" -b "$COOKIE_JAR"
_expect_http_code "401" "GET" "$BASE_URL/enforceai/agents" -b "$COOKIE_JAR"

echo
echo "UI readiness validation OK"
