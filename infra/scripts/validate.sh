#!/bin/bash
# End-to-end validation of a deployed MCP Gateway Registry. Run after
# ./deploy.sh (or post-deploy.sh) to confirm the stack is healthy and the
# admin user can register/manage agents.
#
# Exit codes:
#   0 — all checks passed
#   1 — at least one check failed (details printed)
#
# Tests cleanup after themselves (registers + deletes a test agent).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/_lib.sh"
[ -f "$SCRIPT_DIR/set-env.sh" ] && source "$SCRIPT_DIR/set-env.sh"
AWS_REGION="${AWS_REGION:-us-east-1}"

OUTPUTS="$INFRA_DIR/cdk-outputs.json"
[ -f "$OUTPUTS" ] || { _log_error "$OUTPUTS not found — deploy first"; exit 1; }

REGISTRY_URL=$(jq -r '."Registry-Service".RegistryUrl // empty' "$OUTPUTS")
KEYCLOAK_URL=$(jq -r '."Registry-Service".KeycloakUrl // ."Registry-Auth".KeycloakUrl // empty' "$OUTPUTS")
[ -n "$REGISTRY_URL" ] && [ -n "$KEYCLOAK_URL" ] \
  || { _log_error "Could not read URLs from $OUTPUTS"; exit 1; }

KC_PW="${CDK_KEYCLOAK_ADMIN_PASSWORD:-}"
[ -n "$KC_PW" ] || { _log_error "CDK_KEYCLOAK_ADMIN_PASSWORD not set"; exit 1; }

PASS=0 FAIL=0
_check() {
  local label="$1" exit_code="$2" detail="${3:-}"
  if [ "$exit_code" -eq 0 ]; then
    echo -e "  ${GREEN}[PASS]${NC} $label"
    PASS=$((PASS + 1))
  else
    echo -e "  ${RED}[FAIL]${NC} $label${detail:+ — $detail}"
    FAIL=$((FAIL + 1))
  fi
}

# Probe an HTTP endpoint; pass if status is in the allow-list (default 200).
_http() {
  local label="$1" url="$2" allow="${3:-200}"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
  if echo " $allow " | grep -q " $code "; then
    _check "$label ($code)" 0
  else
    _check "$label" 1 "got $code, want $allow"
  fi
}

echo
echo "============================================"
echo "  MCP Gateway Registry — validation"
echo "============================================"
_log_info "Registry: $REGISTRY_URL"
_log_info "Keycloak: $KEYCLOAK_URL"

# ----------------------------------------------------------------------
# 1. Stack-level: all CFN stacks COMPLETE
# ----------------------------------------------------------------------
echo
_log_info "Stacks"
bad_stacks=$(aws cloudformation list-stacks --region "$AWS_REGION" \
  --query 'StackSummaries[?starts_with(StackName, `Registry-`) && StackStatus != `CREATE_COMPLETE` && StackStatus != `UPDATE_COMPLETE` && StackStatus != `DELETE_COMPLETE`].StackName' \
  --output text 2>/dev/null)
_check "All Registry-* stacks COMPLETE" "$([ -z "$bad_stacks" ] && echo 0 || echo 1)" "$bad_stacks"

# ----------------------------------------------------------------------
# 2. ECS services: desired==running, rollout completed
# ----------------------------------------------------------------------
echo
_log_info "ECS services"
for svc in mcp-gateway-registry mcp-gateway-auth-server; do
  status=$(aws ecs describe-services --region "$AWS_REGION" \
    --cluster mcp-gateway-ecs-cluster --services "$svc" \
    --query 'services[0].[runningCount,desiredCount,deployments[0].rolloutState]' \
    --output text 2>/dev/null)
  read -r running desired rollout <<< "$status"
  if [ "$running" = "$desired" ] && [ "$rollout" = "COMPLETED" ]; then
    _check "$svc ($running/$desired, $rollout)" 0
  else
    _check "$svc" 1 "$running/$desired $rollout"
  fi
done
kc_status=$(aws ecs describe-services --region "$AWS_REGION" \
  --cluster keycloak --services keycloak \
  --query 'services[0].[runningCount,desiredCount,deployments[0].rolloutState]' \
  --output text 2>/dev/null)
read -r kr kd kroll <<< "$kc_status"
if [ "$kr" = "$kd" ] && [ "$kroll" = "COMPLETED" ]; then
  _check "keycloak ($kr/$kd, $kroll)" 0
else
  _check "keycloak" 1 "$kr/$kd $kroll"
fi

# ----------------------------------------------------------------------
# 3. HTTP endpoints
# ----------------------------------------------------------------------
echo
_log_info "HTTP endpoints"
_http "Registry /health" "$REGISTRY_URL/health"
_http "Auth-server /health" "$REGISTRY_URL:8888/health"
_http "Keycloak /" "$KEYCLOAK_URL/" "200 302 303"
_http "Keycloak realm OIDC" "$KEYCLOAK_URL/realms/mcp-gateway/.well-known/openid-configuration"
_http ".well-known/ai-catalog" "$REGISTRY_URL/.well-known/ai-catalog.json"
_http ".well-known/oauth-auth-server" "$REGISTRY_URL/.well-known/oauth-authorization-server"

# ----------------------------------------------------------------------
# 4. Keycloak admin OAuth + token claims
# ----------------------------------------------------------------------
echo
_log_info "Auth"
WEB_SECRET=$(aws secretsmanager get-secret-value --region "$AWS_REGION" \
  --secret-id mcp-gateway-keycloak-client-secret --query SecretString --output text 2>/dev/null \
  | jq -r '.client_secret // empty')
if [ -z "$WEB_SECRET" ] || [ "$WEB_SECRET" = "placeholder-will-be-updated-by-init-script" ]; then
  _check "mcp-gateway-web client secret in SM" 1 "secret missing/placeholder"
else
  _check "mcp-gateway-web client secret in SM" 0
fi

TOKEN=$(curl -s -X POST "$KEYCLOAK_URL/realms/mcp-gateway/protocol/openid-connect/token" \
  -d "username=admin&password=$KC_PW&grant_type=password&client_id=mcp-gateway-web&client_secret=$WEB_SECRET&scope=openid" \
  | jq -r '.access_token // empty')
[ -n "$TOKEN" ] && _check "Admin token issued" 0 || { _check "Admin token issued" 1; FAIL=$((FAIL+1)); }

if [ -n "$TOKEN" ]; then
  # Decode JWT payload (base64-url, padded)
  payload=$(echo "$TOKEN" | cut -d. -f2)
  pad=$(( 4 - ${#payload} % 4 ))
  [ $pad -ne 4 ] && payload="${payload}$(printf '=%.0s' $(seq 1 $pad))"
  groups=$(echo "$payload" | base64 -d 2>/dev/null | jq -r '.groups // [] | join(",")')
  case "$groups" in
    *mcp-registry-admin*) _check "Token includes mcp-registry-admin group" 0 ;;
    *) _check "Token includes mcp-registry-admin group" 1 "got: $groups" ;;
  esac
fi

# ----------------------------------------------------------------------
# 5. End-to-end agent CRUD (catches DocumentDB/scope wiring issues)
# ----------------------------------------------------------------------
echo
_log_info "Agent CRUD"
H="Authorization: Bearer $TOKEN"
TEST_AGENT="validate-test-agent"
PAYLOAD="{\"name\":\"$TEST_AGENT\",\"url\":\"http://example.com\",\"supportedProtocol\":\"a2a\",\"visibility\":\"public\",\"description\":\"validate.sh smoke test\",\"version\":\"1.0.0\"}"

# Best-effort cleanup of any leftover test agent before we start.
curl -s -o /dev/null -X DELETE "$REGISTRY_URL/api/agents/$TEST_AGENT" -H "$H"

reg_code=$(curl -s -o /tmp/.validate.body -w "%{http_code}" -X POST "$REGISTRY_URL/api/agents/register" \
  -H "$H" -H "Content-Type: application/json" -d "$PAYLOAD")
if [ "$reg_code" = "201" ]; then
  _check "POST /api/agents/register" 0
else
  _check "POST /api/agents/register" 1 "$reg_code: $(awk 'NR==1 {print substr($0,1,120)}' /tmp/.validate.body)"
fi

if [ "$reg_code" = "201" ]; then
  list_count=$(curl -s "$REGISTRY_URL/api/agents" -H "$H" | jq -r --arg n "$TEST_AGENT" '[.agents[] | select(.name==$n)] | length')
  _check "GET /api/agents lists test agent" "$([ "$list_count" = "1" ] && echo 0 || echo 1)"

  get_code=$(curl -s -o /dev/null -w "%{http_code}" "$REGISTRY_URL/api/agents/$TEST_AGENT" -H "$H")
  _check "GET /api/agents/{path}" "$([ "$get_code" = "200" ] && echo 0 || echo 1)" "$get_code"

  toggle_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$REGISTRY_URL/api/agents/$TEST_AGENT/toggle?enabled=true" -H "$H")
  _check "POST .../toggle?enabled=true" "$([ "$toggle_code" = "200" ] && echo 0 || echo 1)" "$toggle_code"

  rate_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$REGISTRY_URL/api/agents/$TEST_AGENT/rate" \
    -H "$H" -H "Content-Type: application/json" -d '{"rating":5}')
  _check "POST .../rate" "$([ "$rate_code" = "200" ] && echo 0 || echo 1)" "$rate_code"

  del_code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$REGISTRY_URL/api/agents/$TEST_AGENT" -H "$H")
  _check "DELETE /api/agents/{path}" "$([ "$del_code" = "204" ] && echo 0 || echo 1)" "$del_code"
fi

# ----------------------------------------------------------------------
# 6. ScopesLoader Lambda fired & seeded all UI-Scope docs
# ----------------------------------------------------------------------
echo
_log_info "ScopesLoader"
aws lambda invoke --region "$AWS_REGION" --function-name mcp-gateway-scopes-loader \
  --cli-binary-format raw-in-base64-out --payload '{"debug":true}' /tmp/.validate.lambda > /dev/null 2>&1
doc_count=$(jq -r '.docs | length // 0' /tmp/.validate.lambda 2>/dev/null)
has_admin=$(jq -r '[.docs[] | select(._id=="mcp-registry-admin")] | length' /tmp/.validate.lambda 2>/dev/null)
_check "ScopesLoader Lambda invokable + ${doc_count} UI-Scope docs in DocumentDB" \
  "$([ "$doc_count" -ge 5 ] && echo 0 || echo 1)" "got $doc_count, want >=5"
_check "mcp-registry-admin doc present" \
  "$([ "$has_admin" = "1" ] && echo 0 || echo 1)"

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo
echo "============================================"
if [ "$FAIL" -eq 0 ]; then
  echo -e "  ${GREEN}All $PASS checks passed${NC}"
else
  echo -e "  ${YELLOW}$PASS passed, ${RED}$FAIL failed${NC}"
fi
echo "============================================"
echo

rm -f /tmp/.validate.body /tmp/.validate.lambda
exit "$FAIL"
