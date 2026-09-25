#!/bin/bash
# Post-deploy automation: configure Keycloak, wire up secrets, restart services.
# Run by deploy.sh; needs CDK_KEYCLOAK_ADMIN_PASSWORD and cdk-outputs.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$INFRA_DIR")"

source "$SCRIPT_DIR/_lib.sh"
[ -f "$SCRIPT_DIR/set-env.sh" ] && source "$SCRIPT_DIR/set-env.sh"
AWS_REGION="${AWS_REGION:-us-east-1}"

# ---------------------------------------------------------------------------
# Read endpoints from cdk-outputs.json
# ---------------------------------------------------------------------------

_read_outputs() {
  local outputs_file="$INFRA_DIR/cdk-outputs.json"
  if [ ! -f "$outputs_file" ]; then
    _log_error "cdk-outputs.json not found. Run CDK deploy first."
    exit 1
  fi

  # Keycloak/Registry URLs are HTTPS URLs when CloudFront is enabled (published
  # by Auth/Service stacks after they front their ALBs). sslRequired=external
  # on Keycloak realms blocks admin API calls over plain HTTP, so post-deploy
  # admin ops MUST run against the HTTPS front.
  eval "$(jq -r '@sh "
    KEYCLOAK_URL=\(."Registry-Auth".KeycloakUrl // ."Registry-Service".KeycloakUrl // "")
    REGISTRY_URL=\(."Registry-Service".RegistryUrl // "")
    GRADIO_URL=\(."Registry-Service".GradioUiUrl // "")
    GRAFANA_URL=\(."Registry-Service".GrafanaUrl // "")"' "$outputs_file")"

  if [ -z "$KEYCLOAK_URL" ] || [ -z "$REGISTRY_URL" ]; then
    _log_error "Could not read Keycloak/Registry URLs from cdk-outputs.json"
    exit 1
  fi

  _log_info "Keycloak URL: $KEYCLOAK_URL"
  _log_info "Registry URL: $REGISTRY_URL"
}

# ---------------------------------------------------------------------------
# Wait for Keycloak ALB to be healthy
# ---------------------------------------------------------------------------

_wait_for_keycloak() {
  _log_info "Waiting for Keycloak to be ready at ${KEYCLOAK_URL}..."
  # 180 attempts × 5s = 15 min. CloudFront distribution creation can take
  # ~5-15 min to propagate to all edge locations after CDK reports COMPLETE.
  local max_attempts=180
  local attempt=0

  while [ $attempt -lt $max_attempts ]; do
    local http_code
    http_code=$(curl -sk -o /dev/null -w "%{http_code}" "${KEYCLOAK_URL}/" 2>/dev/null || echo "000")
    if [ "$http_code" = "200" ] || [ "$http_code" = "302" ] || [ "$http_code" = "303" ]; then
      _log_success "Keycloak is ready (HTTP $http_code)"
      return 0
    fi
    sleep 5
    attempt=$((attempt + 1))
  done

  _log_error "Keycloak did not become ready within 15 minutes"
  exit 1
}

# ---------------------------------------------------------------------------
# Get Keycloak admin token
# ---------------------------------------------------------------------------

_get_admin_token() {
  local token
  token=$(curl -s -X POST "${KEYCLOAK_URL}/realms/${KEYCLOAK_ADMIN_REALM}/protocol/openid-connect/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${KC_ADMIN_USER}" \
    -d "password=${KC_ADMIN_PASSWORD}" \
    -d "grant_type=password" \
    -d "client_id=admin-cli" | jq -r '.access_token // empty')

  if [ -z "$token" ]; then
    _log_error "Failed to authenticate with Keycloak"
    exit 1
  fi

  echo "$token"
}

# ---------------------------------------------------------------------------
# Disable sslRequired on a realm via Admin API (requires valid token)
# ---------------------------------------------------------------------------

_disable_ssl_required() {
  local token="$1"
  local realm="$2"

  local http_code
  http_code=$(curl -s -o /dev/null -w "%{http_code}" \
    -X PUT "${KEYCLOAK_URL}/admin/realms/${realm}" \
    -H "Authorization: Bearer ${token}" \
    -H "Content-Type: application/json" \
    -d '{"sslRequired":"EXTERNAL"}')

  if [ "$http_code" = "204" ]; then
    _log_success "Disabled sslRequired on realm: ${realm}"
  else
    _log_warn "Could not disable sslRequired on ${realm} (HTTP ${http_code})"
  fi
}

# ---------------------------------------------------------------------------
# Disable sslRequired on master realm via ECS Exec (kcadm.sh on localhost)
# This bypasses the ALB and avoids the HTTPS-required chicken-and-egg problem
# on fresh deploys where the master realm defaults to sslRequired=EXTERNAL.
# ---------------------------------------------------------------------------

_disable_ssl_via_ecs_exec() {
  _log_info "Disabling SSL requirement on master realm via ECS Exec..."

  local task_arn
  task_arn=$(aws ecs list-tasks --cluster keycloak --service-name keycloak \
    --region "$AWS_REGION" --desired-status RUNNING \
    --query 'taskArns[0]' --output text 2>/dev/null)

  if [ -z "$task_arn" ] || [ "$task_arn" = "None" ]; then
    _log_error "No running Keycloak task found"
    return 1
  fi

  local task_id="${task_arn##*/}"

  local kcadm_cmd="/opt/keycloak/bin/kcadm.sh"
  local script="$kcadm_cmd config credentials --server http://localhost:8080 --realm ${KEYCLOAK_ADMIN_REALM} --user ${KC_ADMIN_USER} --password ${KC_ADMIN_PASSWORD} 2>&1 && $kcadm_cmd update realms/${KEYCLOAK_ADMIN_REALM} -s sslRequired=EXTERNAL 2>&1 && echo SSL_DISABLED_OK"

  local output
  output=$(aws ecs execute-command --cluster keycloak --task "$task_id" \
    --container keycloak --interactive \
    --command "sh -c '${script}'" \
    --region "$AWS_REGION" 2>&1) || true

  if echo "$output" | grep -q "SSL_DISABLED_OK"; then
    _log_success "Disabled sslRequired on ${KEYCLOAK_ADMIN_REALM} realm via ECS Exec"
    return 0
  fi

  _log_warn "ECS Exec output: $output"
  _log_warn "ECS Exec may have timed out but the command may still succeed. Verifying..."

  # Verify by reading realm config via kcadm on loopback (sslRequired=external
  # blocks external HTTP requests, so we cannot verify via the ALB). The
  # inline echo runs only when grep matches, so absence of VERIFY_DONE means
  # sslRequired has not yet been flipped.
  local verify_script="/opt/keycloak/bin/kcadm.sh config credentials --server http://localhost:8080 --realm ${KEYCLOAK_ADMIN_REALM} --user ${KC_ADMIN_USER} --password ${KC_ADMIN_PASSWORD} >/dev/null 2>&1 && /opt/keycloak/bin/kcadm.sh get realms/${KEYCLOAK_ADMIN_REALM} 2>/dev/null | grep -q 'sslRequired.*external' && echo VERIFY_DONE"

  local verify_attempt=0
  local max_verify=6
  while [ $verify_attempt -lt $max_verify ]; do
    sleep 5
    local verify_out
    verify_out=$(aws ecs execute-command --cluster keycloak --task "$task_id" \
      --container keycloak --interactive \
      --command "sh -c '${verify_script}'" \
      --region "$AWS_REGION" 2>&1) || true

    if echo "$verify_out" | grep -q "VERIFY_DONE"; then
      _log_success "Verified: master realm sslRequired is EXTERNAL"
      return 0
    fi
    _log_info "Waiting for SSL flip to verify (attempt $((verify_attempt + 1))/$max_verify)..."
    verify_attempt=$((verify_attempt + 1))
  done

  _log_error "Failed to verify sslRequired=external on master realm after ${max_verify} attempts"
  return 1
}

# ---------------------------------------------------------------------------
# Run init-keycloak.sh to create realm, clients, groups, users
# ---------------------------------------------------------------------------

_init_keycloak() {
  local init_script="$PROJECT_ROOT/keycloak/setup/init-keycloak.sh"
  [ -f "$init_script" ] || { _log_error "init-keycloak.sh not found at $init_script"; exit 1; }

  _log_info "Running Keycloak initialization (realm, clients, groups, users)..."

  # init-keycloak.sh source-loads $PROJECT_ROOT/.env. Write it only if absent;
  # rm-on-exit via a non-local TMP_ENV (bash 3.2 RETURN traps lose locals).
  local tmp_env="$PROJECT_ROOT/.env"
  local cleanup=false
  if [ ! -f "$tmp_env" ]; then
    cleanup=true
    cat > "$tmp_env" <<ENVEOF
KEYCLOAK_ADMIN_URL=${KEYCLOAK_URL}
KEYCLOAK_ADMIN=${KC_ADMIN_USER}
KEYCLOAK_ADMIN_PASSWORD=${KC_ADMIN_PASSWORD}
KEYCLOAK_ADMIN_REALM=${KEYCLOAK_ADMIN_REALM}
REGISTRY_URL=${REGISTRY_URL}
AUTH_SERVER_EXTERNAL_URL=${REGISTRY_URL}
INITIAL_ADMIN_PASSWORD=${KC_ADMIN_PASSWORD}
INITIAL_USER_PASSWORD=testpass123
ENVEOF
  fi

  # Export the CDK-derived values into the child's environment, not just into a
  # .env we may not have written. init-keycloak.sh now PUTs an existing client
  # instead of skipping it, and Keycloak replaces redirectUris/webOrigins
  # wholesale on that PUT, so on a host that already has a Compose .env the
  # file's localhost values would be written over this deployment's URLs and
  # break login with invalid_redirect_uri. The script prefers these exported
  # values over the file.
  local rc=0
  (
    cd "$PROJECT_ROOT" || exit 1
    export KEYCLOAK_ADMIN_URL="$KEYCLOAK_URL"
    export KEYCLOAK_ADMIN="$KC_ADMIN_USER"
    export KEYCLOAK_ADMIN_PASSWORD="$KC_ADMIN_PASSWORD"
    export REGISTRY_URL="$REGISTRY_URL"
    export AUTH_SERVER_EXTERNAL_URL="$REGISTRY_URL"
    export INITIAL_ADMIN_PASSWORD="$KC_ADMIN_PASSWORD"
    bash "$init_script"
  ) || rc=$?
  [ "$cleanup" = true ] && rm -f "$tmp_env"
  [ "$rc" -ne 0 ] && { _log_error "init-keycloak.sh failed"; exit "$rc"; }
  _log_success "Keycloak initialization complete"
}

# ---------------------------------------------------------------------------
# Extract client secrets and update Secrets Manager
# ---------------------------------------------------------------------------

_update_client_secrets() {
  _log_info "Updating Secrets Manager with Keycloak client secrets..."
  local token; token=$(_get_admin_token)

  # macOS bash 3.2 has no assoc arrays — use "client_id|secret_id" pairs.
  for pair in \
    "mcp-gateway-web|mcp-gateway-keycloak-client-secret" \
    "mcp-gateway-m2m|mcp-gateway-keycloak-m2m-client-secret"; do
    local client_id="${pair%%|*}" secret_id="${pair##*|}"
    local uuid; uuid=$(curl -s -H "Authorization: Bearer ${token}" \
      "${KEYCLOAK_URL}/admin/realms/mcp-gateway/clients?clientId=${client_id}" | jq -r '.[0].id // empty')
    [ -z "$uuid" ] && { _log_error "Could not find ${client_id} client"; return 1; }

    local secret; secret=$(curl -s -H "Authorization: Bearer ${token}" \
      "${KEYCLOAK_URL}/admin/realms/mcp-gateway/clients/${uuid}/client-secret" | jq -r '.value // empty')
    [ -z "$secret" ] && { _log_error "Could not retrieve secret for ${client_id}"; return 1; }

    aws secretsmanager put-secret-value --region "$AWS_REGION" \
      --secret-id "$secret_id" \
      --secret-string "{\"client_secret\":\"${secret}\"}" > /dev/null 2>&1
  done
  _log_success "Secrets Manager updated with real client secrets"
}

# ---------------------------------------------------------------------------
# Restart registry and auth-server ECS services
# ---------------------------------------------------------------------------

_restart_services() {
  _log_info "Restarting registry and auth-server to pick up new secrets..."

  local cluster="mcp-gateway-ecs-cluster"

  aws ecs update-service --region "$AWS_REGION" \
    --cluster "$cluster" --service mcp-gateway-registry \
    --force-new-deployment > /dev/null 2>&1

  aws ecs update-service --region "$AWS_REGION" \
    --cluster "$cluster" --service mcp-gateway-auth-server \
    --force-new-deployment > /dev/null 2>&1

  _log_info "Waiting for services to stabilize (this takes 2-4 minutes)..."

  aws ecs wait services-stable --region "$AWS_REGION" \
    --cluster "$cluster" \
    --services mcp-gateway-registry mcp-gateway-auth-server 2>/dev/null || {
    _log_warn "Services did not stabilize within timeout. Check ECS console."
  }

  _log_success "Services restarted"
}

# ---------------------------------------------------------------------------
# Run a Python script in the registry container via ECS Exec.
# Args: <label> <command-line> <success-grep-pattern>
# ---------------------------------------------------------------------------

_run_in_registry() {
  local label="$1" cmd="$2" success="$3"
  local cluster="mcp-gateway-ecs-cluster"
  local task_arn
  task_arn=$(aws ecs list-tasks --cluster "$cluster" --service-name mcp-gateway-registry \
    --region "$AWS_REGION" --desired-status RUNNING \
    --query 'taskArns[0]' --output text 2>/dev/null)
  if [ -z "$task_arn" ] || [ "$task_arn" = "None" ]; then
    _log_error "No running registry task for $label"
    return 1
  fi

  local output
  output=$(aws ecs execute-command --cluster "$cluster" --task "${task_arn##*/}" \
    --container registry --interactive --command "sh -c '$cmd 2>&1'" \
    --region "$AWS_REGION" 2>&1) || true

  if echo "$output" | grep -qE "$success"; then
    _log_success "$label OK"
    return 0
  fi
  _log_warn "$label output: $output"
  return 1
}

# ---------------------------------------------------------------------------
# Validate all endpoints
# ---------------------------------------------------------------------------

_validate_endpoints() {
  _log_info "Validating service endpoints..."
  echo ""

  local all_ok=true

  # Auth-server and Gradio are proxied by the registry container's nginx on
  # port 8080 (behind CloudFront/ALB). Nginx paths:
  #   /oauth2/* → auth-server via Service Connect
  #   /       → gradio (uvicorn on 127.0.0.1:7860, via nginx default location)
  # There are no separate 8888/7860 ALB listeners anymore (TF parity). The
  # auth-server has no plain probe endpoint through nginx; a successful
  # OAuth login flow (post-deploy step 4) is the real integration test.
  for url_label in \
    "Registry|${REGISTRY_URL}/health" \
    "Gradio UI|${REGISTRY_URL}/" \
    "Keycloak|${KEYCLOAK_URL}/" \
    "Keycloak Realm|${KEYCLOAK_URL}/realms/mcp-gateway/.well-known/openid-configuration"; do

    local label="${url_label%%|*}"
    local url="${url_label##*|}"
    if [ -z "$url" ]; then
      echo -e "  ${YELLOW}[SKIP]${NC} $label (URL unset)"
      continue
    fi
    local http_code
    # -k: CloudFront default cert is valid but the ALB HTTP URLs are naked.
    # -L: follow redirects when CloudFront viewer-protocol-policy issues 301.
    http_code=$(curl -skL -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")

    if [ "$http_code" = "200" ] || [ "$http_code" = "302" ] || [ "$http_code" = "303" ]; then
      echo -e "  ${GREEN}[PASS]${NC} $label ($http_code)"
    else
      echo -e "  ${RED}[FAIL]${NC} $label ($http_code)"
      all_ok=false
    fi
  done

  echo ""

  if [ "$all_ok" = false ]; then
    _log_warn "Some endpoints are not responding. Check ECS task logs."
  fi
}

# ---------------------------------------------------------------------------
# Print summary with URLs and credentials
# ---------------------------------------------------------------------------

_print_summary() {
  local grafana_password="${CDK_GRAFANA_ADMIN_PASSWORD:-(unset \xE2\x80\x94 set CDK_GRAFANA_ADMIN_PASSWORD and redeploy)}"
  # printf %b interprets the \033 escape codes that heredoc passes through verbatim.
  printf '%b' "$(cat <<EOF

============================================
  Deployment Complete
============================================

  Service URLs
  ------------
  Registry:          ${GREEN}${REGISTRY_URL}${NC}
  Registry OpenAPI:  ${GREEN}${REGISTRY_URL}/docs${NC}
  Registry API:      ${GREEN}${REGISTRY_URL}/api/agents${NC}  (also /api/audit/*, /api/admin/*, /api/register)
  Gradio UI:         ${GREEN}${REGISTRY_URL}${NC}
  OAuth callback:    ${GREEN}${REGISTRY_URL}/oauth2/callback/keycloak${NC}
  Keycloak:          ${GREEN}${KEYCLOAK_URL}${NC}
  Keycloak Admin:    ${GREEN}${KEYCLOAK_URL}/admin${NC}${GRAFANA_URL:+
  Grafana:           ${GREEN}${GRAFANA_URL}${NC}}

  Login Credentials
  -----------------
  Registry / Gradio UI (Keycloak SSO):
    Admin user:      ${YELLOW}admin${NC} / ${YELLOW}${KC_ADMIN_PASSWORD}${NC}
    Test user:       ${YELLOW}testuser${NC} / ${YELLOW}testpass123${NC}

  Keycloak Admin Console:
    Username:        ${YELLOW}${KC_ADMIN_USER}${NC}
    Password:        ${YELLOW}${KC_ADMIN_PASSWORD}${NC}${GRAFANA_URL:+

  Grafana:
    Username:        ${YELLOW}admin${NC}
    Password:        ${YELLOW}${grafana_password}${NC}}
============================================

EOF
)"
  printf '\n'
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  echo ""
  echo "============================================"
  echo "  Post-Deploy Configuration"
  echo "============================================"
  echo ""

  # Validate required env vars
  KC_ADMIN_USER="admin"
  KC_ADMIN_PASSWORD="${CDK_KEYCLOAK_ADMIN_PASSWORD:-}"
  KEYCLOAK_ADMIN_REALM="${KEYCLOAK_ADMIN_REALM:-master}"

  if [ -z "$KC_ADMIN_PASSWORD" ]; then
    _log_error "CDK_KEYCLOAK_ADMIN_PASSWORD is not set"
    exit 1
  fi

  _read_outputs
  _wait_for_keycloak

  # SSM agent on a fresh task takes ~30s to register, so retry.
  if ! _retry "SSL disable" 3 30 _disable_ssl_via_ecs_exec; then
    _log_error "Failed to disable SSL. Run post-deploy.sh once ECS Exec is available."
    exit 1
  fi

  _init_keycloak

  _log_info "Disabling SSL requirement on mcp-gateway realm..."
  _disable_ssl_required "$(_get_admin_token)" "mcp-gateway"

  _update_client_secrets

  _restart_services
  _validate_endpoints
  _print_summary
}

main "$@"
