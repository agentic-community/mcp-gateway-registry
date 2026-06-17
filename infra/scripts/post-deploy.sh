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

  eval "$(jq -r '@sh "
    KEYCLOAK_URL=\(."Registry-Service".KeycloakUrl // ."Registry-Auth".KeycloakUrl // "")
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
  _log_info "Waiting for Keycloak to be ready..."
  local max_attempts=60
  local attempt=0

  while [ $attempt -lt $max_attempts ]; do
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "${KEYCLOAK_URL}/" 2>/dev/null || echo "000")
    if [ "$http_code" = "200" ] || [ "$http_code" = "302" ] || [ "$http_code" = "303" ]; then
      _log_success "Keycloak is ready (HTTP $http_code)"
      return 0
    fi
    sleep 5
    attempt=$((attempt + 1))
  done

  _log_error "Keycloak did not become ready within 5 minutes"
  exit 1
}

# ---------------------------------------------------------------------------
# Get Keycloak admin token
# ---------------------------------------------------------------------------

_get_admin_token() {
  local token
  token=$(curl -s -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
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
    -d '{"sslRequired":"NONE"}')

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
  local script="$kcadm_cmd config credentials --server http://localhost:8080 --realm master --user ${KC_ADMIN_USER} --password ${KC_ADMIN_PASSWORD} 2>&1 && $kcadm_cmd update realms/master -s sslRequired=NONE 2>&1 && echo SSL_DISABLED_OK"

  local output
  output=$(aws ecs execute-command --cluster keycloak --task "$task_id" \
    --container keycloak --interactive \
    --command "sh -c '${script}'" \
    --region "$AWS_REGION" 2>&1) || true

  if echo "$output" | grep -q "SSL_DISABLED_OK"; then
    _log_success "Disabled sslRequired on master realm via ECS Exec"
    return 0
  fi

  _log_warn "ECS Exec output: $output"
  _log_warn "ECS Exec may have timed out but the command may still succeed. Verifying..."

  local verify_attempt=0
  local max_verify=12
  while [ $verify_attempt -lt $max_verify ]; do
    sleep 5
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
      -H "Content-Type: application/x-www-form-urlencoded" \
      -d "username=${KC_ADMIN_USER}" \
      -d "password=${KC_ADMIN_PASSWORD}" \
      -d "grant_type=password" \
      -d "client_id=admin-cli" 2>/dev/null || echo "000")

    if [ "$http_code" = "200" ]; then
      _log_success "Verified: master realm SSL is now disabled (token request returned 200)"
      return 0
    fi
    _log_info "Waiting for SSL disable to take effect (attempt $((verify_attempt + 1))/$max_verify, HTTP $http_code)..."
    verify_attempt=$((verify_attempt + 1))
  done

  _log_error "Failed to disable SSL on master realm after ${max_verify} attempts"
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
REGISTRY_URL=${REGISTRY_URL}
AUTH_SERVER_EXTERNAL_URL=${REGISTRY_URL}
INITIAL_ADMIN_PASSWORD=${KC_ADMIN_PASSWORD}
INITIAL_USER_PASSWORD=testpass123
ENVEOF
  fi

  local rc=0
  (cd "$PROJECT_ROOT" && bash "$init_script") || rc=$?
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

_load_scopes() {
  _log_info "Loading scopes configuration into DocumentDB..."
  _run_in_registry "scopes load" \
    "/app/.venv/bin/python scripts/load-scopes.py --scopes-file /app/config/scopes.yml" \
    "Successfully loaded|Scopes loading complete|No changes for scope|Updated scope"
}

# ---------------------------------------------------------------------------
# Validate all endpoints
# ---------------------------------------------------------------------------

_validate_endpoints() {
  _log_info "Validating service endpoints..."
  echo ""

  local all_ok=true

  for url_label in \
    "Registry|${REGISTRY_URL}/health" \
    "Gradio UI|${GRADIO_URL:-${REGISTRY_URL}:7860}/health" \
    "Auth Server|${REGISTRY_URL}:8888/health" \
    "Keycloak|${KEYCLOAK_URL}/" \
    "Keycloak Realm|${KEYCLOAK_URL}/realms/mcp-gateway/.well-known/openid-configuration"; do

    local label="${url_label%%|*}"
    local url="${url_label##*|}"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")

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
  Registry API:      ${GREEN}${REGISTRY_URL}/api/v1${NC}
  Gradio UI:         ${GREEN}${GRADIO_URL:-${REGISTRY_URL}:7860}${NC}
  Auth Server:       ${GREEN}${REGISTRY_URL}:8888${NC}
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

  # Top-level scopes (server defs) into DocumentDB. UI-scope group docs are
  # loaded by the ScopesLoader Lambda in the Registry-Service stack.
  _retry "Scopes load" 3 20 _load_scopes \
    || _log_warn "Scopes could not be loaded. Agent registration may return 403."

  _restart_services
  _validate_endpoints
  _print_summary
}

main "$@"
