#!/bin/bash
# Deploy MCP Gateway Registry CDK infrastructure. See --help for usage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/_lib.sh"
[ -f "$SCRIPT_DIR/set-env.sh" ] && source "$SCRIPT_DIR/set-env.sh"
AWS_REGION="${AWS_REGION:-us-east-1}"

_check_prerequisites() {
  local missing=0
  for cmd in aws node npx jq; do
    command -v "$cmd" &>/dev/null || { _log_error "$cmd not found in PATH"; missing=1; }
  done
  [ "$missing" -eq 1 ] && exit 1

  local identity
  if ! identity=$(aws sts get-caller-identity --region "$AWS_REGION" --output json 2>/dev/null); then
    _log_error "AWS credentials not configured or expired."
    _log_error "Run: aws sso login  OR  export AWS_PROFILE=<your-profile>"
    exit 1
  fi
  AWS_ACCOUNT_ID=$(jq -r .Account <<<"$identity")
  _log_info "AWS Account: $AWS_ACCOUNT_ID"
  _log_info "Identity: $(jq -r .Arn <<<"$identity")"
  _log_info "Region: $AWS_REGION"
}

_check_secrets() {
  local bad=0
  for secret in CDK_KEYCLOAK_ADMIN_PASSWORD CDK_KEYCLOAK_DATABASE_PASSWORD CDK_DOCUMENTDB_ADMIN_PASSWORD; do
    local val="${!secret:-}"
    if [ -z "$val" ]; then
      _log_error "$secret not set"
      bad=1
    elif [ "${#val}" -lt 8 ] || echo "$val" | grep -q '[/@" ]'; then
      _log_error "$secret: must be 8+ chars, no '/' '@' '\"' or spaces"
      bad=1
    fi
  done
  [ "$bad" -eq 1 ] && { _log_error "Run '$0 --help' for full env-var list"; exit 1; }
  _log_success "Required secrets are set"
}

_bootstrap_cdk() {
  local bootstrap_status
  bootstrap_status=$(aws cloudformation describe-stacks \
    --region "$AWS_REGION" --stack-name CDKToolkit \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "DOES_NOT_EXIST")

  if [ "$bootstrap_status" = "DOES_NOT_EXIST" ]; then
    _log_info "Bootstrapping CDK in account $AWS_ACCOUNT_ID / region $AWS_REGION..."
    cd "$INFRA_DIR"
    npx cdk bootstrap "aws://${AWS_ACCOUNT_ID}/${AWS_REGION}"
    _log_success "CDK bootstrapped"
  else
    _log_info "CDK already bootstrapped (CDKToolkit: $bootstrap_status)"
  fi
}

_install_deps() {
  (cd "$INFRA_DIR" && npm install --silent)
}

_show_status() {
  _log_info "Checking stack status..."
  echo ""

  aws cloudformation list-stacks \
    --region "$AWS_REGION" \
    --query 'StackSummaries[?starts_with(StackName, `Registry-`) && StackStatus != `DELETE_COMPLETE`].{Name:StackName,Status:StackStatus,Updated:LastUpdatedTime}' \
    --output table 2>/dev/null || echo "No Registry stacks found."

  echo ""
}

_deploy_all() {
  _check_secrets
  _install_deps
  _bootstrap_cdk

  _log_info "Synthesizing CloudFormation templates..."
  cd "$INFRA_DIR"
  npx cdk synth --quiet
  _log_success "Synth complete"

  _log_info "Deploying all stacks (this will take 20-40 minutes)..."
  _timed "All stacks deployed" \
    npx cdk deploy --all --region "$AWS_REGION" \
      --require-approval never --outputs-file "$INFRA_DIR/cdk-outputs.json"
  _show_status

  _log_info "Stack outputs saved to: infra/cdk-outputs.json"
  _log_info "Running post-deploy automation..."
  bash "$SCRIPT_DIR/post-deploy.sh"
}

_deploy_stack() {
  local stack_name="$1"
  _check_secrets
  _install_deps
  _bootstrap_cdk

  _log_info "Deploying $stack_name..."
  cd "$INFRA_DIR"
  _timed "$stack_name deployed" \
    npx cdk deploy "$stack_name" --region "$AWS_REGION" \
      --require-approval never --outputs-file "$INFRA_DIR/cdk-outputs.json"
}

_destroy_all() {
  _log_warn "This will destroy ALL Registry CDK stacks!"
  echo ""
  _show_status

  read -r -p "Are you sure? Type 'yes' to confirm: " confirm
  if [ "$confirm" != "yes" ]; then
    _log_info "Aborted."
    exit 0
  fi

  _log_info "Destroying all stacks (this will take 15-25 minutes)..."
  cd "$INFRA_DIR"
  _timed "All stacks destroyed" npx cdk destroy --all --region "$AWS_REGION" --force
  rm -f "$INFRA_DIR/cdk-outputs.json"
}

_destroy_stack() {
  local stack_name="$1"
  _log_warn "This will destroy stack: $stack_name"

  read -r -p "Are you sure? Type 'yes' to confirm: " confirm
  if [ "$confirm" != "yes" ]; then
    _log_info "Aborted."
    exit 0
  fi

  _log_info "Destroying $stack_name..."
  cd "$INFRA_DIR"

  npx cdk destroy "$stack_name" --region "$AWS_REGION" --force

  _log_success "$stack_name destroyed"
}

_show_endpoints() {
  echo ""
  echo "  Service Endpoints"
  echo "  -----------------"
  if [ ! -f "$INFRA_DIR/cdk-outputs.json" ]; then
    _log_warn "cdk-outputs.json not found — run a deploy first."
    return
  fi
  eval "$(jq -r '@sh "
    registry_url=\(."Registry-Service".RegistryUrl // "")
    keycloak_url=\(."Registry-Auth".KeycloakUrl // ."Registry-Service".KeycloakUrl // "")
    keycloak_admin=\(."Registry-Auth".KeycloakAdminConsole // "")
    gradio_url=\(."Registry-Service".GradioUiUrl // "")
    grafana_url=\(."Registry-Service".GrafanaUrl // "")"' "$INFRA_DIR/cdk-outputs.json")"
  [ -n "$registry_url" ]    && echo -e "  Registry UI:       ${GREEN}${registry_url}${NC}"
  [ -n "$registry_url" ]    && echo -e "  Registry API:      ${GREEN}${registry_url}/api/v1${NC}"
  [ -n "$registry_url" ]    && echo -e "  Registry Health:   ${GREEN}${registry_url}/health${NC}"
  [ -n "$keycloak_url" ]    && echo -e "  Keycloak:          ${GREEN}${keycloak_url}${NC}"
  [ -n "$keycloak_admin" ]  && echo -e "  Keycloak Admin:    ${GREEN}${keycloak_admin}${NC}"
  [ -n "$gradio_url" ]      && echo -e "  Gradio UI:         ${GREEN}${gradio_url}${NC}"
  [ -n "$grafana_url" ]     && echo -e "  Grafana:           ${GREEN}${grafana_url}${NC}"
  echo ""
}

_show_diff() {
  _log_info "Showing pending changes..."
  cd "$INFRA_DIR"
  npx cdk diff --region "$AWS_REGION" 2>&1 || true
}

_show_usage() {
  cat <<EOF
Usage: $0 [--stack <name>] [--destroy|--status|--diff|--endpoints|--validate|--help]

Required env: CDK_KEYCLOAK_ADMIN_PASSWORD CDK_KEYCLOAK_DATABASE_PASSWORD CDK_DOCUMENTDB_ADMIN_PASSWORD
Optional env: AWS_REGION (us-east-1) CDK_EMBEDDINGS_API_KEY CDK_GRAFANA_ADMIN_PASSWORD
              CDK_{ENTRA,OKTA,AUTH0}_CLIENT_SECRET CDK_GITHUB_PAT CDK_OTEL_EXPORTER_OTLP_HEADERS

Stacks: run 'npx cdk list' from $INFRA_DIR
EOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ACTION="deploy"
TARGET_STACK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --destroy)
      ACTION="destroy"
      shift
      ;;
    --stack)
      TARGET_STACK="$2"
      shift 2
      ;;
    --status)
      ACTION="status"
      shift
      ;;
    --diff)
      ACTION="diff"
      shift
      ;;
    --endpoints)
      ACTION="endpoints"
      shift
      ;;
    --validate)
      ACTION="validate"
      shift
      ;;
    --help|-h)
      _show_usage
      exit 0
      ;;
    *)
      _log_error "Unknown option: $1"
      _show_usage
      exit 1
      ;;
  esac
done

echo ""
echo "============================================"
echo "  MCP Gateway Registry - CDK Deployment"
echo "============================================"
echo ""

_check_prerequisites

case "$ACTION" in
  deploy)
    if [ -n "$TARGET_STACK" ]; then
      _deploy_stack "$TARGET_STACK"
    else
      _deploy_all
    fi
    ;;
  destroy)
    if [ -n "$TARGET_STACK" ]; then
      _destroy_stack "$TARGET_STACK"
    else
      _destroy_all
    fi
    ;;
  status)
    _show_status
    ;;
  diff)
    _show_diff
    ;;
  endpoints)
    _show_endpoints
    ;;
  validate)
    bash "$SCRIPT_DIR/validate.sh"
    ;;
esac
