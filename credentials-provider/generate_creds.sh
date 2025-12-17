#!/bin/bash
#
# Gateway Credential + MCP Config Generator
#
# Generates gateway ingress credentials (OIDC client_credentials) and produces MCP client
# configuration files that contain ONLY gateway credentials (no upstream tokens/keys).
#
# Usage:
#   ./credentials-provider/generate_creds.sh
#   ./credentials-provider/generate_creds.sh --ingress-only
#   ./credentials-provider/generate_creds.sh --keycloak-only
#   ./credentials-provider/generate_creds.sh --all
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load env files if present (best-effort)
if [ -f "$SCRIPT_DIR/oauth/.env" ]; then
    # shellcheck disable=SC1090
    source "$SCRIPT_DIR/oauth/.env"
fi
if [ -f "$SCRIPT_DIR/.env" ]; then
    # shellcheck disable=SC1090
    source "$SCRIPT_DIR/.env"
fi
if [ -f "$(dirname "$SCRIPT_DIR")/.env" ]; then
    # shellcheck disable=SC1090
    source "$(dirname "$SCRIPT_DIR")/.env"
fi

RUN_INGRESS=true
RUN_KEYCLOAK=false
VERBOSE=false
FORCE=false

log_info() {
    echo "[INFO] $1"
}

log_warn() {
    echo "[WARN] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

show_help() {
    cat << EOF
Gateway Credential + MCP Config Generator

Generates gateway ingress credentials and MCP client configs that contain ONLY gateway credentials.
Upstream authentication is gateway-managed and is not included in generated client configs.

USAGE:
  ./credentials-provider/generate_creds.sh [OPTIONS]

OPTIONS:
  --ingress-only     Generate gateway ingress token + configs (default)
  --keycloak-only    Generate Keycloak agent tokens + configs
  --all              Run both ingress and Keycloak agent token generation + configs
  --force, -f        Force new token generation (where supported)
  --verbose, -v      Enable verbose logging
  --help, -h         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ingress-only)
            RUN_INGRESS=true
            RUN_KEYCLOAK=false
            shift
            ;;
        --keycloak-only)
            RUN_INGRESS=false
            RUN_KEYCLOAK=true
            shift
            ;;
        --all)
            RUN_INGRESS=true
            RUN_KEYCLOAK=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

run_ingress_auth() {
    log_info "Generating gateway ingress token..."

    local cmd=(uv run python "$SCRIPT_DIR/oauth/ingress_oauth.py")
    if [ "$FORCE" = true ]; then
        cmd+=("--force")
    fi
    if [ "$VERBOSE" = true ]; then
        cmd+=("--verbose")
    fi

    "${cmd[@]}"
    log_info "Ingress token generated."
}

run_keycloak_agent_tokens() {
    log_info "Generating Keycloak agent tokens..."

    local cmd=(uv run python "$SCRIPT_DIR/keycloak/generate_tokens.py" --all-agents)
    if [ "$VERBOSE" = true ]; then
        cmd+=("--verbose")
    fi

    "${cmd[@]}"
    log_info "Keycloak agent tokens generated."
}

generate_mcp_configs() {
    log_info "Generating MCP client configuration files..."

    local cmd=(uv run python "$SCRIPT_DIR/add_services.py")
    if [ "$VERBOSE" = true ]; then
        cmd+=("--verbose")
    fi

    "${cmd[@]}"
    log_info "MCP client configuration files generated under ./.oauth-tokens/."
}

main() {
    log_info "Starting credential generation"

    if [ "$RUN_INGRESS" = true ]; then
        run_ingress_auth
    fi

    if [ "$RUN_KEYCLOAK" = true ]; then
        run_keycloak_agent_tokens
    fi

    generate_mcp_configs
    log_info "Done."
}

main "$@"

