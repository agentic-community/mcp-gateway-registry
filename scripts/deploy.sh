#!/bin/bash
# Deploy services to ECS (build, push, force new deployment)
#
# Usage:
#   ./scripts/deploy.sh [--service registry|auth|both] [--no-cache] [--skip-monitor]
#
# Examples:
#   ./scripts/deploy.sh                          # Deploy both registry and auth server
#   ./scripts/deploy.sh --service registry       # Deploy registry only
#   ./scripts/deploy.sh --service auth           # Deploy auth server only
#   ./scripts/deploy.sh --service both           # Deploy both (default)
#   ./scripts/deploy.sh --no-cache               # Deploy both without Docker cache
#   ./scripts/deploy.sh --service auth --no-cache  # Deploy auth without cache
#   ./scripts/deploy.sh --skip-monitor           # Deploy without monitoring step

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
ECS_CLUSTER="mcp-gateway-ecs-cluster"

# Service configuration mapping
# Format: IMAGE_NAME:ECS_SERVICE_NAME
REGISTRY_IMAGE="registry"
REGISTRY_ECS_SERVICE="mcp-gateway-v2-registry"

AUTH_IMAGE="auth_server"
AUTH_ECS_SERVICE="mcp-gateway-v2-auth"

# go-validate is the /validate fast-path SIDECAR that runs inside the auth task
# (not its own ECS service), so deploying it = build+push its image + force a new
# auth deployment so the task pulls the new sidecar image.
GOVALIDATE_IMAGE="go_validate"

# Defaults
SERVICE="both"
NO_CACHE=""
SKIP_MONITOR="false"


_print_usage() {
    echo "Usage: $0 [--service registry|auth|go-validate|both] [--no-cache] [--skip-monitor]"
    echo ""
    echo "Options:"
    echo "  --service   Service to deploy: registry, auth, go-validate, or both (default: both)"
    echo "              'auth' and 'both' also build+deploy the go-validate sidecar (it"
    echo "              rides in the auth task). 'go-validate' rebuilds only the sidecar"
    echo "              and force-redeploys the auth service to pull it."
    echo "  --no-cache  Build Docker images without cache"
    echo "  --skip-monitor  Skip the deployment monitoring step"
    echo ""
    echo "Examples:"
    echo "  $0                              # Deploy registry + auth (+ go-validate sidecar)"
    echo "  $0 --service registry           # Deploy registry only"
    echo "  $0 --service auth               # Deploy auth server + go-validate sidecar"
    echo "  $0 --service go-validate        # Rebuild the sidecar + redeploy auth"
    echo "  $0 --no-cache --service auth    # Deploy auth (+ sidecar) without cache"
}


_parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --service)
                SERVICE="$2"
                # Accept aliases
                if [[ "$SERVICE" == "auth_server" ]]; then
                    SERVICE="auth"
                fi
                if [[ "$SERVICE" == "go_validate" || "$SERVICE" == "govalidate" ]]; then
                    SERVICE="go-validate"
                fi
                if [[ "$SERVICE" != "registry" && "$SERVICE" != "auth" && "$SERVICE" != "go-validate" && "$SERVICE" != "both" ]]; then
                    echo "Error: --service must be 'registry', 'auth', 'go-validate', or 'both'"
                    _print_usage
                    exit 1
                fi
                shift 2
                ;;
            --no-cache)
                NO_CACHE="true"
                shift
                ;;
            --skip-monitor)
                SKIP_MONITOR="true"
                shift
                ;;
            --help|-h)
                _print_usage
                exit 0
                ;;
            *)
                echo "Error: Unknown option: $1"
                _print_usage
                exit 1
                ;;
        esac
    done
}


_build_and_push() {
    local image_name="$1"
    local display_name="$2"

    echo "Building and pushing ${display_name} image..."
    echo "----------------------------------------"

    cd "$REPO_ROOT"
    if [[ "$NO_CACHE" == "true" ]]; then
        echo "Building without cache (--no-cache)"
        NO_CACHE=true make build-push IMAGE="$image_name"
    else
        make build-push IMAGE="$image_name"
    fi

    echo "${display_name} image built and pushed successfully"
    echo ""
}


_force_new_deployment() {
    local ecs_service="$1"
    local display_name="$2"

    echo "Forcing new deployment for ${display_name} (${ecs_service})..."
    echo "----------------------------------------"

    aws ecs update-service \
        --cluster "$ECS_CLUSTER" \
        --service "$ecs_service" \
        --force-new-deployment \
        --region "$AWS_REGION" \
        --output json | jq '{service: .service.serviceName, status: .service.status, desiredCount: .service.desiredCount}'

    echo "${display_name} deployment triggered"
    echo ""
}


_monitor_deployment() {
    local ecs_services="$1"

    echo "Monitoring deployment status..."
    echo "----------------------------------------"
    echo "Press Ctrl+C to exit monitoring"
    echo ""
    sleep 2

    watch -n 5 'aws ecs describe-services \
      --cluster '"$ECS_CLUSTER"' \
      --services '"$ecs_services"' \
      --region '"$AWS_REGION"' \
      --query "services[*].{Service:serviceName,Status:status,Desired:desiredCount,Running:runningCount,Pending:pendingCount,Deployments:deployments[*].{Status:status,Running:runningCount,Desired:desiredCount,RolloutState:rolloutState}}" \
      --output table'
}


_deploy_services() {
    local step=1
    local monitor_services=""

    # Resolve what to build/deploy. go-validate is a sidecar in the auth task, so
    # it is built for auth/both, and any of auth/go-validate/both redeploys auth.
    local do_registry=false do_auth=false do_govalidate=false
    case "$SERVICE" in
        registry)    do_registry=true ;;
        auth)        do_auth=true; do_govalidate=true ;;
        go-validate) do_govalidate=true ;;
        both)        do_registry=true; do_auth=true; do_govalidate=true ;;
    esac
    local deploy_auth=false
    if [[ "$do_auth" == "true" || "$do_govalidate" == "true" ]]; then
        deploy_auth=true
    fi

    # Calculate total steps (builds + deploys + optional monitor)
    local total_steps=0
    [[ "$do_registry" == "true" ]] && total_steps=$((total_steps + 1))     # build registry
    [[ "$do_auth" == "true" ]] && total_steps=$((total_steps + 1))         # build auth
    [[ "$do_govalidate" == "true" ]] && total_steps=$((total_steps + 1))   # build go-validate
    [[ "$do_registry" == "true" ]] && total_steps=$((total_steps + 1))     # deploy registry
    [[ "$deploy_auth" == "true" ]] && total_steps=$((total_steps + 1))     # deploy auth
    [[ "$SKIP_MONITOR" == "false" ]] && total_steps=$((total_steps + 1))   # monitor

    # Build and push
    if [[ "$do_registry" == "true" ]]; then
        echo "Step ${step}/${total_steps}: Building Registry"
        _build_and_push "$REGISTRY_IMAGE" "Registry"
        step=$((step + 1))
    fi
    if [[ "$do_auth" == "true" ]]; then
        echo "Step ${step}/${total_steps}: Building Auth Server"
        _build_and_push "$AUTH_IMAGE" "Auth Server"
        step=$((step + 1))
    fi
    if [[ "$do_govalidate" == "true" ]]; then
        echo "Step ${step}/${total_steps}: Building go-validate (auth /validate sidecar)"
        _build_and_push "$GOVALIDATE_IMAGE" "go-validate"
        step=$((step + 1))
    fi

    # Force new deployments
    if [[ "$do_registry" == "true" ]]; then
        echo "Step ${step}/${total_steps}: Deploying Registry"
        _force_new_deployment "$REGISTRY_ECS_SERVICE" "Registry"
        monitor_services="$REGISTRY_ECS_SERVICE"
        step=$((step + 1))
    fi
    if [[ "$deploy_auth" == "true" ]]; then
        local auth_label="Auth Server"
        if [[ "$do_govalidate" == "true" ]]; then
            auth_label="Auth Server (incl. go-validate sidecar)"
        fi
        echo "Step ${step}/${total_steps}: Deploying ${auth_label}"
        _force_new_deployment "$AUTH_ECS_SERVICE" "$auth_label"
        if [[ -n "$monitor_services" ]]; then
            monitor_services="$monitor_services $AUTH_ECS_SERVICE"
        else
            monitor_services="$AUTH_ECS_SERVICE"
        fi
        step=$((step + 1))
    fi

    # Monitor
    if [[ "$SKIP_MONITOR" == "false" ]]; then
        echo "Step ${step}/${total_steps}: Monitoring"
        _monitor_deployment "$monitor_services"
    else
        echo "Skipping deployment monitoring (--skip-monitor)"
        echo ""
        echo "To check status manually:"
        echo "  aws ecs describe-services --cluster $ECS_CLUSTER --services $monitor_services --region $AWS_REGION --query 'services[*].{Service:serviceName,Running:runningCount,Desired:desiredCount}' --output table"
    fi
}


# Main
_parse_args "$@"

echo "=========================================="
echo "ECS Deployment Script"
echo "=========================================="
echo "Service:    $SERVICE"
echo "Region:     $AWS_REGION"
echo "Cluster:    $ECS_CLUSTER"
echo "No Cache:   ${NO_CACHE:-false}"
echo "=========================================="
echo ""

_deploy_services
