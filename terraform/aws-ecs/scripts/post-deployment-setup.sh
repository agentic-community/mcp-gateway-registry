#!/bin/bash

################################################################################
# Post-Deployment Setup Script for MCP Gateway
#
# This script automates the post-deployment setup process:
# 1. Saves terraform outputs to JSON
# 2. Validates required resources were created
# 3. Waits for DNS propagation
# 4. Verifies ECS services are running
# 5. Initializes Keycloak (realm, clients, users, groups)
# 6. Initializes MCP scopes on EFS
# 7. Restarts registry and auth services
#
# Usage:
#   ./post-deployment-setup.sh [OPTIONS]
#
# Options:
#   --skip-keycloak        Skip Keycloak initialization
#   --skip-scopes          Skip scopes initialization
#   --skip-restart         Skip service restart
#   --skip-dns-wait        Skip DNS propagation wait
#   --dry-run              Show what would be done without executing
#   --help                 Show this help message
#
# Required Environment Variables:
#   AWS_REGION                    AWS region (default: us-east-1)
#   KEYCLOAK_ADMIN_PASSWORD       Keycloak admin password (or loaded from SSM)
#   INITIAL_ADMIN_PASSWORD        Password for admin user in mcp-gateway realm
#
# Optional Environment Variables:
#   INITIAL_USER_PASSWORD         Password for testuser (default: testpass)
#   LOB1_USER_PASSWORD            Password for lob1-user (default: lob1pass)
#   LOB2_USER_PASSWORD            Password for lob2-user (default: lob2pass)
#
################################################################################

set -euo pipefail

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUTS_FILE="$SCRIPT_DIR/terraform-outputs.json"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Options
SKIP_KEYCLOAK=false
SKIP_SCOPES=false
SKIP_RESTART=false
SKIP_DNS_WAIT=false
DRY_RUN=false

# Counters for summary
STEPS_TOTAL=0
STEPS_PASSED=0
STEPS_FAILED=0
STEPS_SKIPPED=0


log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}


log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}


log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}


log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}


log_step() {
    echo ""
    echo -e "${BOLD}=========================================="
    echo -e "Step $1: $2"
    echo -e "==========================================${NC}"
}


show_help() {
    grep '^#' "$0" | tail -n +2 | head -40 | sed 's/^# //' | sed 's/^#//'
    exit 0
}


_parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-keycloak)
                SKIP_KEYCLOAK=true
                shift
                ;;
            --skip-scopes)
                SKIP_SCOPES=true
                shift
                ;;
            --skip-restart)
                SKIP_RESTART=true
                shift
                ;;
            --skip-dns-wait)
                SKIP_DNS_WAIT=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}


_check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=()

    # Check required tools
    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi

    if ! command -v aws &> /dev/null; then
        missing+=("aws-cli")
    fi

    if ! command -v terraform &> /dev/null; then
        missing+=("terraform")
    fi

    if ! command -v curl &> /dev/null; then
        missing+=("curl")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them before running this script."
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid."
        exit 1
    fi

    log_success "All prerequisites met."
}


_save_terraform_outputs() {
    log_step "1" "Saving Terraform Outputs"
    STEPS_TOTAL=$((STEPS_TOTAL + 1))

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: $SCRIPT_DIR/save-terraform-outputs.sh"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    log_info "Running save-terraform-outputs.sh..."

    if "$SCRIPT_DIR/save-terraform-outputs.sh"; then
        log_success "Terraform outputs saved to $OUTPUTS_FILE"
        STEPS_PASSED=$((STEPS_PASSED + 1))
    else
        log_error "Failed to save terraform outputs"
        STEPS_FAILED=$((STEPS_FAILED + 1))
        return 1
    fi
}


_validate_terraform_outputs() {
    log_step "2" "Validating Terraform Outputs"
    STEPS_TOTAL=$((STEPS_TOTAL + 1))

    if [[ ! -f "$OUTPUTS_FILE" ]]; then
        log_error "Terraform outputs file not found: $OUTPUTS_FILE"
        STEPS_FAILED=$((STEPS_FAILED + 1))
        return 1
    fi

    log_info "Validating required resources..."

    # Core required outputs (always needed)
    local required_outputs=(
        "vpc_id"
        "ecs_cluster_name"
        "ecs_cluster_arn"
        "mcp_gateway_url"
        "mcp_gateway_auth_url"
        "keycloak_url"
        "mcp_gateway_efs_id"
    )
    
    # Note: registry_url is only set in custom domain mode
    # cloudfront_mcp_gateway_url is only set in CloudFront mode
    # At least one of these should be available for a valid deployment

    local missing_outputs=()
    local validation_passed=true

    for output in "${required_outputs[@]}"; do
        local value
        value=$(jq -r ".$output.value // empty" "$OUTPUTS_FILE" 2>/dev/null)

        if [[ -z "$value" || "$value" == "null" ]]; then
            missing_outputs+=("$output")
            validation_passed=false
            log_error "  Missing or empty: $output"
        else
            log_success "  Found: $output = $value"
        fi
    done

    if [[ "$validation_passed" == "true" ]]; then
        log_success "All required terraform outputs validated successfully."
        STEPS_PASSED=$((STEPS_PASSED + 1))

        # Export values for later use
        export KEYCLOAK_ADMIN_URL=$(jq -r '.keycloak_url.value' "$OUTPUTS_FILE")
        export AUTH_SERVER_EXTERNAL_URL=$(jq -r '.mcp_gateway_auth_url.value' "$OUTPUTS_FILE")
        export ECS_CLUSTER_NAME=$(jq -r '.ecs_cluster_name.value' "$OUTPUTS_FILE")
        
        # REGISTRY_URL: prefer custom domain, fallback to CloudFront URL
        local registry_url=$(jq -r '.registry_url.value // empty' "$OUTPUTS_FILE")
        local cloudfront_url=$(jq -r '.cloudfront_mcp_gateway_url.value // empty' "$OUTPUTS_FILE")
        
        if [[ -n "$registry_url" && "$registry_url" != "null" ]]; then
            export REGISTRY_URL="$registry_url"
        elif [[ -n "$cloudfront_url" && "$cloudfront_url" != "null" ]]; then
            export REGISTRY_URL="$cloudfront_url"
            log_info "Using CloudFront URL as REGISTRY_URL (custom domain not configured)"
        else
            export REGISTRY_URL=$(jq -r '.mcp_gateway_url.value' "$OUTPUTS_FILE")
            log_warning "Using ALB URL as REGISTRY_URL (no HTTPS configured)"
        fi
        
        # Also export CloudFront URL if available (for init-keycloak.sh)
        if [[ -n "$cloudfront_url" && "$cloudfront_url" != "null" ]]; then
            export CLOUDFRONT_REGISTRY_URL="$cloudfront_url"
        fi

        log_info "Exported configuration:"
        log_info "  KEYCLOAK_ADMIN_URL: $KEYCLOAK_ADMIN_URL"
        log_info "  REGISTRY_URL: $REGISTRY_URL"
        log_info "  CLOUDFRONT_REGISTRY_URL: ${CLOUDFRONT_REGISTRY_URL:-<not set>}"
        log_info "  AUTH_SERVER_EXTERNAL_URL: $AUTH_SERVER_EXTERNAL_URL"
        log_info "  ECS_CLUSTER_NAME: $ECS_CLUSTER_NAME"

        return 0
    else
        log_error "Missing required outputs: ${missing_outputs[*]}"
        log_error "Please check your terraform apply completed successfully."
        STEPS_FAILED=$((STEPS_FAILED + 1))
        return 1
    fi
}


_wait_for_dns_propagation() {
    log_step "3" "Waiting for DNS Propagation"
    STEPS_TOTAL=$((STEPS_TOTAL + 1))

    if [[ "$SKIP_DNS_WAIT" == "true" ]]; then
        log_warning "Skipping DNS propagation wait (--skip-dns-wait)"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would wait for DNS propagation"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    local endpoints=(
        "$KEYCLOAK_ADMIN_URL"
        "$REGISTRY_URL"
    )

    local max_attempts=60
    local wait_interval=10
    local all_resolved=false

    log_info "Checking DNS resolution for endpoints..."
    log_info "This may take up to 10 minutes for new deployments."

    for attempt in $(seq 1 $max_attempts); do
        all_resolved=true

        for endpoint in "${endpoints[@]}"; do
            # Extract hostname from URL
            local hostname
            hostname=$(echo "$endpoint" | sed 's|https://||' | sed 's|http://||' | cut -d'/' -f1)

            if host "$hostname" &> /dev/null; then
                log_success "  DNS resolved: $hostname"
            else
                log_warning "  DNS not yet resolved: $hostname"
                all_resolved=false
            fi
        done

        if [[ "$all_resolved" == "true" ]]; then
            log_success "All DNS records resolved!"
            STEPS_PASSED=$((STEPS_PASSED + 1))
            return 0
        fi

        if [[ $attempt -lt $max_attempts ]]; then
            log_info "Attempt $attempt/$max_attempts - waiting ${wait_interval}s..."
            sleep $wait_interval
        fi
    done

    log_error "DNS propagation timeout. Some endpoints may not be ready."
    log_warning "You can retry later or use --skip-dns-wait to proceed anyway."
    STEPS_FAILED=$((STEPS_FAILED + 1))
    return 1
}


_verify_ecs_services() {
    log_step "4" "Verifying ECS Services"
    STEPS_TOTAL=$((STEPS_TOTAL + 1))

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would verify ECS services"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    # Services in mcp-gateway-ecs-cluster
    local mcp_gateway_services=(
        "mcp-gateway-v2-registry"
        "mcp-gateway-v2-auth"
        "mcp-gateway-v2-mcpgw"
    )

    # Keycloak runs in its own cluster
    local keycloak_cluster="keycloak"
    local keycloak_service="keycloak"

    local max_attempts=40
    local wait_interval=20

    log_info "Checking ECS services are running..."

    for attempt in $(seq 1 $max_attempts); do
        local all_healthy=true

        # Check MCP Gateway services in mcp-gateway-ecs-cluster
        for service in "${mcp_gateway_services[@]}"; do
            local status
            status=$(aws ecs describe-services \
                --cluster "$ECS_CLUSTER_NAME" \
                --services "$service" \
                --region "$AWS_REGION" \
                --query 'services[0].{running:runningCount,desired:desiredCount}' \
                --output json 2>/dev/null || echo '{}')

            local running
            local desired
            running=$(echo "$status" | jq -r '.running // 0')
            desired=$(echo "$status" | jq -r '.desired // 0')

            if [[ "$running" -ge "$desired" && "$desired" -gt 0 ]]; then
                log_success "  $service: $running/$desired running (cluster: $ECS_CLUSTER_NAME)"
            else
                log_warning "  $service: $running/$desired running (waiting...)"
                all_healthy=false
            fi
        done

        # Check Keycloak in its own cluster
        local kc_status
        kc_status=$(aws ecs describe-services \
            --cluster "$keycloak_cluster" \
            --services "$keycloak_service" \
            --region "$AWS_REGION" \
            --query 'services[0].{running:runningCount,desired:desiredCount}' \
            --output json 2>/dev/null || echo '{}')

        local kc_running
        local kc_desired
        kc_running=$(echo "$kc_status" | jq -r '.running // 0')
        kc_desired=$(echo "$kc_status" | jq -r '.desired // 0')

        if [[ "$kc_running" -ge "$kc_desired" && "$kc_desired" -gt 0 ]]; then
            log_success "  $keycloak_service: $kc_running/$kc_desired running (cluster: $keycloak_cluster)"
        else
            log_warning "  $keycloak_service: $kc_running/$kc_desired running (cluster: $keycloak_cluster, waiting...)"
            all_healthy=false
        fi

        if [[ "$all_healthy" == "true" ]]; then
            log_success "All ECS services are running!"
            STEPS_PASSED=$((STEPS_PASSED + 1))
            return 0
        fi

        if [[ $attempt -lt $max_attempts ]]; then
            log_info "Attempt $attempt/$max_attempts - waiting ${wait_interval}s for services..."
            sleep $wait_interval
        fi
    done

    log_error "ECS services did not reach healthy state in time."
    log_warning "Check CloudWatch logs for errors."
    STEPS_FAILED=$((STEPS_FAILED + 1))
    return 1
}


_initialize_keycloak() {
    log_step "5" "Initializing Keycloak"
    STEPS_TOTAL=$((STEPS_TOTAL + 1))

    if [[ "$SKIP_KEYCLOAK" == "true" ]]; then
        log_warning "Skipping Keycloak initialization (--skip-keycloak)"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: $SCRIPT_DIR/init-keycloak.sh"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    # Try to load INITIAL_ADMIN_PASSWORD from Secrets Manager if not set
    if [[ -z "${INITIAL_ADMIN_PASSWORD:-}" ]]; then
        log_info "INITIAL_ADMIN_PASSWORD not set, attempting to load from Secrets Manager..."
        
        # Find the admin password secret by name pattern (mcp-gateway-v2-admin-password-*)
        local secret_name
        secret_name=$(aws secretsmanager list-secrets \
            --region "$AWS_REGION" \
            --filter Key=name,Values=mcp-gateway-v2-admin-password \
            --query 'SecretList[0].Name' \
            --output text 2>/dev/null)
        
        if [[ -n "$secret_name" && "$secret_name" != "None" ]]; then
            INITIAL_ADMIN_PASSWORD=$(aws secretsmanager get-secret-value \
                --secret-id "$secret_name" \
                --region "$AWS_REGION" \
                --query 'SecretString' \
                --output text 2>/dev/null)
            
            if [[ -n "$INITIAL_ADMIN_PASSWORD" ]]; then
                export INITIAL_ADMIN_PASSWORD
                log_success "Loaded INITIAL_ADMIN_PASSWORD from Secrets Manager ($secret_name)"
            fi
        fi
    fi

    # Final check - if still not set, error out
    if [[ -z "${INITIAL_ADMIN_PASSWORD:-}" ]]; then
        log_error "INITIAL_ADMIN_PASSWORD could not be loaded from Secrets Manager."
        log_error "Either set it manually or ensure the secret exists:"
        log_error "  export INITIAL_ADMIN_PASSWORD='YourSecurePassword123'"
        STEPS_FAILED=$((STEPS_FAILED + 1))
        return 1
    fi

    log_info "Running init-keycloak.sh..."
    log_info "Using KEYCLOAK_ADMIN_URL: $KEYCLOAK_ADMIN_URL"

    # Export variables for init-keycloak.sh
    export KEYCLOAK_ADMIN_URL
    export REGISTRY_URL
    export AUTH_SERVER_EXTERNAL_URL
    export AWS_REGION

    if "$SCRIPT_DIR/init-keycloak.sh"; then
        log_success "Keycloak initialized successfully!"
        STEPS_PASSED=$((STEPS_PASSED + 1))
    else
        log_error "Keycloak initialization failed."
        log_warning "Check the error messages above and try running init-keycloak.sh manually."
        STEPS_FAILED=$((STEPS_FAILED + 1))
        return 1
    fi
}


_initialize_scopes() {
    log_step "6" "Initializing MCP Scopes"
    STEPS_TOTAL=$((STEPS_TOTAL + 1))

    if [[ "$SKIP_SCOPES" == "true" ]]; then
        log_warning "Skipping scopes initialization (--skip-scopes)"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    # Detect storage backend from terraform outputs
    local documentdb_endpoint
    documentdb_endpoint=$(jq -r '.documentdb_cluster_endpoint.value // empty' "$OUTPUTS_FILE" 2>/dev/null)

    if [[ -n "$documentdb_endpoint" && "$documentdb_endpoint" != "null" ]]; then
        # DocumentDB mode
        log_info "Detected DocumentDB storage backend"
        log_info "DocumentDB endpoint: $documentdb_endpoint"

        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would run: $SCRIPT_DIR/run-documentdb-init.sh"
            STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
            return 0
        fi

        log_info "Running DocumentDB initialization (indexes + scopes)..."

        if "$SCRIPT_DIR/run-documentdb-init.sh"; then
            log_success "DocumentDB initialized with indexes and scopes!"
            STEPS_PASSED=$((STEPS_PASSED + 1))
        else
            log_error "DocumentDB initialization failed."
            STEPS_FAILED=$((STEPS_FAILED + 1))
            return 1
        fi
    else
        # EFS mode (default)
        log_info "Using EFS storage backend"

        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would run: $SCRIPT_DIR/run-scopes-init-task.sh --skip-build"
            STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
            return 0
        fi

        log_info "Running scopes initialization task on EFS..."

        if "$SCRIPT_DIR/run-scopes-init-task.sh" --skip-build; then
            log_success "MCP scopes initialized on EFS!"
            STEPS_PASSED=$((STEPS_PASSED + 1))
        else
            log_error "Scopes initialization failed."
            STEPS_FAILED=$((STEPS_FAILED + 1))
            return 1
        fi
    fi
}


_restart_services() {
    log_step "7" "Restarting Registry and Auth Services"
    STEPS_TOTAL=$((STEPS_TOTAL + 1))

    if [[ "$SKIP_RESTART" == "true" ]]; then
        log_warning "Skipping service restart (--skip-restart)"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would restart ECS services"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    local services_to_restart=(
        "mcp-gateway-v2-registry"
        "mcp-gateway-v2-auth"
    )

    log_info "Forcing new deployments for services to pick up new configuration..."

    for service in "${services_to_restart[@]}"; do
        log_info "  Restarting: $service"

        if aws ecs update-service \
            --cluster "$ECS_CLUSTER_NAME" \
            --service "$service" \
            --force-new-deployment \
            --region "$AWS_REGION" &> /dev/null; then
            log_success "  Restart initiated: $service"
        else
            log_error "  Failed to restart: $service"
        fi
    done

    log_info "Waiting for services to stabilize..."

    local max_attempts=40
    local wait_interval=10

    for attempt in $(seq 1 $max_attempts); do
        local all_stable=true

        for service in "${services_to_restart[@]}"; do
            local status
            status=$(aws ecs describe-services \
                --cluster "$ECS_CLUSTER_NAME" \
                --services "$service" \
                --region "$AWS_REGION" \
                --query 'services[0].deployments | length(@)' \
                --output text 2>/dev/null || echo "0")

            if [[ "$status" == "1" ]]; then
                log_success "  $service: deployment complete"
            else
                log_warning "  $service: deployment in progress ($status active)"
                all_stable=false
            fi
        done

        if [[ "$all_stable" == "true" ]]; then
            log_success "All services restarted successfully!"
            STEPS_PASSED=$((STEPS_PASSED + 1))
            return 0
        fi

        if [[ $attempt -lt $max_attempts ]]; then
            log_info "Attempt $attempt/$max_attempts - waiting ${wait_interval}s..."
            sleep $wait_interval
        fi
    done

    log_warning "Services are still deploying. They should complete shortly."
    STEPS_PASSED=$((STEPS_PASSED + 1))
}


_verify_endpoints() {
    log_step "8" "Verifying Application Endpoints"
    STEPS_TOTAL=$((STEPS_TOTAL + 1))

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would verify application endpoints"
        STEPS_SKIPPED=$((STEPS_SKIPPED + 1))
        return 0
    fi

    log_info "Testing endpoint health..."

    local endpoints=(
        "$REGISTRY_URL/health|Registry Health"
        "$KEYCLOAK_ADMIN_URL/admin/|Keycloak Admin"
    )

    local all_healthy=true

    for endpoint_info in "${endpoints[@]}"; do
        local url="${endpoint_info%|*}"
        local name="${endpoint_info#*|}"

        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null || echo "000")

        if [[ "$http_code" =~ ^(200|301|302)$ ]]; then
            log_success "  $name: HTTP $http_code"
        else
            log_warning "  $name: HTTP $http_code (may still be starting)"
            all_healthy=false
        fi
    done

    if [[ "$all_healthy" == "true" ]]; then
        log_success "All endpoints responding!"
        STEPS_PASSED=$((STEPS_PASSED + 1))
    else
        log_warning "Some endpoints not yet responding. They may need more time to start."
        STEPS_PASSED=$((STEPS_PASSED + 1))
    fi
}


_print_summary() {
    echo ""
    echo -e "${BOLD}=========================================="
    echo -e "Post-Deployment Setup Summary"
    echo -e "==========================================${NC}"
    echo ""
    echo -e "Total Steps: $STEPS_TOTAL"
    echo -e "${GREEN}Passed:      $STEPS_PASSED${NC}"
    echo -e "${RED}Failed:      $STEPS_FAILED${NC}"
    echo -e "${YELLOW}Skipped:     $STEPS_SKIPPED${NC}"
    echo ""

    if [[ "$STEPS_FAILED" -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}Post-deployment setup completed successfully!${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Access Keycloak Admin: $KEYCLOAK_ADMIN_URL/admin"
        echo "  2. Access Registry: $REGISTRY_URL"
        echo "  3. Test authentication flow"
        echo ""
    else
        echo -e "${RED}${BOLD}Post-deployment setup completed with errors.${NC}"
        echo ""
        echo "Please review the error messages above and:"
        echo "  1. Check CloudWatch logs for service errors"
        echo "  2. Verify terraform apply completed successfully"
        echo "  3. Re-run this script with appropriate --skip-* flags"
        echo ""
    fi
}


main() {
    _parse_arguments "$@"

    echo -e "${BOLD}=========================================="
    echo -e "MCP Gateway Post-Deployment Setup"
    echo -e "==========================================${NC}"
    echo ""
    echo "AWS Region: $AWS_REGION"
    echo "Terraform Dir: $TERRAFORM_DIR"
    echo "Dry Run: $DRY_RUN"
    echo ""

    _check_prerequisites

    # Step 1: Save terraform outputs
    _save_terraform_outputs || true

    # Step 2: Validate outputs
    if ! _validate_terraform_outputs; then
        log_error "Cannot proceed without valid terraform outputs."
        _print_summary
        exit 1
    fi

    # Step 3: Wait for DNS
    _wait_for_dns_propagation || true

    # Step 4: Verify ECS services
    _verify_ecs_services || true

    # Step 5: Initialize Keycloak
    _initialize_keycloak || true

    # Step 6: Initialize scopes
    _initialize_scopes || true

    # Step 7: Restart services
    _restart_services || true

    # Step 8: Verify endpoints
    _verify_endpoints || true

    # Print summary
    _print_summary

    # Exit with error if any steps failed
    if [[ "$STEPS_FAILED" -gt 0 ]]; then
        exit 1
    fi
}


# Run main function
main "$@"
