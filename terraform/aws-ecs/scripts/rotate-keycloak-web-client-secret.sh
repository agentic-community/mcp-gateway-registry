#!/bin/bash

# Rotate and sync mcp-gateway-web client secret between Keycloak and AWS Secrets Manager
#
# PREREQUISITES:
#   - Keycloak must be fully initialized (run init-keycloak.sh first)
#   - mcp-gateway-web client must exist in Keycloak
#   - Keycloak admin credentials must be configured in terraform.tfvars or .env
#   - AWS Secrets Manager must have mcp-gateway-keycloak-client-secret
#
# This script:
# 1. Connects to Keycloak admin console
# 2. Generates a NEW client secret in Keycloak (Keycloak is source of truth)
# 3. Updates AWS Secrets Manager with the new Keycloak-generated secret
#
# Use this for:
#   - Secret rotation (security best practice)
#   - Syncing Keycloak and AWS Secrets Manager when out of sync
#   - After manual client modifications in Keycloak admin console

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() { echo -e "${GREEN}OK${NC} $1"; }
print_error() { echo -e "${RED}ERROR${NC} $1"; }
print_info() { echo -e "${YELLOW}INFO${NC} $1"; }

detect_aws_region() {
    local region=""

    if [ -n "${AWS_REGION:-}" ]; then
        region="$AWS_REGION"
    elif [ -n "${AWS_DEFAULT_REGION:-}" ]; then
        region="$AWS_DEFAULT_REGION"
    fi

    if [ -z "$region" ] && [ -n "${TERRAFORM_DIR:-}" ] && [ -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
        region="$(awk -F= '/^[[:space:]]*aws_region[[:space:]]*=/{gsub(/"/,"",$2); gsub(/[[:space:]]/,"",$2); print $2; exit}' "$TERRAFORM_DIR/terraform.tfvars" 2>/dev/null || true)"
    fi

    if [ -z "$region" ] && command -v aws >/dev/null 2>&1; then
        region="$(aws configure get region 2>/dev/null || true)"
        if [ "$region" = "None" ]; then
            region=""
        fi
    fi

    if [ -z "$region" ]; then
        region="us-east-1"
    fi

    echo "$region"
}

# Verify that a Secrets Manager secret is valid JSON for ECS JSON-key extraction.
# ECS task definitions reference these secrets via `...:client_secret::`, which
# requires the stored SecretString to be valid JSON with a `client_secret` key.
verify_secretsmanager_client_secret() {
    local secret_id="$1"
    local expected_client_id="$2"
    local expected_client_secret="$3"
    local region="$4"

    local secret_string=""
    secret_string="$(aws secretsmanager get-secret-value \
        --secret-id "$secret_id" \
        --region "$region" \
        --query SecretString \
        --output text 2>/dev/null || true)"

    if [ -z "$secret_string" ]; then
        print_error "Secrets Manager secret '$secret_id' is empty or unreadable"
        return 1
    fi

    if ! echo "$secret_string" | jq -e . >/dev/null 2>&1; then
        print_error "Secrets Manager secret '$secret_id' is not valid JSON"
        echo "SecretString (truncated): ${secret_string:0:200}"
        return 1
    fi

    local actual_client_id=""
    local actual_client_secret=""
    actual_client_id="$(echo "$secret_string" | jq -r '.client_id // empty' 2>/dev/null || true)"
    actual_client_secret="$(echo "$secret_string" | jq -r '.client_secret // empty' 2>/dev/null || true)"

    if [ "$actual_client_id" != "$expected_client_id" ]; then
        print_error "Secret '$secret_id' has unexpected client_id '${actual_client_id}' (expected '${expected_client_id}')"
        return 1
    fi

    if [ -z "$actual_client_secret" ]; then
        print_error "Secret '$secret_id' is missing client_secret"
        return 1
    fi

    if [ "$actual_client_secret" != "$expected_client_secret" ]; then
        print_error "Secret '$secret_id' client_secret does not match the Keycloak-generated value"
        echo "This will cause Keycloak /token to return 401 (invalid_client)."
        return 1
    fi

    return 0
}

_update_secretsmanager_secrets_by_prefix() {
    local secret_name_prefix="$1"
    local secret_payload="$2"
    local region="$3"

    local matches=""
    matches="$(aws secretsmanager list-secrets \
        --region "$region" \
        --query "SecretList[?starts_with(Name, \`${secret_name_prefix}\`)].Name" \
        --output text 2>/dev/null || true)"

    if [ -z "$matches" ]; then
        print_error "No Secrets Manager secrets found with prefix '${secret_name_prefix}' in region '${region}'."
        return 1
    fi

    for secret_id in $matches; do
        if [ -z "$secret_id" ]; then
            continue
        fi
        print_info "Updating Secrets Manager secret: $secret_id"
        aws secretsmanager update-secret \
            --secret-id "$secret_id" \
            --secret-string "$secret_payload" \
            --region "$region" > /dev/null
    done

    return 0
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$TERRAFORM_DIR")"

print_info "Rotating Keycloak client secret for mcp-gateway-web"

# Try to load from .env file first (same as init-keycloak.sh)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    print_info "Loaded configuration from .env file"
fi

# Fall back to terraform.tfvars if .env doesn't have the values
if [ -z "$KEYCLOAK_ADMIN_URL" ]; then
    if [ -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
        KEYCLOAK_ADMIN_URL=$(grep "^keycloak_domain" "$TERRAFORM_DIR/terraform.tfvars" | cut -d'"' -f2)
        if [ -n "$KEYCLOAK_ADMIN_URL" ]; then
            KEYCLOAK_ADMIN_URL="https://${KEYCLOAK_ADMIN_URL}"
        fi
    fi
fi

if [ -z "$KEYCLOAK_ADMIN" ] && [ -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
    KEYCLOAK_ADMIN=$(grep "^keycloak_admin" "$TERRAFORM_DIR/terraform.tfvars" | cut -d'"' -f2)
fi

if [ -z "$KEYCLOAK_ADMIN_PASSWORD" ] && [ -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
    KEYCLOAK_ADMIN_PASSWORD=$(grep "^keycloak_admin_password" "$TERRAFORM_DIR/terraform.tfvars" | cut -d'"' -f2)
fi

# Use KEYCLOAK_ADMIN_URL as the base URL
KEYCLOAK_URL="${KEYCLOAK_ADMIN_URL:-https://kc.mycorp.click}"
REALM="mcp-gateway"
CLIENT_ID="mcp-gateway-web"
AWS_REGION_EFFECTIVE="$(detect_aws_region)"

print_info "Keycloak URL: $KEYCLOAK_URL"
print_info "Realm: $REALM"
print_info "Client ID: $CLIENT_ID"
print_info "AWS region: $AWS_REGION_EFFECTIVE"

# Get the client secret from AWS Secrets Manager
print_info "Retrieving client secret from AWS Secrets Manager..."
SECRET_JSON=$(aws secretsmanager get-secret-value \
    --secret-id mcp-gateway-keycloak-client-secret \
    --region "$AWS_REGION_EFFECTIVE" \
    --query 'SecretString' \
    --output text)

CLIENT_SECRET=$(echo "$SECRET_JSON" | jq -r '.client_secret // empty')

if [ -z "$CLIENT_SECRET" ]; then
    print_error "Could not retrieve client secret from Secrets Manager"
    exit 1
fi

print_success "Client secret retrieved"

# Get admin access token
print_info "Getting Keycloak admin token..."
TOKEN_RESPONSE=$(curl -s -k -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${KEYCLOAK_ADMIN}" \
    -d "password=${KEYCLOAK_ADMIN_PASSWORD}" \
    -d "grant_type=password" \
    -d "client_id=admin-cli")

ADMIN_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.access_token // empty')

if [ -z "$ADMIN_TOKEN" ]; then
    print_error "Failed to get admin token"
    echo "Response:"
    echo "$TOKEN_RESPONSE"
    exit 1
fi

print_success "Admin token obtained"

# Get all clients in the realm
print_info "Fetching clients in realm $REALM..."
CLIENTS_RESPONSE=$(curl -s -k -X GET "${KEYCLOAK_URL}/admin/realms/${REALM}/clients" \
    -H "Authorization: Bearer ${ADMIN_TOKEN}" \
    -H "Content-Type: application/json")

# Find the client UUID
CLIENT_UUID=$(echo "$CLIENTS_RESPONSE" | jq -r ".[] | select(.clientId == \"${CLIENT_ID}\") | .id" | head -1)

if [ -z "$CLIENT_UUID" ]; then
    print_error "Client $CLIENT_ID not found in realm $REALM"
    print_info "Available clients:"
    echo "$CLIENTS_RESPONSE" | jq -r '.[].clientId'
    exit 1
fi

print_success "Found client UUID: $CLIENT_UUID"

# Generate a new client secret in Keycloak
print_info "Generating new client secret in Keycloak..."
SECRET_RESPONSE=$(curl -s -k -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}/client-secret" \
    -H "Authorization: Bearer ${ADMIN_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{}')

GENERATED_SECRET=$(echo "$SECRET_RESPONSE" | jq -r '.value // empty')

if [ -z "$GENERATED_SECRET" ]; then
    print_error "Failed to generate client secret"
    echo "Response: $SECRET_RESPONSE" | jq '.'
    exit 1
fi

print_success "New client secret generated in Keycloak"

# Update the secret in AWS Secrets Manager with the Keycloak-generated secret
print_info "Updating AWS Secrets Manager with Keycloak-generated secret..."
SECRET_PAYLOAD="$(jq -n \
    --arg client_id "${CLIENT_ID}" \
    --arg client_secret "${GENERATED_SECRET}" \
    '{client_id: $client_id, client_secret: $client_secret}' \
)"
_update_secretsmanager_secrets_by_prefix \
    "mcp-gateway-keycloak-client-secret" \
    "$SECRET_PAYLOAD" \
    "$AWS_REGION_EFFECTIVE"

print_success "Secrets Manager updated"
for secret_id in $(aws secretsmanager list-secrets \
    --region "$AWS_REGION_EFFECTIVE" \
    --query "SecretList[?starts_with(Name, \`mcp-gateway-keycloak-client-secret\`)].Name" \
    --output text 2>/dev/null || true); do
    if [ -z "$secret_id" ]; then
        continue
    fi
    if ! verify_secretsmanager_client_secret \
        "$secret_id" \
        "$CLIENT_ID" \
        "$GENERATED_SECRET" \
        "$AWS_REGION_EFFECTIVE"; then
        print_error "Secrets Manager verification failed for '$secret_id'. Aborting."
        exit 1
    fi
done

# Verify the client is configured correctly
print_info "Verifying client configuration..."
CLIENT_CONFIG=$(curl -s -k -X GET "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}" \
    -H "Authorization: Bearer ${ADMIN_TOKEN}" \
    -H "Content-Type: application/json")

print_success "Client configuration verified"

echo ""
echo "=================================================="
echo "Keycloak Client Secret Rotation Complete!"
echo "=================================================="
echo ""
echo "Client Details:"
echo "  Client ID: $CLIENT_ID"
echo "  Realm: $REALM"
echo "  Client UUID: $CLIENT_UUID"
echo ""
echo "Configuration:"
echo "  Enabled: $(echo "$CLIENT_CONFIG" | jq -r '.enabled')"
echo "  Auth Type: $(echo "$CLIENT_CONFIG" | jq -r '.clientAuthenticatorType')"
echo "  Public Client: $(echo "$CLIENT_CONFIG" | jq -r '.publicClient')"
echo ""
echo "Secret Sync Status:"
echo "  ✓ New secret generated in Keycloak"
echo "  ✓ AWS Secrets Manager updated"
echo ""
echo "Next Steps:"
echo "  1. Restart registry ECS tasks to pick up new secret from Secrets Manager"
echo "  2. Verify login functionality at: https://registry.mycorp.click"
echo ""
