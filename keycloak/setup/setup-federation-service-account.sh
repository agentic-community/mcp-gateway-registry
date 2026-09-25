#!/bin/bash
# Setup Federation Service Account in Keycloak
#
# Creates a dedicated M2M client for peer-to-peer federation with a
# 6-month access token lifetime. This client is separate from the main
# mcp-gateway-m2m client so it does not affect other token lifetimes.
#
# The service account is assigned to the federation-service group, which
# grants read-only access to servers and agents for peer sync.
#
# Prerequisites:
#   - Keycloak running and initialized (init-keycloak.sh completed)
#   - KEYCLOAK_ADMIN_PASSWORD environment variable set
#   - jq installed
#
# Usage:
#   export KEYCLOAK_ADMIN_PASSWORD="your-password"
#   bash keycloak/setup/setup-federation-service-account.sh

set -e

# Configuration
ADMIN_URL="${KEYCLOAK_ADMIN_URL:-http://localhost:8080}"
REALM="mcp-gateway"
ADMIN_USER="${KEYCLOAK_ADMIN:-admin}"
ADMIN_PASS="${KEYCLOAK_ADMIN_PASSWORD}"

FEDERATION_CLIENT_ID="federation-peer-m2m"
FEDERATION_GROUP="federation-service"
SERVICE_ACCOUNT="service-account-${FEDERATION_CLIENT_ID}"

# Token lifetime: 6 months in seconds (180 days * 24 hours * 60 minutes * 60 seconds)
TOKEN_LIFETIME_SECONDS=15552000

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Setting up Federation Service Account for Keycloak"
echo "=============================================="
echo "Client ID: $FEDERATION_CLIENT_ID"
echo "Group: $FEDERATION_GROUP"
echo "Token Lifetime: 180 days (6 months)"
echo ""


# --- Private functions ---


_get_admin_token() {
    echo "Getting admin token..."
    KEYCLOAK_ADMIN_REALM="${KEYCLOAK_ADMIN_REALM:-master}"
    TOKEN=$(curl -s -X POST "${ADMIN_URL}/realms/${KEYCLOAK_ADMIN_REALM}/protocol/openid-connect/token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=${ADMIN_USER}" \
        -d "password=${ADMIN_PASS}" \
        -d "grant_type=password" \
        -d "client_id=admin-cli" | jq -r '.access_token // empty')

    if [ -z "$TOKEN" ]; then
        echo -e "${RED}Failed to get admin token. Check KEYCLOAK_ADMIN_PASSWORD.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Admin token obtained${NC}"
}


_ensure_federation_group() {
    echo "Checking if federation-service group exists..."
    GROUP_ID=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/groups" | \
        jq -r ".[] | select(.name==\"${FEDERATION_GROUP}\") | .id")

    if [ -n "$GROUP_ID" ] && [ "$GROUP_ID" != "null" ]; then
        echo -e "${GREEN}Group '${FEDERATION_GROUP}' exists (ID: ${GROUP_ID})${NC}"
        return 0
    fi

    echo "Creating group '${FEDERATION_GROUP}'..."
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${ADMIN_URL}/admin/realms/${REALM}/groups" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${FEDERATION_GROUP}\"}")

    if [ "$RESPONSE" = "201" ]; then
        echo -e "${GREEN}Group created${NC}"
        GROUP_ID=$(curl -s -H "Authorization: Bearer $TOKEN" \
            "${ADMIN_URL}/admin/realms/${REALM}/groups" | \
            jq -r ".[] | select(.name==\"${FEDERATION_GROUP}\") | .id")
    else
        echo -e "${RED}Failed to create group. HTTP: ${RESPONSE}${NC}"
        exit 1
    fi
}


_create_federation_client() {
    echo "Checking if federation client exists..."
    EXISTING_CLIENT=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/clients?clientId=${FEDERATION_CLIENT_ID}" | \
        jq -r '.[0].id // empty')

    if [ -n "$EXISTING_CLIENT" ] && [ "$EXISTING_CLIENT" != "null" ]; then
        echo -e "${GREEN}Client '${FEDERATION_CLIENT_ID}' already exists (ID: ${EXISTING_CLIENT})${NC}"
        CLIENT_UUID="$EXISTING_CLIENT"
        return 0
    fi

    echo "Creating federation M2M client..."
    CLIENT_JSON='{
        "clientId": "'"${FEDERATION_CLIENT_ID}"'",
        "name": "Federation Peer M2M Client",
        "description": "Machine-to-machine client for peer registry federation sync with extended token lifetime",
        "enabled": true,
        "clientAuthenticatorType": "client-secret",
        "serviceAccountsEnabled": true,
        "standardFlowEnabled": false,
        "implicitFlowEnabled": false,
        "directAccessGrantsEnabled": false,
        "publicClient": false,
        "protocol": "openid-connect",
        "attributes": {
            "access.token.lifespan": "'"${TOKEN_LIFETIME_SECONDS}"'"
        }
    }'

    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${ADMIN_URL}/admin/realms/${REALM}/clients" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "$CLIENT_JSON")

    if [ "$RESPONSE" = "201" ]; then
        echo -e "${GREEN}Federation client created${NC}"
    else
        echo -e "${RED}Failed to create client. HTTP: ${RESPONSE}${NC}"
        exit 1
    fi

    # Get the client UUID
    CLIENT_UUID=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/clients?clientId=${FEDERATION_CLIENT_ID}" | \
        jq -r '.[0].id')
    echo "Client UUID: $CLIENT_UUID"
}


_get_client_secret() {
    echo "Retrieving client secret..."
    CLIENT_SECRET=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}/client-secret" | \
        jq -r '.value // empty')

    if [ -z "$CLIENT_SECRET" ] || [ "$CLIENT_SECRET" = "null" ]; then
        echo "Generating new client secret..."
        CLIENT_SECRET=$(curl -s -X POST \
            -H "Authorization: Bearer $TOKEN" \
            "${ADMIN_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}/client-secret" | \
            jq -r '.value // empty')
    fi

    if [ -z "$CLIENT_SECRET" ] || [ "$CLIENT_SECRET" = "null" ]; then
        echo -e "${RED}Failed to retrieve client secret${NC}"
        exit 1
    fi
    echo -e "${GREEN}Client secret retrieved${NC}"
}


_setup_service_account() {
    echo "Setting up service account..."

    # Get the service account user for this client
    SA_USER=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}/service-account-user")
    SA_USER_ID=$(echo "$SA_USER" | jq -r '.id // empty')

    if [ -z "$SA_USER_ID" ] || [ "$SA_USER_ID" = "null" ]; then
        echo -e "${RED}Service account user not found for client${NC}"
        exit 1
    fi
    echo "Service account user ID: $SA_USER_ID"

    # Assign to federation-service group
    echo "Assigning service account to '${FEDERATION_GROUP}' group..."
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${ADMIN_URL}/admin/realms/${REALM}/users/${SA_USER_ID}/groups/${GROUP_ID}" \
        -H "Authorization: Bearer $TOKEN")

    if [ "$RESPONSE" = "204" ]; then
        echo -e "${GREEN}Service account assigned to '${FEDERATION_GROUP}' group${NC}"
    else
        echo -e "${RED}Failed to assign to group. HTTP: ${RESPONSE}${NC}"
        exit 1
    fi
}


_add_groups_mapper() {
    echo "Adding groups mapper to federation client..."

    # Check if mapper already exists
    EXISTING_MAPPER=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}/protocol-mappers/models" | \
        jq -r '.[] | select(.name=="groups") | .id')

    if [ -n "$EXISTING_MAPPER" ] && [ "$EXISTING_MAPPER" != "null" ]; then
        echo -e "${GREEN}Groups mapper already exists${NC}"
        return 0
    fi

    GROUPS_MAPPER='{
        "name": "groups",
        "protocol": "openid-connect",
        "protocolMapper": "oidc-group-membership-mapper",
        "consentRequired": false,
        "config": {
            "full.path": "false",
            "id.token.claim": "true",
            "access.token.claim": "true",
            "claim.name": "groups",
            "userinfo.token.claim": "true"
        }
    }'

    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${ADMIN_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}/protocol-mappers/models" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "$GROUPS_MAPPER")

    if [ "$RESPONSE" = "201" ] || [ "$RESPONSE" = "409" ]; then
        echo -e "${GREEN}Groups mapper configured${NC}"
    else
        echo -e "${RED}Failed to add groups mapper. HTTP: ${RESPONSE}${NC}"
        exit 1
    fi
}


_save_credentials() {
    # Save credentials to .oauth-tokens directory
    CREDS_DIR="$(dirname "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")")/.oauth-tokens"
    mkdir -p "$CREDS_DIR"

    CREDS_FILE="${CREDS_DIR}/${FEDERATION_CLIENT_ID}.json"
    cat > "$CREDS_FILE" <<CREDENTIALS_EOF
{
    "client_id": "${FEDERATION_CLIENT_ID}",
    "client_secret": "${CLIENT_SECRET}",
    "token_endpoint": "${ADMIN_URL}/realms/${REALM}/protocol/openid-connect/token",
    "grant_type": "client_credentials",
    "token_lifetime_seconds": ${TOKEN_LIFETIME_SECONDS}
}
CREDENTIALS_EOF

    echo -e "${GREEN}Credentials saved to: ${CREDS_FILE}${NC}"
}


_verify_token() {
    echo "Verifying token generation..."
    VERIFY_RESPONSE=$(curl -s -X POST \
        "${ADMIN_URL}/realms/${REALM}/protocol/openid-connect/token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "grant_type=client_credentials" \
        -d "client_id=${FEDERATION_CLIENT_ID}" \
        -d "client_secret=${CLIENT_SECRET}")

    ACCESS_TOKEN=$(echo "$VERIFY_RESPONSE" | jq -r '.access_token // empty')
    EXPIRES_IN=$(echo "$VERIFY_RESPONSE" | jq -r '.expires_in // empty')

    if [ -z "$ACCESS_TOKEN" ] || [ "$ACCESS_TOKEN" = "null" ]; then
        echo -e "${RED}Failed to obtain test token${NC}"
        echo "Response: $VERIFY_RESPONSE"
        exit 1
    fi

    # Decode and show token groups
    TOKEN_GROUPS=$(echo "$ACCESS_TOKEN" | cut -d'.' -f2 | base64 -d 2>/dev/null | jq -r '.groups // empty')

    echo -e "${GREEN}Token verification successful${NC}"
    echo "  Token expires_in: ${EXPIRES_IN}s"
    echo "  Token groups: ${TOKEN_GROUPS}"
}


# --- Main function ---


main() {
    # Check required environment variables
    if [ -z "$ADMIN_PASS" ]; then
        echo -e "${RED}Error: KEYCLOAK_ADMIN_PASSWORD environment variable is required${NC}"
        echo "Usage: export KEYCLOAK_ADMIN_PASSWORD=\"your-password\""
        exit 1
    fi

    _get_admin_token
    _ensure_federation_group
    _create_federation_client
    _get_client_secret
    _setup_service_account
    _add_groups_mapper
    _save_credentials
    _verify_token

    echo ""
    echo "=============================================="
    echo -e "${GREEN}Federation service account setup complete${NC}"
    echo ""
    echo "Client ID:     ${FEDERATION_CLIENT_ID}"
    echo "Client Secret: ${CLIENT_SECRET}"
    echo "Token Endpoint: ${ADMIN_URL}/realms/${REALM}/protocol/openid-connect/token"
    echo "Token Lifetime: 180 days (${TOKEN_LIFETIME_SECONDS}s)"
    echo ""
    echo "Add these to your registry .env file:"
    echo "  FEDERATION_TOKEN_ENDPOINT=${ADMIN_URL}/realms/${REALM}/protocol/openid-connect/token"
    echo "  FEDERATION_CLIENT_ID=${FEDERATION_CLIENT_ID}"
    echo "  FEDERATION_CLIENT_SECRET=${CLIENT_SECRET}"
}


main
