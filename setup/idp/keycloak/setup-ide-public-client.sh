#!/bin/bash
# Setup IDE Public OAuth Client in Keycloak (for PR #1224 - IDE_OAUTH_CLIENT_ID)
#
# Creates a PUBLIC authorization-code + PKCE client that IDEs (Cursor,
# Claude Code, Codex) use to run the interactive gateway login flow. This is
# fundamentally different from the mcp-gateway-m2m client:
#
#   - mcp-gateway-m2m: confidential + client_credentials grant (machine login,
#     no human, has a secret). CANNOT be used for the IDE login button.
#   - this client:     public + authorization_code grant + PKCE (a human logs
#     in through the browser, no secret). Its client id is meant to be
#     advertised in plaintext Connect configs and is NOT a secret.
#
# After running this, set the printed client id as IDE_OAUTH_CLIENT_ID in your
# registry .env (or as a per-server oauth_client_id on a server entry).
#
# Prerequisites:
#   - Keycloak running and initialized (init-keycloak.sh completed)
#   - KEYCLOAK_ADMIN_PASSWORD environment variable set
#   - jq installed
#
# Usage:
#   export KEYCLOAK_ADMIN_PASSWORD="your-password"
#   bash setup/idp/keycloak/setup-ide-public-client.sh
#
#   # Optional overrides:
#   IDE_CLIENT_ID="my-ide-client" bash setup/idp/keycloak/setup-ide-public-client.sh

set -e

# Configuration
ADMIN_URL="${KEYCLOAK_ADMIN_URL:-http://localhost:8080}"
REALM="${KEYCLOAK_REALM:-mcp-gateway}"
ADMIN_USER="${KEYCLOAK_ADMIN:-admin}"
ADMIN_PASS="${KEYCLOAK_ADMIN_PASSWORD}"

# The public client id IDEs will advertise. Override with IDE_CLIENT_ID.
IDE_CLIENT_ID="${IDE_CLIENT_ID:-mcp-gateway-ide}"

# Loopback redirect URIs the IDEs spin up local listeners on. Keycloak matches
# these with wildcards so any ephemeral port is accepted. We intentionally scope
# to loopback only - a public client must never allow arbitrary external
# redirect URIs (that is the real attack surface for a public client).
REDIRECT_URIS='[
        "http://localhost/*",
        "http://127.0.0.1/*"
    ]'

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Setting up IDE Public OAuth Client for Keycloak"
echo "=============================================="
echo "Client ID: $IDE_CLIENT_ID"
echo "Realm:     $REALM"
echo "Type:      public (authorization_code + PKCE, no secret)"
echo ""


# --- Private functions ---


_get_admin_token() {
    echo "Getting admin token..."
    TOKEN=$(curl -s -X POST "${ADMIN_URL}/realms/master/protocol/openid-connect/token" \
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


_create_ide_client() {
    echo "Checking if IDE client exists..."
    EXISTING_CLIENT=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/clients?clientId=${IDE_CLIENT_ID}" | \
        jq -r '.[0].id // empty')

    # The public client + PKCE config. publicClient=true means no secret;
    # standardFlowEnabled=true is the authorization_code (browser login) flow;
    # serviceAccountsEnabled=false because this is a human-login client, not M2M.
    # pkce.code.challenge.method=S256 forces PKCE, which is required for a public
    # client to be safe without a secret.
    CLIENT_JSON='{
        "clientId": "'"${IDE_CLIENT_ID}"'",
        "name": "MCP Gateway IDE Public Client",
        "description": "Public PKCE client for IDE (Cursor/Claude Code/Codex) gateway login. Client id is public, not a secret.",
        "enabled": true,
        "protocol": "openid-connect",
        "publicClient": true,
        "standardFlowEnabled": true,
        "implicitFlowEnabled": false,
        "directAccessGrantsEnabled": false,
        "serviceAccountsEnabled": false,
        "redirectUris": '"${REDIRECT_URIS}"',
        "attributes": {
            "pkce.code.challenge.method": "S256"
        }
    }'

    if [ -n "$EXISTING_CLIENT" ] && [ "$EXISTING_CLIENT" != "null" ]; then
        echo -e "${YELLOW}Client '${IDE_CLIENT_ID}' already exists (ID: ${EXISTING_CLIENT}); updating config${NC}"
        CLIENT_UUID="$EXISTING_CLIENT"
        RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
            -X PUT "${ADMIN_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}" \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            -d "$CLIENT_JSON")

        if [ "$RESPONSE" = "204" ]; then
            echo -e "${GREEN}IDE client updated${NC}"
        else
            echo -e "${RED}Failed to update client. HTTP: ${RESPONSE}${NC}"
            exit 1
        fi
        return 0
    fi

    echo "Creating IDE public client..."
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${ADMIN_URL}/admin/realms/${REALM}/clients" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "$CLIENT_JSON")

    if [ "$RESPONSE" = "201" ]; then
        echo -e "${GREEN}IDE client created${NC}"
    else
        echo -e "${RED}Failed to create client. HTTP: ${RESPONSE}${NC}"
        exit 1
    fi

    CLIENT_UUID=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/clients?clientId=${IDE_CLIENT_ID}" | \
        jq -r '.[0].id')
    echo "Client UUID: $CLIENT_UUID"
}


_add_groups_mapper() {
    # The gateway derives a user's access from their groups claim, so the IDE
    # token must carry groups just like the web-login token does.
    echo "Adding groups mapper to IDE client..."

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


_verify_client() {
    # A public authorization_code + PKCE client cannot be exercised
    # non-interactively (it needs a browser redirect), so we verify by reading
    # the client config back and confirming the safety-critical flags.
    echo "Verifying client configuration..."
    CONFIG=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${ADMIN_URL}/admin/realms/${REALM}/clients/${CLIENT_UUID}")

    IS_PUBLIC=$(echo "$CONFIG" | jq -r '.publicClient')
    STD_FLOW=$(echo "$CONFIG" | jq -r '.standardFlowEnabled')
    PKCE=$(echo "$CONFIG" | jq -r '.attributes."pkce.code.challenge.method" // empty')

    if [ "$IS_PUBLIC" != "true" ] || [ "$STD_FLOW" != "true" ] || [ "$PKCE" != "S256" ]; then
        echo -e "${RED}Client config check failed:${NC}"
        echo "  publicClient=${IS_PUBLIC} (want true)"
        echo "  standardFlowEnabled=${STD_FLOW} (want true)"
        echo "  pkce=${PKCE} (want S256)"
        exit 1
    fi
    echo -e "${GREEN}Client verified: public + authorization_code + PKCE(S256)${NC}"
}


# --- Main function ---


main() {
    if [ -z "$ADMIN_PASS" ]; then
        echo -e "${RED}Error: KEYCLOAK_ADMIN_PASSWORD environment variable is required${NC}"
        echo "Usage: export KEYCLOAK_ADMIN_PASSWORD=\"your-password\""
        exit 1
    fi

    _get_admin_token
    _create_ide_client
    _add_groups_mapper
    _verify_client

    echo ""
    echo "=============================================="
    echo -e "${GREEN}IDE public OAuth client setup complete${NC}"
    echo ""
    echo "Client ID: ${IDE_CLIENT_ID}  (public - safe to share, NOT a secret)"
    echo "Realm:     ${REALM}"
    echo "Flow:      authorization_code + PKCE (S256)"
    echo ""
    echo "Add this to your registry .env file, then restart the registry:"
    echo "  IDE_OAUTH_CLIENT_ID=${IDE_CLIENT_ID}"
    echo ""
    echo "Or set it per-server as 'oauth_client_id' on a server entry to override"
    echo "the registry-wide default for that one server."
}


main
