#!/bin/bash
# Setup IDE Public OAuth Client in Amazon Cognito (for PR #1224 - IDE_OAUTH_CLIENT_ID)
#
# Cognito equivalent of setup/idp/keycloak/setup-ide-public-client.sh. Creates a
# PUBLIC user-pool app client (authorization_code + PKCE, no secret) that IDEs
# (Cursor, Claude Code, Codex) use to run the gateway login flow.
#
# The resulting client_id is PUBLIC and is NOT a secret - advertise it via
# IDE_OAUTH_CLIENT_ID.
#
# IMPORTANT - Cognito needs a FIXED callback port. Like Okta (and unlike
# Keycloak's wildcard redirect), Cognito matches the callback URL literally,
# including the port, and does NOT allow wildcards on http://localhost. So you
# MUST pin the IDE callback port and register that exact URL here. Set
# IDE_OAUTH_CALLBACK_PORT in the registry .env to the same value so the Connect
# dialog emits --callback-port (Claude Code). The callback path is always
# /callback. Note: Codex/Cursor cannot pin the port, so IDE login via Cognito
# only fully works with Claude Code.
#
# IMPORTANT - access comes from GROUPS. Cognito must put the user's group
# membership into the token (the "cognito:groups" claim is included by default
# in Cognito access tokens), and those group names must be mapped to a registry
# scope in the mcp_scopes collection. Without a matching group->scope mapping the
# user authenticates but is denied on every server.
#
# CONFIGURATION MODEL: env-var driven (matching the Keycloak/Okta scripts), but
# Cognito uses the AWS API, so it relies on standard AWS credentials/region from
# the environment (AWS_PROFILE / AWS_REGION / instance role) rather than an admin
# token.
#
# Required env vars:
#   COGNITO_USER_POOL_ID   e.g. us-east-1_XXXXXXXXX
#   COGNITO_DOMAIN         the user pool domain prefix or full domain (for the
#                          hosted-UI authorize URL; informational here)
#
# Optional env vars:
#   AWS_REGION             AWS region (default: us-east-1, or from AWS config)
#   IDE_CLIENT_NAME        App client name (default: "mcp-gateway-ide")
#   IDE_CALLBACK_PORT      Fixed loopback callback port (default: 56789). MUST
#                          match the IDE's --callback-port and IDE_OAUTH_CALLBACK_PORT.
#   IDE_REDIRECT_URIS      Full override of callback URL list (space-separated).
#
# Usage:
#   export COGNITO_USER_POOL_ID="us-east-1_XXXXXXXXX"
#   export AWS_REGION="us-east-1"
#   bash setup/idp/cognito/setup-ide-public-client.sh

set -e

# Configuration
USER_POOL_ID="${COGNITO_USER_POOL_ID}"
REGION="${AWS_REGION:-us-east-1}"
CLIENT_NAME="${IDE_CLIENT_NAME:-mcp-gateway-ide}"
IDE_CALLBACK_PORT="${IDE_CALLBACK_PORT:-56789}"
# Cognito matches callback URLs literally, so register a FIXED-port /callback URL.
REDIRECT_URIS_RAW="${IDE_REDIRECT_URIS:-http://localhost:${IDE_CALLBACK_PORT}/callback http://127.0.0.1:${IDE_CALLBACK_PORT}/callback}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Setting up IDE Public OAuth Client in Amazon Cognito"
echo "=============================================="
echo "User pool:  $USER_POOL_ID"
echo "Region:     $REGION"
echo "Client:     $CLIENT_NAME"
echo "Type:       public client (authorization_code + PKCE, no secret)"
echo ""


# --- Private functions ---


_check_env() {
    local missing=""
    [ -z "$USER_POOL_ID" ] && missing="$missing COGNITO_USER_POOL_ID"
    if [ -n "$missing" ]; then
        echo -e "${RED}Error: missing required env vars:${missing}${NC}"
        exit 1
    fi
    if ! aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
        echo -e "${RED}Error: AWS credentials not available. Configure AWS_PROFILE / role / keys.${NC}"
        exit 1
    fi
}


_build_callback_args() {
    # AWS CLI takes repeated --callback-urls values; build a space-separated list.
    CALLBACK_ARGS=""
    for uri in $REDIRECT_URIS_RAW; do
        CALLBACK_ARGS="$CALLBACK_ARGS $uri"
    done
}


_find_existing_client() {
    echo "Checking if app client '${CLIENT_NAME}' already exists..."
    EXISTING_CLIENT_ID=$(aws cognito-idp list-user-pool-clients \
        --user-pool-id "$USER_POOL_ID" --region "$REGION" --max-results 60 \
        --query "UserPoolClients[?ClientName=='${CLIENT_NAME}'].ClientId | [0]" \
        --output text 2>/dev/null)
    [ "$EXISTING_CLIENT_ID" = "None" ] && EXISTING_CLIENT_ID=""
}


_create_or_update_client() {
    # Public client = no secret (omit --generate-secret). PKCE is enforced by
    # Cognito for public clients on the authorization_code flow. Group membership
    # rides the access token as the default "cognito:groups" claim.
    # shellcheck disable=SC2086
    if [ -n "$EXISTING_CLIENT_ID" ]; then
        echo -e "${YELLOW}App client exists (${EXISTING_CLIENT_ID}); updating${NC}"
        APP_CLIENT_ID=$(aws cognito-idp update-user-pool-client \
            --user-pool-id "$USER_POOL_ID" --region "$REGION" \
            --client-id "$EXISTING_CLIENT_ID" \
            --client-name "$CLIENT_NAME" \
            --allowed-o-auth-flows code \
            --allowed-o-auth-scopes openid email profile \
            --allowed-o-auth-flows-user-pool-client \
            --supported-identity-providers COGNITO \
            --callback-urls $CALLBACK_ARGS \
            --explicit-auth-flows ALLOW_REFRESH_TOKEN_AUTH ALLOW_USER_SRP_AUTH \
            --query 'UserPoolClient.ClientId' --output text 2>&1)
    else
        echo "Creating Cognito public app client..."
        APP_CLIENT_ID=$(aws cognito-idp create-user-pool-client \
            --user-pool-id "$USER_POOL_ID" --region "$REGION" \
            --client-name "$CLIENT_NAME" \
            --no-generate-secret \
            --allowed-o-auth-flows code \
            --allowed-o-auth-scopes openid email profile \
            --allowed-o-auth-flows-user-pool-client \
            --supported-identity-providers COGNITO \
            --callback-urls $CALLBACK_ARGS \
            --explicit-auth-flows ALLOW_REFRESH_TOKEN_AUTH ALLOW_USER_SRP_AUTH \
            --query 'UserPoolClient.ClientId' --output text 2>&1)
    fi

    if [ -z "$APP_CLIENT_ID" ] || echo "$APP_CLIENT_ID" | grep -qiE "error|exception"; then
        echo -e "${RED}Failed to create/update app client.${NC}"
        echo "Response: $APP_CLIENT_ID"
        exit 1
    fi
    echo -e "${GREEN}App client ready${NC}"
}


_verify_client() {
    echo "Verifying client configuration..."
    HAS_SECRET=$(aws cognito-idp describe-user-pool-client \
        --user-pool-id "$USER_POOL_ID" --region "$REGION" --client-id "$APP_CLIENT_ID" \
        --query 'UserPoolClient.ClientSecret' --output text 2>/dev/null)
    if [ -n "$HAS_SECRET" ] && [ "$HAS_SECRET" != "None" ]; then
        echo -e "${RED}Client unexpectedly has a secret - not a public client${NC}"
        exit 1
    fi
    echo -e "${GREEN}Client verified: public (no secret) + authorization_code${NC}"
}


# --- Main function ---


main() {
    _check_env
    _build_callback_args
    _find_existing_client
    _create_or_update_client
    _verify_client

    echo ""
    echo "=============================================="
    echo -e "${GREEN}Cognito IDE public client setup complete${NC}"
    echo ""
    echo "Client ID: ${APP_CLIENT_ID}  (public - NOT a secret)"
    echo "User pool: ${USER_POOL_ID}"
    echo "Callback:  ${REDIRECT_URIS_RAW}"
    echo ""
    echo "Add these to your registry .env (or terraform.tfvars) and restart:"
    echo "  IDE_OAUTH_CLIENT_ID=${APP_CLIENT_ID}"
    echo "  IDE_OAUTH_CALLBACK_PORT=${IDE_CALLBACK_PORT}"
    echo "  MCP_ADVERTISED_SCOPES=\"openid email profile\""
    echo ""
    echo "Run the IDE with the matching FIXED callback port (Cognito matches literally):"
    echo "  claude mcp add --transport http --client-id ${APP_CLIENT_ID} \\"
    echo "    --callback-port ${IDE_CALLBACK_PORT} <name> <gateway-url>/<server>/mcp"
    echo ""
    echo -e "${YELLOW}Reminders:${NC}"
    echo "  - Cognito does not allow wildcard localhost callbacks; the port is fixed."
    echo "  - Only Claude Code can pin the port; Codex/Cursor will use a random port"
    echo "    and fail Cognito's literal callback match."
    echo "  - The user's Cognito groups (cognito:groups claim) must map to a registry"
    echo "    scope in mcp_scopes, or access is denied after a successful login."
}


main
