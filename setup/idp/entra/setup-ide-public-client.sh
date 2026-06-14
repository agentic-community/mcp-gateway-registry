#!/bin/bash
# Setup IDE Public OAuth Client in Microsoft Entra ID (for PR #1224 - IDE_OAUTH_CLIENT_ID)
#
# Entra equivalent of setup/idp/keycloak/setup-ide-public-client.sh. Creates a
# PUBLIC client (authorization_code + PKCE, no secret) that IDEs (Cursor,
# Claude Code, Codex) use to run the gateway login flow. In Entra terms this is
# an app registration with a "Mobile and desktop applications" platform, the
# loopback redirect URI, and public-client flows allowed.
#
# Like the Keycloak script, the resulting Application (client) ID is PUBLIC and
# is NOT a secret - you advertise it via IDE_OAUTH_CLIENT_ID.
#
# CONFIGURATION MODEL: this script is env-var driven (matching the Keycloak
# script). It needs a Microsoft Graph token, which it obtains via client
# credentials from an app that has the Application.ReadWrite.All permission.
# Secrets are taken from env vars only (never CLI args, which leak into shell
# history and the process list).
#
# Required env vars:
#   ENTRA_TENANT_ID            Azure AD tenant (directory) ID
#   ENTRA_GRAPH_CLIENT_ID      App (client) ID of an app with
#                              Application.ReadWrite.All (admin-consented)
#   ENTRA_GRAPH_CLIENT_SECRET  That app's client secret
#
# Optional env vars:
#   IDE_CLIENT_DISPLAY_NAME    Display name (default: "MCP Gateway IDE Public Client")
#   IDE_REDIRECT_URIS          Space-separated loopback redirect URIs
#                              (default: "http://localhost http://127.0.0.1")
#
# Usage:
#   export ENTRA_TENANT_ID="..."
#   export ENTRA_GRAPH_CLIENT_ID="..."
#   export ENTRA_GRAPH_CLIENT_SECRET="..."
#   bash setup/idp/entra/setup-ide-public-client.sh
#
# NOTE: the existing ENTRA_CLIENT_ID/SECRET in your .env is the gateway's WEB
# auth app. It is a different app and almost certainly does NOT have
# Application.ReadWrite.All, so do not assume you can reuse it here. Either grant
# that permission to a dedicated automation app, or create the IDE app by hand
# in the Azure portal (see the any-IdP FAQ) and just set IDE_OAUTH_CLIENT_ID.

set -e

# Configuration
TENANT_ID="${ENTRA_TENANT_ID}"
GRAPH_CLIENT_ID="${ENTRA_GRAPH_CLIENT_ID}"
GRAPH_CLIENT_SECRET="${ENTRA_GRAPH_CLIENT_SECRET}"
DISPLAY_NAME="${IDE_CLIENT_DISPLAY_NAME:-MCP Gateway IDE Public Client}"
REDIRECT_URIS_RAW="${IDE_REDIRECT_URIS:-http://localhost http://127.0.0.1}"

GRAPH="https://graph.microsoft.com/v1.0"
LOGIN="https://login.microsoftonline.com"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Setting up IDE Public OAuth Client in Microsoft Entra ID"
echo "=============================================="
echo "Tenant:       $TENANT_ID"
echo "Display name: $DISPLAY_NAME"
echo "Type:         public client (authorization_code + PKCE, no secret)"
echo ""


# --- Private functions ---


_check_env() {
    local missing=""
    [ -z "$TENANT_ID" ] && missing="$missing ENTRA_TENANT_ID"
    [ -z "$GRAPH_CLIENT_ID" ] && missing="$missing ENTRA_GRAPH_CLIENT_ID"
    [ -z "$GRAPH_CLIENT_SECRET" ] && missing="$missing ENTRA_GRAPH_CLIENT_SECRET"
    if [ -n "$missing" ]; then
        echo -e "${RED}Error: missing required env vars:${missing}${NC}"
        echo "The Graph app must have Application.ReadWrite.All (admin-consented)."
        exit 1
    fi
}


_get_graph_token() {
    echo "Getting Microsoft Graph token..."
    TOKEN=$(curl -s -X POST "${LOGIN}/${TENANT_ID}/oauth2/v2.0/token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "client_id=${GRAPH_CLIENT_ID}" \
        -d "client_secret=${GRAPH_CLIENT_SECRET}" \
        -d "scope=https://graph.microsoft.com/.default" \
        -d "grant_type=client_credentials" | jq -r '.access_token // empty')

    if [ -z "$TOKEN" ]; then
        echo -e "${RED}Failed to get Graph token. Check tenant/app id/secret and that"
        echo -e "the app has Application.ReadWrite.All with admin consent.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Graph token obtained${NC}"
}


_build_redirect_uris_json() {
    # Turn the space-separated list into a JSON array.
    REDIRECT_URIS_JSON="["
    local first=1
    for uri in $REDIRECT_URIS_RAW; do
        if [ $first -eq 1 ]; then first=0; else REDIRECT_URIS_JSON="${REDIRECT_URIS_JSON},"; fi
        REDIRECT_URIS_JSON="${REDIRECT_URIS_JSON}\"${uri}\""
    done
    REDIRECT_URIS_JSON="${REDIRECT_URIS_JSON}]"
}


_create_or_update_app() {
    echo "Checking if app '${DISPLAY_NAME}' already exists..."
    EXISTING_APP_ID=$(curl -s -H "Authorization: Bearer $TOKEN" \
        "${GRAPH}/applications?\$filter=displayName eq '${DISPLAY_NAME}'" | \
        jq -r '.value[0].id // empty')

    # publicClient.redirectUris = loopback; isFallbackPublicClient=true is
    # Entra's "Allow public client flows" toggle (required for a secret-less
    # authorization_code+PKCE desktop client). groupMembershipClaims puts the
    # user's security groups into the token so the registry can map them to
    # scopes (same role groups serve in the Keycloak setup).
    APP_JSON='{
        "displayName": "'"${DISPLAY_NAME}"'",
        "signInAudience": "AzureADMyOrg",
        "isFallbackPublicClient": true,
        "publicClient": { "redirectUris": '"${REDIRECT_URIS_JSON}"' },
        "groupMembershipClaims": "SecurityGroup"
    }'

    if [ -n "$EXISTING_APP_ID" ] && [ "$EXISTING_APP_ID" != "null" ]; then
        echo -e "${YELLOW}App exists (objectId: ${EXISTING_APP_ID}); updating${NC}"
        # PATCH returns 204 (no body), so read appId back from the existing app.
        RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
            -X PATCH "${GRAPH}/applications/${EXISTING_APP_ID}" \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            -d "$APP_JSON")
        if [ "$RESPONSE" != "204" ]; then
            echo -e "${RED}Failed to update app. HTTP: ${RESPONSE}${NC}"
            exit 1
        fi
        APP_OBJECT_ID="$EXISTING_APP_ID"
        APP_CLIENT_ID=$(curl -s -H "Authorization: Bearer $TOKEN" \
            "${GRAPH}/applications/${APP_OBJECT_ID}" | jq -r '.appId // empty')
    else
        echo "Creating Entra app registration..."
        # The create POST RESPONSE already contains both id (objectId) and appId
        # (the client id). Read them from it directly. Do NOT do a follow-up GET:
        # Graph is eventually consistent and a GET-by-id immediately after create
        # can 404, which previously caused a spurious "Could not read appId" error
        # even though the app was created fine.
        CREATE_RESP=$(curl -s -X POST "${GRAPH}/applications" \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            -d "$APP_JSON")
        APP_OBJECT_ID=$(echo "$CREATE_RESP" | jq -r '.id // empty')
        APP_CLIENT_ID=$(echo "$CREATE_RESP" | jq -r '.appId // empty')
        if [ -z "$APP_OBJECT_ID" ] || [ "$APP_OBJECT_ID" = "null" ]; then
            echo -e "${RED}Failed to create app.${NC}"
            echo "Response: $CREATE_RESP"
            exit 1
        fi
        echo -e "${GREEN}App created${NC}"
    fi

    # The Application (client) ID is appId, NOT the objectId. This is the value
    # that goes into IDE_OAUTH_CLIENT_ID.
    if [ -z "$APP_CLIENT_ID" ] || [ "$APP_CLIENT_ID" = "null" ]; then
        echo -e "${RED}Could not determine appId for the app${NC}"
        exit 1
    fi
}


# --- Main function ---


main() {
    _check_env
    _get_graph_token
    _build_redirect_uris_json
    _create_or_update_app

    echo ""
    echo "=============================================="
    echo -e "${GREEN}Entra IDE public client setup complete${NC}"
    echo ""
    echo "Application (client) ID: ${APP_CLIENT_ID}  (public - NOT a secret)"
    echo "Tenant:                  ${TENANT_ID}"
    echo "Redirect URIs:           ${REDIRECT_URIS_RAW}"
    echo ""
    echo "Add this to your registry .env file, then restart the registry:"
    echo "  IDE_OAUTH_CLIENT_ID=${APP_CLIENT_ID}"
    echo "  MCP_ADVERTISED_SCOPES=openid email profile offline_access"
    echo ""
    echo "Entra reminders:"
    echo "  - Confirm 'Allow public client flows' shows enabled in the portal."
    echo "  - You may need admin consent + group claims config so the token"
    echo "    carries the user's security groups (the registry maps groups -> scopes)."
}


main
