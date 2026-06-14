#!/bin/bash
# Setup IDE Public OAuth Client in Okta (for PR #1224 - IDE_OAUTH_CLIENT_ID)
#
# Okta equivalent of setup/idp/keycloak/setup-ide-public-client.sh. Creates a PUBLIC
# OIDC client (authorization_code + PKCE, no secret) that IDEs (Cursor, Claude
# Code, Codex) use to run the gateway login flow. In Okta terms this is an OIDC
# "native" application with token_endpoint_auth_method=none and PKCE required.
#
# The resulting client_id is PUBLIC and is NOT a secret - advertise it via
# IDE_OAUTH_CLIENT_ID.
#
# IMPORTANT - Okta needs a FIXED callback port. Unlike Keycloak (which accepts a
# wildcard redirect URI like http://localhost/*), Okta matches the redirect_uri
# LITERALLY, including the port. The IDE uses an ephemeral port by default, so
# you MUST pin it: run the IDE with a fixed callback port and register that exact
# URI here. For Claude Code:
#   claude mcp add --transport http --client-id <id> --callback-port 56789 \
#     <name> https://gateway.example.com/<server>/mcp
# This script registers http://localhost:<IDE_CALLBACK_PORT>/callback (and the
# 127.0.0.1 form) to match. The callback path is always /callback.
#
# IMPORTANT - two more Okta steps the API cannot fully do for you (the script
# does what it can; the rest is called out at the end):
#   1. ASSIGN the user (or a group) to this app, or login fails 'user_not_assigned'.
#   2. ADD a 'groups' claim to the authorization server, or the token carries no
#      groups and the registry denies every server (login succeeds, all 403).
#
# CONFIGURATION MODEL: env-var driven (matching the Keycloak script). It calls
# the Okta management API, which needs an Okta org URL and an admin API token.
# Secrets are taken from env vars only (never CLI args).
#
# Required env vars:
#   OKTA_ORG_URL        e.g. https://dev-12345.okta.com  (no trailing slash)
#   OKTA_API_TOKEN      Okta admin API token (sent as "Authorization: SSWS ...")
#
# Optional env vars:
#   IDE_CLIENT_LABEL    App label (default: "MCP Gateway IDE Public Client")
#   IDE_CALLBACK_PORT   Fixed loopback callback port to register (default: 56789).
#                       MUST match the IDE's --callback-port.
#   IDE_REDIRECT_URIS   Full override of the redirect URI list (space-separated).
#                       When set, IDE_CALLBACK_PORT is ignored.
#
# Usage:
#   export OKTA_ORG_URL="https://dev-12345.okta.com"
#   export OKTA_API_TOKEN="00abc..."
#   bash setup/idp/okta/setup-ide-public-client.sh

set -e

# Configuration
ORG_URL="${OKTA_ORG_URL%/}"
API_TOKEN="${OKTA_API_TOKEN}"
LABEL="${IDE_CLIENT_LABEL:-MCP Gateway IDE Public Client}"
IDE_CALLBACK_PORT="${IDE_CALLBACK_PORT:-56789}"
# Okta matches redirect_uri literally (port included), so register a FIXED-port
# /callback URI. Override the whole list with IDE_REDIRECT_URIS if needed.
REDIRECT_URIS_RAW="${IDE_REDIRECT_URIS:-http://localhost:${IDE_CALLBACK_PORT}/callback http://127.0.0.1:${IDE_CALLBACK_PORT}/callback}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Setting up IDE Public OAuth Client in Okta"
echo "=============================================="
echo "Org:    $ORG_URL"
echo "Label:  $LABEL"
echo "Type:   native/public client (authorization_code + PKCE, no secret)"
echo ""


# --- Private functions ---


_check_env() {
    local missing=""
    [ -z "$ORG_URL" ] && missing="$missing OKTA_ORG_URL"
    [ -z "$API_TOKEN" ] && missing="$missing OKTA_API_TOKEN"
    if [ -n "$missing" ]; then
        echo -e "${RED}Error: missing required env vars:${missing}${NC}"
        exit 1
    fi
}


_build_redirect_uris_json() {
    REDIRECT_URIS_JSON="["
    local first=1
    for uri in $REDIRECT_URIS_RAW; do
        if [ $first -eq 1 ]; then first=0; else REDIRECT_URIS_JSON="${REDIRECT_URIS_JSON},"; fi
        REDIRECT_URIS_JSON="${REDIRECT_URIS_JSON}\"${uri}\""
    done
    REDIRECT_URIS_JSON="${REDIRECT_URIS_JSON}]"
}


_find_existing_app() {
    echo "Checking if app '${LABEL}' already exists..."
    EXISTING_APP_ID=$(curl -s \
        -H "Authorization: SSWS ${API_TOKEN}" \
        -H "Accept: application/json" \
        "${ORG_URL}/api/v1/apps?q=$(echo "$LABEL" | sed 's/ /%20/g')&limit=1" | \
        jq -r '.[0].id // empty')
}


_create_app() {
    # application_type=native + token_endpoint_auth_method=none is Okta's public
    # client. grant_types authorization_code + refresh_token, PKCE is implied for
    # native/none clients. Okta puts group membership into the token via a groups
    # claim configured on the authorization server (see reminder in main); the
    # registry maps those groups to scopes.
    APP_JSON='{
        "name": "oidc_client",
        "label": "'"${LABEL}"'",
        "signOnMode": "OPENID_CONNECT",
        "credentials": {
            "oauthClient": { "token_endpoint_auth_method": "none" }
        },
        "settings": {
            "oauthClient": {
                "application_type": "native",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "redirect_uris": '"${REDIRECT_URIS_JSON}"'
            }
        }
    }'

    if [ -n "$EXISTING_APP_ID" ] && [ "$EXISTING_APP_ID" != "null" ]; then
        echo -e "${YELLOW}App exists (id: ${EXISTING_APP_ID}); updating${NC}"
        RESP=$(curl -s -X PUT "${ORG_URL}/api/v1/apps/${EXISTING_APP_ID}" \
            -H "Authorization: SSWS ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -H "Accept: application/json" \
            -d "$APP_JSON")
    else
        echo "Creating Okta OIDC native (public) app..."
        RESP=$(curl -s -X POST "${ORG_URL}/api/v1/apps" \
            -H "Authorization: SSWS ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -H "Accept: application/json" \
            -d "$APP_JSON")
    fi

    APP_CLIENT_ID=$(echo "$RESP" | jq -r '.credentials.oauthClient.client_id // empty')
    if [ -z "$APP_CLIENT_ID" ] || [ "$APP_CLIENT_ID" = "null" ]; then
        echo -e "${RED}Failed to create/update Okta app or read client_id${NC}"
        echo "Response: $RESP"
        exit 1
    fi
    echo -e "${GREEN}Okta app ready${NC}"
}


# --- Main function ---


main() {
    _check_env
    _build_redirect_uris_json
    _find_existing_app
    _create_app

    echo ""
    echo "=============================================="
    echo -e "${GREEN}Okta IDE public client setup complete${NC}"
    echo ""
    echo "Client ID: ${APP_CLIENT_ID}  (public - NOT a secret)"
    echo "Org:       ${ORG_URL}"
    echo "Redirect URIs: ${REDIRECT_URIS_RAW}"
    echo ""
    echo "Add this to your registry .env file, then restart the registry:"
    echo "  IDE_OAUTH_CLIENT_ID=${APP_CLIENT_ID}"
    echo "  MCP_ADVERTISED_SCOPES=\"openid email profile offline_access\""
    echo ""
    echo "Run the IDE with the matching FIXED callback port (Okta matches literally):"
    echo "  claude mcp add --transport http --client-id ${APP_CLIENT_ID} \\"
    echo "    --callback-port ${IDE_CALLBACK_PORT} <name> <gateway-url>/<server>/mcp"
    echo ""
    echo -e "${YELLOW}REQUIRED Okta steps this script cannot fully automate:${NC}"
    echo "  1. ASSIGN your user (or a group) to this app in the Okta admin console,"
    echo "     or login fails with 'user_not_assigned'."
    echo "  2. ADD a 'groups' claim to the authorization server used by the gateway"
    echo "     (claim name 'groups', value type Groups, filter regex .*, include in"
    echo "     the ACCESS token). WITHOUT THIS, login SUCCEEDS but the registry denies"
    echo "     every server (the token carries no groups -> no scopes). This is the"
    echo "     #1 non-obvious Okta gotcha."
    echo "  3. Ensure your Okta group is mapped to a registry scope in mcp_scopes"
    echo "     (e.g. group 'registry-admins' -> scope 'registry-admins')."
}


main
