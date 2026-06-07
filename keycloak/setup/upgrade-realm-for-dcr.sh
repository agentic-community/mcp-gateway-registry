#!/bin/bash
# Upgrade an existing Keycloak realm to support MCP Dynamic Client Registration
#
# This standalone script applies the three configuration changes required for
# MCP clients (Claude Code, Claude.ai connectors, Cursor) to register dynamically
# and produce tokens with a `groups` claim for per-user authorization at the
# gateway. Use this on existing deployments where the realm was created before
# these changes landed in init-keycloak.sh.
#
# The script is idempotent: re-running it is safe.
#
# Usage:
#   bash keycloak/setup/upgrade-realm-for-dcr.sh
#
# Reads .env from repo root for KEYCLOAK_ADMIN_URL, KEYCLOAK_ADMIN, and
# KEYCLOAK_ADMIN_PASSWORD.

set -e

REALM="mcp-gateway"
# Do not unset KEYCLOAK_ADMIN / KEYCLOAK_ADMIN_PASSWORD here — main() needs to
# detect whether the caller already exported them so it can skip .env loading
# and run against a non-.env Keycloak (e.g. ECS).
KEYCLOAK_URL=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}MCP DCR upgrade for existing Keycloak realm '${REALM}'${NC}"
echo "============================================================"

get_admin_token() {
    local response=$(curl -s -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=${KEYCLOAK_ADMIN}" \
        -d "password=${KEYCLOAK_ADMIN_PASSWORD}" \
        -d "grant_type=password" \
        -d "client_id=admin-cli")

    echo "$response" | jq -r '.access_token // empty'
}

setup_dcr_groups_mapper() {
    local token=$1

    echo ""
    echo "[1/4] Attaching Groups protocol mapper to the 'basic' client-scope..."

    local basic_scope_id=$(curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes" | \
        jq -r '.[] | select(.name=="basic") | .id')

    if [ -z "$basic_scope_id" ] || [ "$basic_scope_id" = "null" ]; then
        # Older Keycloak versions and custom realm imports do not include a
        # 'basic' client-scope. The Groups mapper is recommended but not
        # required for DCR to succeed; skip gracefully so the policy fixes
        # in steps 3 and 4 still run.
        echo -e "${YELLOW}WARNING: 'basic' client-scope not found in realm '${REALM}' — skipping groups mapper. Add the mapper manually if you need group claims in DCR'd-client tokens.${NC}"
        return 0
    fi

    local existing=$(curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes/${basic_scope_id}" | \
        jq -r '.protocolMappers[]? | select(.name=="groups") | .id')

    if [ -n "$existing" ] && [ "$existing" != "null" ]; then
        echo -e "${YELLOW}Groups mapper already attached to 'basic' scope. Skipping.${NC}"
        return 0
    fi

    local groups_mapper_json='{
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

    local response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes/${basic_scope_id}/protocol-mappers/models" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$groups_mapper_json")

    if [ "$response" = "201" ]; then
        echo -e "${GREEN}OK: groups mapper attached to 'basic'.${NC}"
    else
        echo -e "${RED}FAILED: HTTP status ${response}${NC}"
        return 1
    fi
}

setup_dcr_audience_mapper() {
    local token=$1

    echo ""
    echo "[2/4] Attaching Audience protocol mapper to the 'basic' client-scope..."

    local basic_scope_id=$(curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes" | \
        jq -r '.[] | select(.name=="basic") | .id')

    if [ -z "$basic_scope_id" ] || [ "$basic_scope_id" = "null" ]; then
        # See setup_dcr_groups_mapper() for the rationale: skip when the
        # realm does not have a 'basic' client-scope so the policy fixes
        # in steps 3 and 4 can still run.
        echo -e "${YELLOW}WARNING: 'basic' client-scope not found — skipping audience mapper. Add the mapper manually if DCR'd-client tokens need aud=\"mcp-gateway\".${NC}"
        return 0
    fi

    local existing=$(curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes/${basic_scope_id}" | \
        jq -r '.protocolMappers[]? | select(.name=="mcp-gateway-audience") | .id')

    if [ -n "$existing" ] && [ "$existing" != "null" ]; then
        echo -e "${YELLOW}Audience mapper already attached. Skipping.${NC}"
        return 0
    fi

    local audience_mapper_json='{
        "name": "mcp-gateway-audience",
        "protocol": "openid-connect",
        "protocolMapper": "oidc-audience-mapper",
        "consentRequired": false,
        "config": {
            "included.custom.audience": "mcp-gateway",
            "id.token.claim": "false",
            "access.token.claim": "true"
        }
    }'

    local response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes/${basic_scope_id}/protocol-mappers/models" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$audience_mapper_json")

    if [ "$response" = "201" ]; then
        echo -e "${GREEN}OK: audience mapper attached. DCR'd-client tokens will carry aud=\"mcp-gateway\".${NC}"
    else
        echo -e "${RED}FAILED: HTTP status ${response}${NC}"
        return 1
    fi
}

_widen_allowed_scopes_policy() {
    # Widen one subType's "Allowed Client Scopes" policy to include every
    # client-scope defined in the realm. Both 'anonymous' and 'authenticated'
    # subTypes need this because Keycloak picks one based on whether the DCR
    # request carries an initial-access / registration token. Claude Code,
    # Claude.ai connectors, and Cursor have been observed to hit either path.
    local token=$1
    local sub_type=$2  # "anonymous" or "authenticated"
    local components=$3
    local all_scope_names=$4

    local policy_id=$(echo "$components" | \
        jq -r ".[] | select(.name==\"Allowed Client Scopes\" and .subType==\"${sub_type}\") | .id")
    local parent_id=$(echo "$components" | \
        jq -r ".[] | select(.name==\"Allowed Client Scopes\" and .subType==\"${sub_type}\") | .parentId")

    if [ -z "$policy_id" ] || [ "$policy_id" = "null" ]; then
        echo -e "${YELLOW}WARNING: ${sub_type} 'Allowed Client Scopes' policy not found — skipping.${NC}"
        return 0
    fi

    local policy_json=$(jq -n \
        --arg name "Allowed Client Scopes" \
        --arg providerId "allowed-client-templates" \
        --arg providerType "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy" \
        --arg parentId "$parent_id" \
        --arg subType "$sub_type" \
        --argjson allowedScopes "$all_scope_names" \
        '{
            name: $name,
            providerId: $providerId,
            providerType: $providerType,
            parentId: $parentId,
            subType: $subType,
            config: {
                "allow-default-scopes": ["true"],
                "allowed-client-scopes": $allowedScopes
            }
        }')

    local response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/components/${policy_id}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$policy_json")

    if [ "$response" = "204" ]; then
        echo -e "${GREEN}OK: ${sub_type} Allowed Client Scopes policy includes all realm scopes.${NC}"
    else
        echo -e "${RED}FAILED (${sub_type}): HTTP status ${response}${NC}"
        return 1
    fi
}

configure_dcr_allowed_scopes() {
    local token=$1

    echo ""
    echo "[3/4] Widening 'Allowed Client Scopes' DCR policy (anonymous + authenticated)..."

    local components=$(curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy")

    local all_scope_names=$(curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes" | \
        jq -c '[.[].name] + ["openid"]')

    _widen_allowed_scopes_policy "$token" "anonymous" "$components" "$all_scope_names" || return 1
    _widen_allowed_scopes_policy "$token" "authenticated" "$components" "$all_scope_names" || return 1
}

configure_dcr_trusted_hosts() {
    local token=$1

    echo ""
    echo "[4/4] Relaxing anonymous-DCR 'Trusted Hosts' policy..."

    local components=$(curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy")

    local policy_id=$(echo "$components" | \
        jq -r '.[] | select(.name=="Trusted Hosts" and .subType=="anonymous") | .id')
    local parent_id=$(echo "$components" | \
        jq -r '.[] | select(.name=="Trusted Hosts" and .subType=="anonymous") | .parentId')

    if [ -z "$policy_id" ] || [ "$policy_id" = "null" ]; then
        echo -e "${RED}Error: Could not find anonymous 'Trusted Hosts' policy${NC}"
        return 1
    fi

    local policy_json='{
        "name": "Trusted Hosts",
        "providerId": "trusted-hosts",
        "providerType": "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy",
        "parentId": "'"$parent_id"'",
        "subType": "anonymous",
        "config": {
            "host-sending-registration-request-must-match": ["false"],
            "client-uris-must-match": ["true"],
            "trusted-hosts": ["localhost", "127.0.0.1", "claude.ai"]
        }
    }'

    local response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/components/${policy_id}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$policy_json")

    if [ "$response" = "204" ]; then
        echo -e "${GREEN}OK: Trusted Hosts policy relaxed (IP check off, URI check on, localhost allowed).${NC}"
    else
        echo -e "${RED}FAILED: HTTP status ${response}${NC}"
        return 1
    fi
}

main() {
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
    ENV_FILE="$PROJECT_ROOT/.env"

    # Prefer already-exported KEYCLOAK_ADMIN_URL + KEYCLOAK_ADMIN_PASSWORD so
    # this script can run against a non-.env Keycloak (e.g. ECS) without
    # mutating the developer's local .env. Fall back to .env when those
    # variables are not exported.
    if [ -n "$KEYCLOAK_ADMIN_URL" ] && [ -n "$KEYCLOAK_ADMIN_PASSWORD" ]; then
        echo "Using KEYCLOAK_ADMIN_URL / KEYCLOAK_ADMIN_PASSWORD from environment (skipping .env load)"
    elif [ -f "$ENV_FILE" ]; then
        echo "Loading environment from $ENV_FILE..."
        set -a
        source "$ENV_FILE"
        set +a
    else
        echo -e "${RED}Error: export KEYCLOAK_ADMIN_URL and KEYCLOAK_ADMIN_PASSWORD, or provide .env at $ENV_FILE${NC}"
        exit 1
    fi

    KEYCLOAK_URL="${KEYCLOAK_ADMIN_URL:-http://localhost:8080}"
    KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"

    if [ -z "$KEYCLOAK_ADMIN_PASSWORD" ]; then
        echo -e "${RED}Error: KEYCLOAK_ADMIN_PASSWORD must be set (export or .env)${NC}"
        exit 1
    fi

    if ! command -v jq >/dev/null 2>&1; then
        echo -e "${RED}Error: this script requires 'jq'. Install with: sudo apt-get install jq${NC}"
        exit 1
    fi

    echo "Using Keycloak API URL: $KEYCLOAK_URL"
    echo "Realm: $REALM"

    echo ""
    echo "Authenticating as Keycloak admin..."
    TOKEN=$(get_admin_token)
    if [ -z "$TOKEN" ]; then
        echo -e "${RED}Error: failed to obtain admin token${NC}"
        exit 1
    fi
    echo -e "${GREEN}Authentication successful.${NC}"

    setup_dcr_groups_mapper "$TOKEN"
    setup_dcr_audience_mapper "$TOKEN"
    configure_dcr_allowed_scopes "$TOKEN"
    configure_dcr_trusted_hosts "$TOKEN"

    echo ""
    echo -e "${GREEN}DCR upgrade complete.${NC}"
    echo ""
    echo "What this script attempts:"
    echo "  - Groups protocol mapper on 'basic' client-scope (skipped if 'basic' missing)"
    echo "  - Audience mapper on 'basic' client-scope, aud=mcp-gateway (skipped if 'basic' missing)"
    echo "  - Allowed Client Scopes policy widened to all realm scopes"
    echo "  - Trusted Hosts policy: IP check OFF, URI check ON, 'localhost' trusted"
    echo ""
    echo "Check the output above to see which steps actually ran for this realm."
    echo "Existing DCR'd clients will pick up the policy changes immediately."
}

main "$@"
