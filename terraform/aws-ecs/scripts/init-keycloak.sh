#!/bin/bash
# Initialize Keycloak with MCP Gateway configuration
# This script sets up the initial realm, clients, groups, and users
#
# Usage:
#   KEYCLOAK_ADMIN_URL=https://your-keycloak-url \
#   KEYCLOAK_ADMIN=admin \
#   KEYCLOAK_ADMIN_PASSWORD=your-admin-password \
#   AUTH_SERVER_EXTERNAL_URL=https://your-auth-server-url \
#   REGISTRY_URL=https://your-registry-url \
#   ./init-keycloak.sh
#
# Or set these in a .env file in the project root

set -eo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../keycloak/setup" && pwd)/group-path-preflight.sh"

# These will be set properly after loading .env in main()
KEYCLOAK_URL=""  # Will be overridden with KEYCLOAK_ADMIN_URL after .env is loaded
REALM="mcp-gateway"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Emit full group paths in mcpgw_group_paths, distinct from legacy flat groups.
# The auth-server removes exactly one leading slash only from this claim.
GROUPS_MAPPER_JSON='{
    "name": "groups",
    "protocol": "openid-connect",
    "protocolMapper": "oidc-group-membership-mapper",
    "consentRequired": false,
    "config": {
        "full.path": "true",
        "id.token.claim": "true",
        "access.token.claim": "true",
        "claim.name": "mcpgw_group_paths",
        "userinfo.token.claim": "true"
    }
}'

# Security baseline merged over a realm's current representation (the jq
# input): a stronger setting already in place is kept, a missing or weaker one
# is raised to the baseline. $gateway (true for the gateway realm) adds the
# password policy and the offline-session cap; the master realm gets the
# quick-login settings instead. adminEventsDetailsEnabled is always false so
# request bodies, which can carry secrets, are not stored in admin events.
# The same program is in keycloak/setup/init-keycloak.sh,
# keycloak/setup/upgrade-realm-for-dcr.sh and the Helm Job script.
REALM_SECURITY_BASELINE_JQ='
    if type != "object" then error("not a realm representation") else . end
    | . as $r
    | {
        bruteForceProtected: true,
        permanentLockout: ($r.permanentLockout == true),
        failureFactor: (if $r.bruteForceProtected == true and ($r.failureFactor // 0) > 0
                        then ([$r.failureFactor, 5] | min) else 5 end),
        waitIncrementSeconds: ([($r.waitIncrementSeconds // 0), 60] | max),
        maxFailureWaitSeconds: ([($r.maxFailureWaitSeconds // 0), 900] | max),
        eventsEnabled: true,
        eventsExpiration: (if $r.eventsExpiration == 0 or ($r.eventsExpiration // 0) >= 2592000
                           then $r.eventsExpiration else 2592000 end),
        adminEventsEnabled: true,
        adminEventsDetailsEnabled: false
      }
    + (if $adminExpiry then {attributes: (($r.attributes // {}) + {
            adminEventsExpiration: (if $r.attributes.adminEventsExpiration == "0"
                or (($r.attributes.adminEventsExpiration // "0") | tonumber) >= 2592000
                then $r.attributes.adminEventsExpiration else "2592000" end)})} else {} end)
    + (if $gateway then {
        passwordPolicy: (if (($r.passwordPolicy // "") | length) > 0
                         then $r.passwordPolicy else "length(8) and notUsername and notEmail" end),
        offlineSessionMaxLifespanEnabled: true,
        offlineSessionMaxLifespan: (if $r.offlineSessionMaxLifespanEnabled == true
                                       and ($r.offlineSessionMaxLifespan // 0) > 0
                                       and $r.offlineSessionMaxLifespan <= 2592000
                                    then $r.offlineSessionMaxLifespan else 2592000 end)
      } else {
        minimumQuickLoginWaitSeconds: ([($r.minimumQuickLoginWaitSeconds // 0), 60] | max),
        quickLoginCheckMilliSeconds: ([($r.quickLoginCheckMilliSeconds // 0), 1000] | max),
        maxDeltaTimeSeconds: ([($r.maxDeltaTimeSeconds // 0), 43200] | max)
      } end)
    | with_entries(select(.value != null))'

echo -e "${YELLOW}Keycloak initialization script for MCP Gateway Registry${NC}"
echo "=============================================="

# Function to wait for Keycloak to be ready
wait_for_keycloak() {
    echo -n "Waiting for Keycloak to be ready..."
    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        # Try to access the admin console which indicates Keycloak is ready
        if curl -f -s "${KEYCLOAK_URL}/admin/" > /dev/null 2>&1; then
            echo -e " ${GREEN}Ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 5
        attempt=$((attempt + 1))
    done

    echo -e " ${RED}Timeout!${NC}"
    echo "Keycloak did not become ready within 5 minutes"
    exit 1
}

# admin token can expire, rather than once per high-level operation.
_realm_fresh_token() {
    local now saved=0 token="" cache="$CREDENTIAL_IO_DIR/realm-token"
    now=$(date +%s) || return 1
    if [ -f "$cache" ]; then
        { IFS= read -r saved; IFS= read -r token; } < "$cache" || return 1
    fi
    if [ -z "$token" ] || [ "$((now - saved))" -ge 20 ]; then
        token=$(get_admin_token) || return 1
        if [ -z "$token" ]; then
            echo "Could not refresh Keycloak administrator credentials." >&2
            return 1
        fi
        printf '%s\n%s\n' "$now" "$token" | credential_write "$cache" || return 1
    fi
    printf '%s' "$token"
}

_admin_event_expiry_supported() {
    _keycloak_policy_version_supported "$KEYCLOAK_URL" "$1"
}

get_admin_token() {
    # Parse JSON structurally; whitespace in an OAuth response is insignificant.
    local response
    response=$(printf '%s' "$KEYCLOAK_ADMIN_PASSWORD" | \
        credential_curl -fsS -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
            -H "Content-Type: application/x-www-form-urlencoded" \
            --data-urlencode "username=${KEYCLOAK_ADMIN}" \
            --data-urlencode "password@-" \
            --data-urlencode "grant_type=password" \
            --data-urlencode "client_id=admin-cli") || return 1
    jq -er '.access_token | select(type == "string" and length > 0)' <<< "$response"
}


# URL-encode a single query-parameter value.
_uri_encode() {
    jq -rn --arg v "$1" '$v|@uri'
}

# GET an Admin API path (relative to /admin/realms/) and print the body only on
# HTTP 200. Any other status returns 1 so callers fail closed instead of parsing
# an error body as "nothing found".
_kc_get() {
    local token=$1
    local path=$2
    local response status
    response=$(realm_curl -sS -w $'\n%{http_code}' \
        -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${path}") || return 1
    status="${response##*$'\n'}"
    if [ "$status" != "200" ]; then
        echo -e "${RED}Error: GET /admin/realms/${path} returned HTTP ${status}${NC}" >&2
        return 1
    fi
    printf '%s' "${response%$'\n'*}"
}

_find_client_uuid() {
    local token=$1
    local client_id=$2
    local body
    body=$(_kc_get "$token" "${REALM}/clients?clientId=$(_uri_encode "$client_id")") || return 1
    printf '%s' "$body" | jq -r --arg c "$client_id" \
        'if type == "array" then ([.[] | select(.clientId == $c) | .id] | first // "") else error("unexpected clients response") end'
}

_upsert_groups_mapper() {
    local token=$1
    local client_uuid=$2
    local label=$3
    local mappers_path="${REALM}/clients/${client_uuid}/protocol-mappers/models"
    local mappers mapper_ids mapper_id mapper_json status


    mappers=$(_kc_get "$token" "$mappers_path") || return 1
    mapper_ids=$(printf '%s' "$mappers" | jq -r '
        def gateway_groups: .protocol == "openid-connect"
            and .protocolMapper == "oidc-group-membership-mapper"
            and (.config["claim.name"] == "groups" or .config["claim.name"] == "mcpgw_group_paths");
        if type != "array" then error("unexpected mappers response")
        elif any(.[]; (.name == "groups" or .config["claim.name"] == "mcpgw_group_paths") and (gateway_groups | not))
        then error("conflicting groups mapper")
        else .[] | select(gateway_groups) | .id
            | if type == "string" and length > 0 then . else error("groups mapper has no id") end end') || return 1

    _check_client_group_path_mappers "$KEYCLOAK_URL" "$REALM" "$token" "$client_uuid" || return 1
    if [ -n "$mapper_ids" ] && jq -e 'all(.[] | select(.protocol == "openid-connect"
        and .protocolMapper == "oidc-group-membership-mapper"
        and (.config["claim.name"] == "groups" or .config["claim.name"] == "mcpgw_group_paths"));
        .config["full.path"] == "true" and .config["claim.name"] == "mcpgw_group_paths")' >/dev/null <<< "$mappers"; then
        return 0
    fi
    _check_group_paths "$KEYCLOAK_URL" "$REALM" "$token" || return 1
    if [ -z "$mapper_ids" ]; then
        status=$(realm_curl -sS -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}/admin/realms/${mappers_path}" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$GROUPS_MAPPER_JSON") || status="000"
        if [ "$status" = "201" ]; then
            echo "  - Groups mapper created for ${label}"
            return 0
        fi
        echo -e "${RED}  - Failed to create groups mapper for ${label} (HTTP ${status})${NC}" >&2
        return 1
    fi

    for mapper_id in $mapper_ids; do
        mapper_json=$(printf '%s' "$mappers" | jq -c --arg id "$mapper_id" \
            '.[] | select(.id == $id)') || return 1
        if jq -e '.config["full.path"] == "true" and .config["claim.name"] == "mcpgw_group_paths"' >/dev/null <<< "$mapper_json"; then
            continue
        fi
        mapper_json=$(jq -c '.config = ((.config // {}) + {"full.path": "true", "claim.name": "mcpgw_group_paths"})' <<< "$mapper_json") || return 1
        status=$(realm_curl -sS -o /dev/null -w "%{http_code}" \
            -X PUT "${KEYCLOAK_URL}/admin/realms/${mappers_path}/${mapper_id}" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$mapper_json") || status="000"
        if [ "$status" != "204" ]; then
            echo -e "${RED}  - Failed to update groups mapper ${mapper_id} for ${label} (HTTP ${status})${NC}" >&2
            return 1
        fi
        echo "  - Groups mapper ${mapper_id} updated to mcpgw_group_paths (full.path=true) for ${label}"
    done
}

# Print the security settings to PUT on realm $2: REALM_SECURITY_BASELINE_JQ
# applied to the realm's current representation. $3 is true for the gateway
# realm. Returns 1 if the realm cannot be read, so nothing is written.
_realm_security_settings() {
    local token=$1
    local realm=$2
    local gateway=$3
    local current admin_expiry

    admin_expiry=$(_admin_event_expiry_supported "$token") || return 1
    current=$(_kc_get "$token" "$realm") || return 1
    if ! jq -ec --argjson gateway "$gateway" --argjson adminExpiry "$admin_expiry" "$REALM_SECURITY_BASELINE_JQ" <<< "$current"; then
        echo -e "${RED}Error: could not read the current settings of realm ${realm}${NC}" >&2
        return 1
    fi
}

# PUT the bootstrap's realm settings ($2) onto the existing gateway realm, with
# the security fields merged over its current values so a stronger setting the
# operator chose is kept. Partial PUT: other fields keep their values.
_update_realm_settings() {
    local token=$1
    local realm_json=$2
    local settings body update_response update_status

    settings=$(_realm_security_settings "$token" "$REALM" true) || return 1
    body=$(jq -c --argjson s "$settings" '. + $s' <<< "$realm_json")
    update_response=$(realm_curl -sS -w $'\n%{http_code}' \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$body") || update_response=$'\n000'
    update_status="${update_response##*$'\n'}"
    if [ "$update_status" = "204" ]; then
        echo -e "${GREEN}Realm already exists; settings updated (stronger existing settings kept).${NC}"
        return 0
    fi
    echo -e "${RED}Failed to update existing realm (HTTP ${update_status})${NC}" >&2
    echo "  response: ${update_response%$'\n'*}" >&2
    return 1
}

harden_master_realm() {
    local token=$1
    local status master_json
    echo "Hardening master realm..."
    master_json=$(_realm_security_settings "$token" "master" false) || return 1
    status=$(realm_curl -sS -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/master" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$master_json") || status="000"
    case "$status" in
        2??)
            echo -e "${GREEN}Master realm hardened (stronger existing settings kept).${NC}"
            return 0
            ;;
    esac
    echo -e "${RED}Failed to harden master realm (HTTP ${status})${NC}" >&2
    return 1
}

# Function to refresh admin token (call before each major operation)
refresh_token() {
    TOKEN=$(get_admin_token)
    if [ -z "$TOKEN" ]; then
        echo -e "${RED}Error: Failed to refresh authentication token${NC}"
        exit 1
    fi
}

# Check whether the realm exists: returns 0 if it does, 2 if Keycloak answers
# 404, and 1 on any other response so an auth or server error is never read as
# "absent".
realm_exists() {
    local token=$1
    local response
    response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}")

    case "$response" in
        200) return 0 ;;
        404) return 2 ;;
    esac
    echo -e "${RED}Could not check whether realm ${REALM} exists (HTTP ${response})${NC}" >&2
    return 1
}

# Function to create realm step by step
create_realm() {
    local token=$1

    echo "Creating MCP Gateway realm..."

    # Realm settings the bootstrap owns, as created. On an existing realm the
    # security fields are merged over its current values instead (see
    # _update_realm_settings).
    local realm_json='{
        "realm": "mcp-gateway",
        "enabled": true,
        "registrationAllowed": false,
        "loginWithEmailAllowed": true,
        "duplicateEmailsAllowed": false,
        "resetPasswordAllowed": true,
        "editUsernameAllowed": false,
        "bruteForceProtected": true,
        "failureFactor": 5,
        "waitIncrementSeconds": 60,
        "maxFailureWaitSeconds": 900,
        "permanentLockout": false,
        "passwordPolicy": "length(8) and notUsername and notEmail",
        "eventsEnabled": true,
        "eventsExpiration": 2592000,
        "adminEventsEnabled": true,
        "adminEventsDetailsEnabled": false,
        "offlineSessionMaxLifespanEnabled": true,
        "offlineSessionMaxLifespan": 2592000
    }'
    local admin_expiry
    admin_expiry=$(_admin_event_expiry_supported "$token") || return 1
    if [ "$admin_expiry" = true ]; then
        realm_json=$(jq -c '.attributes = {adminEventsExpiration: "2592000"}' <<< "$realm_json") || return 1
    fi

    # An existing realm is UPDATED, not skipped. Skipping meant a security setting
    # added here (e.g. bruteForceProtected) never reached a deployment whose realm
    # already existed, while the script still reported success.
    local exists_rc=0
    realm_exists "$token" || exists_rc=$?
    if [ "$exists_rc" = "1" ]; then
        return 1
    fi
    if [ "$exists_rc" = "0" ]; then
        _update_realm_settings "$token" "$realm_json"
        return
    fi

    local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$realm_json")

    if [ "$response" = "201" ]; then
        echo -e "${GREEN}Realm created successfully!${NC}"
        return 0
    elif [ "$response" = "409" ]; then
        # Lost a race with a concurrent bootstrap; the realm exists, so apply the
        # settings rather than continuing with an unknown configuration.
        _update_realm_settings "$token" "$realm_json"
        return
    else
        echo -e "${RED}Failed to create realm. HTTP status: ${response}${NC}" >&2
        return 1
    fi
}

# Function to create clients
create_clients() {
    local token=$1

    echo "Creating OAuth2 clients..."

    # Create web client
    # Build redirect URIs based on deployment mode
    # - Custom domain mode: use REGISTRY_URL
    # - CloudFront mode: use CLOUDFRONT_REGISTRY_URL
    # - Both modes: include both URLs

    local redirect_uris='"http://localhost:7860/*", "http://localhost:8888/*"'
    # No "+" wildcard: it expands to the origin of every redirect URI, which in
    # CloudFront/custom-domain mode silently widens CORS to whatever redirect was
    # registered. The browser never calls Keycloak via XHR (the auth-server does
    # the code exchange server-side), so origins are listed explicitly.
    local web_origins='"http://localhost:7860", "http://localhost:8888"'
    # Post-logout landing pages, "##"-delimited (Keycloak multi-value attribute).
    # Set explicitly so the allowlist does NOT fall back to redirectUris, which
    # includes the auth-server OAuth callback -- not a valid logout destination.
    local post_logout_uris='http://localhost:7860/*##http://localhost:8888/*'

    # Add custom domain URLs if available
    if [ -n "$REGISTRY_URL" ] && [ "$REGISTRY_URL" != "http://localhost:7860" ]; then
        redirect_uris="${redirect_uris}, \"${REGISTRY_URL}/oauth2/callback/keycloak\", \"${REGISTRY_URL}/*\""
        web_origins="${web_origins}, \"${REGISTRY_URL}\""
        post_logout_uris="${post_logout_uris}##${REGISTRY_URL}/*"
        echo "  - Adding custom domain redirect URIs: ${REGISTRY_URL}"
    fi

    # Add CloudFront URLs if available
    if [ -n "$CLOUDFRONT_REGISTRY_URL" ]; then
        redirect_uris="${redirect_uris}, \"${CLOUDFRONT_REGISTRY_URL}/oauth2/callback/keycloak\", \"${CLOUDFRONT_REGISTRY_URL}/*\""
        web_origins="${web_origins}, \"${CLOUDFRONT_REGISTRY_URL}\""
        post_logout_uris="${post_logout_uris}##${CLOUDFRONT_REGISTRY_URL}/*"
        echo "  - Adding CloudFront redirect URIs: ${CLOUDFRONT_REGISTRY_URL}"
    fi

    # If neither is set, use localhost as fallback
    if [ -z "$REGISTRY_URL" ] && [ -z "$CLOUDFRONT_REGISTRY_URL" ]; then
        echo "  - Using localhost fallback for redirect URIs"
    fi

    local web_client_json='{
        "clientId": "mcp-gateway-web",
        "name": "MCP Gateway Web Client",
        "enabled": true,
        "clientAuthenticatorType": "client-secret",
        "redirectUris": ['"${redirect_uris}"'],
        "webOrigins": ['"${web_origins}"'],
        "attributes": {
            "post.logout.redirect.uris": "'"${post_logout_uris}"'"
        },
        "protocol": "openid-connect",
        "standardFlowEnabled": true,
        "implicitFlowEnabled": false,
        "directAccessGrantsEnabled": false,
        "serviceAccountsEnabled": false,
        "fullScopeAllowed": false,
        "publicClient": false
    }'

    local web_response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$web_client_json")

    if [ "$web_response" = "201" ]; then
        echo "  - Web client created"
    elif [ "$web_response" = "409" ]; then
        echo "  - Web client already exists, updating..."
        local web_client_uuid=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/clients?clientId=mcp-gateway-web" 2>/dev/null | \
            jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)
        if [ -z "$web_client_uuid" ] || [ "$web_client_uuid" = "null" ]; then
            # Silently skipping here used to leave the client unmodified while the
            # script printed success, so the webOrigins/post-logout hardening never
            # landed on an existing realm.
            echo -e "${RED}  - Could not resolve mcp-gateway-web client id; cannot apply client settings${NC}" >&2
            return 1
        fi
        local update_response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${web_client_uuid}" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$web_client_json")
        if [ "$update_response" != "204" ]; then
            echo -e "  - ${RED}Failed to update web client (HTTP $update_response)${NC}" >&2
            return 1
        fi
        echo -e "  - ${GREEN}Web client updated successfully${NC}"
    else
        echo -e "${RED}  - Failed to create web client (HTTP $web_response)${NC}" >&2
        return 1
    fi

    # Create M2M client
    local m2m_client_json='{
        "clientId": "mcp-gateway-m2m",
        "name": "MCP Gateway M2M Client",
        "enabled": true,
        "clientAuthenticatorType": "client-secret",
        "protocol": "openid-connect",
        "standardFlowEnabled": false,
        "implicitFlowEnabled": false,
        "directAccessGrantsEnabled": false,
        "serviceAccountsEnabled": true,
        "fullScopeAllowed": false,
        "publicClient": false
    }'

    local m2m_response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$m2m_client_json")

    if [ "$m2m_response" = "201" ]; then
        echo "  - M2M client created"
    elif [ "$m2m_response" = "409" ]; then
        echo "  - M2M client already exists"
    else
        echo -e "${RED}  - Failed to create M2M client (HTTP $m2m_response)${NC}"
    fi

    echo -e "${GREEN}Clients configured successfully!${NC}"
}

# Function to create groups
create_groups() {
    local token=$1

    echo "Creating user groups..."

    local groups=(
        "mcp-registry-admin"
        "mcp-registry-user"
        "mcp-registry-developer"
        "mcp-registry-operator"
        "mcp-servers-unrestricted"
        "mcp-servers-restricted"
        "a2a-agent-admin"
        "a2a-agent-publisher"
        "a2a-agent-user"
        "registry-admins"
        "registry-users-lob1"
        "registry-users-lob2"
    )

    for group in "${groups[@]}"; do
        local group_json='{
            "name": "'$group'",
            "attributes": {
                "description": ["'$group' group for MCP Gateway access"]
            }
        }'

        realm_curl -s -X POST "${KEYCLOAK_URL}/admin/realms/mcp-gateway/groups" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$group_json" > /dev/null
    done

    echo -e "${GREEN}Groups created successfully!${NC}"
}

# Function to create custom scopes
create_scopes() {
    # Refresh token to ensure it's valid
    refresh_token
    local token=$TOKEN

    echo "Creating custom MCP scopes..."

    local scopes=("mcp-servers-unrestricted/read" "mcp-servers-unrestricted/execute" "mcp-servers-restricted/read" "mcp-servers-restricted/execute")

    for scope in "${scopes[@]}"; do
        local scope_json='{
            "name": "'$scope'",
            "description": "MCP Gateway scope for '$scope' access",
            "protocol": "openid-connect"
        }'

        local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$scope_json")

        if [ "$response" = "201" ]; then
            echo "  - Created scope: $scope"
        elif [ "$response" = "409" ]; then
            echo "  - Scope already exists: $scope"
        else
            echo -e "${RED}  - Failed to create scope: $scope (HTTP $response)${NC}"
        fi
    done

    echo -e "${GREEN}Custom scopes created successfully!${NC}"
}

# Function to assign scopes to M2M client
setup_m2m_scopes() {
    # Refresh token to ensure it's valid
    refresh_token
    local token=$TOKEN

    echo "Setting up M2M client scopes..."

    # Get M2M client ID
    local m2m_client_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/clients?clientId=mcp-gateway-m2m" 2>/dev/null | \
        jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

    if [ -z "$m2m_client_id" ] || [ "$m2m_client_id" = "null" ]; then
        echo -e "${RED}Error: Could not find mcp-gateway-m2m client${NC}"
        return 1
    fi

    # Get all available client scopes
    local scopes=("mcp-servers-unrestricted/read" "mcp-servers-unrestricted/execute" "mcp-servers-restricted/read" "mcp-servers-restricted/execute")

    for scope in "${scopes[@]}"; do
        # Get scope ID
        local scope_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes" 2>/dev/null | \
            jq -r 'if type == "array" then (.[] | select(.name=="'$scope'") | .id) else empty end' 2>/dev/null)

        if [ ! -z "$scope_id" ] && [ "$scope_id" != "null" ]; then
            # Add scope as default client scope
            local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
                -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${m2m_client_id}/default-client-scopes/${scope_id}" \
                -H "Authorization: Bearer ${token}")

            if [ "$response" = "204" ]; then
                echo "  - Assigned scope: $scope"
            else
                echo -e "${YELLOW}  - Warning: Could not assign scope $scope (HTTP $response)${NC}"
            fi
        else
            echo -e "${RED}  - Error: Could not find scope: $scope${NC}"
        fi
    done

    echo -e "${GREEN}M2M client scopes configured successfully!${NC}"
}

# Function to create service account user for M2M client
create_service_account_user() {
    # Refresh token to ensure it's valid
    refresh_token
    local token=$TOKEN
    local service_account_username="service-account-mcp-gateway-m2m"

    echo "Creating service account user: $service_account_username"

    # Check if user already exists
    local existing_user=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=$service_account_username" 2>/dev/null | \
        jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

    if [ ! -z "$existing_user" ]; then
        echo -e "${YELLOW}Service account user already exists with ID: $existing_user${NC}"
        return 0
    fi

    # Create service account user
    local user_json='{
        "username": "'$service_account_username'",
        "enabled": true,
        "emailVerified": true,
        "serviceAccountClientId": "mcp-gateway-m2m"
    }'

    local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/users" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$user_json")

    if [ "$response" = "201" ]; then
        echo -e "${GREEN}Service account user created successfully!${NC}"

        # Get the newly created user ID
        local user_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=$service_account_username" 2>/dev/null | \
            jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

        echo "Created service account user with ID: $user_id"

        # Assign user to mcp-servers-unrestricted group
        local group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" 2>/dev/null | \
            jq -r 'if type == "array" then (.[] | select(.name=="mcp-servers-unrestricted") | .id) else empty end' 2>/dev/null)

        if [ ! -z "$group_id" ] && [ "$group_id" != "null" ]; then
            local group_response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
                -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/$user_id/groups/$group_id" \
                -H "Authorization: Bearer ${token}")

            if [ "$group_response" = "204" ]; then
                echo -e "${GREEN}Service account assigned to mcp-servers-unrestricted group!${NC}"
            else
                echo -e "${YELLOW}Warning: Could not assign service account to mcp-servers-unrestricted group (HTTP $group_response)${NC}"
            fi
        else
            echo -e "${RED}Error: Could not find mcp-servers-unrestricted group${NC}"
        fi

        # Assign user to a2a-agent-admin group for A2A agent access
        local a2a_group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" 2>/dev/null | \
            jq -r 'if type == "array" then (.[] | select(.name=="a2a-agent-admin") | .id) else empty end' 2>/dev/null)

        if [ ! -z "$a2a_group_id" ] && [ "$a2a_group_id" != "null" ]; then
            local a2a_response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
                -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/$user_id/groups/$a2a_group_id" \
                -H "Authorization: Bearer ${token}")

            if [ "$a2a_response" = "204" ]; then
                echo -e "${GREEN}Service account assigned to a2a-agent-admin group!${NC}"
            else
                echo -e "${YELLOW}Warning: Could not assign service account to a2a-agent-admin group (HTTP $a2a_response)${NC}"
            fi
        else
            echo -e "${YELLOW}Warning: a2a-agent-admin group not found. Create it manually if A2A agent support is needed.${NC}"
        fi

        return 0
    elif [ "$response" = "409" ]; then
        echo -e "${YELLOW}Service account user already exists. Continuing...${NC}"
        return 0
    else
        echo -e "${RED}Failed to create service account user. HTTP status: ${response}${NC}"
        return 1
    fi
}

# Function to create service account clients for A2A agents
create_service_account_clients() {
    # Refresh token to ensure it's valid
    refresh_token
    local token=$TOKEN

    echo "Creating service account clients for A2A agents..."

    # Define service account clients
    local clients=("registry-admin-bot" "lob1-bot" "lob2-bot")
    local groups=("registry-admins" "registry-users-lob1" "registry-users-lob2")

    for i in "${!clients[@]}"; do
        local client_name="${clients[$i]}"
        local group_name="${groups[$i]}"

        echo "  Creating client: $client_name"

        # Create M2M service account client
        local client_json='{
            "clientId": "'$client_name'",
            "name": "'$client_name' Service Account",
            "description": "Service account for '$client_name' operations",
            "enabled": true,
            "serviceAccountsEnabled": true,
            "standardFlowEnabled": false,
            "implicitFlowEnabled": false,
            "directAccessGrantsEnabled": false,
            "publicClient": false,
            "protocol": "openid-connect"
        }'

        local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$client_json")

        if [ "$response" = "201" ]; then
            echo "    - Client created successfully"
        elif [ "$response" = "409" ]; then
            echo "    - Client already exists"
        else
            echo -e "${RED}    - Failed to create client (HTTP $response)${NC}"
            continue
        fi

        # Get the client UUID
        local client_uuid=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/clients?clientId=${client_name}" 2>/dev/null | \
            jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

        if [ -z "$client_uuid" ] || [ "$client_uuid" = "null" ]; then
            echo -e "${RED}    - Error: Could not find client UUID${NC}"
            continue
        fi

        # Get the service account user ID
        local service_account_user=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${client_uuid}/service-account-user" 2>/dev/null | \
            jq -r '.id // empty' 2>/dev/null)

        if [ -z "$service_account_user" ] || [ "$service_account_user" = "null" ]; then
            echo -e "${RED}    - Error: Could not get service account user ID${NC}"
            continue
        fi

        # Get the group ID
        local group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" 2>/dev/null | \
            jq -r 'if type == "array" then (.[] | select(.name=="'$group_name'") | .id) else empty end' 2>/dev/null)

        if [ -z "$group_id" ] || [ "$group_id" = "null" ]; then
            echo -e "${RED}    - Error: Could not find group: $group_name${NC}"
            continue
        fi

        # Assign service account to the group
        local group_response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/${service_account_user}/groups/${group_id}" \
            -H "Authorization: Bearer ${token}")

        if [ "$group_response" = "204" ]; then
            echo "    - Service account assigned to group: $group_name"
        else
            echo -e "${YELLOW}    - Warning: Could not assign to group (HTTP $group_response)${NC}"
        fi

        # Upsert full paths without changing the service-account bootstrap.
        _upsert_groups_mapper "$token" "$client_uuid" "$client_name" || return 1
    done

    echo -e "${GREEN}Service account clients created successfully!${NC}"
}

# Function to update user password (for existing users)
update_user_password() {
    local token=$1
    local username=$2
    local password=$3

    # Get user ID
    local user_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=${username}" 2>/dev/null | \
        jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

    if [ -z "$user_id" ] || [ "$user_id" = "null" ]; then
        return 1
    fi

    # Reset password (temporary: true forces a change on next login)
    local password_json='{
        "type": "password",
        "value": "'"${password}"'",
        "temporary": true
    }'

    local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/${user_id}/reset-password" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$password_json")

    [ "$response" = "204" ]
}

# Function to create test users
create_users() {
    # Refresh token to ensure it's valid
    refresh_token
    local token=$TOKEN

    echo "Creating test users..."

    # SA-9: LOB demo users must not fall back to weak default passwords. Require
    # explicit passwords for any LOB user that will be created.
    if [ -z "$LOB1_USER_PASSWORD" ] || [ -z "$LOB2_USER_PASSWORD" ]; then
        echo -e "${RED}Error: LOB1_USER_PASSWORD and LOB2_USER_PASSWORD are required${NC}"
        echo "These set the passwords for the lob1-user / lob2-user demo accounts."
        echo "Export both before running this script, e.g.:"
        echo "  export LOB1_USER_PASSWORD='YourSecurePassword1'"
        echo "  export LOB2_USER_PASSWORD='YourSecurePassword2'"
        exit 1
    fi

    # Define usernames for consistency
    local admin_username="admin"
    local lob1_username="lob1-user"
    local lob2_username="lob2-user"

    # Create admin user (temporary: true forces a password change on first login)
    local admin_user_json='{
        "username": "'$admin_username'",
        "email": "'$admin_username'@example.com",
        "enabled": true,
        "emailVerified": true,
        "firstName": "Admin",
        "lastName": "User",
        "credentials": [
            {
                "type": "password",
                "value": "'${INITIAL_ADMIN_PASSWORD}'",
                "temporary": true
            }
        ]
    }'

    local admin_response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/users" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$admin_user_json")

    if [ "$admin_response" = "201" ]; then
        echo "  - Created admin user with password from Secrets Manager"
    elif [ "$admin_response" = "409" ]; then
        echo "  - Admin user already exists, updating password..."
        if update_user_password "$token" "$admin_username" "$INITIAL_ADMIN_PASSWORD"; then
            echo "  - Admin password updated successfully"
        else
            echo -e "${YELLOW}  - Warning: Could not update admin password${NC}"
        fi
    else
        echo -e "${RED}  - Failed to create admin user (HTTP $admin_response)${NC}"
    fi

    # Create lob1-user (temporary: true forces a password change on first login)
    local lob1_user_json='{
        "username": "'$lob1_username'",
        "email": "'$lob1_username'@example.com",
        "enabled": true,
        "emailVerified": true,
        "firstName": "LOB1",
        "lastName": "User",
        "credentials": [
            {
                "type": "password",
                "value": "'${LOB1_USER_PASSWORD}'",
                "temporary": true
            }
        ]
    }'

    realm_curl -s -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/users" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$lob1_user_json" > /dev/null

    # Create lob2-user (temporary: true forces a password change on first login)
    local lob2_user_json='{
        "username": "'$lob2_username'",
        "email": "'$lob2_username'@example.com",
        "enabled": true,
        "emailVerified": true,
        "firstName": "LOB2",
        "lastName": "User",
        "credentials": [
            {
                "type": "password",
                "value": "'${LOB2_USER_PASSWORD}'",
                "temporary": true
            }
        ]
    }'

    realm_curl -s -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/users" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$lob2_user_json" > /dev/null

    echo "Assigning users to groups..."

    # Get user IDs
    local admin_user_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=$admin_username" 2>/dev/null | \
        jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

    local lob1_user_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=$lob1_username" 2>/dev/null | \
        jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

    local lob2_user_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=$lob2_username" 2>/dev/null | \
        jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

    # Get the group IDs the admin and LOB users need
    local admin_group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" 2>/dev/null | \
        jq -r 'if type == "array" then (.[] | select(.name=="mcp-registry-admin") | .id) else empty end' 2>/dev/null)

    local unrestricted_group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" 2>/dev/null | \
        jq -r 'if type == "array" then (.[] | select(.name=="mcp-servers-unrestricted") | .id) else empty end' 2>/dev/null)

    local lob1_group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" 2>/dev/null | \
        jq -r 'if type == "array" then (.[] | select(.name=="registry-users-lob1") | .id) else empty end' 2>/dev/null)

    local lob2_group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" 2>/dev/null | \
        jq -r 'if type == "array" then (.[] | select(.name=="registry-users-lob2") | .id) else empty end' 2>/dev/null)

    # Define usernames for consistent logging
    local admin_username="admin"
    local lob1_username="lob1-user"
    local lob2_username="lob2-user"

    # Get registry-admins group ID for admin user
    local registry_admins_group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" 2>/dev/null | \
        jq -r 'if type == "array" then (.[] | select(.name=="registry-admins") | .id) else empty end' 2>/dev/null)

    # Assign admin user to admin group and unrestricted servers group
    if [ ! -z "$admin_user_id" ] && [ ! -z "$admin_group_id" ]; then
        realm_curl -s -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/$admin_user_id/groups/$admin_group_id" \
            -H "Authorization: Bearer ${token}" > /dev/null
        echo "  - $admin_username assigned to mcp-registry-admin group"
    fi

    # Also assign admin to unrestricted servers group for full access
    if [ ! -z "$admin_user_id" ] && [ ! -z "$unrestricted_group_id" ]; then
        realm_curl -s -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/$admin_user_id/groups/$unrestricted_group_id" \
            -H "Authorization: Bearer ${token}" > /dev/null
        echo "  - $admin_username assigned to mcp-servers-unrestricted group"
    fi

    # Assign admin to registry-admins group for full UI permissions
    if [ ! -z "$admin_user_id" ] && [ ! -z "$registry_admins_group_id" ]; then
        realm_curl -s -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/$admin_user_id/groups/$registry_admins_group_id" \
            -H "Authorization: Bearer ${token}" > /dev/null
        echo "  - $admin_username assigned to registry-admins group"
    fi

    # Assign lob1-user to registry-users-lob1 group
    if [ ! -z "$lob1_user_id" ] && [ ! -z "$lob1_group_id" ]; then
        realm_curl -s -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/$lob1_user_id/groups/$lob1_group_id" \
            -H "Authorization: Bearer ${token}" > /dev/null
        echo "  - $lob1_username assigned to registry-users-lob1 group"
    fi

    # Assign lob2-user to registry-users-lob2 group
    if [ ! -z "$lob2_user_id" ] && [ ! -z "$lob2_group_id" ]; then
        realm_curl -s -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/users/$lob2_user_id/groups/$lob2_group_id" \
            -H "Authorization: Bearer ${token}" > /dev/null
        echo "  - $lob2_username assigned to registry-users-lob2 group"
    fi

    echo -e "${GREEN}Users created and assigned to groups successfully!${NC}"
}

# Function to create client secrets
setup_client_secrets() {
    # Refresh token to ensure it's valid
    refresh_token
    local token=$TOKEN

    echo "Setting up client secrets..."

    # Get web client ID
    local web_client_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/clients?clientId=mcp-gateway-web" 2>/dev/null | \
        jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

    # Generate secret for web client
    realm_curl -s -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${web_client_id}/client-secret" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" > /dev/null

    local web_secret_response=$(realm_curl -s "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${web_client_id}/client-secret" \
        -H "Authorization: Bearer ${token}")
    web_secret=$(echo "$web_secret_response" | jq -r '.value // empty')

    # Get M2M client ID
    local m2m_client_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/clients?clientId=mcp-gateway-m2m" 2>/dev/null | \
        jq -r 'if type == "array" then (.[0].id // empty) else empty end' 2>/dev/null)

    # Generate secret for M2M client
    realm_curl -s -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${m2m_client_id}/client-secret" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" > /dev/null

    local m2m_secret_response=$(realm_curl -s "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${m2m_client_id}/client-secret" \
        -H "Authorization: Bearer ${token}")
    m2m_secret=$(echo "$m2m_secret_response" | jq -r '.value // empty')

    echo -e "${GREEN}Client secrets generated!${NC}"

    # Save web client secret to AWS Secrets Manager
    if [ -n "$web_secret" ] && command -v aws &> /dev/null; then
        echo "Saving web client secret to AWS Secrets Manager..."
        if aws secretsmanager update-secret \
            --secret-id mcp-gateway-keycloak-client-secret \
            --secret-string "{\"client_id\": \"mcp-gateway-web\", \"client_secret\": \"${web_secret}\"}" \
            --region "${AWS_REGION}" &>/dev/null; then
            echo -e "${GREEN}Web client secret saved to AWS Secrets Manager!${NC}"
        else
            echo -e "${YELLOW}Warning: Could not save web client secret to Secrets Manager${NC}"
            echo "You can manually update it with:"
            echo "  aws secretsmanager update-secret --secret-id mcp-gateway-keycloak-client-secret \\"
            echo "    --secret-string '{\"client_id\": \"mcp-gateway-web\", \"client_secret\": \"${web_secret}\"}' \\"
            echo "    --region \${AWS_REGION}"
        fi
    fi

    # Save M2M client secret to AWS Secrets Manager
    if [ -n "$m2m_secret" ] && command -v aws &> /dev/null; then
        echo "Saving M2M client secret to AWS Secrets Manager..."
        if aws secretsmanager update-secret \
            --secret-id mcp-gateway-keycloak-m2m-client-secret \
            --secret-string "{\"client_id\": \"mcp-gateway-m2m\", \"client_secret\": \"${m2m_secret}\"}" \
            --region "${AWS_REGION}" &>/dev/null; then
            echo -e "${GREEN}M2M client secret saved to AWS Secrets Manager!${NC}"
        else
            echo -e "${YELLOW}Warning: Could not save M2M client secret to Secrets Manager${NC}"
            echo "You can manually update it with:"
            echo "  aws secretsmanager update-secret --secret-id mcp-gateway-keycloak-m2m-client-secret \\"
            echo "    --secret-string '{\"client_id\": \"mcp-gateway-m2m\", \"client_secret\": \"${m2m_secret}\"}' \\"
            echo "    --region \${AWS_REGION}"
        fi
    fi

    echo ""
    echo "=============================================="
    echo -e "${YELLOW}Client credentials have been created.${NC}"
    echo "=============================================="
    echo ""
    echo "Web Client:"
    echo "  Client ID: mcp-gateway-web"
    echo "  Secret: ${web_secret}"
    echo ""
    echo "M2M Client:"
    echo "  Client ID: mcp-gateway-m2m"
    echo "  Secret: ${m2m_secret}"
    echo ""
    echo -e "${GREEN}Note: Both client secrets have been saved to AWS Secrets Manager${NC}"
    echo "  - mcp-gateway-keycloak-client-secret (web client)"
    echo "  - mcp-gateway-keycloak-m2m-client-secret (M2M client)"
    echo "=============================================="
}

# Function to setup groups mapper for OAuth2 clients
setup_groups_mapper() {
    # Refresh token to ensure it's valid
    refresh_token
    local token=$TOKEN

    echo "Setting up groups mapper for OAuth2 clients..."

    local client_id client_uuid
    for client_id in "mcp-gateway-web" "mcp-gateway-m2m"; do
        echo "Setting up groups mapper for ${client_id} client..."
        client_uuid=$(_find_client_uuid "$token" "$client_id") || return 1
        if [ -z "$client_uuid" ]; then
            echo -e "${RED}Error: Could not find ${client_id} client${NC}" >&2
            return 1
        fi
        _upsert_groups_mapper "$token" "$client_uuid" "$client_id" || return 1
    done
    echo -e "${GREEN}Groups mappers configured (full.path=true).${NC}"
}

# Function to load values from terraform-outputs.json
load_from_terraform_outputs() {
    local terraform_outputs="$SCRIPT_DIR/terraform-outputs.json"

    if [ ! -f "$terraform_outputs" ]; then
        echo -e "${YELLOW}Warning: terraform-outputs.json not found in $SCRIPT_DIR${NC}"
        return 1
    fi

    echo "Loading values from terraform-outputs.json..."

    # Extract values from JSON
    if command -v jq &> /dev/null; then
        # Load KEYCLOAK_ADMIN_URL if not set
        if [ -z "$KEYCLOAK_ADMIN_URL" ]; then
            local keycloak_url=$(jq -r '.keycloak_url.value // empty' "$terraform_outputs" 2>/dev/null)
            if [ -n "$keycloak_url" ] && [ "$keycloak_url" != "null" ]; then
                KEYCLOAK_ADMIN_URL="$keycloak_url"
                echo "  - Loaded KEYCLOAK_ADMIN_URL: $KEYCLOAK_ADMIN_URL"
            fi
        fi

        # Load AUTH_SERVER_EXTERNAL_URL if not set
        if [ -z "$AUTH_SERVER_EXTERNAL_URL" ]; then
            local auth_url=$(jq -r '.mcp_gateway_auth_url.value // empty' "$terraform_outputs" 2>/dev/null)
            if [ -n "$auth_url" ] && [ "$auth_url" != "null" ]; then
                AUTH_SERVER_EXTERNAL_URL="$auth_url"
                echo "  - Loaded AUTH_SERVER_EXTERNAL_URL: $AUTH_SERVER_EXTERNAL_URL"
            fi
        fi

        # Load REGISTRY_URL if not set (custom domain mode)
        if [ -z "$REGISTRY_URL" ]; then
            local registry_url=$(jq -r '.registry_url.value // empty' "$terraform_outputs" 2>/dev/null)
            if [ -n "$registry_url" ] && [ "$registry_url" != "null" ]; then
                REGISTRY_URL="$registry_url"
                echo "  - Loaded REGISTRY_URL: $REGISTRY_URL"
            fi
        fi

        # Load CLOUDFRONT_REGISTRY_URL if not set (CloudFront mode)
        if [ -z "$CLOUDFRONT_REGISTRY_URL" ]; then
            local cloudfront_url=$(jq -r '.cloudfront_mcp_gateway_url.value // empty' "$terraform_outputs" 2>/dev/null)
            if [ -n "$cloudfront_url" ] && [ "$cloudfront_url" != "null" ]; then
                CLOUDFRONT_REGISTRY_URL="$cloudfront_url"
                echo "  - Loaded CLOUDFRONT_REGISTRY_URL: $CLOUDFRONT_REGISTRY_URL"
            fi
        fi

        # Load deployment mode to understand which URLs are active
        if [ -z "$DEPLOYMENT_MODE" ]; then
            local deployment_mode=$(jq -r '.deployment_mode.value // empty' "$terraform_outputs" 2>/dev/null)
            if [ -n "$deployment_mode" ] && [ "$deployment_mode" != "null" ]; then
                DEPLOYMENT_MODE="$deployment_mode"
                echo "  - Loaded DEPLOYMENT_MODE: $DEPLOYMENT_MODE"
            fi
        fi

        # Check if we successfully loaded values
        if [ -n "$AUTH_SERVER_EXTERNAL_URL" ] || [ -n "$REGISTRY_URL" ] || [ -n "$KEYCLOAK_ADMIN_URL" ] || [ -n "$CLOUDFRONT_REGISTRY_URL" ]; then
            return 0
        fi
    else
        echo -e "${YELLOW}Warning: jq not found. Skipping terraform-outputs.json parsing${NC}"
        return 1
    fi

    return 1
}

# Main execution
main() {
    # Get script directory and find .env file
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
    ENV_FILE="$PROJECT_ROOT/.env"

    # Load environment variables from .env file if it exists
    if [ -f "$ENV_FILE" ]; then
        echo "Loading environment variables from $ENV_FILE..."
        local caller_vars="AWS_REGION KEYCLOAK_ADMIN_URL KEYCLOAK_ADMIN KEYCLOAK_ADMIN_PASSWORD REGISTRY_URL AUTH_SERVER_EXTERNAL_URL CLOUDFRONT_REGISTRY_URL DEPLOYMENT_MODE INITIAL_ADMIN_PASSWORD LOB1_USER_PASSWORD LOB2_USER_PASSWORD ENABLE_DCR_CONFIG ENABLE_CIMD_CONFIG DCR_KEEP_CLIENTS DCR_SWEEP_ACKNOWLEDGED CIMD_SWEEP_ACKNOWLEDGED DCR_ALLOWED_SCOPES CIMD_TRUSTED_DOMAINS REGISTRATION_TRUSTED_CLIENT_IDS"
        local name present saved
        local -a caller_names=() caller_values=()
        for name in $caller_vars; do
            eval 'present=${'"$name"'+set}'
            if [ "$present" = set ]; then
                caller_names+=("$name")
                caller_values+=("${!name}")
            fi
        done
        set -a
        source "$ENV_FILE"
        set +a
        local index
        for ((index=0; index<${#caller_names[@]}; index++)); do
            export "${caller_names[$index]}=${caller_values[$index]}"
        done
        echo "Environment variables loaded successfully"
    else
        echo "No .env file found at $ENV_FILE"
    fi

    # Check for AWS_REGION - required for SSM and Secrets Manager operations
    if [ -z "$AWS_REGION" ]; then
        echo -e "${RED}Error: AWS_REGION environment variable is required${NC}"
        echo "Please set AWS_REGION before running this script:"
        echo "  export AWS_REGION=us-east-1"
        echo "  # or"
        echo "  AWS_REGION=us-east-1 $0"
        exit 1
    fi
    echo "Using AWS Region: $AWS_REGION"

    # Try to load missing values from terraform-outputs.json
    if [ -z "$AUTH_SERVER_EXTERNAL_URL" ] || [ -z "$REGISTRY_URL" ] || [ -z "$KEYCLOAK_ADMIN_URL" ]; then
        echo "Attempting to load missing values from terraform-outputs.json..."
        load_from_terraform_outputs || true
    fi

    # Override KEYCLOAK_URL with KEYCLOAK_ADMIN_URL for API calls
    KEYCLOAK_URL="${KEYCLOAK_ADMIN_URL:-}"
    if [ -z "$KEYCLOAK_URL" ]; then
        echo -e "${RED}Error: KEYCLOAK_ADMIN_URL is required${NC}"
        echo "Please set KEYCLOAK_ADMIN_URL in your .env file or environment,"
        echo "or ensure terraform-outputs.json contains keycloak_url."
        exit 1
    fi
    KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN-admin}"
    if [ -z "$KEYCLOAK_ADMIN" ]; then
        echo "KEYCLOAK_ADMIN must not be empty." >&2
        return 1
    fi
    echo "Using Keycloak API URL: $KEYCLOAK_URL"

    # Display loaded configuration
    echo ""
    echo "Configuration:"
    echo "  - KEYCLOAK_URL: $KEYCLOAK_URL"
    echo "  - AUTH_SERVER_EXTERNAL_URL: ${AUTH_SERVER_EXTERNAL_URL:-<not set>}"
    echo "  - REGISTRY_URL: ${REGISTRY_URL:-<not set>}"
    echo ""

    # Try to load admin credentials from SSM Parameter Store if not set
    if [ "${KEYCLOAK_ADMIN_PASSWORD+x}" != x ]; then
        echo "Attempting to load KEYCLOAK_ADMIN_PASSWORD from SSM Parameter Store..."
        if command -v aws &> /dev/null; then
            SSM_PASSWORD=$(aws ssm get-parameter --name "/keycloak/admin_password" --with-decryption --query 'Parameter.Value' --output text --region "${AWS_REGION}" 2>/dev/null)
            if [ -n "$SSM_PASSWORD" ] && [ "$SSM_PASSWORD" != "null" ]; then
                KEYCLOAK_ADMIN_PASSWORD="$SSM_PASSWORD"
                echo -e "${GREEN}Loaded KEYCLOAK_ADMIN_PASSWORD from SSM Parameter Store${NC}"
            fi
        fi
    fi

    # Check if admin password is set (from env var or SSM)
    if [ -z "$KEYCLOAK_ADMIN_PASSWORD" ]; then
        echo -e "${RED}Error: KEYCLOAK_ADMIN_PASSWORD not found${NC}"
        echo "Please either:"
        echo "  1. Export KEYCLOAK_ADMIN_PASSWORD environment variable"
        echo "  2. Ensure AWS credentials are configured and SSM parameter '/keycloak/admin_password' exists"
        exit 1
    fi

    # Check if initial admin password is set (for realm admin user creation)
    if [ -z "$INITIAL_ADMIN_PASSWORD" ]; then
        echo -e "${RED}Error: INITIAL_ADMIN_PASSWORD environment variable is required${NC}"
        echo "This password will be used for the 'admin' user in the mcp-gateway realm."
        echo "Please export INITIAL_ADMIN_PASSWORD before running this script:"
        echo "  export INITIAL_ADMIN_PASSWORD='YourSecurePassword123'"
        exit 1
    fi

    # Wait for Keycloak to be ready
    wait_for_keycloak

    # Get admin token
    echo "Authenticating with Keycloak..."
    TOKEN=$(get_admin_token)

    if [ -z "$TOKEN" ]; then
        echo -e "${RED}Error: Failed to authenticate with Keycloak${NC}"
        echo "Please check your admin credentials"
        exit 1
    fi

    echo -e "${GREEN}Authentication successful!${NC}"

    DCR_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../keycloak/setup" && pwd)/upgrade-realm-for-dcr.sh"
    if ! KEYCLOAK_ADMIN_URL="$KEYCLOAK_URL" KEYCLOAK_ADMIN="$KEYCLOAK_ADMIN" KEYCLOAK_ADMIN_PASSWORD="$KEYCLOAK_ADMIN_PASSWORD" \
        ENABLE_DCR_CONFIG="${ENABLE_DCR_CONFIG:-false}" ENABLE_CIMD_CONFIG="${ENABLE_CIMD_CONFIG:-false}" \
        DCR_KEEP_CLIENTS="${DCR_KEEP_CLIENTS:-}" DCR_SWEEP_ACKNOWLEDGED="${DCR_SWEEP_ACKNOWLEDGED-true}" CIMD_SWEEP_ACKNOWLEDGED="${CIMD_SWEEP_ACKNOWLEDGED:-false}" \
        DCR_ALLOWED_SCOPES="${DCR_ALLOWED_SCOPES:-}" CIMD_TRUSTED_DOMAINS="${CIMD_TRUSTED_DOMAINS:-}" \
        REGISTRATION_TRUSTED_CLIENT_IDS="${REGISTRATION_TRUSTED_CLIENT_IDS:-[]}" \
        bash "$DCR_SCRIPT" --reconcile-registration; then
        exit 1
    fi
    if ! harden_master_realm "$TOKEN"; then
        exit 1
    fi

    # Create realm and configure it step by step
    # Refresh token before each operation to prevent expiration
    if create_realm "$TOKEN"; then

        # Constrain registration before any client emits the reserved group claim.
        if ! KEYCLOAK_ADMIN_URL="$KEYCLOAK_URL" KEYCLOAK_ADMIN="$KEYCLOAK_ADMIN" KEYCLOAK_ADMIN_PASSWORD="$KEYCLOAK_ADMIN_PASSWORD" \
            ENABLE_DCR_CONFIG="${ENABLE_DCR_CONFIG:-false}" ENABLE_CIMD_CONFIG="${ENABLE_CIMD_CONFIG:-false}" \
            DCR_KEEP_CLIENTS="${DCR_KEEP_CLIENTS:-}" DCR_SWEEP_ACKNOWLEDGED="${DCR_SWEEP_ACKNOWLEDGED-true}" CIMD_SWEEP_ACKNOWLEDGED="${CIMD_SWEEP_ACKNOWLEDGED:-false}" \
            DCR_ALLOWED_SCOPES="${DCR_ALLOWED_SCOPES:-}" CIMD_TRUSTED_DOMAINS="${CIMD_TRUSTED_DOMAINS:-}" \
            REGISTRATION_TRUSTED_CLIENT_IDS="${REGISTRATION_TRUSTED_CLIENT_IDS:-[]}" \
            bash "$DCR_SCRIPT" --restrict-group-claims; then
            exit 1
        fi
        refresh_token
        create_clients "$TOKEN"

        refresh_token
        create_scopes "$TOKEN"

        refresh_token
        create_groups "$TOKEN"

        refresh_token
        create_users "$TOKEN"

        refresh_token
        create_service_account_user "$TOKEN"

        refresh_token
        create_service_account_clients "$TOKEN"

        refresh_token
        setup_client_secrets "$TOKEN"

        refresh_token
        setup_groups_mapper "$TOKEN"

        refresh_token
        setup_m2m_scopes "$TOKEN"

        # MCP Dynamic Client Registration (DCR) support: groups + audience
        # mappers on the 'basic' client-scope, an explicit Allowed Client
        # Scopes allowlist, a relaxed Trusted Hosts policy, no client-supplied
        # mappers and no device grant, so MCP clients (Claude Code, claude.ai
        # connectors, Cursor) can self-register safely. Delegates to the
        # canonical, idempotent script (single source of truth) rather than
        # duplicating it.
        refresh_token
        DCR_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../keycloak/setup" 2>/dev/null && pwd)/upgrade-realm-for-dcr.sh"
        if [ ! -f "$DCR_SCRIPT" ]; then
            echo -e "${RED}Error: DCR script not found at $DCR_SCRIPT. Run this script from a full repository checkout.${NC}"
            exit 1
        fi
        echo "Applying MCP Dynamic Client Registration (DCR) realm configuration..."
        if ! KEYCLOAK_ADMIN_URL="$KEYCLOAK_URL" KEYCLOAK_ADMIN="$KEYCLOAK_ADMIN" KEYCLOAK_ADMIN_PASSWORD="$KEYCLOAK_ADMIN_PASSWORD" \
        ENABLE_DCR_CONFIG="${ENABLE_DCR_CONFIG:-false}" ENABLE_CIMD_CONFIG="${ENABLE_CIMD_CONFIG:-false}" \
        DCR_KEEP_CLIENTS="${DCR_KEEP_CLIENTS:-}" DCR_SWEEP_ACKNOWLEDGED="${DCR_SWEEP_ACKNOWLEDGED-true}" CIMD_SWEEP_ACKNOWLEDGED="${CIMD_SWEEP_ACKNOWLEDGED:-false}" \
        DCR_ALLOWED_SCOPES="${DCR_ALLOWED_SCOPES:-}" CIMD_TRUSTED_DOMAINS="${CIMD_TRUSTED_DOMAINS:-}" \
        REGISTRATION_TRUSTED_CLIENT_IDS="${REGISTRATION_TRUSTED_CLIENT_IDS:-[]}" \
            bash "$DCR_SCRIPT"; then
            echo -e "${RED}Error: DCR realm configuration failed (see the [n/7] step above). Fix the cause and re-run this script; it is idempotent.${NC}"
            exit 1
        fi
    else
        exit 1
    fi

    echo ""
    echo -e "${GREEN}Keycloak initialization complete!${NC}"
    echo ""
    echo "You can now access Keycloak at: ${KEYCLOAK_URL}"
    echo "Admin console: ${KEYCLOAK_URL}/admin"
    echo "Realm: ${REALM}"
    echo ""
    echo "Users created (passwords are temporary; each user is prompted to reset on first login):"
    echo "  - admin (realm admin - all groups including mcp-registry-admin)"
    echo "  - lob1-user (LOB1 user - registry-users-lob1 group)"
    echo "  - lob2-user (LOB2 user - registry-users-lob2 group)"
    echo "  - service-account-mcp-gateway-m2m (service account for M2M access)"
    echo ""
    echo "Service Account Clients (M2M):"
    echo "  - registry-admin-bot (in registry-admins group)"
    echo "  - lob1-bot (in registry-users-lob1 group)"
    echo "  - lob2-bot (in registry-users-lob2 group)"
    echo ""
    echo "Groups created:"
    echo "  - mcp-registry-admin, mcp-registry-user, mcp-registry-developer"
    echo "  - mcp-registry-operator, mcp-servers-unrestricted, mcp-servers-restricted"
    echo "  - a2a-agent-admin, a2a-agent-publisher, a2a-agent-user"
    echo "  - registry-admins, registry-users-lob1, registry-users-lob2"
    echo ""
    echo "OAuth2 Clients:"
    echo "  - mcp-gateway-web (for UI authentication)"
    echo "  - mcp-gateway-m2m (for service-to-service authentication)"
    echo ""
    echo -e "${YELLOW}Remember to change the default passwords!${NC}"
}

# Run main function
main
