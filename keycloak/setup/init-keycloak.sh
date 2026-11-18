#!/bin/bash
# Initialize Keycloak with MCP Gateway configuration
# This script sets up the initial realm, clients, groups, and users

set -eo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/group-path-preflight.sh"

# These will be set properly after loading .env in main()
KEYCLOAK_URL=""  # Will be overridden with KEYCLOAK_ADMIN_URL after .env is loaded
REALM="mcp-gateway"

# Client scopes that carry gateway authorization rather than OIDC identity.
# The gateway resolves these names into server and tool access, so a
# self-registered (DCR) client must never be able to request one: it would be
# choosing its own privilege.
#
# This list is created in the realm and assigned to the M2M client, which is
# a trusted first-party client.
PRIVILEGED_CLIENT_SCOPES=(
    "mcp-servers-unrestricted/read"
    "mcp-servers-unrestricted/execute"
    "mcp-servers-restricted/read"
    "mcp-servers-restricted/execute"
)

# The complete set of client scopes a self-registered (DCR) client may hold.
#  Keycloak attaches the realm's default client scopes (basic, profile, email,
#  roles, web-origins, acr) to every new client and checks them against this
#  list too, so they have to be listed. `openid` is not a client scope
#  but MCP clients send it at registration. Override with DCR_ALLOWED_SCOPES (space-separated).
DCR_DEFAULT_ALLOWED_SCOPES="openid basic profile email roles web-origins acr offline_access"

# Refused even when an operator lists them in DCR_ALLOWED_SCOPES: the gateway
# privilege scopes created above, the two admin scope documents (the `_id` of
# a scope document granting `server: "*"` with every action; the Helm chart
# creates them as client scopes), and Keycloak's Admin API scope.
DCR_FORBIDDEN_SCOPES=(
    "${PRIVILEGED_CLIENT_SCOPES[@]}"
    "mcp-registry-admin"
    "registry-admins"
    "kc-admin-api"
)

# Page size for walking the realm's clients.
CLIENT_PAGE_SIZE=100

# Every user token carries the custom audience "mcp-gateway" (the audience
# mapper on the `basic` scope). A client whose clientId equals that audience is
# inside every token's aud, and on Keycloak 25.0 and 26.2.4 it can exchange any
# user token it holds for its own token and a refresh token. The setup holds
# the clientId with this disabled bearer-only placeholder, so no other client
# (for example one registered while DCR is on) can take it.
GATEWAY_AUDIENCE_CLIENT_ID="mcp-gateway"
GATEWAY_AUDIENCE_PLACEHOLDER_JSON='{
    "clientId": "mcp-gateway",
    "name": "Reserved: MCP Gateway token audience",
    "description": "Reserves the clientId equal to the mcp-gateway audience that every user token carries. Must stay disabled; the setup disables it on every run.",
    "protocol": "openid-connect",
    "enabled": false,
    "bearerOnly": true,
    "publicClient": false,
    "standardFlowEnabled": false,
    "implicitFlowEnabled": false,
    "directAccessGrantsEnabled": false,
    "serviceAccountsEnabled": false,
    "defaultClientScopes": [],
    "optionalClientScopes": [],
    "redirectUris": [],
    "webOrigins": []
}'

# True when a client representation already has the placeholder's shape.
GATEWAY_AUDIENCE_PLACEHOLDER_SHAPE_JQ='
    .enabled == false and .bearerOnly == true and .publicClient != true
    and .standardFlowEnabled != true and .implicitFlowEnabled != true
    and .directAccessGrantsEnabled != true and .serviceAccountsEnabled != true
    and ((.redirectUris // []) | length) == 0'

# Security baseline merged over a realm's current representation (the jq
# input): a stronger setting already in place is kept, a missing or weaker one
# is raised to the baseline. $gateway (true for the gateway realm) adds the
# password policy and the offline-session cap; the master realm gets the
# quick-login settings instead. adminEventsDetailsEnabled is always false so
# request bodies, which can carry secrets, are not stored in admin events.
# The same program is in upgrade-realm-for-dcr.sh, the Helm Job script and
# terraform/aws-ecs/scripts/init-keycloak.sh.
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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Function to get admin token. The password goes in on stdin, not argv, so it
# is not visible in `ps`.
# The private cache survives command substitutions; refresh before a 60-second
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
    printf '%s' "$KEYCLOAK_ADMIN_PASSWORD" | \
        realm_curl -s -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
            -H "Content-Type: application/x-www-form-urlencoded" \
            --data-urlencode "username=${KEYCLOAK_ADMIN}" \
            --data-urlencode "password@-" \
            --data-urlencode "grant_type=password" \
            --data-urlencode "client_id=admin-cli" | \
        jq -r '.access_token // empty' 2>/dev/null || true
}

# Get a fresh admin token into TOKEN. Master-realm admin tokens live 60
# seconds by default, so long steps start with a new one.
_refresh_admin_token() {
    TOKEN=$(get_admin_token)
    if [ -z "$TOKEN" ]; then
        echo -e "${RED}Error: Failed to re-authenticate with Keycloak${NC}"
        exit 1
    fi
}

# GET an Admin API path (starting with /admin/). Prints the body; returns 1
# unless the response is HTTP 200, so a caller never mistakes an error body or
# an empty response for "nothing found".
_admin_get() {
    local token=$1
    local path=$2

    local response
    if ! response=$(realm_curl -sS -w $'\n%{http_code}' \
        -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}${path}"); then
        echo -e "${RED}GET ${path} failed: could not reach Keycloak${NC}" >&2
        return 1
    fi

    local status="${response##*$'\n'}"
    if [ "$status" != "200" ]; then
        echo -e "${RED}GET ${path} returned HTTP ${status}${NC}" >&2
        return 1
    fi
    printf '%s' "${response%$'\n'*}"
}

# Print the internal id of the client with exactly this clientId, or nothing if
# there is none. Returns 1 if the lookup itself fails.
_find_client_uuid() {
    local token=$1
    local client_id=$2

    local body
    body=$(_admin_get "$token" "/admin/realms/${REALM}/clients?clientId=${client_id}") || return 1
    if ! echo "$body" | jq -er --arg c "$client_id" \
        'if type == "array" then ([.[] | select(.clientId == $c) | .id] | first // "") else error("not an array") end'; then
        echo -e "${RED}Unexpected response looking up client ${client_id}${NC}" >&2
        return 1
    fi
}

_rotate_registration_access_token() {
    local token=$1
    local client_uuid=$2
    local client_id=$3

    local status
    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${client_uuid}/registration-access-token" \
        -H "Authorization: Bearer ${token}")
    if [[ "$status" != 2* ]]; then
        echo -e "  ${RED}Failed to invalidate the registration access token of ${client_id} (HTTP ${status})${NC}" >&2
        return 1
    fi
    echo "  Registration access token of ${client_id} invalidated"
}

_upsert_groups_mapper() {
    local token=$1
    local models_path=$2
    local label=$3


    local mappers
    mappers=$(_admin_get "$token" "$models_path") || return 1
    _check_group_path_mapper_models "$mappers" || return 1
    if [[ "$models_path" == "/admin/realms/${REALM}/clients/"* ]]; then
        local client_uuid=${models_path#"/admin/realms/${REALM}/clients/"}
        _check_client_group_path_mappers "$KEYCLOAK_URL" "$REALM" "$token" "${client_uuid%%/*}" || return 1
    fi

    local existing
    if ! existing=$(echo "$mappers" | jq -ec \
        'if type != "array" then error("not an array")
         elif ([.[] | select(.name == "groups")] | length) > 1
             or any(.[]; .name == "groups" and
                 (.protocol != "openid-connect" or .protocolMapper != "oidc-group-membership-mapper"
                  or (.config["claim.name"] != "groups" and .config["claim.name"] != "mcpgw_group_paths")))
             or any(.[]; .config["claim.name"] == "mcpgw_group_paths" and .name != "groups")
         then error("conflicting groups mapper")
         else ([.[] | select(.name == "groups")] | first // {}) end'); then
        echo -e "${RED}Unexpected response listing protocol mappers of ${label}${NC}" >&2
        return 1
    fi

    if [ -n "$existing" ] && jq -e --argjson desired "$GROUPS_MAPPER_JSON" '
        .protocol == $desired.protocol and .protocolMapper == $desired.protocolMapper
        and (.config as $config | all($desired.config | to_entries[]; $config[.key] == .value))' >/dev/null <<< "$existing"; then
        return 0
    fi
    _check_group_paths "$KEYCLOAK_URL" "$REALM" "$token" || return 1

    local status
    if [ "$existing" = "{}" ]; then
        status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}${models_path}" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$GROUPS_MAPPER_JSON")
        if [ "$status" = "201" ]; then
            echo -e "${GREEN}Groups mapper created for ${label}${NC}"
            return 0
        fi
        echo -e "${RED}Failed to create groups mapper for ${label}. HTTP status: ${status}${NC}" >&2
        return 1
    fi

    local mapper_id
    mapper_id=$(echo "$existing" | jq -r '.id')
    local updated
    updated=$(echo "$existing" | jq -c --argjson want "$GROUPS_MAPPER_JSON" \
        '. + {protocol: $want.protocol, protocolMapper: $want.protocolMapper, config: ((.config // {}) + $want.config)}')
    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}${models_path}/${mapper_id}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$updated")
    if [ "$status" = "204" ]; then
        echo -e "${GREEN}Groups mapper updated for ${label} (full.path=true)${NC}"
        return 0
    fi
    echo -e "${RED}Failed to update groups mapper for ${label}. HTTP status: ${status}${NC}" >&2
    return 1
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
    current=$(_admin_get "$token" "/admin/realms/${realm}") || return 1
    if ! jq -ec --argjson gateway "$gateway" --argjson adminExpiry "$admin_expiry" "$REALM_SECURITY_BASELINE_JQ" <<< "$current"; then
        echo -e "${RED}Could not read the current settings of realm ${realm}${NC}" >&2
        return 1
    fi
}

harden_master_realm() {
    local token=$1

    echo "Hardening the master realm (brute-force protection, event logging)..."

    local master_json
    master_json=$(_realm_security_settings "$token" "master" false) || return 1

    local response
    response=$(realm_curl -sS -w $'\n%{http_code}' \
        -X PUT "${KEYCLOAK_URL}/admin/realms/master" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$master_json")
    local status="${response##*$'\n'}"
    if [[ "$status" != 2* ]]; then
        echo -e "${RED}Failed to harden the master realm (HTTP ${status})${NC}" >&2
        echo "  response: ${response%$'\n'*}" >&2
        return 1
    fi
    echo -e "${GREEN}Master realm hardened (stronger existing settings kept).${NC}"
}

# PUT the bootstrap's realm settings ($2) onto the existing realm, with the
# security fields merged over the realm's current values so a stronger setting
# the operator chose is kept. Partial PUT: other fields keep their values.
_update_realm_settings() {
    local token=$1
    local realm_json=$2
    local settings body response status

    settings=$(_realm_security_settings "$token" "$REALM" true) || return 1
    body=$(jq -c --argjson s "$settings" '. + $s' <<< "$realm_json")
    response=$(realm_curl -sS -w $'\n%{http_code}' \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$body")
    status="${response##*$'\n'}"
    if [ "$status" = "204" ]; then
        echo -e "${GREEN}Realm already exists; settings updated (stronger existing settings kept).${NC}"
        return 0
    fi
    # Print the body: a 4xx on a realm representation names the rejected field,
    # and a bare status is not actionable.
    echo -e "${RED}Failed to update existing realm (HTTP ${status})${NC}" >&2
    echo "  response: ${response%$'\n'*}" >&2
    return 1
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

# Create the client if absent, otherwise PUT the full representation so attribute
# changes (webOrigins, post.logout.redirect.uris) reach deployments whose realm
# already exists. A bare POST silently 409s on a re-run, which would make every
# hardening change here a no-op for existing installs. Mirrors upsert_client in
# charts/keycloak-configure/templates/configmap.yaml and the 409->PUT path in
# terraform/aws-ecs/scripts/init-keycloak.sh.
upsert_client() {
    local token=$1
    local client_id=$2
    local client_json=$3

    local existing_id
    existing_id=$(_find_client_uuid "$token" "$client_id") || return 1

    local status
    if [ -z "$existing_id" ]; then
        local create_response
        create_response=$(realm_curl -sS -w $'\n%{http_code}' \
            -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$client_json")
        status="${create_response##*$'\n'}"
        if [ "$status" = "201" ]; then
            echo -e "  ${GREEN}Client ${client_id} created${NC}"
        else
            echo -e "  ${RED}Failed to create client ${client_id} (HTTP ${status})${NC}" >&2
            echo "  response: ${create_response%$'\n'*}" >&2
            return 1
        fi
    else
        local update_response
        update_response=$(realm_curl -sS -w $'\n%{http_code}' \
            -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${existing_id}" \
            -H "Authorization: Bearer ${token}" \
            -H "Content-Type: application/json" \
            -d "$client_json")
        status="${update_response##*$'\n'}"
        if [ "$status" = "204" ]; then
            echo -e "  ${GREEN}Client ${client_id} updated${NC}"
        else
            echo -e "  ${RED}Failed to update client ${client_id} (HTTP ${status})${NC}" >&2
            echo "  response: ${update_response%$'\n'*}" >&2
            return 1
        fi
        _rotate_registration_access_token "$token" "$existing_id" "$client_id" || return 1
    fi
}

# Function to create clients
create_clients() {
    local token=$1

    echo "Creating OAuth2 clients..."

    # Create web client
    local web_client_json='{
        "clientId": "mcp-gateway-web",
        "name": "MCP Gateway Web Client",
        "enabled": true,
        "clientAuthenticatorType": "client-secret",
        "redirectUris": [
            "'${AUTH_SERVER_EXTERNAL_URL:-http://localhost:8888}'/oauth2/callback/keycloak",
            "'${REGISTRY_URL:-http://localhost:7860}'/*",
            "http://localhost:7860/*",
            "http://localhost:8888/*"
        ],
        "webOrigins": [
            "'${REGISTRY_URL:-http://localhost:7860}'",
            "http://localhost:7860",
            "http://localhost:8888"
        ],
        "attributes": {
            "post.logout.redirect.uris": "'${REGISTRY_URL:-http://localhost:7860}'/*##http://localhost:7860/*##http://localhost:8888/*"
        },
        "protocol": "openid-connect",
        "standardFlowEnabled": true,
        "implicitFlowEnabled": false,
        "directAccessGrantsEnabled": false,
        "serviceAccountsEnabled": false,
        "fullScopeAllowed": false,
        "publicClient": false
    }'

    upsert_client "$token" "mcp-gateway-web" "$web_client_json" || return 1

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

    upsert_client "$token" "mcp-gateway-m2m" "$m2m_client_json" || return 1

    echo -e "${GREEN}Clients created successfully!${NC}"
}

# Function to create groups
create_groups() {
    local token=$1

    echo "Creating user groups..."

    local groups=("mcp-registry-admin" "mcp-registry-user" "mcp-registry-developer" "mcp-registry-operator" "mcp-servers-unrestricted" "mcp-servers-restricted" "a2a-agent-admin" "a2a-agent-publisher" "a2a-agent-user")

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
    local token=$1

    echo "Creating custom MCP scopes..."

    local scopes=("${PRIVILEGED_CLIENT_SCOPES[@]}")

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
    local token=$1

    echo "Setting up M2M client scopes..."

    # Get M2M client ID
    local m2m_client_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/clients?clientId=mcp-gateway-m2m" | \
        jq -r '.[0].id')

    if [ -z "$m2m_client_id" ] || [ "$m2m_client_id" = "null" ]; then
        echo -e "${RED}Error: Could not find mcp-gateway-m2m client${NC}"
        return 1
    fi

    # The M2M client is a trusted first-party client, so unlike a DCR client it
    # does get the privileged scopes.
    local scopes=("${PRIVILEGED_CLIENT_SCOPES[@]}")

    for scope in "${scopes[@]}"; do
        # Get scope ID
        local scope_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes" | \
            jq -r '.[] | select(.name=="'$scope'") | .id')

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
    local token=$1
    local service_account_username="service-account-mcp-gateway-m2m"

    echo "Creating service account user: $service_account_username"

    # Check if user already exists
    local existing_user=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=$service_account_username" | \
        jq -r '.[0].id // empty')

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
            "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=$service_account_username" | \
            jq -r '.[0].id')

        echo "Created service account user with ID: $user_id"

        # Assign user to mcp-servers-unrestricted group
        local group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" | \
            jq -r '.[] | select(.name=="mcp-servers-unrestricted") | .id')

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
            "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" | \
            jq -r '.[] | select(.name=="a2a-agent-admin") | .id')

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

# Function to create the realm admin user
create_users() {
    local token=$1

    # SA-9: never create the admin user with a weak default password. Require an
    # explicit INITIAL_ADMIN_PASSWORD and force a reset on first login.
    if [ -z "$INITIAL_ADMIN_PASSWORD" ]; then
        echo -e "${RED}Error: INITIAL_ADMIN_PASSWORD environment variable is required${NC}"
        echo "This password is used for the realm 'admin' user in the ${REALM} realm."
        echo "Set it in .env or export it before running this script:"
        echo "  export INITIAL_ADMIN_PASSWORD='YourSecurePassword'"
        exit 1
    fi

    echo "Creating realm admin user..."

    # Define usernames for consistency
    local admin_username="admin"

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

    realm_curl -s -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/users" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$admin_user_json" > /dev/null

    echo "Assigning users to groups..."

    # Get user IDs
    local admin_user_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/users?username=$admin_username" | \
        jq -r '.[0].id')

    # Get the group IDs the admin user needs
    local admin_group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" | \
        jq -r '.[] | select(.name=="mcp-registry-admin") | .id')

    local unrestricted_group_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/groups" | \
        jq -r '.[] | select(.name=="mcp-servers-unrestricted") | .id')

    # Define usernames for consistent logging
    local admin_username="admin"

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

    echo -e "${GREEN}Admin user created and assigned to groups successfully!${NC}"
}

# Function to create client secrets
setup_client_secrets() {
    local token=$1

    echo "Setting up client secrets..."

    # Get web client ID
    local web_client_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/clients?clientId=mcp-gateway-web" | \
        jq -r '.[0].id')

    # Generate secret for web client
    realm_curl -s -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${web_client_id}/client-secret" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" > /dev/null

    local web_secret_response=$(realm_curl -s "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${web_client_id}/client-secret" \
        -H "Authorization: Bearer ${token}")
    web_secret=$(echo "$web_secret_response" | jq -r '.value // empty')

    # Get M2M client ID
    local m2m_client_id=$(realm_curl -s -H "Authorization: Bearer ${token}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/clients?clientId=mcp-gateway-m2m" | \
        jq -r '.[0].id')

    # Generate secret for M2M client
    realm_curl -s -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${m2m_client_id}/client-secret" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" > /dev/null

    local m2m_secret_response=$(realm_curl -s "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${m2m_client_id}/client-secret" \
        -H "Authorization: Bearer ${token}")
    m2m_secret=$(echo "$m2m_secret_response" | jq -r '.value // empty')

    echo -e "${GREEN}Client secrets generated!${NC}"
    echo ""
    echo "=============================================="
    echo -e "${YELLOW}Client credentials have been created.${NC}"
    echo "=============================================="
    echo ""
    echo -e "${GREEN}To retrieve all client credentials, run:${NC}"
    echo "  ./keycloak/setup/get-all-client-credentials.sh"
    echo ""
    echo "This will save all credentials to .oauth-tokens/"
    echo "=============================================="
}

# Create or update the mcpgw_group_paths mapper on the web and M2M clients.
setup_groups_mapper() {
    local token=$1

    echo "Setting up groups mapper for OAuth2 clients..."

    local client_id client_uuid
    for client_id in "mcp-gateway-web" "mcp-gateway-m2m"; do
        echo "Setting up groups mapper for ${client_id} client..."
        client_uuid=$(_find_client_uuid "$token" "$client_id") || return 1
        if [ -z "$client_uuid" ]; then
            echo -e "${RED}Error: Could not find ${client_id} client${NC}" >&2
            return 1
        fi
        _upsert_groups_mapper "$token" \
            "/admin/realms/${REALM}/clients/${client_uuid}/protocol-mappers/models" \
            "$client_id" || return 1
    done
}

# Print every client of the realm as one JSON array, read page by page.
# Returns 1 if any page cannot be read. Pages are merged on stdin, not argv,
# so a large page cannot overflow the argument size limit.
_list_all_clients() {
    local token=$1
    local clients="[]" page first=0

    while :; do
        page=$(_admin_get "$token" "/admin/realms/${REALM}/clients?first=${first}&max=${CLIENT_PAGE_SIZE}") || return 1
        if ! jq -e 'type == "array"' >/dev/null 2>&1 <<< "$page"; then
            echo -e "${RED}Unexpected response listing the clients of realm '${REALM}'${NC}" >&2
            return 1
        fi
        clients=$(printf '%s\n%s\n' "$clients" "$page" | jq -sc 'add')
        if [ "$(jq 'length' <<< "$page")" -lt "$CLIENT_PAGE_SIZE" ]; then
            break
        fi
        first=$((first + CLIENT_PAGE_SIZE))
    done
    printf '%s\n' "$clients"
}

# PUT an existing groups mapper back with full paths in mcpgw_group_paths,
# keeping the rest of its config. $2 is its protocol-mappers/models
# collection path, $3 the mapper JSON, $4 a label for the owner.
_ensure_groups_mapper_full_path() {
    if ! jq -e '.protocol == "openid-connect" and .protocolMapper == "oidc-group-membership-mapper"
        and (.config["claim.name"] == "groups" or .config["claim.name"] == "mcpgw_group_paths")' >/dev/null <<< "$3"; then
        echo "Refusing to repurpose a non-gateway groups mapper." >&2
        return 1
    fi
    _check_group_path_mapper_collection "$1" "${KEYCLOAK_URL}${2%/protocol-mappers/models}" || return 1
    if [[ "$2" == "/admin/realms/${REALM}/clients/"* ]]; then
        local client_uuid=${2#"/admin/realms/${REALM}/clients/"}
        _check_client_group_path_mappers "$KEYCLOAK_URL" "$REALM" "$1" "${client_uuid%%/*}" || return 1
    fi
    if jq -e '.config["full.path"] == "true" and .config["claim.name"] == "mcpgw_group_paths"' >/dev/null <<< "$3"; then
        return 0
    fi
    _check_group_paths "$KEYCLOAK_URL" "$REALM" "$1" || return 1
    _put_groups_mapper_full_path "$@"
}

_put_groups_mapper_full_path() {
    local token=$1
    local models_path=$2
    local mapper_json=$3

    local owner=$4
    local mapper_id mapper_name

    mapper_id=$(echo "$mapper_json" | jq -r '.id // empty')
    mapper_name=$(echo "$mapper_json" | jq -r '(.name // "") | gsub("[^A-Za-z0-9._:/@ -]"; "?")')
    if [ -z "$mapper_id" ]; then
        echo -e "${RED}Groups mapper on ${owner} has no id${NC}" >&2
        return 1
    fi
    if jq -e '.config["full.path"] == "true" and .config["claim.name"] == "mcpgw_group_paths"' >/dev/null <<< "$mapper_json"; then
        return 0
    fi

    local updated status
    updated=$(echo "$mapper_json" | jq -c '.config = ((.config // {}) + {"full.path": "true", "claim.name": "mcpgw_group_paths"})')
    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}${models_path}/${mapper_id}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$updated")
    if [[ "$status" != 2* ]]; then
        echo -e "${RED}Failed to switch groups mapper ${mapper_name} on ${owner} to full.path=true (HTTP ${status})${NC}" >&2
        return 1
    fi
    echo -e "${GREEN}${owner}: groups mapper ${mapper_name} now uses full.path=true${NC}"
}

# Migrate client and client-scope group membership mappers emitting groups or
# mcpgw_group_paths to the dedicated full-path claim. Leave unrelated group
# claims alone and reject foreign emitters of the reserved claim.
update_client_groups_mappers() {
    local token=$1 clients entries entry client_uuid client_id mapper collection
    local scopes scope scope_id scope_name models scope_entries
    _check_realm_group_path_mappers "$KEYCLOAK_URL" "$REALM" "$token" || return 1
    clients=$(_list_all_clients "$token") || return 1
    entries=$(jq -c '.[]
        | if any(.protocolMappers[]?; .config["claim.name"] == "mcpgw_group_paths" and
            (.protocol != "openid-connect" or .protocolMapper != "oidc-group-membership-mapper"))
          then error("conflicting mcpgw_group_paths mapper") else . end
        | {id, clientId, collection: "clients", mapper: (.protocolMappers[]?
            | select(.protocol == "openid-connect" and .protocolMapper == "oidc-group-membership-mapper"
                and (.config["claim.name"] == "groups" or .config["claim.name"] == "mcpgw_group_paths")
                and (.config["full.path"] != "true" or .config["claim.name"] != "mcpgw_group_paths")))}' <<< "$clients") || return 1
    scopes=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes") || return 1
    scopes=$(jq -c '.[] | {id, name}' <<< "$scopes") || return 1
    while IFS= read -r scope; do
        [ -n "$scope" ] || continue
        scope_id=$(jq -er '.id' <<< "$scope") || return 1
        scope_name=$(jq -r '.name' <<< "$scope") || return 1
        models=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes/${scope_id}/protocol-mappers/models") || return 1
        _check_group_path_mapper_models "$models" || return 1
        scope_entries=$(jq -c --arg id "$scope_id" --arg name "$scope_name" '
            .[] | select(.protocol == "openid-connect" and .protocolMapper == "oidc-group-membership-mapper"
                and (.config["claim.name"] == "groups" or .config["claim.name"] == "mcpgw_group_paths")
                and (.config["full.path"] != "true" or .config["claim.name"] != "mcpgw_group_paths"))
            | {id: $id, clientId: $name, collection: "client-scopes", mapper: .}' <<< "$models") || return 1
        if [ -n "$scope_entries" ]; then
            entries="${entries}${entries:+$'\n'}${scope_entries}"
        fi
    done <<< "$scopes"
    [ -n "$entries" ] || return 0
    _check_group_paths "$KEYCLOAK_URL" "$REALM" "$token" || return 1
    while IFS= read -r entry; do
        client_uuid=$(jq -er '.id' <<< "$entry") || return 1
        client_id=$(jq -r '.clientId | gsub("[^A-Za-z0-9._:/@-]"; "?")' <<< "$entry") || return 1
        mapper=$(jq -c '.mapper' <<< "$entry") || return 1
        collection=$(jq -r '.collection' <<< "$entry") || return 1
        _put_groups_mapper_full_path "$token" \
            "/admin/realms/${REALM}/${collection}/${client_uuid}/protocol-mappers/models" \
            "$mapper" "${collection} ${client_id}" || return 1
    done <<< "$entries"
}

# Create the disabled placeholder that holds the clientId GATEWAY_AUDIENCE_CLIENT_ID,
# or put an existing client of that clientId back into the placeholder shape
# (disabled) and invalidate its registration access token. Fails closed on
# any read or write error.
reserve_gateway_audience_client() {
    local token=$1
    local clients existing uuid status display

    echo "Reserving clientId '${GATEWAY_AUDIENCE_CLIENT_ID}' (the audience every user token carries)..."
    clients=$(_admin_get "$token" "/admin/realms/${REALM}/clients?clientId=${GATEWAY_AUDIENCE_CLIENT_ID}") || return 1
    if ! existing=$(jq -ec --arg c "$GATEWAY_AUDIENCE_CLIENT_ID" \
        'if type == "array" then ([.[] | select(.clientId == $c)] | first // {}) else error("not an array") end' <<< "$clients"); then
        echo -e "${RED}Unexpected response looking up client ${GATEWAY_AUDIENCE_CLIENT_ID}${NC}" >&2
        return 1
    fi

    if [ "$existing" = "{}" ]; then
        status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients" \
            -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
            -d "$GATEWAY_AUDIENCE_PLACEHOLDER_JSON")
        if [ "$status" != "201" ]; then
            echo -e "${RED}Failed to create the disabled placeholder client ${GATEWAY_AUDIENCE_CLIENT_ID} (HTTP ${status})${NC}" >&2
            return 1
        fi
        echo -e "${GREEN}Created the disabled placeholder client ${GATEWAY_AUDIENCE_CLIENT_ID}${NC}"
        return 0
    fi

    uuid=$(jq -r '.id' <<< "$existing")
    display="${GATEWAY_AUDIENCE_CLIENT_ID} (id ${uuid})"
    if ! jq -e "$GATEWAY_AUDIENCE_PLACEHOLDER_SHAPE_JQ" >/dev/null <<< "$existing"; then
        echo -e "${YELLOW}Warning: client ${display} was enabled or not in the placeholder shape; disabling it. A client with this clientId is inside the audience of every user token.${NC}"
    fi
    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${uuid}" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d "$GATEWAY_AUDIENCE_PLACEHOLDER_JSON")
    if [ "$status" != "204" ]; then
        echo -e "${RED}Failed to disable client ${display} (HTTP ${status})${NC}" >&2
        return 1
    fi
    _rotate_registration_access_token "$token" "$uuid" "$GATEWAY_AUDIENCE_CLIENT_ID" || return 1
    existing=$(_admin_get "$token" "/admin/realms/${REALM}/clients/${uuid}") || return 1
    if ! jq -e "$GATEWAY_AUDIENCE_PLACEHOLDER_SHAPE_JQ" >/dev/null <<< "$existing"; then
        echo -e "${RED}Client ${display} is still not a disabled placeholder after the update${NC}" >&2
        return 1
    fi
    echo -e "${GREEN}Client ${GATEWAY_AUDIENCE_CLIENT_ID} is a disabled placeholder${NC}"
}

# =============================================================================
# Dynamic Client Registration (RFC 7591) support
# =============================================================================
# MCP clients (Claude Code, Claude.ai connectors, Cursor, etc.) register
# themselves anonymously and receive tokens that carry the user's groups for
# per-user authorization at the gateway.
#
# 1. setup_dcr_groups_mapper() - attach the Groups protocol mapper to the
#    `basic` client-scope (or migrate its groups claim to mcpgw_group_paths).
#    `basic` is auto-attached to every client (including DCR'd ones), so this
#    is the only reliable place to put a mapper that runs for clients we don't
#    pre-create.
#
# 2. configure_dcr_allowed_scopes() - set the anonymous + authenticated DCR
#    "Allowed Client Scopes" policies to exactly the allowlist above.
#
# 3. configure_dcr_trusted_hosts() - relax the anonymous DCR "Trusted Hosts"
#    policy so MCP clients running on diverse cloud egress IPs (Claude Code's
#    cloud listener, etc.) can register. The client-URI check stays on with
#    `localhost` allowed for the OAuth callback redirect URI.
#
# 4. configure_dcr_allowed_mapper_types() - no client-supplied protocol
#    mappers (anonymous + authenticated). The gateway's own groups/audience
#    mappers live on the server-side `basic` client-scope, so this does not
#    affect legitimate MCP.
#
# 5. configure_dcr_reject_device_grant() - refuse the device authorization
#    grant realm-wide.
#
# 6. verify_dcr_baseline_policies() - fail unless Keycloak's own anonymous
#    policies that these steps rely on (Consent Required, Full Scope
#    Disabled, Max Clients Limit, Allowed Protocol Mapper Types) are present.

setup_dcr_groups_mapper() {
    local token=$1

    echo "Attaching Groups protocol mapper to the 'basic' client-scope..."

    # Find the 'basic' client-scope id
    local scopes
    scopes=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes") || return 1
    local basic_scope_id
    if ! basic_scope_id=$(echo "$scopes" | jq -er \
        'if type == "array" then ([.[] | select(.name == "basic") | .id] | first // "") else error("not an array") end'); then
        echo -e "${RED}Unexpected response listing client scopes${NC}" >&2
        return 1
    fi

    if [ -z "$basic_scope_id" ]; then
        echo -e "${RED}Error: Could not find 'basic' client-scope${NC}" >&2
        return 1
    fi

    _upsert_groups_mapper "$token" \
        "/admin/realms/${REALM}/client-scopes/${basic_scope_id}/protocol-mappers/models" \
        "the 'basic' client scope" || return 1
}

setup_dcr_audience_mapper() {
    local token=$1

    echo "Attaching Audience protocol mapper to the 'basic' client-scope..."

    # By default, Keycloak does NOT include an `aud` claim on access tokens
    # minted for DCR'd clients. Without an `aud` claim, RFC 8707 audience
    # binding can't be enforced and the gateway's validator has to relax to
    # an issuer-only check. Attaching this mapper to the `basic` scope (the
    # only scope reliably attached to every DCR'd client) guarantees every
    # token issued by this realm carries `aud="mcp-gateway"` so the gateway
    # can do strict audience validation.

    local scopes basic_scope_id
    scopes=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes") || return 1
    basic_scope_id=$(jq -er '[.[] | select(.name == "basic") | .id] | first // empty' <<< "$scopes") || return 1

    if [ -z "$basic_scope_id" ] || [ "$basic_scope_id" = "null" ]; then
        echo -e "${RED}Error: Could not find 'basic' client-scope${NC}"
        return 1
    fi

    local scope existing
    scope=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes/${basic_scope_id}") || return 1
    existing=$(jq -r '[.protocolMappers[]? | select(.name == "mcp-gateway-audience") | .id] | first // empty' <<< "$scope") || return 1

    if [ -n "$existing" ] && [ "$existing" != "null" ]; then
        echo -e "${YELLOW}Audience mapper already attached to 'basic' scope. Skipping.${NC}"
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

    local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/client-scopes/${basic_scope_id}/protocol-mappers/models" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$audience_mapper_json")

    if [ "$response" = "201" ]; then
        echo -e "${GREEN}Audience mapper attached to 'basic' scope. DCR'd-client tokens will carry aud=\"mcp-gateway\".${NC}"
    else
        echo -e "${RED}Failed to attach Audience mapper to 'basic'. HTTP status: ${response}${NC}"
        return 1
    fi
}

# Set one subType's "Allowed Client Scopes" policy to exactly the given list.
_set_allowed_scopes_policy() {
    local token=$1
    local sub_type=$2
    local components=$3
    local allowed_json=$4

    local policy_id=$(echo "$components" | \
        jq -r ".[] | select(.name==\"Allowed Client Scopes\" and .subType==\"${sub_type}\") | .id")
    local parent_id=$(echo "$components" | \
        jq -r ".[] | select(.name==\"Allowed Client Scopes\" and .subType==\"${sub_type}\") | .parentId")

    if [ -z "$policy_id" ] || [ "$policy_id" = "null" ]; then
        echo -e "${RED}${sub_type} 'Allowed Client Scopes' policy not found; refusing to leave DCR scopes unrestricted.${NC}"
        return 1
    fi

    local response=$(_put_allowed_scopes_policy "$token" "$policy_id" "$parent_id" "$sub_type" "$allowed_json")

    if [ "$response" = "400" ] && echo "$allowed_json" | jq -e 'index("openid")' >/dev/null; then
        echo -e "${YELLOW}${sub_type}: Keycloak refused 'openid' in the allowlist; retrying without it.${NC}"
        response=$(_put_allowed_scopes_policy "$token" "$policy_id" "$parent_id" "$sub_type" \
            "$(echo "$allowed_json" | jq -c 'map(select(. != "openid"))')")
    fi

    if [ "$response" = "204" ]; then
        echo -e "${GREEN}${sub_type} Allowed Client Scopes policy set to the explicit allowlist.${NC}"
    else
        echo -e "${RED}Failed to update ${sub_type} Allowed Client Scopes policy. HTTP status: ${response}${NC}"
        return 1
    fi
}

# PUT the Allowed Client Scopes component; prints the HTTP status.
_put_allowed_scopes_policy() {
    local token=$1
    local policy_id=$2
    local parent_id=$3
    local sub_type=$4
    local allowed_json=$5

    local policy_json=$(jq -n \
        --arg name "Allowed Client Scopes" \
        --arg providerId "allowed-client-templates" \
        --arg providerType "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy" \
        --arg parentId "$parent_id" \
        --arg subType "$sub_type" \
        --argjson allowedScopes "$allowed_json" \
        '{
            name: $name,
            providerId: $providerId,
            providerType: $providerType,
            parentId: $parentId,
            subType: $subType,
            config: {
                "allow-default-scopes": ["false"],
                "allowed-client-scopes": $allowedScopes
            }
        }')

    realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/components/${policy_id}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$policy_json"
}

configure_dcr_allowed_scopes() {
    local token=$1

    echo ""
    echo "[3/7] Setting 'Allowed Client Scopes' DCR policy (anonymous + authenticated)..."

    # Read here, not at the top of the script, so a value set in .env counts.
    local -a DCR_ALLOWED_SCOPE_LIST
    read -r -a DCR_ALLOWED_SCOPE_LIST <<< "${DCR_ALLOWED_SCOPES:-$DCR_DEFAULT_ALLOWED_SCOPES}"
    local scope forbidden
    for scope in "${DCR_ALLOWED_SCOPE_LIST[@]}"; do
        if [[ "$scope" == */* ]]; then
            echo -e "${RED}FAILED: DCR_ALLOWED_SCOPES contains '${scope}'; scope names with '/' are gateway privilege scopes and are never allowed.${NC}"
            return 1
        fi
        for forbidden in "${DCR_FORBIDDEN_SCOPES[@]}"; do
            if [ "$scope" = "$forbidden" ]; then
                echo -e "${RED}FAILED: DCR_ALLOWED_SCOPES contains '${scope}', which a self-registered client must never hold.${NC}"
                return 1
            fi
        done
    done

    local allowed_json=$(printf '%s\n' "${DCR_ALLOWED_SCOPE_LIST[@]}" | jq -R . | jq -sc 'unique')
    echo "  DCR clients may hold only: $(echo "$allowed_json" | jq -r 'join(", ")')"

    local components
    components=$(_admin_get "$token" \
        "/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy") || return 1

    _set_allowed_scopes_policy "$token" "anonymous" "$components" "$allowed_json" || return 1
    _set_allowed_scopes_policy "$token" "authenticated" "$components" "$allowed_json" || return 1
}

# Native DCR accepts full ClientRepresentation; reject self-assigned default
# roles for anonymous registrations AND authenticated registration-token updates.
configure_dcr_default_roles() {
    local token=$1 info components realm_json realm_id subtype ids id body status
    local type="org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy"
    info=$(_admin_get "$token" "/admin/serverinfo") || return 1
    if ! jq -e --arg type "$type" 'any(.componentTypes[$type][]?; .id == "default-roles")' >/dev/null <<< "$info"; then
        if [ "${2:-true}" = false ]; then
            return 0
        fi
        echo "DCR requires Keycloak 26.8.2 or newer with the Default Roles Disabled registration policy. Keep ENABLE_DCR_CONFIG=false until the server is upgraded." >&2
        return 1
    fi
    realm_json=$(_admin_get "$token" "/admin/realms/${REALM}") || return 1
    realm_id=$(jq -er '.id | select(type == "string" and length > 0)' <<< "$realm_json") || return 1
    components=$(_admin_get "$token" "/admin/realms/${REALM}/components?type=$type") || return 1
    jq -e 'type == "array"' >/dev/null <<< "$components" || return 1
    for subtype in anonymous authenticated; do
        ids=$(jq -r --arg subtype "$subtype" '.[] | select(.providerId == "default-roles" and .subType == $subtype) | .id' <<< "$components") || return 1
        if [ -z "$ids" ]; then
            body=$(jq -nc --arg parent "$realm_id" --arg subtype "$subtype" --arg type "$type" '{
                name: "Default Roles Disabled", providerId: "default-roles", providerType: $type,
                parentId: $parent, subType: $subtype, config: {}}') || return 1
            status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X POST \
                "$KEYCLOAK_URL/admin/realms/$REALM/components" \
                -H "Authorization: Bearer $token" -H 'Content-Type: application/json' -d "$body") || return 1
            if [ "$status" != 201 ]; then
                echo "Could not install Default Roles Disabled ($subtype): HTTP $status" >&2
                return 1
            fi
        fi
    done
    components=$(_admin_get "$token" "/admin/realms/${REALM}/components?type=$type") || return 1
    for subtype in anonymous authenticated; do
        if ! jq -e --arg subtype "$subtype" --arg parent "$realm_id" --arg type "$type" \
            'type == "array" and any(.[]; .providerId == "default-roles" and .providerType == $type
                and .subType == $subtype and .parentId == $parent)' >/dev/null <<< "$components"; then
            echo "Default Roles Disabled verification failed ($subtype); DCR was not enabled." >&2
            return 1
        fi
    done
}

configure_dcr_trusted_hosts() {
    local token=$1
    configure_registration_security_policy "$token" || return 1
    reconcile_registered_client_scopes "$token" || return 1
    configure_dcr_default_roles "$token" || return 1

    echo ""
    echo "[4/7] Relaxing anonymous-DCR 'Trusted Hosts' policy..."

    local components
    components=$(_admin_get "$token" \
        "/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy") || return 1

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

    local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
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

# Set one subType's "Allowed Protocol Mapper Types" policy to the empty list
# (no client-supplied protocol mappers). A realm without the policy gets one:
# without it a registered client may define any mapper, including one that
# writes the reserved `mcpgw_group_paths` claim.
_set_allowed_mapper_types_policy() {
    local token=$1
    local sub_type=$2  # "anonymous" or "authenticated"
    local components=$3
    local realm_id=$4

    local policy_id
    policy_id=$(echo "$components" | jq -r --arg st "$sub_type" \
        '[.[] | select(.providerId=="allowed-protocol-mappers" and .subType==$st) | .id] | first // empty')

    local policy_json
    policy_json=$(jq -n --arg parentId "$realm_id" --arg subType "$sub_type" '{
        name: "Allowed Protocol Mapper Types",
        providerId: "allowed-protocol-mappers",
        providerType: "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy",
        parentId: $parentId,
        subType: $subType,
        config: {"allowed-protocol-mapper-types": []}
    }')

    local response action
    if [ -n "$policy_id" ]; then
        action="narrowed down"
        response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/components/${policy_id}" \
            -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
            -d "$(echo "$policy_json" | jq --arg id "$policy_id" '. + {id: $id}')")
    else
        action="created (the realm had none)"
        response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/components" \
            -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
            -d "$policy_json")
    fi

    if [ "$response" = "204" ] || [ "$response" = "201" ]; then
        echo -e "${GREEN}Allowed Protocol Mapper Types (${sub_type}) ${action}: no client-supplied mappers permitted.${NC}"
    else
        echo -e "${RED}Failed to set Allowed Protocol Mapper Types (${sub_type}). HTTP status: ${response}${NC}"
        return 1
    fi
}

configure_dcr_allowed_mapper_types() {
    local token=$1

    echo "Setting 'Allowed Protocol Mapper Types' DCR policy (anonymous + authenticated)..."

    local components realm_body realm_id
    components=$(_admin_get "$token" "/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy") || return 1
    if ! echo "$components" | jq -e 'type == "array"' >/dev/null 2>&1; then
        echo -e "${RED}Unexpected client registration policy list for realm '${REALM}'.${NC}"
        return 1
    fi
    realm_body=$(_admin_get "$token" "/admin/realms/${REALM}") || return 1
    realm_id=$(echo "$realm_body" | jq -r '.id // empty')
    if [ -z "$realm_id" ]; then
        echo -e "${RED}Could not read the id of realm '${REALM}'.${NC}"
        return 1
    fi

    local sub_type
    for sub_type in anonymous authenticated; do
        _set_allowed_mapper_types_policy "$token" "$sub_type" "$components" "$realm_id" || return 1
    done
}

# Refuse the device authorization grant for every client in the realm.
configure_dcr_reject_device_grant() {
    local token=$1

    echo "Rejecting the device authorization grant realm-wide..."

    local serverinfo
    serverinfo=$(_admin_get "$token" "/admin/serverinfo") || return 1
    local device_flow=$(echo "$serverinfo" | \
        jq -r '[.features[]? | select(.name=="DEVICE_FLOW") | .enabled | tostring] | first // "unknown"')
    if [ "$device_flow" = "false" ]; then
        echo -e "${GREEN}Device authorization grant is disabled server-wide (device-flow feature off).${NC}"
        return 0
    fi
    local has_condition=$(echo "$serverinfo" | \
        jq -r '[.componentTypes["org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"][]? | select(.id=="grant-type")] | length')
    if [ "$has_condition" = "0" ] || [ -z "$has_condition" ]; then
        echo -e "${RED}This Keycloak cannot refuse the device grant by client policy (no grant-type condition). Start it with KC_FEATURES_DISABLED=device-flow.${NC}"
        return 1
    fi

    local profiles
    profiles=$(_admin_get "$token" "/admin/realms/${REALM}/client-policies/profiles") || return 1
    local new_profiles=$(echo "$profiles" | jq '
        {profiles: (((.profiles // []) | map(select(.name != "mcp-reject-request"))) + [{
            name: "mcp-reject-request",
            description: "Reject the request outright",
            executors: [{executor: "reject-request", configuration: {}}]
        }])}')
    local r1=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/client-policies/profiles" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d "$new_profiles")

    local policies
    policies=$(_admin_get "$token" "/admin/realms/${REALM}/client-policies/policies") || return 1
    local new_policies=$(echo "$policies" | jq '
        {policies: (((.policies // []) | map(select(.name != "mcp-reject-device-code-grant"))) + [{
            name: "mcp-reject-device-code-grant",
            description: "Device grant bypasses the DCR Trusted Hosts redirect check",
            enabled: true,
            conditions: [{
                condition: "grant-type",
                configuration: {"grant_types": ["urn:ietf:params:oauth:grant-type:device_code"]}
            }],
            profiles: ["mcp-reject-request"]
        }])}')
    local r2=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/client-policies/policies" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d "$new_policies")

    if { [ "$r1" = "204" ] || [ "$r1" = "200" ]; } && { [ "$r2" = "204" ] || [ "$r2" = "200" ]; }; then
        echo -e "${GREEN}Device authorization grant rejected (profile HTTP $r1, policy HTTP $r2).${NC}"
    else
        echo -e "${RED}Failed to configure the device-grant rejection policy (profile HTTP $r1, policy HTTP $r2).${NC}"
        return 1
    fi
}

verify_dcr_baseline_policies() {
    local token=$1

    echo "Verifying Keycloak's anonymous DCR baseline policies..."

    local components
    components=$(_admin_get "$token" "/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy") || return 1

    local provider missing=""
    for provider in consent-required scope max-clients allowed-protocol-mappers; do
        local found=$(echo "$components" | \
            jq -r --arg p "$provider" '[.[] | select(.subType=="anonymous" and .providerId==$p)] | length')
        if [ "$found" = "0" ]; then
            missing="${missing} ${provider}"
        fi
    done

    if [ -n "$missing" ]; then
        echo -e "${RED}Anonymous DCR policies missing from realm '${REALM}':${missing}. Restore them under Clients -> Client registration.${NC}"
        return 1
    fi
    echo -e "${GREEN}Anonymous DCR baseline policies present.${NC}"
}

# Restrict OAuth token exchange (RFC 8693) to clients that hold a client role
# named 'token-exchange'.
restrict_token_exchange() {
    local token=$1

    echo "Restricting token exchange to clients with the 'token-exchange' client role..."

    local serverinfo
    serverinfo=$(_admin_get "$token" "/admin/serverinfo") || return 1
    local has_condition=$(echo "$serverinfo" | \
        jq -r '[.componentTypes["org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"][]? | select(.id=="grant-type")] | length')
    local standard_exchange=$(echo "$serverinfo" | \
        jq -r '[.features[]? | select(.name=="TOKEN_EXCHANGE_STANDARD_V2") | .enabled | tostring] | first // "false"')
    if [ "$standard_exchange" != true ]; then
        echo "Standard token exchange is disabled; existing exchange policy retained."
        return 0
    fi
    if [ "$has_condition" = "0" ] || [ -z "$has_condition" ]; then
        echo "Standard token exchange requires the grant-type client-policy condition." >&2
        return 1
    fi

    _require_supported_keycloak "$KEYCLOAK_URL" "$token" || return 1

    local web_id
    web_id=$(_find_client_uuid "$token" "mcp-gateway-web") || return 1
    if [ -z "$web_id" ]; then
        echo -e "${RED}Client mcp-gateway-web not found; cannot mark it for token exchange.${NC}"
        return 1
    fi
    local role_status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${web_id}/roles" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d '{"name": "token-exchange", "description": "This client may use OAuth token exchange"}')
    if [ "$role_status" != "201" ] && [ "$role_status" != "409" ]; then
        echo -e "${RED}Failed to create the 'token-exchange' role on mcp-gateway-web. HTTP status: ${role_status}${NC}"
        return 1
    fi

    local profiles
    profiles=$(_admin_get "$token" "/admin/realms/${REALM}/client-policies/profiles") || return 1
    local new_profiles=$(echo "$profiles" | jq '
        {profiles: (((.profiles // []) | map(select(.name != "mcp-reject-request"))) + [{
            name: "mcp-reject-request",
            description: "Reject the request outright",
            executors: [{executor: "reject-request", configuration: {}}]
        }])}')
    local r1=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/client-policies/profiles" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d "$new_profiles")

    local policies
    policies=$(_admin_get "$token" "/admin/realms/${REALM}/client-policies/policies") || return 1
    local new_policies=$(echo "$policies" | jq '
        {policies: (((.policies // []) | map(select(.name != "mcp-restrict-token-exchange"))) + [{
            name: "mcp-restrict-token-exchange",
            description: "Only clients with a token-exchange client role may exchange tokens",
            enabled: true,
            mode: "STRICT",
            conditions: [
                {condition: "grant-type", configuration: {"grant_types": ["urn:ietf:params:oauth:grant-type:token-exchange"]}},
                {condition: "client-roles", configuration: {"roles": ["token-exchange"], "is-negative-logic": true}}
            ],
            profiles: ["mcp-reject-request"]
        }])}')
    local r2=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/client-policies/policies" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d "$new_policies")

    if { [ "$r1" = "204" ] || [ "$r1" = "200" ]; } && { [ "$r2" = "204" ] || [ "$r2" = "200" ]; }; then
        echo -e "${GREEN}Token exchange restricted (profile HTTP $r1, policy HTTP $r2).${NC}"
    else
        echo -e "${RED}Failed to configure the token-exchange restriction (profile HTTP $r1, policy HTTP $r2).${NC}"
        return 1
    fi
}

# =============================================================================
# Client ID Metadata Documents (CIMD) support
# =============================================================================
# Lets MCP clients whose client_id is an https URL (e.g. VS Code, Claude Code)
# authenticate by publishing their own metadata document -- Keycloak fetches +
# validates it, no pre-registration. Requires the server 'cimd' feature
# (KC_FEATURES=cimd; Keycloak >= 26.8.2). Configures a client policy: a
# `client-id-uri` condition (scheme https + permitted domains) applying a
# `client-id-metadata-document` executor with the native scope rejection policy.
#
# NOTE: the executor's permitted-domains must list EVERY host a trusted client's
# document references -- the client_id host, its loopback redirect hosts
# (127.0.0.1/localhost), and any logo_uri/client_uri CDN hosts -- or the fetch
# is rejected ("not trusted domain: host = ..."). The condition, which decides
# which client_id URLs Keycloak fetches at all, gets the same list WITHOUT the
# loopback hosts: they are only ever redirect hosts, and a loopback client_id
# would make Keycloak fetch a "metadata document" from itself.
configure_cimd_client_policy() {
    local ENABLE_CIMD_CONFIG=true
    _configure_cimd_client_policy "$1"
}

# ENABLE_CIMD_CONFIG=false: remove the policy and profile
# configure_cimd_client_policy writes, preserving every other client
# policy/profile. The policy goes first because it references the profile.
remove_cimd_client_policy() {
    local ENABLE_CIMD_CONFIG=false
    configure_registration_security_policy "$1" || return 1
    _remove_cimd_client_policy "$1"
}

# ENABLE_DCR_CONFIG=false: put the anonymous Trusted Hosts policy back to
# Keycloak's default (the registering host must be trusted, and no host is),
# so anonymous registration is refused from everywhere. Clients registered
# while DCR was on are disabled next by
# disable_unlisted_clients.
disable_anonymous_dcr() {
    local token=$1

    echo "Disabling anonymous DCR ('Trusted Hosts': no trusted hosts)..."

    local components realm_body
    components=$(_admin_get "$token" "/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy") || return 1
    local policy_id=$(echo "$components" | \
        jq -r '[.[] | select(.providerId=="trusted-hosts" and .subType=="anonymous") | .id] | first // empty')
    realm_body=$(_admin_get "$token" "/admin/realms/${REALM}") || return 1
    local realm_id=$(echo "$realm_body" | jq -r '.id // empty')
    if [ -z "$realm_id" ]; then
        echo -e "${RED}Could not read the id of realm '${REALM}'.${NC}"
        return 1
    fi

    local policy_json=$(jq -n --arg parentId "$realm_id" '{
        name: "Trusted Hosts",
        providerId: "trusted-hosts",
        providerType: "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy",
        parentId: $parentId,
        subType: "anonymous",
        config: {
            "host-sending-registration-request-must-match": ["true"],
            "client-uris-must-match": ["true"],
            "trusted-hosts": []
        }
    }')

    local response
    if [ -n "$policy_id" ]; then
        response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/components/${policy_id}" \
            -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
            -d "$(echo "$policy_json" | jq --arg id "$policy_id" '. + {id: $id}')")
    else
        response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/components" \
            -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
            -d "$policy_json")
    fi

    if [ "$response" = "204" ] || [ "$response" = "201" ]; then
        echo -e "${GREEN}Anonymous DCR disabled (HTTP ${response}).${NC}"
    else
        echo -e "${RED}Failed to disable anonymous DCR. HTTP status: ${response}${NC}"
        return 1
    fi
}

# Keycloak's built-in clients. They stay enabled when DCR is off and are the
# only kept clients whose registration access token is left alone.
DCR_OFF_BUILTIN_CLIENTS=(
    account account-console admin-cli admin-permissions broker realm-management security-admin-console
)

# Clients kept by name when DCR is off: the built-in clients, mcp-gateway-web,
# and every clientId listed in DCR_KEEP_CLIENTS (comma or space
# separated). M2M clients (mcp-gateway-m2m, agent-<id>-m2m,
# federation-peer-m2m, the terraform bots, the registry's service accounts) are
# not kept by name, because a registrant can choose any clientId; they are kept
# by the service-account-only shape the setup scripts give them.
DCR_OFF_KEEP_CLIENTS=(
    "${DCR_OFF_BUILTIN_CLIENTS[@]}"
    mcp-gateway-web
)

# Sorts each enabled client into builtin, keep (by name), shape (a confidential
# service-account-only client using client-secret) or disable. Prints
# tab-separated action, uuid and a printable clientId (registrants choose
# clientIds).
DCR_OFF_CLASSIFY_JQ='
    .[]
    | select(.enabled == true)
    | .clientId as $c
    | (if ($builtin | any(. == $c)) then "builtin"
       elif ($keep | any(. == $c)) then "keep"
       elif ($c == "mcp-gateway-ide" and .protocol == "openid-connect"
             and .publicClient == true and .standardFlowEnabled == true
             and .implicitFlowEnabled != true and .directAccessGrantsEnabled != true
             and .serviceAccountsEnabled != true and .bearerOnly != true
             and .attributes["pkce.code.challenge.method"] == "S256"
             and ((.redirectUris // []) | length) > 0
             and all(.redirectUris[]; test("^http://(localhost|127\\.0\\.0\\.1):[0-9]+/[^*]*$"))) then "ide"
       elif ($c != "mcp-gateway-ide" and .publicClient != true
             and .serviceAccountsEnabled == true
             and .clientAuthenticatorType == "client-secret"
             and .standardFlowEnabled != true
             and .implicitFlowEnabled != true
             and .directAccessGrantsEnabled != true
             and ((.redirectUris // []) | length) == 0) then "shape"
       else "disable" end) as $action
    | [$action, .id, ($c | gsub("[^A-Za-z0-9._:/@-]"; "?"))] | @tsv'

_disable_client() {
    local token=$1
    local uuid=$2
    local display=$3
    local status

    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${uuid}" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d '{"enabled": false}')
    if [ "$status" != "204" ]; then
        echo -e "${RED}Failed to disable client ${display} (HTTP ${status}).${NC}"
        return 1
    fi
    echo "  - disabled ${display}"
}

# ENABLE_DCR_CONFIG=false: disable every client not kept above, and invalidate
# the registration access token of every client except the built-in ones, so
# clients MCP tools registered while DCR was on stop working (no new logins,
# no token refresh) and no registrant can modify a client any more, including
# one that is kept. Disabled, not deleted: re-enable one in the admin console
# if it was needed. Fails closed: if the client list cannot be read, nothing
# is disabled and the run fails.
_dcr_sweep_targets() {
    local token=$1 clients keep builtin extra_keep=()
    clients=$(_list_all_clients "$token") || return 1
    # CIMD has its own off/trust inventory, regardless of DCR keep settings.
    clients=$(jq '[.[] | select(((.attributes // {}) | has("cimd.cache.expiry.time.in.sec")) | not)]' <<< "$clients") || return 1
    read -r -a extra_keep <<< "${DCR_KEEP_CLIENTS//,/ }"
    keep=$(printf '%s\n' "${DCR_OFF_KEEP_CLIENTS[@]}" "${extra_keep[@]}" | jq -R . | jq -sc 'map(select(length > 0))') || return 1
    builtin=$(printf '%s\n' "${DCR_OFF_BUILTIN_CLIENTS[@]}" | jq -R . | jq -sc '.') || return 1
    jq -r --argjson keep "$keep" --argjson builtin "$builtin" "$DCR_OFF_CLASSIFY_JQ" <<< "$clients"
}

_confirm_dcr_sweep() {
    local targets=$1 action uuid display destructive=false
    echo "DCR-off inventory (action, client UUID, client ID):"
    printf '%s\n' "$targets"
    while IFS=$'\t' read -r action uuid display; do
        [ "$action" != disable ] || destructive=true
    done <<< "$targets"
    if [ "$destructive" = true ] && [ "${DCR_SWEEP_ACKNOWLEDGED:-false}" != true ]; then
        echo "Refusing to disable clients without DCR_SWEEP_ACKNOWLEDGED=true (Helm: dcr.sweepAcknowledged=true). Review this inventory and set DCR_KEEP_CLIENTS to exact client IDs to preserve. No sweep was performed." >&2
        return 1
    fi
}

preflight_dcr_upgrade() {
    local token=$1 status targets info type="org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy"
    _preflight_registration_security "$token" || return 1
    _preflight_token_exchange_support "$KEYCLOAK_URL" "$token" || return 1
    if [ "${ENABLE_CIMD_CONFIG:-false}" = true ]; then
        _preflight_cimd_support "$KEYCLOAK_URL" "$token" || return 1
    fi
    if [ "${ENABLE_DCR_CONFIG:-false}" = true ]; then
        _require_supported_keycloak "$KEYCLOAK_URL" "$token" || return 1
        info=$(_admin_get "$token" "/admin/serverinfo") || return 1
        if ! jq -e --arg type "$type" 'any(.componentTypes[$type][]?; .id == "default-roles")' >/dev/null <<< "$info"; then
            echo "DCR requires Keycloak 26.8.2+ with Default Roles Disabled; keep ENABLE_DCR_CONFIG=false or upgrade before enabling DCR. No realm settings changed." >&2
            return 1
        fi
    fi
    status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $token" "$KEYCLOAK_URL/admin/realms/$REALM") || return 1
    [ "$status" != 404 ] || return 0
    [ "$status" = 200 ] || { echo "Could not inspect realm before DCR sweep (HTTP $status)." >&2; return 1; }
    targets=$(_cimd_sweep_targets "$token") || return 1
    _confirm_cimd_sweep "$targets" || return 1
    [ "${ENABLE_DCR_CONFIG:-false}" != true ] || return 0
    targets=$(_dcr_sweep_targets "$token") || return 1
    _confirm_dcr_sweep "$targets"
}

disable_unlisted_clients() {
    local token=$1 targets
    targets=$(_dcr_sweep_targets "$token") || return 1
    _confirm_dcr_sweep "$targets" || return 1

    local disabled=0 rotated=0 action uuid display
    while IFS=$'\t' read -r action uuid display; do
        if [ -z "$uuid" ] || [ "$action" = "builtin" ]; then
            continue
        fi
        if [ "$action" = "disable" ]; then
            _disable_client "$token" "$uuid" "$display" || return 1
            disabled=$((disabled + 1))
        elif [ "$action" = "shape" ]; then
            echo -e "  ${YELLOW}- kept ${display}: confidential service-account-only client using client-secret (disable it by hand if this deployment did not create it)${NC}"
        fi
        _rotate_registration_access_token "$token" "$uuid" "$display" || return 1
        rotated=$((rotated + 1))
    done <<< "$targets"

    echo -e "${GREEN}Disabled ${disabled} client(s) not on the keep-list (built-in, mcp-gateway-web, safe public PKCE IDE, DCR_KEEP_CLIENTS, service-account-only with client-secret); invalidated the registration access token of ${rotated} non-built-in client(s).${NC}"
}

# Main execution
main() {
    # Get script directory and find .env file
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
    ENV_FILE="$PROJECT_ROOT/.env"

    # Load environment variables from .env file if it exists.
    #
    # The caller's environment WINS over the file for every variable that decides
    # WHICH Keycloak we talk to or WHAT we write into it. This matters now that an
    # existing client is PUT rather than skipped, because Keycloak replaces
    # redirectUris/webOrigins wholesale on that PUT: on a host that also has a
    # Compose .env, the file's localhost values would be written over a deployed
    # realm and break login with invalid_redirect_uri. KEYCLOAK_ADMIN_URL is in the
    # list for the same reason in reverse -- a stale .env would otherwise retarget
    # the entire run at a local Keycloak.
    CALLER_PRECEDENCE_VARS="KEYCLOAK_ADMIN_URL REGISTRY_URL AUTH_SERVER_EXTERNAL_URL ENABLE_DCR_CONFIG ENABLE_CIMD_CONFIG DCR_KEEP_CLIENTS DCR_SWEEP_ACKNOWLEDGED DCR_ALLOWED_SCOPES KEYCLOAK_ADMIN KEYCLOAK_ADMIN_PASSWORD INITIAL_ADMIN_PASSWORD CIMD_TRUSTED_DOMAINS REGISTRATION_TRUSTED_CLIENT_IDS AWS_REGION"
    if [ -f "$ENV_FILE" ]; then
        echo "Loading environment variables from $ENV_FILE..."
        # Snapshot the caller's values before `set -a; source` overwrites them.
        local name present
        local -a caller_names=() caller_values=()
        for name in $CALLER_PRECEDENCE_VARS; do
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
        echo "Current directory: $(pwd)"
        echo "Script directory: $SCRIPT_DIR"
        echo "Project root: $PROJECT_ROOT"
    fi

    # Override KEYCLOAK_URL with KEYCLOAK_ADMIN_URL for API calls
    KEYCLOAK_URL="${KEYCLOAK_ADMIN_URL-http://localhost:8080}"
    KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN-admin}"
    if [ -z "$KEYCLOAK_URL" ] || [ -z "$KEYCLOAK_ADMIN" ]; then
        echo "KEYCLOAK_ADMIN_URL and KEYCLOAK_ADMIN must not be empty." >&2
        return 1
    fi
    echo "Using Keycloak API URL: $KEYCLOAK_URL"

    # Check if admin password is set
    if [ -z "$KEYCLOAK_ADMIN_PASSWORD" ]; then
        echo -e "${RED}Error: KEYCLOAK_ADMIN_PASSWORD environment variable is not set${NC}"
        echo "Please set it in .env file or export it before running this script"
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

    preflight_dcr_upgrade "$TOKEN" || exit 1
    reconcile_registration_security "$TOKEN" || exit 1
    harden_master_realm "$TOKEN" || exit 1

    # Create realm and configure it step by step
    if create_realm "$TOKEN"; then
        reconcile_registration_security "$TOKEN" || exit 1
        configure_dcr_allowed_scopes "$TOKEN" || exit 1
        configure_dcr_allowed_mapper_types "$TOKEN" || exit 1
        _check_realm_group_path_mappers "$KEYCLOAK_URL" "$REALM" "$TOKEN" || exit 1
        create_clients "$TOKEN" || exit 1
        create_scopes "$TOKEN" || exit 1
        configure_registration_security_policy "$TOKEN" || exit 1
        create_groups "$TOKEN"
        create_users "$TOKEN" || exit 1
        create_service_account_user "$TOKEN" || exit 1
        setup_client_secrets "$TOKEN" || exit 1
        setup_groups_mapper "$TOKEN" || exit 1
        setup_m2m_scopes "$TOKEN"

        # Groups + audience mappers on the `basic` scope run whether or not
        # DCR is on: `basic` is attached to every client, and the
        # aud="mcp-gateway" mapper is what puts the gateway's audience in the
        # web and M2M clients' tokens too. Without it the gateway rejects
        # every token.
        setup_dcr_groups_mapper "$TOKEN" || exit 1
        setup_dcr_audience_mapper "$TOKEN" || exit 1

        # The master admin token lives 60 seconds by default, and the steps
        # above can take longer on a large realm. Every read below fails the
        # run on an error response, so each long step starts with a fresh token.
        _refresh_admin_token
        update_client_groups_mappers "$TOKEN" || exit 1
        _refresh_admin_token

        # Runs with DCR on or off, before the DCR steps: no client may hold the
        # clientId that equals the audience every user token carries.
        reserve_gateway_audience_client "$TOKEN" || exit 1

        # MCP DCR support, gated by ENABLE_DCR_CONFIG (default false): an
        # explicit Allowed Client Scopes allowlist, Trusted Hosts relaxed for
        # cloud-egress MCP clients, no client-supplied mappers, no device
        # grant. Off-state revocation already ran before unrelated migrations.
        if [ "${ENABLE_DCR_CONFIG:-false}" = "true" ]; then
            configure_dcr_reject_device_grant "$TOKEN" || exit 1
            verify_dcr_baseline_policies "$TOKEN" || exit 1
            configure_dcr_trusted_hosts "$TOKEN" || exit 1
        else
            echo -e "${YELLOW}DCR disabled (ENABLE_DCR_CONFIG=${ENABLE_DCR_CONFIG:-false}): turning off anonymous registration.${NC}"
            configure_dcr_default_roles "$TOKEN" false || exit 1
        fi

        # Only the gateway client may use token exchange (runs with DCR on or off).
        restrict_token_exchange "$TOKEN" || exit 1

        # MCP CIMD support (client-id-metadata-document client policy). Requires
        # the Keycloak server 'cimd' feature (KC_FEATURES=cimd; >= 26.8.2). Opt-in
        # via ENABLE_CIMD_CONFIG; when false, a previously written CIMD policy
        # is removed.
        if [ "${ENABLE_CIMD_CONFIG:-false}" = "true" ]; then
            configure_cimd_client_policy "$TOKEN" || exit 1
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
    echo "Users created:"
    echo "  - admin (admin access) — password from INITIAL_ADMIN_PASSWORD"
    echo ""
    echo -e "${YELLOW}The admin password is set to 'temporary': you will be prompted to${NC}"
    echo -e "${YELLOW}change it on first login.${NC}"
}

# Run main function
main
