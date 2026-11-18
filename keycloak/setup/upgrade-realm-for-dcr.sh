#!/bin/bash
# Upgrade an existing Keycloak realm to support MCP Dynamic Client Registration
#
# This standalone script applies the configuration changes required for MCP
# clients (Claude Code, Claude.ai connectors, Cursor) to register dynamically
# and produce tokens with a `mcpgw_group_paths` claim for per-user authorization at the
# gateway. Use this on existing deployments where the realm was created before
# these changes landed in init-keycloak.sh.
#
# What a self-registered client is entitled to, stated positively: exactly the
# client scopes named in DCR_DEFAULT_ALLOWED_SCOPES below (or DCR_ALLOWED_SCOPES),
# and nothing else. It may not define protocol mappers, because a mapper
# chooses what a claim says and the gateway derives authorization from the
# `mcpgw_group_paths` claim. These limits are realm policy, which is where Keycloak
# enforces them; there is deliberately no gateway-side setting, because a
# setting in the gateway could not constrain what the realm hands out.
#
# The script is idempotent: re-running it is safe.
#
# Usage:
#   bash keycloak/setup/upgrade-realm-for-dcr.sh
#
# Reads .env from repo root for KEYCLOAK_ADMIN_URL, KEYCLOAK_ADMIN, and
# KEYCLOAK_ADMIN_PASSWORD.
#
# ENABLE_DCR_CONFIG (default false): true sets up anonymous DCR for MCP
# clients; false turns DCR off in the realm instead:
# anonymous registration is disabled, and the scope and mapper
# restrictions still apply. ENABLE_CIMD_CONFIG (default false): unless true,
# the trusted-domain CIMD policy is removed. When true, the policy is upserted
# after the server capability preflight succeeds.

set -eo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/group-path-preflight.sh"

REALM="mcp-gateway"
# Do not unset KEYCLOAK_ADMIN / KEYCLOAK_ADMIN_PASSWORD here — main() needs to
# detect whether the caller already exported them so it can skip .env loading
# and run against a non-.env Keycloak (e.g. ECS).
KEYCLOAK_URL=""

# The complete set of client scopes a self-registered (DCR) client may hold.
DCR_DEFAULT_ALLOWED_SCOPES="openid basic profile email roles web-origins acr offline_access"

# Refused even when an operator lists them in DCR_ALLOWED_SCOPES: the scopes
# that carry gateway authorization rather than OIDC identity (the union of
# what the three provisioning surfaces create; the last two are the `_id` of a
# scope document granting `server: "*"` with every action), and Keycloak's
# Admin API scope. Any name containing "/" is refused as well.
DCR_FORBIDDEN_SCOPES=(
    "mcp-servers-unrestricted/read"
    "mcp-servers-unrestricted/execute"
    "mcp-servers-restricted/read"
    "mcp-servers-restricted/execute"
    "mcp-registry-admin"
    "registry-admins"
    "kc-admin-api"
)

# First-party clients whose registration access token is rotated every run.
FIRST_PARTY_CLIENTS=("mcp-gateway-web" "mcp-gateway-m2m")

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
# The same program is in init-keycloak.sh, the Helm Job script and
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

# Variables a caller-supplied value keeps over .env (see main).
CALLER_PRECEDENCE_VARS="KEYCLOAK_ADMIN_URL ENABLE_DCR_CONFIG ENABLE_CIMD_CONFIG DCR_KEEP_CLIENTS DCR_SWEEP_ACKNOWLEDGED DCR_ALLOWED_SCOPES KEYCLOAK_ADMIN KEYCLOAK_ADMIN_PASSWORD INITIAL_ADMIN_PASSWORD CIMD_TRUSTED_DOMAINS REGISTRATION_TRUSTED_CLIENT_IDS AWS_REGION"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}MCP DCR upgrade for existing Keycloak realm '${REALM}'${NC}"
echo "============================================================"

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
    # The password goes in on stdin, not argv, so it is not visible in `ps`.
    printf '%s' "$KEYCLOAK_ADMIN_PASSWORD" | \
        realm_curl -s -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
            -H "Content-Type: application/x-www-form-urlencoded" \
            --data-urlencode "username=${KEYCLOAK_ADMIN}" \
            --data-urlencode "password@-" \
            --data-urlencode "grant_type=password" \
            --data-urlencode "client_id=admin-cli" | \
        jq -r '.access_token // empty' 2>/dev/null
}

# Get a fresh admin token into TOKEN (call before each step).
refresh_token() {
    TOKEN=$(get_admin_token)
    if [ -z "$TOKEN" ]; then
        echo -e "${RED}Error: failed to obtain admin token${NC}"
        exit 1
    fi
}

# GET an Admin API path and print the body. Returns 1 (status on stderr)
# unless the response is HTTP 200 with a JSON object or array, so an expired
# token or an error page is never read as "nothing configured".
_admin_get() {
    local token=$1
    local path=$2
    local response status body

    if ! response=$(realm_curl -s -w '\n%{http_code}' \
        -H "Authorization: Bearer ${token}" "${KEYCLOAK_URL}${path}"); then
        echo -e "${RED}FAILED: GET ${path}: request error${NC}" >&2
        return 1
    fi
    status=${response##*$'\n'}
    body=${response%$'\n'*}

    if [ "$status" != "200" ] || \
        ! echo "$body" | jq -e 'type == "object" or type == "array"' >/dev/null 2>&1; then
        echo -e "${RED}FAILED: GET ${path} returned HTTP ${status}${NC}" >&2
        return 1
    fi
    printf '%s\n' "$body"
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
        echo -e "${RED}FAILED: could not read the current settings of realm ${realm}.${NC}" >&2
        return 1
    fi
}

# PUT the security baseline onto realm $2, merged over its current values.
_apply_realm_security_settings() {
    local token=$1
    local realm=$2
    local gateway=$3
    local settings_json status

    settings_json=$(_realm_security_settings "$token" "$realm" "$gateway") || return 1
    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${realm}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$settings_json")

    if [[ "$status" == 2[0-9][0-9] ]]; then
        echo -e "${GREEN}OK: realm ${realm} security settings applied, stronger existing settings kept (HTTP ${status}).${NC}"
    else
        echo -e "${RED}FAILED: apply realm ${realm} security settings. HTTP status: ${status}${NC}"
        return 1
    fi
}

# Brute-force protection and event logging on the master realm, where the
# Keycloak admin account lives. sslRequired is deliberately left alone.
harden_master_realm() {
    local token=$1

    echo ""
    echo "[master] Enabling brute-force protection and event logging on the master realm..."
    _apply_realm_security_settings "$token" "master" false
}

# Brute-force protection, password policy, login/admin event logging and the
# offline-session cap on the gateway realm, with the baseline init-keycloak.sh
# applies. A partial PUT: other realm settings are kept.
configure_realm_security_settings() {
    local token=$1

    echo ""
    echo "[realm] Applying brute-force protection, password policy, event logging and offline-session cap to '${REALM}'..."
    _apply_realm_security_settings "$token" "$REALM" true
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
    mapper_name=$(echo "$mapper_json" | jq -r '.name // empty')
    if [ -z "$mapper_id" ]; then
        printf '%bFAILED: groups mapper on %s has no id.%b\n' "$RED" "$owner" "$NC"
        return 1
    fi

    if jq -e '.config["full.path"] == "true" and .config["claim.name"] == "mcpgw_group_paths"' >/dev/null <<< "$mapper_json"; then
        printf '  %s: groups mapper %s already emits mcpgw_group_paths with full.path=true.\n' "$owner" "$mapper_name"
        return 0
    fi

    local updated status
    updated=$(echo "$mapper_json" | jq -c '.config = ((.config // {}) + {"full.path": "true", "claim.name": "mcpgw_group_paths"})')
    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}${models_path}/${mapper_id}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d "$updated")

    if [[ "$status" == 2[0-9][0-9] ]]; then
        printf '%bOK: %s: groups mapper %s now uses full.path=true.%b\n' "$GREEN" "$owner" "$mapper_name" "$NC"
    else
        printf '%bFAILED: %s: update groups mapper %s. HTTP status: %s%b\n' "$RED" "$owner" "$mapper_name" "$status" "$NC"
        return 1
    fi
}

setup_dcr_groups_mapper() {
    local token=$1


    echo ""
    echo "[1/7] Attaching Groups protocol mapper (mcpgw_group_paths, full.path=true) to the 'basic' client-scope..."

    local scopes basic_scope_id
    scopes=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes") || return 1
    basic_scope_id=$(echo "$scopes" | jq -r '[.[] | select(.name=="basic") | .id] | first // empty')

    if [ -z "$basic_scope_id" ]; then
        echo "Required 'basic' client scope is missing from realm '$REALM'." >&2
        return 1
    fi

    local scope existing
    scope=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes/${basic_scope_id}") || return 1
    local mappers
    mappers=$(jq -c '.protocolMappers // []' <<< "$scope") || return 1
    _check_group_path_mapper_models "$mappers" || return 1
    existing=$(echo "$scope" | jq -c '.protocolMappers // []
        | if type != "array" then error("invalid mapper list")
          elif ([.[] | select(.name == "groups")] | length) > 1
            or any(.[]; .name == "groups" and
                (.protocol != "openid-connect" or .protocolMapper != "oidc-group-membership-mapper"
                 or (.config["claim.name"] != "groups" and .config["claim.name"] != "mcpgw_group_paths")))
            or any(.[]; .config["claim.name"] == "mcpgw_group_paths" and .name != "groups")
          then error("conflicting groups mapper")
          else [.[] | select(.name == "groups")] | first // empty end') || return 1

    if [ -n "$existing" ]; then
        _ensure_groups_mapper_full_path "$token" \
            "/admin/realms/${REALM}/client-scopes/${basic_scope_id}/protocol-mappers/models" \
            "$existing" "'basic' client-scope"
        return
    fi

    _check_group_paths "$KEYCLOAK_URL" "$REALM" "$token" || return 1

    local groups_mapper_json='{
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

    local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
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

# Enumerate clients for migration of legacy groups and mcpgw_group_paths
# membership mappers; unrelated custom group claims remain unchanged.
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

    echo ""
    echo "[clients] Reserving clientId '${GATEWAY_AUDIENCE_CLIENT_ID}' (the audience every user token carries)..."
    clients=$(_admin_get "$token" "/admin/realms/${REALM}/clients?clientId=${GATEWAY_AUDIENCE_CLIENT_ID}") || return 1
    if ! existing=$(jq -ec --arg c "$GATEWAY_AUDIENCE_CLIENT_ID" \
        'if type == "array" then ([.[] | select(.clientId == $c)] | first // {}) else error("not an array") end' <<< "$clients"); then
        echo -e "${RED}FAILED: unexpected response looking up client ${GATEWAY_AUDIENCE_CLIENT_ID}.${NC}"
        return 1
    fi

    if [ "$existing" = "{}" ]; then
        status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients" \
            -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
            -d "$GATEWAY_AUDIENCE_PLACEHOLDER_JSON")
        if [ "$status" != "201" ]; then
            echo -e "${RED}FAILED: create the disabled placeholder client ${GATEWAY_AUDIENCE_CLIENT_ID} (HTTP ${status}).${NC}"
            return 1
        fi
        echo -e "${GREEN}OK: created the disabled placeholder client ${GATEWAY_AUDIENCE_CLIENT_ID}.${NC}"
        return 0
    fi

    uuid=$(jq -r '.id' <<< "$existing")
    display="${GATEWAY_AUDIENCE_CLIENT_ID} (id ${uuid})"
    if ! jq -e "$GATEWAY_AUDIENCE_PLACEHOLDER_SHAPE_JQ" >/dev/null <<< "$existing"; then
        echo -e "${YELLOW}WARNING: client ${display} was enabled or not in the placeholder shape; disabling it. A client with this clientId is inside the audience of every user token.${NC}"
    fi
    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${uuid}" \
        -H "Authorization: Bearer ${token}" -H "Content-Type: application/json" \
        -d "$GATEWAY_AUDIENCE_PLACEHOLDER_JSON")
    if [ "$status" != "204" ]; then
        echo -e "${RED}FAILED: disable client ${display} (HTTP ${status}).${NC}"
        return 1
    fi
    _rotate_registration_access_token "$token" "$uuid" "$GATEWAY_AUDIENCE_CLIENT_ID" || return 1
    existing=$(_admin_get "$token" "/admin/realms/${REALM}/clients/${uuid}") || return 1
    if ! jq -e "$GATEWAY_AUDIENCE_PLACEHOLDER_SHAPE_JQ" >/dev/null <<< "$existing"; then
        echo -e "${RED}FAILED: client ${display} is still not a disabled placeholder after the update.${NC}"
        return 1
    fi
    echo -e "${GREEN}OK: client ${GATEWAY_AUDIENCE_CLIENT_ID} is a disabled placeholder.${NC}"
}

_find_client_uuid() {
    local token=$1
    local client_id=$2
    local clients

    clients=$(_admin_get "$token" "/admin/realms/${REALM}/clients?clientId=${client_id}") || return 1
    echo "$clients" | jq -r --arg c "$client_id" '[.[] | select(.clientId == $c) | .id] | first // empty'
}

disable_web_client_direct_grants() {
    local token=$1

    echo ""
    echo "[clients] Disabling the password grant (direct access grants) on mcp-gateway-web..."

    local web_id
    web_id=$(_find_client_uuid "$token" "mcp-gateway-web") || return 1
    if [ -z "$web_id" ]; then
        echo -e "${RED}FAILED: client mcp-gateway-web not found in realm '${REALM}'.${NC}"
        return 1
    fi

    local status
    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X PUT "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${web_id}" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -d '{"directAccessGrantsEnabled": false}')
    if [[ "$status" != 2[0-9][0-9] ]]; then
        echo -e "${RED}FAILED: update mcp-gateway-web. HTTP status: ${status}${NC}"
        return 1
    fi

    local client
    client=$(_admin_get "$token" "/admin/realms/${REALM}/clients/${web_id}") || return 1
    if [ "$(echo "$client" | jq -r '.directAccessGrantsEnabled')" != "false" ]; then
        echo -e "${RED}FAILED: mcp-gateway-web still allows direct access grants after the update.${NC}"
        return 1
    fi
    echo -e "${GREEN}OK: mcp-gateway-web no longer accepts the password grant.${NC}"
}

rotate_first_party_registration_tokens() {
    local token=$1

    echo ""
    echo "[clients] Rotating registration access tokens of first-party clients..."

    local client_id client_uuid status
    for client_id in "${FIRST_PARTY_CLIENTS[@]}"; do
        client_uuid=$(_find_client_uuid "$token" "$client_id") || return 1
        if [ -z "$client_uuid" ]; then
            echo -e "${YELLOW}${client_id} not present in realm '${REALM}'; nothing to rotate.${NC}"
            continue
        fi

        status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${client_uuid}/registration-access-token" \
            -H "Authorization: Bearer ${token}")
        if [[ "$status" == 2[0-9][0-9] ]]; then
            echo -e "${GREEN}OK: ${client_id} registration access token rotated.${NC}"
        else
            echo -e "${RED}FAILED: rotate the registration access token of ${client_id}. HTTP status: ${status}${NC}"
            return 1
        fi
    done
}

setup_dcr_audience_mapper() {
    local token=$1

    echo ""
    echo "[2/7] Attaching Audience protocol mapper to the 'basic' client-scope..."

    local scopes basic_scope_id
    scopes=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes") || return 1
    basic_scope_id=$(echo "$scopes" | jq -r '[.[] | select(.name=="basic") | .id] | first // empty')

    if [ -z "$basic_scope_id" ] || [ "$basic_scope_id" = "null" ]; then
        echo "Required 'basic' client scope is missing from realm '$REALM'." >&2
        return 1
    fi

    local scope existing
    scope=$(_admin_get "$token" "/admin/realms/${REALM}/client-scopes/${basic_scope_id}") || return 1
    existing=$(echo "$scope" | jq -r '[.protocolMappers[]? | select(.name=="mcp-gateway-audience") | .id] | first // empty')

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

    local response=$(realm_curl -s -o /dev/null -w "%{http_code}" \
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

# Set one subType's "Allowed Client Scopes" policy to exactly the given list,
# with allow-default-scopes off so realm scopes outside the list are refused.
# Both 'anonymous' and 'authenticated' subTypes need this because Keycloak
# picks one based on whether the DCR request carries an initial-access /
# registration token. Claude Code, Claude.ai connectors, and Cursor have been
# observed to hit either path.
_set_allowed_scopes_policy() {
    local token=$1
    local sub_type=$2  # "anonymous" or "authenticated"
    local components=$3
    local allowed_json=$4

    local policy_id=$(echo "$components" | \
        jq -r ".[] | select(.name==\"Allowed Client Scopes\" and .subType==\"${sub_type}\") | .id")
    local parent_id=$(echo "$components" | \
        jq -r ".[] | select(.name==\"Allowed Client Scopes\" and .subType==\"${sub_type}\") | .parentId")

    if [ -z "$policy_id" ] || [ "$policy_id" = "null" ]; then
        echo -e "${RED}FAILED: ${sub_type} 'Allowed Client Scopes' policy not found; refusing to leave DCR scopes unrestricted.${NC}"
        return 1
    fi

    local response=$(_put_allowed_scopes_policy "$token" "$policy_id" "$parent_id" "$sub_type" "$allowed_json")

    if [ "$response" = "400" ] && echo "$allowed_json" | jq -e 'index("openid")' >/dev/null; then
        echo -e "${YELLOW}${sub_type}: Keycloak refused 'openid' in the allowlist; retrying without it.${NC}"
        response=$(_put_allowed_scopes_policy "$token" "$policy_id" "$parent_id" "$sub_type" \
            "$(echo "$allowed_json" | jq -c 'map(select(. != "openid"))')")
    fi

    if [ "$response" = "204" ]; then
        echo -e "${GREEN}OK: ${sub_type} Allowed Client Scopes policy set to the explicit allowlist.${NC}"
    else
        echo -e "${RED}FAILED (${sub_type}): HTTP status ${response}${NC}"
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
        echo -e "${GREEN}OK: Allowed Protocol Mapper Types (${sub_type}) ${action}: no client-supplied mappers.${NC}"
    else
        echo -e "${RED}FAILED: Allowed Protocol Mapper Types (${sub_type}). HTTP status: ${response}${NC}"
        return 1
    fi
}

configure_dcr_allowed_mapper_types() {
    # Replace the allowed types with an empty list on every run, repairing
    # policies that an operator widened after setup.
    local token=$1

    echo ""
    echo "[5/7] Narrowing down DCR 'Allowed Protocol Mapper Types' policy (anonymous + authenticated)..."

    local components realm realm_id
    components=$(_admin_get "$token" \
        "/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy") || return 1
    if ! echo "$components" | jq -e 'type == "array"' >/dev/null 2>&1; then
        echo -e "${RED}FAILED: unexpected client registration policy list for realm '${REALM}'.${NC}"
        return 1
    fi
    realm=$(_admin_get "$token" "/admin/realms/${REALM}") || return 1
    realm_id=$(echo "$realm" | jq -r '.id // empty')
    if [ -z "$realm_id" ]; then
        echo -e "${RED}FAILED: could not read the id of realm '${REALM}'.${NC}"
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

    echo ""
    echo "[6/7] Rejecting the device authorization grant realm-wide..."

    local serverinfo
    serverinfo=$(_admin_get "$token" "/admin/serverinfo") || return 1
    local device_flow=$(echo "$serverinfo" | \
        jq -r '[.features[]? | select(.name=="DEVICE_FLOW") | .enabled | tostring] | first // "unknown"')
    if [ "$device_flow" = "false" ]; then
        echo -e "${GREEN}OK: device authorization grant is disabled server-wide (device-flow feature off).${NC}"
        return 0
    fi
    local has_condition=$(echo "$serverinfo" | \
        jq -r '[.componentTypes["org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"][]? | select(.id=="grant-type")] | length')
    if [ "$has_condition" = "0" ] || [ -z "$has_condition" ]; then
        echo -e "${RED}FAILED: this Keycloak cannot refuse the device grant by client policy (no grant-type condition). Start it with KC_FEATURES_DISABLED=device-flow.${NC}"
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
        echo -e "${GREEN}OK: device authorization grant rejected (profile HTTP $r1, policy HTTP $r2).${NC}"
    else
        echo -e "${RED}FAILED: device-grant rejection policy (profile HTTP $r1, policy HTTP $r2).${NC}"
        return 1
    fi
}

verify_dcr_baseline_policies() {
    local token=$1

    echo ""
    echo "[7/7] Verifying Keycloak's anonymous DCR baseline policies..."

    local components
    components=$(_admin_get "$token" \
        "/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy") || return 1

    local provider missing=""
    for provider in consent-required scope max-clients allowed-protocol-mappers; do
        local found=$(echo "$components" | \
            jq -r --arg p "$provider" '[.[] | select(.subType=="anonymous" and .providerId==$p)] | length')
        if [ "$found" = "0" ]; then
            missing="${missing} ${provider}"
        fi
    done

    if [ -n "$missing" ]; then
        echo -e "${RED}FAILED: anonymous DCR policies missing from realm '${REALM}':${missing}. Restore them under Clients -> Client registration.${NC}"
        return 1
    fi
    echo -e "${GREEN}OK: anonymous DCR baseline policies present.${NC}"
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

# ENABLE_DCR_CONFIG=false: put the anonymous Trusted Hosts policy back to
# Keycloak's default (the registering host must be trusted, and
# no host is), so anonymous registration is refused from everywhere.
disable_anonymous_dcr() {
    local token=$1

    echo ""
    echo "[DCR off] Disabling anonymous DCR ('Trusted Hosts': no trusted hosts)..."

    local components realm policy_id realm_id
    components=$(_admin_get "$token" \
        "/admin/realms/${REALM}/components?type=org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy") || return 1
    policy_id=$(echo "$components" | \
        jq -r '[.[] | select(.providerId=="trusted-hosts" and .subType=="anonymous") | .id] | first // empty')
    realm=$(_admin_get "$token" "/admin/realms/${REALM}") || return 1
    realm_id=$(echo "$realm" | jq -r '.id // empty')
    if [ -z "$realm_id" ]; then
        echo -e "${RED}FAILED: could not read the id of realm '${REALM}'.${NC}"
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
        echo -e "${GREEN}OK: anonymous DCR disabled (HTTP ${response}).${NC}"
    else
        echo -e "${RED}FAILED: disable anonymous DCR. HTTP status: ${response}${NC}"
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

# POST a new registration access token for a client, invalidating the old one.
# The response carries the new token; it is discarded, never printed.
_rotate_registration_access_token() {
    local token=$1
    local uuid=$2
    local display=$3
    local status

    status=$(realm_curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${uuid}/registration-access-token" \
        -H "Authorization: Bearer ${token}")
    if [[ "$status" != 2* ]]; then
        echo -e "${RED}FAILED: invalidate the registration access token of ${display} (HTTP ${status}).${NC}"
        return 1
    fi
}

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
        echo -e "${RED}FAILED: disable client ${display} (HTTP ${status}).${NC}"
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

    DCR_OFF_DISABLED_COUNT=$disabled
    echo -e "${GREEN}OK: disabled ${disabled} client(s) not on the keep-list (built-in, mcp-gateway-web, safe public PKCE IDE, DCR_KEEP_CLIENTS, service-account-only with client-secret); invalidated the registration access token of ${rotated} non-built-in client(s).${NC}"
}

# ENABLE_CIMD_CONFIG!=true: remove the CIMD policy and profile the other setup
# scripts write, preserving every other client policy/profile. The policy goes
# first because it references the profile.
remove_cimd_client_policy() {
    local ENABLE_CIMD_CONFIG=false
    configure_registration_security_policy "$1" || return 1
    _remove_cimd_client_policy "$1"
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

main() {
    if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != --preflight ] && [ "$1" != --restrict-group-claims ] && [ "$1" != --reconcile-registration ]; }; then
        echo "Usage: $0 [--preflight|--restrict-group-claims|--reconcile-registration]" >&2
        echo "  Internal Terraform modes: preflight, registration cleanup, or registration guard and group-claim restrictions." >&2
        return 1
    fi
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
        # Settings the caller passes on the command line (e.g.
        # `ENABLE_DCR_CONFIG=true bash upgrade-realm-for-dcr.sh`, as the docs
        # show) win over .env, the same rule init-keycloak.sh applies. Without
        # this a .env value silently replaces the caller's choice, and DCR-off
        # disables every client not on the keep-list.
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
    else
        echo -e "${RED}Error: export KEYCLOAK_ADMIN_URL and KEYCLOAK_ADMIN_PASSWORD, or provide .env at $ENV_FILE${NC}"
        exit 1
    fi

    KEYCLOAK_URL="${KEYCLOAK_ADMIN_URL-http://localhost:8080}"
    KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN-admin}"
    if [ -z "$KEYCLOAK_URL" ] || [ -z "$KEYCLOAK_ADMIN" ]; then
        echo "KEYCLOAK_ADMIN_URL and KEYCLOAK_ADMIN must not be empty." >&2
        return 1
    fi

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
    refresh_token
    echo -e "${GREEN}Authentication successful.${NC}"

    # Each step gets a fresh token: master-realm admin tokens live 60 seconds.
    preflight_dcr_upgrade "$TOKEN" || exit 1
    if [ "${1:-}" = --preflight ]; then
        return 0
    fi
    reconcile_registration_security "$TOKEN" || exit 1
    if [ "${1:-}" = --reconcile-registration ]; then
        return 0
    fi
    refresh_token
    configure_dcr_allowed_scopes "$TOKEN" || exit 1
    refresh_token
    configure_dcr_allowed_mapper_types "$TOKEN" || exit 1
    _check_realm_group_path_mappers "$KEYCLOAK_URL" "$REALM" "$TOKEN" || exit 1
    if [ "${1:-}" = --restrict-group-claims ]; then
        return 0
    fi
    harden_master_realm "$TOKEN"

    refresh_token
    configure_realm_security_settings "$TOKEN"

    # The `basic` scope mappers run whether or not DCR is on: the
    # aud="mcp-gateway" mapper is what puts the gateway's audience in the web
    # and M2M clients' tokens too. Without it the gateway rejects every token.
    refresh_token
    setup_dcr_groups_mapper "$TOKEN"
    refresh_token
    setup_dcr_audience_mapper "$TOKEN"

    # Runs with DCR on or off, before the DCR steps: no client may hold the
    # clientId that equals the audience every user token carries.
    refresh_token
    reserve_gateway_audience_client "$TOKEN"

    if [ "${ENABLE_DCR_CONFIG:-false}" = "true" ]; then
        refresh_token
        configure_dcr_reject_device_grant "$TOKEN" || exit 1
        refresh_token
        verify_dcr_baseline_policies "$TOKEN" || exit 1
        configure_dcr_trusted_hosts "$TOKEN" || exit 1
    else
        echo -e "${YELLOW}DCR disabled (ENABLE_DCR_CONFIG=${ENABLE_DCR_CONFIG:-false}): turning off anonymous registration. Set ENABLE_DCR_CONFIG=true to enable it.${NC}"
        configure_dcr_default_roles "$TOKEN" false || exit 1
    fi

    # Only the gateway client may use token exchange (runs with DCR on or off).
    refresh_token
    restrict_token_exchange "$TOKEN"

    refresh_token
    update_client_groups_mappers "$TOKEN"
    refresh_token
    disable_web_client_direct_grants "$TOKEN"
    refresh_token
    rotate_first_party_registration_tokens "$TOKEN"

    refresh_token
    if [ "${ENABLE_CIMD_CONFIG:-false}" = "true" ]; then
        configure_cimd_client_policy "$TOKEN"
    fi

    echo ""
    echo -e "${GREEN}Realm configuration complete.${NC}"
    echo ""
    echo "Applied on every run:"
    echo "  - Master realm: brute-force protection, login and admin event logging"
    echo "  - Realm '${REALM}': brute-force protection, password policy, login and admin event logging, 30-day offline-session cap"
    echo "  - Groups protocol mapper on 'basic' client-scope emits mcpgw_group_paths with full.path=true"
    echo "  - Audience mapper on 'basic' client-scope, aud=mcp-gateway"
    echo "  - Token exchange restricted to clients with the 'token-exchange' client role"
    echo "  - Legacy client-level groups mappers migrated to mcpgw_group_paths with full.path=true"
    echo "  - Password grant (direct access grants) disabled on mcp-gateway-web"
    echo "  - Registration access tokens rotated on: ${FIRST_PARTY_CLIENTS[*]}"
    echo "  - clientId '${GATEWAY_AUDIENCE_CLIENT_ID}' held by a disabled placeholder client"
    if [ "${ENABLE_DCR_CONFIG:-false}" = "true" ]; then
        echo ""
        echo "DCR configuration:"
        echo "  - Allowed Client Scopes policy set to exactly: ${DCR_ALLOWED_SCOPES:-$DCR_DEFAULT_ALLOWED_SCOPES}"
        echo "  - Trusted Hosts policy: IP check OFF, URI check ON, 'localhost' trusted"
        echo "  - Allowed Protocol Mapper Types policy emptied (no client-supplied mappers)"
        echo "  - Device authorization grant refused (server feature off, or realm client policy)"
        echo "  - Consent Required / Full Scope Disabled / Max Clients Limit / Allowed Protocol Mapper Types verified present"
        echo ""
        echo "Check the output above to see which steps actually ran for this realm."
        echo "Existing DCR'd clients will pick up the policy changes immediately."
    else
        echo ""
        echo "DCR disabled: anonymous registration turned off; scope and mapper restrictions applied;"
        echo "  ${DCR_OFF_DISABLED_COUNT:-0} client(s) not on the keep-list disabled (set DCR_KEEP_CLIENTS to keep more);"
        echo "  registration access tokens of all other non-built-in clients invalidated."
    fi
}

main "$@"
