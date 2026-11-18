#!/bin/bash
# Sourced by repository setup scripts before changing any groups mapper.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/credential-io.sh"
if [ -z "${CREDENTIAL_IO_DIR:-}" ]; then
    credential_io_init || return 1
fi

# Cache in the private per-run directory so command substitutions share it.
# Keep less than the advertised lifetime, never more than twenty seconds.
_account_admin_token() {
    local url=${1%/} user=$2 password=$3 now expires=0 token="" response ttl
    local cache="$CREDENTIAL_IO_DIR/account-token"
    now=$(date +%s) || return 1
    if [ -f "$cache" ]; then
        { IFS= read -r expires; IFS= read -r token; } < "$cache" || return 1
    fi
    if [ -n "$token" ] && [[ "$expires" =~ ^[0-9]+$ ]] && [ "$now" -lt "$expires" ]; then
        printf '%s' "$token"
        return 0
    fi
    response=$(printf '%s' "$password" | credential_curl -fsS --connect-timeout 10 --max-time 30 \
        -X POST "$url/realms/master/protocol/openid-connect/token" \
        --data-urlencode "username=$user" --data-urlencode "password@-" \
        --data-urlencode grant_type=password --data-urlencode client_id=admin-cli) || return 1
    token=$(jq -er '.access_token | select(type == "string" and length > 0)' <<< "$response") || return 1
    ttl=$(jq -er '.expires_in | select(type == "number" and . > 1) | floor | . - 1
        | if . > 20 then 20 else . end' <<< "$response") || return 1
    printf '%s\n%s\n' "$((now + ttl))" "$token" | credential_write "$cache" || return 1
    printf '%s' "$token"
}

_keycloak_policy_version_supported() {
    local info version
    info=$(realm_curl -fsS --connect-timeout 10 --max-time 30 \
        -H "Authorization: Bearer $2" "${1%/}/admin/serverinfo") || return 1
    version=$(jq -er '.systemInfo.version | select(type == "string")' <<< "$info") || return 1
    jq -n --arg version "$version" 'try ($version | capture(
        "^(?<major>[0-9]+)\\.(?<minor>[0-9]+)\\.(?<patch>[0-9]+)(?:$|[-+])")
        | [.major, .minor, .patch] | map(tonumber) | . >= [26,8,2]) catch false'
}

_require_supported_keycloak() {
    local supported
    supported=$(_keycloak_policy_version_supported "$1" "$2") || return 1
    if [ "$supported" != true ]; then
        echo "This policy requires Keycloak 26.8.2 or newer; upgrade before enabling it." >&2
        return 1
    fi
}

_preflight_token_exchange_support() {
    local info
    info=$(realm_curl -fsS -H "Authorization: Bearer $2" "${1%/}/admin/serverinfo") || return 1
    if jq -e 'any(.features[]?;
        .name == "TOKEN_EXCHANGE_STANDARD_V2" and .enabled == true)' <<< "$info" >/dev/null; then
        _require_supported_keycloak "$1" "$2" || return 1
        local condition="org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"
        if ! jq -e --arg condition "$condition" \
            'any(.componentTypes[$condition][]?; .id == "grant-type")' <<< "$info" >/dev/null; then
            echo "Standard token exchange is enabled without the required policy condition." >&2
            echo "Upgrade Keycloak before running realm setup; no realm settings changed." >&2
            return 1
        fi
    fi
}

_preflight_cimd_support() {
    local info
    _require_supported_keycloak "$1" "$2" || return 1
    info=$(realm_curl -fsS --connect-timeout 10 --max-time 30 \
        -H "Authorization: Bearer $2" "${1%/}/admin/serverinfo") || return 1
    if ! jq -e '
        any(.features[]?; .name == "CIMD" and .enabled == true) and
        any(.componentTypes["org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"][]?;
            .id == "client-id-uri") and
        any(.componentTypes["org.keycloak.services.clientpolicy.executor.ClientPolicyExecutorProvider"][]?;
            .id == "client-id-metadata-document")' <<< "$info" >/dev/null; then
        echo "CIMD requires stock Keycloak 26.8.2+ with KC_FEATURES=cimd and the native metadata-document executor." >&2
        return 1
    fi
}

# Keep credentials out of subprocess arguments, including long paginated scans.
realm_curl() {
    local option value
    local args=()
    while [ "$#" -gt 0 ]; do
        option=$1; shift
        case "$option" in
            -H|--header)
                [ "$#" -gt 0 ] || return 2
                value=$1; shift
                if [[ "$value" == "Authorization: Bearer "* ]] && declare -F _realm_fresh_token >/dev/null; then
                    value=$(_realm_fresh_token) || return 1
                    [ -n "$value" ] || return 1
                    value="Authorization: Bearer $value"
                fi
                args+=("$option" "$value") ;;
            *) args+=("$option") ;;
        esac
    done
    credential_curl "${args[@]}"
}

_check_group_path_mapper_models() {
    if ! jq -e 'type == "array" and all(.[];
        if ((.config["claim.name"] // "") | split(".")[0]) == "mcpgw_group_paths" then
            .protocol == "openid-connect" and
            .protocolMapper == "oidc-group-membership-mapper" and
            .config["claim.name"] == "mcpgw_group_paths" and
            .config["full.path"] == "true"
        else true end)' <<< "$1" >/dev/null; then
        echo "The reserved mcpgw_group_paths claim has an unsafe mapper. Review local and inherited mappers before migration." >&2
        return 1
    fi
}

_check_group_path_mapper_collection() {
    local models
    models=$(realm_curl -fsS --connect-timeout 10 --max-time 30 \
        -H "Authorization: Bearer $1" "$2/protocol-mappers/models") || return 1
    _check_group_path_mapper_models "$models"
}

_check_client_group_path_mappers() {
    local url=${1%/} realm token=$3 uuid=$4 kind scopes ids id
    realm=$(jq -rn --arg value "$2" '$value | @uri') || return 1
    uuid=$(jq -rn --arg value "$uuid" '$value | @uri') || return 1
    local base="$url/admin/realms/$realm"
    _check_group_path_mapper_collection "$token" "$base/clients/$uuid" || return 1
    for kind in default optional; do
        scopes=$(realm_curl -fsS --connect-timeout 10 --max-time 30 \
            -H "Authorization: Bearer $token" "$base/clients/$uuid/$kind-client-scopes") || return 1
        ids=$(jq -er 'if type == "array" and all(.[]; .id | type == "string" and length > 0)
            then [.[] | .id | @uri] | join("\n") else error("invalid client scopes") end' <<< "$scopes") || return 1
        while IFS= read -r id; do
            [ -n "$id" ] || continue
            _check_group_path_mapper_collection "$token" "$base/client-scopes/$id" || return 1
        done <<< "$ids"
    done
}

_check_realm_group_path_mappers() {
    local url=${1%/} realm token=$3 scopes ids id page first=0 count
    realm=$(jq -rn --arg value "$2" '$value | @uri') || return 1
    local base="$url/admin/realms/$realm"
    scopes=$(realm_curl -fsS --connect-timeout 10 --max-time 30 \
        -H "Authorization: Bearer $token" "$base/client-scopes") || return 1
    ids=$(jq -er 'if type == "array" and all(.[]; .id | type == "string" and length > 0)
        then [.[] | .id | @uri] | join("\n") else error("invalid realm scopes") end' <<< "$scopes") || return 1
    while IFS= read -r id; do
        [ -n "$id" ] || continue
        _check_group_path_mapper_collection "$token" "$base/client-scopes/$id" || return 1
    done <<< "$ids"
    while :; do
        page=$(realm_curl -fsS --connect-timeout 10 --max-time 30 \
            -H "Authorization: Bearer $token" "$base/clients?first=$first&max=100") || return 1
        ids=$(jq -er 'if type == "array" and all(.[]; .id | type == "string" and length > 0)
            then [.[] | .id | @uri] | join("\n") else error("invalid realm clients") end' <<< "$page") || return 1
        while IFS= read -r id; do
            [ -n "$id" ] || continue
            _check_group_path_mapper_collection "$token" "$base/clients/$id" || return 1
        done <<< "$ids"
        count=$(jq 'length' <<< "$page") || return 1
        [ "$count" -eq 100 ] || break
        first=$((first + count))
    done
}

_group_paths_get() {
    local token=$1 url=$2 response status
    response=$(realm_curl -sS --connect-timeout 10 --max-time 30 -w $'\n%{http_code}' \
        -H "Authorization: Bearer ${token}" "$url") || return 1
    status=${response##*$'\n'}
    if [[ "$status" != 2[0-9][0-9] ]]; then
        printf 'Could not enumerate Keycloak groups (HTTP %s).\n' "$status" >&2
        return 1
    fi
    printf '%s' "${response%$'\n'*}"
}

# A count identifies paginated child collections on newer Keycloak versions.
# Older versions embed subGroups instead; walk those records recursively.
_group_paths_entries() {
    local admin_url=$1 realm=$2 token=$3 page=$4 parent=$5
    local entries entry id child_path count children
    if ! jq -e 'type == "array" and all(.[];
        (.id | type == "string" and length > 0) and
        (.name | type == "string" and length > 0) and
        ((has("subGroupCount") | not) or (.subGroupCount | type == "number" and . >= 0 and . == floor)) and
        ((has("subGroups") | not) or (.subGroups | type == "array")))' \
        >/dev/null 2>&1 <<< "$page"; then
        echo "Could not enumerate Keycloak groups: invalid group representation." >&2
        return 1
    fi
    jq -c --arg parent "$parent" '.[] | {id, name, path: ($parent + "/" + .name)}' <<< "$page" || return 1
    entries=$(jq -c '.[] | select(
        if has("subGroupCount") then .subGroupCount > 0
        elif has("subGroups") then (.subGroups | length) > 0
        else true end)' <<< "$page") || return 1
    while IFS= read -r entry; do
        [ -z "$entry" ] && continue
        id=$(jq -r '.id | @uri' <<< "$entry") || return 1
        # Keep trailing newlines in names: command substitution strips them.
        child_path=$(jq -r --arg parent "$parent" '$parent + "/" + .name + "."' <<< "$entry") || return 1
        child_path=${child_path%.}
        count=$(jq -r '.subGroupCount // empty' <<< "$entry") || return 1
        if [ -n "$count" ]; then
            _group_paths_walk "$admin_url" "$realm" "$token" \
                "/admin/realms/${realm}/groups/${id}/children" "$child_path" || return 1
        elif jq -e 'has("subGroups")' >/dev/null <<< "$entry"; then
            children=$(jq -c '.subGroups' <<< "$entry") || return 1
            _group_paths_entries "$admin_url" "$realm" "$token" "$children" "$child_path" || return 1
        else
            # Missing child metadata is not proof that this is a leaf.
            _group_paths_walk "$admin_url" "$realm" "$token" \
                "/admin/realms/${realm}/groups/${id}/children" "$child_path" || return 1
        fi
    done <<< "$entries"
}

_group_paths_walk() {
    local admin_url=$1 realm=$2 token=$3 path=$4 parent=$5
    local first=0 page count page_size=100
    while :; do
        page=$(_group_paths_get "$token" \
            "${admin_url}${path}?first=${first}&max=${page_size}&briefRepresentation=false") || return 1
        _group_paths_entries "$admin_url" "$realm" "$token" "$page" "$parent" || return 1
        count=$(jq 'length' <<< "$page") || return 1
        [ "$count" -lt "$page_size" ] && break
        first=$((first + count))
    done
}

# Full paths are ambiguous only when distinct groups emit the same path.
# A legacy slash-bearing name without a collision remains supported.
_check_group_paths() {
    local admin_url=${1%/} realm=$2 token=$3 groups collisions slashed
    realm=$(jq -rn --arg realm "$realm" '$realm | @uri') || return 1
    if ! groups=$(_group_paths_walk "$admin_url" "$realm" "$token" "/admin/realms/${realm}/groups" ""); then
        echo "Could not read every Keycloak group; refusing to change groups mappers." >&2
        return 1
    fi
    collisions=$(jq -s '[group_by(.path)[] | select((unique_by(.id) | length) > 1)] | length' <<< "$groups") || return 1
    if [ "$collisions" -gt 0 ]; then
        echo "Different Keycloak groups share a full path; refusing to change groups mappers. Rename the conflicting groups and rerun setup." >&2
        return 1
    fi
    slashed=$(jq -s '[.[] | select(.name | contains("/"))] | length' <<< "$groups") || return 1
    if [ "$slashed" -gt 0 ]; then
        echo "Warning: ${slashed} Keycloak group names contain '/' but have no full-path collision. Consider renaming them." >&2
    fi
}

# Keep the registration allowlist independent of first-party client scope grants.
_registration_allowed_scopes() {
    local names=() scope forbidden
    read -r -a names <<< "${DCR_ALLOWED_SCOPES:-openid basic profile email roles web-origins acr offline_access}"
    for scope in "${names[@]}"; do
        if [[ "$scope" == */* ]]; then
            echo "DCR_ALLOWED_SCOPES must not include gateway privilege scope $scope." >&2
            return 1
        fi
        for forbidden in mcp-registry-admin registry-admins kc-admin-api; do
            if [ "$scope" = "$forbidden" ]; then
                echo "DCR_ALLOWED_SCOPES must not include privilege scope $scope." >&2
                return 1
            fi
        done
    done
    printf '%s\n' "${names[@]}" | jq -Rsc 'split("\n") | map(select(length > 0)) | unique'
}

_registration_trusted_clients() {
    local extra="${REGISTRATION_TRUSTED_CLIENT_IDS:-[]}"
    jq -cen --argjson extra "$extra" 'if ($extra | type == "array" and all(.[]; type == "string" and length > 0)) then
        (["account", "account-console", "admin-cli", "admin-permissions", "broker", "realm-management",
          "security-admin-console", "mcp-gateway-web", "mcp-gateway-m2m"] + $extra | unique)
        else error("REGISTRATION_TRUSTED_CLIENT_IDS must be a JSON array of exact reviewed client IDs") end'
}

_cimd_domains() {
    local domains="${CIMD_TRUSTED_DOMAINS:-[\"vscode.dev\",\"code.visualstudio.com\",\"claude.ai\",\"127.0.0.1\",\"localhost\"]}"
    jq -ce 'if type == "array" and all(.[]; type == "string" and length > 0)
        then map(ascii_downcase) | unique else error("CIMD_TRUSTED_DOMAINS must be an array of domain strings") end' <<< "$domains"
}

_cimd_client_id_domains() {
    local domains
    domains=$(_cimd_domains) || return 1
    jq -ce 'map(select(test("^(localhost|\\[?::1\\]?|127\\.[0-9.]+)$"; "i") | not))
        | if length > 0 then . else error("CIMD_TRUSTED_DOMAINS lists only loopback hosts; add vendor domains") end' <<< "$domains"
}

# The native format accepts arbitrary attributes, including CIMD's cache receipt.
# Disable that provider at server startup; standard OIDC registration stays enabled.
_require_oidc_registration_only() {
    local info
    info=$(_admin_get "$1" /admin/serverinfo) || return 1
    if ! jq -e '.providers["client-registration"].providers | type == "object" and
        has("openid-connect") and (has("default") | not)' <<< "$info" >/dev/null; then
        echo "Disable native DCR with KC_SPI_CLIENT_REGISTRATION__DEFAULT__ENABLED=false and restart Keycloak before enabling DCR/CIMD. Keep the openid-connect provider enabled." >&2
        return 1
    fi
}

# Cached entries from the unrestricted native provider cannot prove provenance.
# Recreate them through verified metadata once; never re-enable disabled records.
_reset_untrusted_cimd_cache() {
    local token=$1 realm clients entries entry id status attributes
    _require_oidc_registration_only "$token" || return 1
    realm=$(_admin_get "$token" "/admin/realms/$REALM") || return 1
    [ "$(jq -r '.attributes["mcpgw.cimd.provenance.v1"] // "false"' <<< "$realm")" != true ] || return 0
    clients=$(_list_all_clients "$token") || return 1
    entries=$(jq -c '.[] | select((.attributes // {}) | has("cimd.cache.expiry.time.in.sec")) | {id, clientId, enabled}' <<< "$clients") || return 1
    if [ -n "$entries" ]; then
        echo "CIMD cache reset inventory (old registrations will be removed):"
        jq -sr '.[] | [.id, (.clientId | gsub("[^A-Za-z0-9._:/@-]"; "?"))] | @tsv' <<< "$entries"
        if [ "${DCR_SWEEP_ACKNOWLEDGED:-false}" != true ]; then
            echo "Review the one-time CIMD cache removal and set DCR_SWEEP_ACKNOWLEDGED=true. Old cache attributes could have been supplied by native DCR." >&2
            return 1
        fi
        # Stop new metadata creation while removing the untrusted cache records.
        _registration_policy_write "$token" policies cimd-vendors-policy null || return 1
        while IFS= read -r entry; do
            id=$(jq -er '.id | @uri' <<< "$entry") || return 1
            if [ "$(jq -r '.enabled' <<< "$entry")" = false ]; then
                echo "Disabled CIMD client $id requires administrator review before cache reset; remove it explicitly or retain CIMD disabled." >&2
                return 1
            fi
            status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X DELETE \
                "$KEYCLOAK_URL/admin/realms/$REALM/clients/$id" -H "Authorization: Bearer $token") || return 1
            [ "$status" = 204 ] || return 1
            status=$(realm_curl -sS -o /dev/null -w '%{http_code}' \
                "$KEYCLOAK_URL/admin/realms/$REALM/clients/$id" -H "Authorization: Bearer $token") || return 1
            [ "$status" = 404 ] || return 1
        done <<< "$entries"
    fi
    attributes=$(jq -c '{attributes: ((.attributes // {}) + {"mcpgw.cimd.provenance.v1": "true"})}' <<< "$realm") || return 1
    status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X PUT \
        "$KEYCLOAK_URL/admin/realms/$REALM" -H "Authorization: Bearer $token" \
        -H 'Content-Type: application/json' -d "$attributes") || return 1
    [ "$status" = 204 ] || return 1
    realm=$(_admin_get "$token" "/admin/realms/$REALM") || return 1
    jq -e '.attributes["mcpgw.cimd.provenance.v1"] == "true"' <<< "$realm" >/dev/null
}

_configure_public_pkce_policy() {
    local token=$1 profile policy
    profile='{"name":"mcp-public-pkce","executors":[{"executor":"pkce-enforcer","configuration":{"auto-configure":true}}]}'
    policy='{"name":"mcp-public-pkce-policy","enabled":true,"mode":"STRICT","conditions":[{"condition":"client-access-type","configuration":{"type":["public"]}},{"condition":"client-type","configuration":{"protocol":"openid-connect"}}],"profiles":["mcp-public-pkce"]}'
    _registration_policy_write "$token" profiles mcp-public-pkce "$profile" || return 1
    _registration_policy_write "$token" policies mcp-public-pkce-policy "$policy"
}

_registration_security_available() {
    local info
    info=$(_admin_get "$1" /admin/serverinfo) || return 1
    jq -r '(.componentTypes["org.keycloak.services.clientpolicy.condition.ClientPolicyConditionProvider"] // [] | map(.id)) as $conditions
        | ($conditions | index("client-scopes") != null and index("client-roles") != null
            and index("client-access-type") != null and index("client-type") != null)
        and ((.componentTypes["org.keycloak.services.clientpolicy.executor.ClientPolicyExecutorProvider"] // [] | map(.id)) as $executors
            | ($executors | index("reject-request") != null and index("pkce-enforcer") != null))
        and any(.componentTypes["org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy"][]?;
            .id == "default-roles")' <<< "$info"
}

_preflight_registration_security() {
    local token=$1 available response status
    _registration_allowed_scopes >/dev/null || return 1
    _registration_trusted_clients >/dev/null || return 1
    if [ "${ENABLE_CIMD_CONFIG:-false}" = true ]; then
        _cimd_client_id_domains >/dev/null || return 1
        _preflight_cimd_support "$KEYCLOAK_URL" "$token" || return 1
    fi
    if [ "${ENABLE_DCR_CONFIG:-false}" = true ] || [ "${ENABLE_CIMD_CONFIG:-false}" = true ]; then
        _require_oidc_registration_only "$token" || return 1
        _require_supported_keycloak "$KEYCLOAK_URL" "$token" || return 1
        available=$(_registration_security_available "$token") || return 1
        if [ "$available" != true ]; then
            echo "DCR/CIMD requires native scope, role, public-client, protocol, PKCE and default-roles policy providers." >&2
            return 1
        fi
        # A missing realm is normal on first install. Inspect again after creation.
        response=$(realm_curl -sS -w $'\n%{http_code}' -H "Authorization: Bearer $token" \
            "$KEYCLOAK_URL/admin/realms/$REALM/client-scopes") || return 1
        status=${response##*$'\n'}
        if [ "$status" != 404 ]; then
            [ "$status" = 200 ] || return 1
            if ! jq -e 'type == "array" and all(.[];
                (.attributes["is.parameterized.scope"] // "false" | tostring) != "true" and
                (.attributes["is.dynamic.scope"] // "false" | tostring) != "true")' <<< "${response%$'\n'*}" >/dev/null; then
                echo "DCR/CIMD scope enforcement requires ordinary client scopes. Disable parameterized/dynamic scopes in this realm before enabling registration; native client-scopes conditions cannot match their values." >&2
                return 1
            fi
        fi
    fi
}

# Merge and read back only the owned entry. Preserve every unrelated profile/policy.
_registration_policy_write() {
    local token=$1 collection=$2 name=$3 entry=$4 current body status actual
    current=$(_admin_get "$token" "/admin/realms/$REALM/client-policies/$collection") || return 1
    body=$(jq -ce --arg key "$collection" --arg name "$name" --argjson entry "$entry" '
        if (.[$key] | type) != "array" then error("invalid policy collection") else
        {($key): ((.[$key] | map(select(.name != $name))) + (if $entry == null then [] else [$entry] end))} end' <<< "$current") || return 1
    status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X PUT \
        "$KEYCLOAK_URL/admin/realms/$REALM/client-policies/$collection" \
        -H "Authorization: Bearer $token" -H 'Content-Type: application/json' -d "$body") || return 1
    [[ "$status" == 2[0-9][0-9] ]] || { echo "Could not write $name: HTTP $status" >&2; return 1; }
    actual=$(_admin_get "$token" "/admin/realms/$REALM/client-policies/$collection") || return 1
    if ! jq -e --arg key "$collection" --arg name "$name" --argjson expected "$entry" '
        [.[$key][]? | select(.name == $name)] | if $expected == null then length == 0 else
        length == 1 and (.[0] | with_entries(select(.key as $k | $expected | has($k)))) == $expected end' <<< "$actual" >/dev/null; then
        echo "Required policy readback failed: $name. Registration must stay closed." >&2
        return 1
    fi
}

# ClientRolesCondition checks roles DEFINED on the client, not user role mappings.
# Only the Admin API creates this role; default-roles registration policy rejects
# attempts to create it via native DCR. Never grant the exception to a CIMD client.
_reconcile_registration_trust_roles() {
    local token=$1 clients trusted entries entry uuid permitted response status role
    clients=$(_list_all_clients "$token") || return 1
    trusted=$(_registration_trusted_clients) || return 1
    entries=$(jq -c --argjson trusted "$trusted" '.[] | .clientId as $id
        | select(["account", "account-console", "admin-cli", "admin-permissions", "broker",
            "realm-management", "security-admin-console"] | index($id) | not)
        | {id, trusted: (($trusted | index($id) != null) and
            (((.attributes // {}) | has("cimd.cache.expiry.time.in.sec")) | not))}' <<< "$clients") || return 1
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        uuid=$(jq -er '.id | @uri' <<< "$entry") || return 1
        permitted=$(jq -r '.trusted' <<< "$entry") || return 1
        role="/admin/realms/$REALM/clients/$uuid/roles/mcp-registration-trusted"
        response=$(realm_curl -sS -w $'\n%{http_code}' -H "Authorization: Bearer $token" "$KEYCLOAK_URL$role") || return 1
        status=${response##*$'\n'}
        if [ "$permitted" = true ]; then
            if [ "$status" = 404 ]; then
                status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X POST \
                    "$KEYCLOAK_URL/admin/realms/$REALM/clients/$uuid/roles" \
                    -H "Authorization: Bearer $token" -H 'Content-Type: application/json' \
                    -d '{"name":"mcp-registration-trusted","description":"Operator-reviewed first-party scope exception"}') || return 1
                [ "$status" = 201 ] || { echo "Could not create registration trust role for client $uuid: HTTP $status" >&2; return 1; }
            else
                [ "$status" = 200 ] || { echo "Could not read registration trust role for client $uuid: HTTP $status" >&2; return 1; }
            fi
            response=$(_admin_get "$token" "$role") || return 1
            jq -e '.name == "mcp-registration-trusted" and .clientRole == true and .composite != true' <<< "$response" >/dev/null || { echo "Registration trust role readback failed for client $uuid" >&2; return 1; }
        else
            if [ "$status" = 200 ]; then
                status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X DELETE \
                    "$KEYCLOAK_URL$role" -H "Authorization: Bearer $token") || return 1
                [ "$status" = 204 ] || { echo "Could not remove registration trust role for client $uuid: HTTP $status" >&2; return 1; }
                status=$(realm_curl -sS -o /dev/null -w '%{http_code}' \
                    "$KEYCLOAK_URL$role" -H "Authorization: Bearer $token") || return 1
            fi
            [ "$status" = 404 ] || { echo "Registration trust role removal readback failed for client $uuid: HTTP $status" >&2; return 1; }
        fi
    done <<< "$entries"
}

# These realm associations affect NEW clients only; existing first-party client
# associations are preserved. Both sets matter even when DCR omits scope fields.
reconcile_registration_default_scopes() {
    local token=$1 allowed kind scopes removals scope status current
    allowed=$(_registration_allowed_scopes) || return 1
    for kind in default optional; do
        scopes=$(_admin_get "$token" "/admin/realms/$REALM/default-$kind-client-scopes") || return 1
        removals=$(jq -er --argjson allowed "$allowed" 'if type != "array" then error("invalid realm defaults") else
            [.[] | .name as $name | select($allowed | index($name) | not) | .id | @uri] | join("\n") end' <<< "$scopes") || return 1
        while IFS= read -r scope; do
            [ -n "$scope" ] || continue
            status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X DELETE \
                "$KEYCLOAK_URL/admin/realms/$REALM/default-$kind-client-scopes/$scope" \
                -H "Authorization: Bearer $token") || return 1
            [ "$status" = 204 ] || return 1
        done <<< "$removals"
        current=$(_admin_get "$token" "/admin/realms/$REALM/default-$kind-client-scopes") || return 1
        jq -e --argjson allowed "$allowed" 'type == "array" and all(.[]; .name as $name | $allowed | index($name))' <<< "$current" >/dev/null || return 1
    done
}

configure_registration_security_policy() {
    local token=$1 available allowed scopes forbidden profile policy
    _preflight_registration_security "$token" || return 1
    available=$(_registration_security_available "$token") || return 1
    if [ "$available" != true ]; then
        echo "Registration is off; applying cleanup without native scope policy support."
        reconcile_registration_default_scopes "$token" || return 1
        return 0
    fi
    allowed=$(_registration_allowed_scopes) || return 1
    scopes=$(_admin_get "$token" "/admin/realms/$REALM/client-scopes") || return 1
    forbidden=$(jq -ce --argjson allowed "$allowed" 'if type != "array" then error("invalid realm scopes") else
        [.[] | .name | select(. as $name | $allowed | index($name) | not)] | unique end' <<< "$scopes") || return 1
    configure_dcr_default_roles "$token" || return 1
    _configure_public_pkce_policy "$token" || return 1
    _reconcile_registration_trust_roles "$token" || return 1
    profile='{"name":"mcp-registration-scope-guard","executors":[{"executor":"reject-request","configuration":{}}]}'
    # STRICT is essential: on registration/pre-auth events client-scopes abstains.
    # ScopeParameterContext covers authorization, code/refresh responses, password,
    # and service-account requests. Re-run setup whenever the scope catalog changes.
    policy=$(jq -nc --argjson scopes "$forbidden" '{name: "mcp-registration-scope-guard-policy", enabled: true, mode: "STRICT",
        conditions: [{condition: "client-scopes", configuration: {type: "Any", scopes: $scopes}},
            {condition: "client-roles", configuration: {roles: ["mcp-registration-trusted"], "is-negative-logic": true}}],
        profiles: ["mcp-registration-scope-guard"]}') || return 1
    _registration_policy_write "$token" profiles mcp-registration-scope-guard "$profile" || return 1
    _registration_policy_write "$token" policies mcp-registration-scope-guard-policy "$policy" || return 1
    reconcile_registration_default_scopes "$token"
}

# Native CIMD leaves the cache-expiry attribute. An HTTPS ID alone is not provenance.
_cimd_sweep_targets() {
    local clients domains='[]'
    clients=$(_list_all_clients "$1") || return 1
    if [ "${ENABLE_CIMD_CONFIG:-false}" = true ]; then
        domains=$(_cimd_client_id_domains) || return 1
    fi
    jq -r --argjson domains "$domains" --argjson enabled "${ENABLE_CIMD_CONFIG:-false}" '
        .[] | select((.attributes // {}) | has("cimd.cache.expiry.time.in.sec"))
        | .clientId as $client
        | ((try ($client | capture("^https://(?<host>[^/:?#@]+)(?::[0-9]+)?(?:[/?#]|$)"; "i").host | ascii_downcase) catch "") // "") as $host
        | select(($enabled | not) or ($domains | any(. as $domain |
            if startswith("*.") then ($host == ($domain | ltrimstr("*.")) or ($host | endswith($domain[1:])))
            else $host == $domain end) | not))
        | [.id, ($client | gsub("[^A-Za-z0-9._:/@-]"; "?")), (.enabled != false | tostring)] | @tsv' <<< "$clients"
}

_confirm_cimd_sweep() {
    local targets=$1
    [ -n "$targets" ] || return 0
    echo "CIMD revocation inventory (UUID, client ID, currently enabled):"
    printf '%s\n' "$targets"
    if grep -q $'\ttrue$' <<< "$targets" && [ "${DCR_SWEEP_ACKNOWLEDGED:-false}" != true ]; then
        echo "Review CIMD revocations and set DCR_SWEEP_ACKNOWLEDGED=true (Helm: dcr.sweepAcknowledged=true). DCR_KEEP_CLIENTS does not override CIMD off or withdrawn host trust." >&2
        return 1
    fi
}

reconcile_cimd_clients() {
    local token=$1 targets uuid display enabled status now current body
    targets=$(_cimd_sweep_targets "$token") || return 1
    _confirm_cimd_sweep "$targets" || return 1
    now=$(date +%s) || return 1
    while IFS=$'\t' read -r uuid display enabled; do
        [ -n "$uuid" ] || continue
        uuid=$(jq -rn --arg id "$uuid" '$id | @uri') || return 1
        current=$(_admin_get "$token" "/admin/realms/$REALM/clients/$uuid") || return 1
        # Keycloak exposes user-session deletion, not isolated client-session deletion.
        # A client not-before cutoff revokes its old tokens without logging out other clients.
        body=$(jq -nc --argjson now "$now" --argjson current "$current" \
            '{enabled: false, notBefore: ([$now, ($current.notBefore // 0)] | max)}') || return 1
        status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X PUT \
            "$KEYCLOAK_URL/admin/realms/$REALM/clients/$uuid" \
            -H "Authorization: Bearer $token" -H 'Content-Type: application/json' -d "$body") || return 1
        [ "$status" = 204 ] || { echo "CIMD revocation failed for $display: HTTP $status" >&2; return 1; }
        current=$(_admin_get "$token" "/admin/realms/$REALM/clients/$uuid") || return 1
        jq -e --argjson cutoff "$now" '.enabled == false and .notBefore >= $cutoff' <<< "$current" >/dev/null || return 1
    done <<< "$targets"
}

reconcile_registered_client_scopes() {
    local token=$1 clients trusted allowed entries entry uuid display provenance kind scopes removals scope status current
    clients=$(_list_all_clients "$token") || return 1
    trusted=$(_registration_trusted_clients) || return 1
    allowed=$(_registration_allowed_scopes) || return 1
    # Admin GET omits Keycloak's registration token. Unmarked legacy clients need review.
    entries=$(jq -c --argjson trusted "$trusted" '.[] | .clientId as $id
        | if ((.attributes // {}) | has("cimd.cache.expiry.time.in.sec")) then . + {provenance: "registered"}
          elif $id == "mcp-gateway" and .enabled == false and .bearerOnly == true then . + {provenance: "registered"}
          elif $trusted | index($id) then . + {provenance: "trusted"}
          else . + {provenance: "unknown"} end
        | select(.provenance != "trusted") | {id, clientId, provenance}' <<< "$clients") || return 1
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        uuid=$(jq -er '.id | @uri' <<< "$entry") || return 1
        display=$(jq -r '.clientId | gsub("[^A-Za-z0-9._:/@-]"; "?")' <<< "$entry") || return 1
        provenance=$(jq -r '.provenance' <<< "$entry") || return 1
        for kind in default optional; do
            scopes=$(_admin_get "$token" "/admin/realms/$REALM/clients/$uuid/$kind-client-scopes") || return 1
            removals=$(jq -er --argjson allowed "$allowed" 'if type != "array" then error("invalid client scopes") else
                [.[] | .name as $name | select($allowed | index($name) | not) | .id | @uri] | join("\n") end' <<< "$scopes") || return 1
            [ -n "$removals" ] || continue
            if [ "$provenance" = unknown ]; then
                echo "Scope reconciliation cannot establish registration provenance for $display (UUID $uuid). Review its $kind scopes and add this exact ID to REGISTRATION_TRUSTED_CLIENT_IDS only if first-party, or remove disallowed scopes in the admin console." >&2
                return 1
            fi
            while IFS= read -r scope; do
                status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -X DELETE \
                    "$KEYCLOAK_URL/admin/realms/$REALM/clients/$uuid/$kind-client-scopes/$scope" \
                    -H "Authorization: Bearer $token") || return 1
                [ "$status" = 204 ] || return 1
            done <<< "$removals"
            current=$(_admin_get "$token" "/admin/realms/$REALM/clients/$uuid/$kind-client-scopes") || return 1
            jq -e --argjson allowed "$allowed" 'type == "array" and all(.[]; .name as $name | $allowed | index($name))' <<< "$current" >/dev/null || return 1
        done
    done <<< "$entries"
}

_configure_cimd_client_policy() {
    local token=$1 domains hosts profile policy
    _preflight_cimd_support "$KEYCLOAK_URL" "$token" || return 1
    domains=$(_cimd_domains) || return 1
    hosts=$(_cimd_client_id_domains) || return 1
    _reset_untrusted_cimd_cache "$token" || return 1
    configure_registration_security_policy "$token" || return 1
    reconcile_cimd_clients "$token" || return 1
    reconcile_registered_client_scopes "$token" || return 1
    profile=$(jq -nc --argjson domains "$domains" '{name: "cimd-vendors",
        executors: [{executor: "client-id-metadata-document", configuration: {
            "cimd-allow-permitted-domains": $domains}}]}') || return 1
    policy=$(jq -nc --argjson hosts "$hosts" '{name: "cimd-vendors-policy", enabled: true, mode: "STRICT",
        conditions: [{condition: "client-id-uri", configuration: {"client-id-uri-scheme": ["https"],
            "client-id-uri-allow-permitted-domains": $hosts}}], profiles: ["cimd-vendors"]}') || return 1
    _registration_policy_write "$token" profiles cimd-vendors "$profile" || return 1
    _registration_policy_write "$token" policies cimd-vendors-policy "$policy"
}

_remove_cimd_client_policy() {
    local token=$1
    reconcile_cimd_clients "$token" || return 1
    _registration_policy_write "$token" policies cimd-vendors-policy null || return 1
    _registration_policy_write "$token" profiles cimd-vendors null
}

# Run before group migration or first-party updates, including on DCR-on deployments.
reconcile_registration_security() {
    local token=$1 status scope_status=0
    [ "${REGISTRATION_SECURITY_RECONCILED:-false}" != true ] || return 0
    status=$(realm_curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $token" \
        "$KEYCLOAK_URL/admin/realms/$REALM") || return 1
    [ "$status" != 404 ] || return 0
    [ "$status" = 200 ] || return 1
    # Close anonymous registration during the cutover, including already-open realms.
    disable_anonymous_dcr "$token" || return 1
    if [ "${ENABLE_CIMD_CONFIG:-false}" = true ]; then
        _reset_untrusted_cimd_cache "$token" || return 1
        reconcile_cimd_clients "$token" || return 1
    else
        _remove_cimd_client_policy "$token" || return 1
    fi
    configure_registration_security_policy "$token" || return 1
    # Keep the pre-rotation provenance; scope errors must not skip off-state revocation.
    reconcile_registered_client_scopes "$token" || scope_status=$?
    if [ "${ENABLE_DCR_CONFIG:-false}" != true ]; then
        disable_unlisted_clients "$token" || return 1
    fi
    [ "$scope_status" = 0 ] || return "$scope_status"
    REGISTRATION_SECURITY_RECONCILED=true
}
