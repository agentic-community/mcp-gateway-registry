#!/bin/bash
#
# validate-extra-env.sh - Preflight validation for Docker Compose extra_env files (Issue #1000).
#
# Scans ${HOME}/mcp-gateway/extra_env/{registry,auth-server,mcpgw}.env against
# the chart-managed reserved name lists in charts/<subchart>/reserved-env-names.txt.
# Rejects reserved-name collisions, warns on malformed lines, and logs the
# number of custom variables applied per service.
#
# Usage:
#   scripts/validate-extra-env.sh              # validate all three services
#   scripts/validate-extra-env.sh registry     # validate only the registry service
#
# Exit codes:
#   0 - all extra_env files are valid (or absent)
#   1 - one or more files contains a reserved name collision
#
# Source this from build_and_run.sh (or run directly from CI / pre-commit).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default to extra_env/ at the repo root so operators find it next to .env.
# Override with MCP_EXTRA_ENV_DIR for non-default locations (CI, shared hosts).
EXTRA_ENV_DIR="${MCP_EXTRA_ENV_DIR:-${REPO_ROOT}/extra_env}"

# Log helper that matches build_and_run.sh's `log` if available, else falls
# back to a plain echo so the script works standalone.
if ! declare -F log > /dev/null; then
    log() {
        echo "$(date '+%Y-%m-%d %H:%M:%S') $*"
    }
fi


_normalize_key() {
    # Upper-case and strip whitespace so "secret_key" and "SECRET_KEY " both
    # match the reserved-name list. Env var names are conventionally
    # upper-case; compare case-insensitively to catch operator typos.
    local raw="$1"
    echo "${raw}" | tr '[:lower:]' '[:upper:]' | tr -d '[:space:]'
}


_load_reserved_names() {
    # Read one name per line from the given file into the RESERVED_NAMES array.
    # Skips blank lines and comments.
    local reserved_file="$1"
    RESERVED_NAMES=()
    while IFS= read -r name || [ -n "$name" ]; do
        [[ -z "$name" || "$name" =~ ^# ]] && continue
        RESERVED_NAMES+=("$(_normalize_key "$name")")
    done < "$reserved_file"
}


_is_reserved() {
    local candidate="$1"
    local reserved
    for reserved in "${RESERVED_NAMES[@]}"; do
        if [[ "$candidate" == "$reserved" ]]; then
            return 0
        fi
    done
    return 1
}


_validate_one_service() {
    # Validate a single service's extra_env file.
    #
    # Side effects:
    #   - Appends any reserved-name collisions to the caller-visible
    #     COLLISIONS array.
    #   - Sets LAST_VALID_COUNT to the number of non-reserved custom vars
    #     the caller should treat as "accepted".
    #
    # Runs in the caller's shell (no subshell / stdout capture) so array
    # mutations are visible to the caller. Never exits; the caller decides
    # whether to fail.
    local service_name="$1"
    local reserved_file="$2"
    local env_file="${EXTRA_ENV_DIR}/${service_name}.env"

    LAST_VALID_COUNT=0

    if [ ! -f "$env_file" ]; then
        return 0
    fi

    _load_reserved_names "$reserved_file"

    local line_num=0
    local line trimmed raw_key key
    while IFS= read -r line || [ -n "$line" ]; do
        line_num=$((line_num + 1))

        # Strip CR (DOS line endings) + leading/trailing whitespace
        line="${line%$'\r'}"
        trimmed="${line#"${line%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"

        [[ -z "$trimmed" || "$trimmed" =~ ^# ]] && continue

        # Malformed line: no '=' means it cannot be a KEY=VALUE entry.
        if [[ "$trimmed" != *"="* ]]; then
            log "WARNING: extra_env/${service_name}.env line ${line_num} does not look like KEY=VALUE (no '='); skipping: ${trimmed}"
            continue
        fi

        # Malformed line: '=' at column 0 means empty key.
        if [[ "${trimmed%%=*}" == "" ]]; then
            log "WARNING: extra_env/${service_name}.env line ${line_num} has an empty key; skipping"
            continue
        fi

        raw_key="${trimmed%%=*}"
        key="$(_normalize_key "$raw_key")"

        if _is_reserved "$key"; then
            COLLISIONS+=("${service_name}.env:${line_num}: '${raw_key}' is reserved (chart-managed). Use the canonical setting instead.")
            continue
        fi

        LAST_VALID_COUNT=$((LAST_VALID_COUNT + 1))
    done < "$env_file"

    return 0
}


validate_extra_env_all() {
    # Validate all three services, collecting every collision before
    # deciding pass/fail. Logs a summary line with the per-service
    # custom-variable count. Returns 0 on success, 1 if any collision was
    # found.
    COLLISIONS=()
    local registry_count auth_count mcpgw_count

    _validate_one_service "registry" \
        "$REPO_ROOT/charts/registry/reserved-env-names.txt"
    registry_count=$LAST_VALID_COUNT

    _validate_one_service "auth-server" \
        "$REPO_ROOT/charts/auth-server/reserved-env-names.txt"
    auth_count=$LAST_VALID_COUNT

    _validate_one_service "mcpgw" \
        "$REPO_ROOT/charts/mcpgw/reserved-env-names.txt"
    mcpgw_count=$LAST_VALID_COUNT

    if [ ${#COLLISIONS[@]} -gt 0 ]; then
        log "ERROR: extra_env validation found ${#COLLISIONS[@]} reserved-name collision(s) in ${EXTRA_ENV_DIR}:"
        local collision
        for collision in "${COLLISIONS[@]}"; do
            log "  - ${collision}"
        done
        log "       Reserved lists: $REPO_ROOT/charts/{registry,auth-server,mcpgw}/reserved-env-names.txt"
        return 1
    fi

    log "extra_env validation passed (registry=${registry_count}, auth-server=${auth_count}, mcpgw=${mcpgw_count} custom var(s))"
    return 0
}


_main() {
    local service="${1:-all}"

    case "$service" in
        all)
            validate_extra_env_all
            ;;
        registry|auth-server|mcpgw)
            COLLISIONS=()
            _validate_one_service "$service" "$REPO_ROOT/charts/${service}/reserved-env-names.txt"
            if [ ${#COLLISIONS[@]} -gt 0 ]; then
                log "ERROR: extra_env validation found ${#COLLISIONS[@]} reserved-name collision(s) for ${service}:"
                local collision
                for collision in "${COLLISIONS[@]}"; do
                    log "  - ${collision}"
                done
                return 1
            fi
            log "extra_env validation passed for ${service} (${LAST_VALID_COUNT} custom var(s))"
            ;;
        *)
            log "ERROR: unknown service '${service}'. Expected one of: all, registry, auth-server, mcpgw"
            return 2
            ;;
    esac
}


# Only run _main when the script is executed directly, not when sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _main "$@"
    exit $?
fi
