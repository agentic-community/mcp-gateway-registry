#!/bin/bash
# Delete stale DCR-registered Keycloak clients from the mcp-gateway realm.
#
# Each time an MCP client (Claude Code, Cursor, etc.) runs the OAuth flow
# against the gateway, it registers itself via RFC 7591 Dynamic Client
# Registration and Keycloak creates a fresh client record with a UUID-format
# clientId. Over time these accumulate — re-running `claude mcp add` produces
# a new record on every invocation, and old ones never get cleaned up.
#
# This script identifies UUID-clientId clients with zero active sessions and
# deletes them. Pre-defined clients (mcp-gateway-web, mcp-gateway-m2m, and
# anything else you registered manually) are never touched — only entries
# whose clientId matches the canonical 8-4-4-4-12 hex pattern are eligible.
#
# Usage:
#   bash keycloak/setup/cleanup-stale-dcr-clients.sh           # actually delete
#   bash keycloak/setup/cleanup-stale-dcr-clients.sh --dry-run # preview only
#
# Reads .env from repo root for KEYCLOAK_ADMIN_URL, KEYCLOAK_ADMIN, and
# KEYCLOAK_ADMIN_PASSWORD.

set -e

REALM="mcp-gateway"
KEYCLOAK_URL=""
KEYCLOAK_ADMIN=""
KEYCLOAK_ADMIN_PASSWORD=""
DRY_RUN=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse args
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n)
            DRY_RUN=1
            ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            exit 1
            ;;
    esac
done

if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}DRY-RUN MODE: nothing will be deleted${NC}"
else
    echo -e "${YELLOW}LIVE MODE: stale DCR'd clients will be deleted${NC}"
fi
echo "============================================================"

get_admin_token() {
    KEYCLOAK_ADMIN_REALM="${KEYCLOAK_ADMIN_REALM:-master}"
    local response=$(curl -s -X POST "${KEYCLOAK_URL}/realms/${KEYCLOAK_ADMIN_REALM}/protocol/openid-connect/token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=${KEYCLOAK_ADMIN}" \
        -d "password=${KEYCLOAK_ADMIN_PASSWORD}" \
        -d "grant_type=password" \
        -d "client_id=admin-cli")

    echo "$response" | jq -r '.access_token // empty'
}

main() {
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
    ENV_FILE="$PROJECT_ROOT/.env"

    if [ -f "$ENV_FILE" ]; then
        echo "Loading environment from $ENV_FILE..."
        set -a
        source "$ENV_FILE"
        set +a
    else
        echo -e "${RED}Error: .env not found at $ENV_FILE${NC}"
        exit 1
    fi

    KEYCLOAK_URL="${KEYCLOAK_ADMIN_URL:-http://localhost:8080}"
    KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"

    if [ -z "$KEYCLOAK_ADMIN_PASSWORD" ]; then
        echo -e "${RED}Error: KEYCLOAK_ADMIN_PASSWORD must be set in .env${NC}"
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

    # Pull all clients in one shot, then filter to DCR'd UUIDs locally.
    local all_clients=$(curl -s -H "Authorization: Bearer ${TOKEN}" \
        "${KEYCLOAK_URL}/admin/realms/${REALM}/clients")

    # UUIDs only
    local dcr_uuids=$(echo "$all_clients" | jq -r \
        '.[] | select(.clientId|test("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")) | .id')

    local total=0
    local stale=0
    local active=0
    local deleted=0
    local failed=0

    for client_uuid in $dcr_uuids; do
        total=$((total + 1))

        local count=$(curl -s -H "Authorization: Bearer ${TOKEN}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${client_uuid}/session-count" | \
            jq -r '.count // 0')

        if [ "$count" != "0" ]; then
            active=$((active + 1))
            continue
        fi

        stale=$((stale + 1))

        if [ $DRY_RUN -eq 1 ]; then
            echo "  would delete: ${client_uuid} (0 active sessions)"
            continue
        fi

        local del_status=$(curl -s -o /dev/null -w "%{http_code}" \
            -X DELETE \
            -H "Authorization: Bearer ${TOKEN}" \
            "${KEYCLOAK_URL}/admin/realms/${REALM}/clients/${client_uuid}")

        if [ "$del_status" = "204" ]; then
            deleted=$((deleted + 1))
        else
            failed=$((failed + 1))
            echo -e "${RED}  FAILED to delete ${client_uuid}: HTTP ${del_status}${NC}"
        fi
    done

    echo ""
    echo -e "${GREEN}Summary${NC}"
    echo "  Total DCR'd clients in realm: ${total}"
    echo "  With active sessions (kept): ${active}"
    echo "  Stale (no active sessions):  ${stale}"
    if [ $DRY_RUN -eq 1 ]; then
        echo -e "  ${YELLOW}DRY-RUN: re-run without --dry-run to actually delete${NC}"
    else
        echo "  Deleted: ${deleted}"
        if [ $failed -gt 0 ]; then
            echo -e "  ${RED}Failed:  ${failed}${NC}"
            exit 1
        fi
    fi
}

main "$@"
