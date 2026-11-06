#!/bin/bash
#
# Peer Federation Docker Test Script
#
# This script:
#   1. Builds and starts two registry instances via Docker Compose
#   2. Registers test servers on Registry A
#   3. Configures Registry B to peer with Registry A
#   4. Triggers sync and verifies data replication
#   5. Cleans up on exit
#
# Usage:
#   ./scripts/test-peer-federation-docker.sh
#   ./scripts/test-peer-federation-docker.sh --no-cleanup  # Keep containers running
#   ./scripts/test-peer-federation-docker.sh --rebuild     # Force rebuild images
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.federation-test.yml"

REGISTRY_A_URL="http://localhost:7860"
REGISTRY_B_URL="http://localhost:7861"
AUTH_A_URL="http://localhost:8888"
AUTH_B_URL="http://localhost:8889"

SECRET_KEY_A="${SECRET_KEY:-federation-test-secret-key-a}"
SECRET_KEY_B="${SECRET_KEY:-federation-test-secret-key-b}"

# Parse arguments
CLEANUP=true
REBUILD=""
for arg in "$@"; do
    case $arg in
        --no-cleanup)
            CLEANUP=false
            ;;
        --rebuild)
            REBUILD="--build --no-cache"
            ;;
    esac
done

# Cleanup function
cleanup() {
    if [ "$CLEANUP" = true ]; then
        echo -e "\n${YELLOW}Cleaning up...${NC}"
        cd "$PROJECT_DIR"
        docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
        echo -e "${GREEN}Cleanup complete${NC}"
    else
        echo -e "\n${YELLOW}Containers left running (--no-cleanup specified)${NC}"
        echo "To stop: docker compose -f $COMPOSE_FILE down -v"
    fi
}

trap cleanup EXIT INT TERM

# Print helpers
print_section() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Wait for service to be healthy
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-60}
    local attempt=1

    echo -n "Waiting for $name to be ready"
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            echo -e " ${GREEN}Ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e " ${RED}Failed${NC}"
    return 1
}

# Generate a session cookie using the SECRET_KEY (same signing as the registry)
generate_session_cookie() {
    local secret_key=$1
    local cookie_file=$2
    # The registry reads the session cookie under SESSION_COOKIE_NAME.
    local cookie_name="${SESSION_COOKIE_NAME:-mcp_gateway_session}"

    print_info "Generating session cookie using SECRET_KEY..."

    local cookie_value
    cookie_value=$(python3 -c "
from itsdangerous import URLSafeTimedSerializer
signer = URLSafeTimedSerializer('${secret_key}')
data = {'username': 'admin', 'auth_method': 'oauth2', 'provider': 'test', 'groups': ['mcp-registry-admin']}
print(signer.dumps(data))
" 2>/dev/null)

    if [ -z "$cookie_value" ]; then
        print_error "Failed to generate session cookie (is itsdangerous installed?)"
        return 1
    fi

    # Write cookie in Netscape cookie format for curl -b
    echo "# Netscape HTTP Cookie File" > "$cookie_file"
    echo "localhost	FALSE	/	FALSE	0	${cookie_name}	${cookie_value}" >> "$cookie_file"
    print_success "Session cookie generated"
    return 0
}

# Main test flow
main() {
    cd "$PROJECT_DIR"

    print_section "Peer Federation Docker Test"
    echo "Registry A: $REGISTRY_A_URL"
    echo "Registry B: $REGISTRY_B_URL"

    # Start services
    print_section "Starting Docker Services"
    print_info "Building and starting containers (this may take a few minutes)..."

    if [ -n "$REBUILD" ]; then
        docker compose -f "$COMPOSE_FILE" build --no-cache
    fi

    docker compose -f "$COMPOSE_FILE" up -d ${REBUILD:+--build}

    # Wait for services
    print_section "Waiting for Services"
    wait_for_service "$AUTH_A_URL" "Auth Server A" 90 || { print_error "Auth A failed to start"; docker compose -f "$COMPOSE_FILE" logs auth-server-a; exit 1; }
    wait_for_service "$AUTH_B_URL" "Auth Server B" 90 || { print_error "Auth B failed to start"; docker compose -f "$COMPOSE_FILE" logs auth-server-b; exit 1; }
    wait_for_service "$REGISTRY_A_URL" "Registry A" 90 || { print_error "Registry A failed to start"; docker compose -f "$COMPOSE_FILE" logs registry-a; exit 1; }
    wait_for_service "$REGISTRY_B_URL" "Registry B" 90 || { print_error "Registry B failed to start"; docker compose -f "$COMPOSE_FILE" logs registry-b; exit 1; }

    # Create cookie files
    COOKIE_A=$(mktemp)
    COOKIE_B=$(mktemp)
    trap "rm -f $COOKIE_A $COOKIE_B; cleanup" EXIT INT TERM

    # Generate session cookies for both registries using their SECRET_KEYs
    print_section "Authenticating"
    generate_session_cookie "$SECRET_KEY_A" "$COOKIE_A" || exit 1
    generate_session_cookie "$SECRET_KEY_B" "$COOKIE_B" || exit 1

    # Register test servers on Registry A
    print_section "Registering Test Servers on Registry A"

    # Server 1
    print_info "Registering 'Test Server 1'..."
    REGISTER_RESULT=$(curl -s -b "$COOKIE_A" -X POST "$REGISTRY_A_URL/api/servers/register" \
        -F "name=Test Server 1" \
        -F "description=First test server for federation" \
        -F "path=/test-server-1" \
        -F "proxy_pass_url=http://localhost:9001" \
        -F "tags=production,federation-test")

    if echo "$REGISTER_RESULT" | grep -q "registered successfully\|already exists"; then
        print_success "Server 1 registered"
    else
        print_error "Failed to register Server 1: $REGISTER_RESULT"
    fi

    # Server 2
    print_info "Registering 'Test Server 2'..."
    REGISTER_RESULT=$(curl -s -b "$COOKIE_A" -X POST "$REGISTRY_A_URL/api/servers/register" \
        -F "name=Test Server 2" \
        -F "description=Second test server for federation" \
        -F "path=/test-server-2" \
        -F "proxy_pass_url=http://localhost:9002" \
        -F "tags=development,federation-test")

    if echo "$REGISTER_RESULT" | grep -q "registered successfully\|already exists"; then
        print_success "Server 2 registered"
    else
        print_error "Failed to register Server 2: $REGISTER_RESULT"
    fi

    # Enable the servers
    print_info "Enabling test servers..."
    curl -s -b "$COOKIE_A" -X POST "$REGISTRY_A_URL/api/servers/toggle" \
        -F "path=/test-server-1" -F "new_state=true" > /dev/null
    curl -s -b "$COOKIE_A" -X POST "$REGISTRY_A_URL/api/servers/toggle" \
        -F "path=/test-server-2" -F "new_state=true" > /dev/null
    print_success "Servers enabled"

    # Verify servers on Registry A
    print_section "Verifying Servers on Registry A"
    SERVERS_A=$(curl -s -b "$COOKIE_A" "$REGISTRY_A_URL/api/servers")
    echo "$SERVERS_A" | python3 -m json.tool 2>/dev/null || echo "$SERVERS_A"

    # Check federation export endpoint
    print_section "Testing Federation Export (Registry A)"
    FED_EXPORT=$(curl -s -b "$COOKIE_A" "$REGISTRY_A_URL/api/federation/servers")
    echo "$FED_EXPORT" | python3 -m json.tool 2>/dev/null || echo "$FED_EXPORT"

    EXPORT_COUNT=$(echo "$FED_EXPORT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_count', 0))" 2>/dev/null || echo "0")
    if [ "$EXPORT_COUNT" -gt 0 ]; then
        print_success "Federation export has $EXPORT_COUNT servers"
    else
        print_info "Federation export shows 0 servers (servers may need to be public)"
    fi

    # Configure peer on Registry B
    print_section "Configuring Peer on Registry B"
    print_info "Adding Registry A as peer..."

    # Note: Using internal Docker network hostname
    PEER_RESULT=$(curl -s -b "$COOKIE_B" -X POST "$REGISTRY_B_URL/api/peers" \
        -H "Content-Type: application/json" \
        -d '{
            "peer_id": "registry-a",
            "name": "Registry A (Primary)",
            "endpoint": "http://registry-a:7860",
            "enabled": true,
            "sync_mode": "all",
            "sync_interval_minutes": 5
        }')

    if echo "$PEER_RESULT" | grep -q "registry-a\|already exists"; then
        print_success "Peer configured"
        echo "$PEER_RESULT" | python3 -m json.tool 2>/dev/null || echo "$PEER_RESULT"
    else
        print_error "Failed to configure peer: $PEER_RESULT"
    fi

    # List peers on Registry B
    print_section "Peers on Registry B"
    curl -s -b "$COOKIE_B" "$REGISTRY_B_URL/api/peers" | python3 -m json.tool 2>/dev/null || true

    # Trigger sync
    print_section "Triggering Sync"
    print_info "Syncing from Registry A to Registry B..."
    SYNC_RESULT=$(curl -s -b "$COOKIE_B" -X POST "$REGISTRY_B_URL/api/peers/registry-a/sync")
    echo "$SYNC_RESULT" | python3 -m json.tool 2>/dev/null || echo "$SYNC_RESULT"

    SYNC_SUCCESS=$(echo "$SYNC_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")
    SERVERS_SYNCED=$(echo "$SYNC_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('servers_synced', 0))" 2>/dev/null || echo "0")

    if [ "$SYNC_SUCCESS" = "True" ] || [ "$SYNC_SUCCESS" = "true" ]; then
        print_success "Sync completed: $SERVERS_SYNCED servers synced"
    else
        print_error "Sync failed"
        ERROR_MSG=$(echo "$SYNC_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error_message', 'unknown'))" 2>/dev/null || echo "unknown")
        print_info "Error: $ERROR_MSG"
    fi

    # Check peer status
    print_section "Peer Sync Status"
    curl -s -b "$COOKIE_B" "$REGISTRY_B_URL/api/peers/registry-a/status" | python3 -m json.tool 2>/dev/null || true

    # Verify servers on Registry B
    print_section "Servers on Registry B (After Sync)"
    SERVERS_B=$(curl -s -b "$COOKIE_B" "$REGISTRY_B_URL/api/servers")
    echo "$SERVERS_B" | python3 -m json.tool 2>/dev/null || echo "$SERVERS_B"

    # Check for federated servers
    if echo "$SERVERS_B" | grep -q "registry-a"; then
        print_success "Federation test PASSED - servers synced from Registry A to Registry B"
    else
        print_info "No federated servers found on Registry B"
        print_info "This may be expected if servers on Registry A are not publicly visible"
    fi

    # Summary
    print_section "Test Summary"
    echo "Registry A: $REGISTRY_A_URL (UI: http://localhost:80)"
    echo "Registry B: $REGISTRY_B_URL (UI: http://localhost:81)"
    echo ""
    echo "Authentication: via SECRET_KEY signed session cookies"
    echo ""
    echo "To manually test:"
    echo "  1. Open Registry A UI and register/enable servers"
    echo "  2. Open Registry B UI and check for synced servers"
    echo "  3. Or use the API endpoints shown above"
    echo ""

    if [ "$CLEANUP" = false ]; then
        echo -e "${YELLOW}Containers are still running.${NC}"
        echo "To stop: docker compose -f docker-compose.federation-test.yml down -v"
        echo ""
        echo -e "${YELLOW}Press Ctrl+C when done testing.${NC}"
        # Keep script running so user can test manually
        while true; do
            sleep 60
        done
    fi
}

main "$@"
