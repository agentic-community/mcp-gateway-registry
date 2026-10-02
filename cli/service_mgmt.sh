#!/bin/bash

# DEPRECATED: This script is deprecated in favor of the Registry Management API
# Use: uv run python api/registry_management.py OR cli/registry_cli_wrapper.py
# See: api/README.md for documentation
#
# Service Management Script for MCP Gateway Registry
# Usage: ./cli/service_mgmt.sh {add|delete|monitor|test|add-to-groups|remove-from-groups|create-group|delete-group|list-groups} [args...]

echo "WARNING: This script is DEPRECATED. Please use the Registry Management API instead:"
echo "  uv run python api/registry_management.py --help"
echo "  OR cli/registry_cli_wrapper.py --help"
echo "See api/README.md for full documentation."
echo ""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Unicode symbols
CHECK_MARK="✓"
CROSS_MARK="✗"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Data-only helper for handling untrusted server config / scanner / health
# output. The config JSON is piped to it on stdin so registrant-controlled
# values never reach an interpreter or shell code position (no "python3 -c"
# string interpolation, no eval).
HELPER="$SCRIPT_DIR/_service_config.py"

# Load environment variables from .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a  # automatically export all variables
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Gateway URL (can be overridden with GATEWAY_URL environment variable)
GATEWAY_URL="${GATEWAY_URL:-http://localhost}"

# Default service name
DEFAULT_SERVICE="example-server"

print_success() {
    echo -e "${GREEN}${CHECK_MARK} $1${NC}"
}

print_error() {
    echo -e "${RED}${CROSS_MARK} $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check and refresh credentials if needed
    if ! "$PROJECT_ROOT/credentials-provider/check_and_refresh_creds.sh"; then
        print_error "Failed to setup credentials"
        exit 1
    fi
    print_success "Credentials ready"
}

run_mcp_command() {
    local tool="$1"
    local args="$2"
    local description="$3"

    print_info "$description"

    # Print the exact command being executed
    echo "🔍 Executing: uv run cli/mcp_client.py --url ${GATEWAY_URL}/mcpgw/mcp call --tool $tool --args '$args'"

    if output=$(cd "$PROJECT_ROOT" && uv run cli/mcp_client.py --url "${GATEWAY_URL}/mcpgw/mcp" call --tool "$tool" --args "$args" 2>&1); then
        print_success "$description completed"
        echo "$output"
        return 0
    else
        print_error "$description failed"
        echo "$output"
        return 1
    fi
}

verify_server_in_list() {
    local service_name="$1"
    local should_exist="$2"  # "true" or "false"

    print_info "Checking server in service list..."

    if output=$(cd "$PROJECT_ROOT" && uv run cli/mcp_client.py --url "${GATEWAY_URL}/mcpgw/mcp" call --tool list_services --args '{}' 2>&1); then
        if echo "$output" | grep -Fq -- "$service_name"; then
            if [ "$should_exist" = "true" ]; then
                print_success "Server found in service list"
                echo "$output" | grep -F -A2 -B2 -- "$service_name"
                return 0
            else
                print_error "Server still exists in service list (should be removed)"
                return 1
            fi
        else
            if [ "$should_exist" = "false" ]; then
                print_success "Server not found in service list (expected)"
                return 0
            else
                print_error "Server not found in service list"
                return 1
            fi
        fi
    else
        print_error "Failed to check service list"
        echo "$output"
        return 1
    fi
}


parse_health_output() {
    local json_output="$1"
    local service_filter="$2"

    # Parse/format via the data-only helper: health output arrives on stdin and
    # the (registrant-derived) service filter as an argv value, so neither is
    # ever interpolated into interpreter source.
    printf '%s' "$json_output" | python3 "$HELPER" format-health "$service_filter"
}

run_health_check() {
    local service_name="$1"

    print_info "Running health check..."

    if output=$(cd "$PROJECT_ROOT" && uv run cli/mcp_client.py --url "${GATEWAY_URL}/mcpgw/mcp" call --tool healthcheck --args '{}' 2>&1); then
        print_success "Health check completed"
        echo ""

        # Parse and display formatted output
        if ! parse_health_output "$output" "$service_name"; then
            print_error "Failed to parse health check output"
            echo "Raw output:"
            echo "$output"
            return 1
        fi
        return 0
    else
        print_error "Health check failed"
        echo "$output"
        return 1
    fi
}

validate_config() {
    local config_json="$1"

    # Validate/normalize via the data-only helper. The untrusted config is piped
    # on stdin so a value containing triple-quotes or shell metacharacters cannot
    # break out into interpreter/shell code. Emits the normalized config JSON on
    # line one and the derived service name on line two; exits non-zero on failure.
    printf '%s' "$config_json" | python3 "$HELPER" validate
}

add_service() {
    local config_file="${1}"
    local analyzers="${2:-yara}"

    if [ -z "$config_file" ]; then
        print_error "Usage: $0 add <config-file> [analyzers]"
        print_error "Example: $0 add cli/examples/example-server-config.json"
        print_error "Example: $0 add cli/examples/example-server-config.json yara,llm"
        exit 1
    fi

    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        print_error "Full path searched: $(pwd)/$config_file"
        exit 1
    fi

    print_info "Loading config from: $config_file"
    local config_json
    config_json="$(cat "$config_file")"

    # Validate config and extract service name
    local validation_output service_name modified_config
    if ! validation_output=$(validate_config "$config_json"); then
        print_error "Config validation failed"
        echo "$validation_output"  # This contains error message
        exit 1
    fi

    # Parse the two-line output: first line is modified config, second is service name
    modified_config=$(echo "$validation_output" | head -n 1)
    service_name=$(echo "$validation_output" | tail -n 1)

    # Use the modified config for registration
    config_json="$modified_config"

    # Extract service_path from config for later use
    local service_path
    service_path=$(printf '%s' "$config_json" | python3 "$HELPER" get path)

    echo "=== Adding Service: $service_name ==="

    # Check prerequisites
    check_prerequisites

    # Extract proxy_pass_url for security scanning
    local proxy_pass_url
    proxy_pass_url=$(printf '%s' "$config_json" | python3 "$HELPER" get proxy_pass_url)

    # Extract headers from config if present
    local headers_json
    headers_json=$(printf '%s' "$config_json" | python3 "$HELPER" get --omit-falsy headers)

    # Check if LLM analyzer is requested and API key is available
    if [[ "$analyzers" == *"llm"* ]]; then
        if [ -z "$MCP_SCANNER_LLM_API_KEY" ] || [[ "$MCP_SCANNER_LLM_API_KEY" == *"your_"* ]] || [[ "$MCP_SCANNER_LLM_API_KEY" == *"placeholder"* ]]; then
            echo ""
            print_error "LLM analyzer requested but MCP_SCANNER_LLM_API_KEY is not configured"
            print_info "Current value: ${MCP_SCANNER_LLM_API_KEY:-<not set>}"
            print_info ""
            print_info "Options:"
            print_info "  1. Add real API key to .env file: MCP_SCANNER_LLM_API_KEY=sk-..."
            print_info "  2. Set environment variable: export MCP_SCANNER_LLM_API_KEY=sk-..."
            print_info "  3. Use only YARA analyzer: $0 add $config_file yara"
            exit 1
        fi
    fi

    # Run security scan
    echo ""
    echo "=== Security Scan ==="
    print_info "Scanning server for security vulnerabilities..."
    print_info "Using analyzers: $analyzers"

    local is_safe="true"
    local scan_output=""

    # Prepare scan URL - append /mcp if not already present
    local scan_url="$proxy_pass_url"
    if [[ ! "$scan_url" =~ /mcp/?$ ]] && [[ ! "$scan_url" =~ /sse/?$ ]]; then
        # Remove trailing slash if present, then add /mcp
        scan_url="${scan_url%/}/mcp"
        print_info "Appending /mcp to scan URL: $scan_url"
    fi

    # Run scan using Python CLI and capture JSON output
    # Note: Scanner exits with code 1 when unsafe, so we need to capture both success and "failure" cases
    local scan_exit_code=0
    # Build the scan command as an argv array so untrusted values (scan_url,
    # headers) are passed as data and never re-parsed by a shell via eval.
    local -a scan_cmd=(uv run cli/mcp_security_scanner.py --server-url "$scan_url" --analyzers "$analyzers" --json)

    # Add headers if present in config
    if [ -n "$headers_json" ]; then
        print_info "Using custom headers from config for security scan"
        scan_cmd+=(--headers "$headers_json")
    fi

    # Scrub the operator's ambient scan bearer token from the scan subshell: the
    # server being added is not yet vetted and its proxy_pass_url is registrant/
    # remote-controlled, so the external scanner must never forward a stored
    # credential to that URL. Authenticated scans of a trusted server go through
    # the standalone "scan" subcommand or an explicit config header instead.
    scan_output=$(cd "$PROJECT_ROOT" && env -u MCP_SCAN_BEARER_TOKEN "${scan_cmd[@]}" 2>&1) || scan_exit_code=$?
    print_info "scan_exit_code - $scan_exit_code"

    # Fail closed: only a clean scan (exit code 0) may register. A non-zero exit
    # (1 = critical/high findings, 2+ = scanner error) means the server is NOT
    # verified safe, so it is NOT registered at all. Registration auto-enables a
    # server and there is no atomic "register-disabled" path in the API, so a
    # register-then-disable flow would briefly expose the auto-enabled,
    # registrant-controlled backend. The detailed scan report is written under
    # security_scans/ for operator review; fix the findings and re-run.
    if [ "$scan_exit_code" -ne 0 ]; then
        if [ "$scan_exit_code" -eq 1 ]; then
            print_error "Security scan failed - Server has critical or high severity issues"
        else
            print_error "Security scan encountered an error (exit code: $scan_exit_code)"
        fi
        print_error "Failing closed: the server was NOT registered because its security scan did not pass."
        print_info "Review the security scan report under security_scans/ and re-run once the server passes."
        exit 1
    fi

    print_success "Security scan passed - Server is SAFE"

    echo ""

    # Register the service (only reached after a clean security scan).
    if ! run_mcp_command "register_service" "$config_json" "Registering service"; then
        exit 1
    fi

    # Verify registration
    echo ""
    echo "=== Verifying Registration ==="

    if ! verify_server_in_list "$service_path" "true"; then
        exit 1
    fi

    # Run health check
    echo ""
    echo "=== Health Check ==="
    if ! run_health_check "$service_name"; then
        exit 1
    fi

    echo ""
    print_success "Service $service_name successfully added and verified!"
}

delete_service() {
    local service_path="${1}"
    local service_name="${2}"

    if [ -z "$service_path" ] || [ -z "$service_name" ]; then
        print_error "Usage: $0 delete <service-path> <service-name>"
        print_error "Example: $0 delete /example-server example-server"
        exit 1
    fi

    echo "=== Deleting Service: $service_name (path: $service_path) ==="

    # Check prerequisites
    check_prerequisites

    # Remove the service
    if ! run_mcp_command "remove_service" "{\"service_path\": \"$service_path\"}" "Removing service"; then
        exit 1
    fi

    # Verify deletion
    echo ""
    echo "=== Verifying Deletion ==="

    if ! verify_server_in_list "$service_path" "false"; then
        exit 1
    fi

    echo ""
    print_success "Service $service_name successfully deleted and verified!"
}

test_service() {
    local config_file="${1}"

    if [ -z "$config_file" ]; then
        print_error "Usage: $0 test <config-file>"
        print_error "Example: $0 test cli/examples/example-server-config.json"
        exit 1
    fi

    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        print_error "Full path searched: $(pwd)/$config_file"
        exit 1
    fi

    print_info "Loading config from: $config_file"
    local config_json
    config_json="$(cat "$config_file")"

    # Validate config and extract service info
    local validation_output service_name modified_config
    if ! validation_output=$(validate_config "$config_json"); then
        print_error "Config validation failed"
        echo "$validation_output"  # This contains error message
        exit 1
    fi

    # Parse the two-line output: first line is modified config, second is service name
    modified_config=$(echo "$validation_output" | head -n 1)
    service_name=$(echo "$validation_output" | tail -n 1)

    # Use the modified config
    config_json="$modified_config"

    # Extract description and tags for testing
    local description tags_json
    description=$(printf '%s' "$config_json" | python3 "$HELPER" get description)
    tags_json=$(printf '%s' "$config_json" | python3 "$HELPER" get tags)

    echo "=== Testing Service: $service_name ==="

    # Check prerequisites
    check_prerequisites

    # Test intelligent tool finder with description
    if [ -n "$description" ]; then
        print_info "Testing search with description: \"$description\""
        if ! run_mcp_command "intelligent_tool_finder" "{\"natural_language_query\": \"$description\"}" "Searching with description"; then
            print_error "Failed to search with description"
        else
            print_success "Search with description completed"
        fi
        echo ""
    fi

    # Test intelligent tool finder with tags only
    if [ "$tags_json" != "[]" ]; then
        print_info "Testing search with tags: $tags_json"
        if ! run_mcp_command "intelligent_tool_finder" "{\"tags\": $tags_json}" "Searching with tags"; then
            print_error "Failed to search with tags"
        else
            print_success "Search with tags completed"
        fi
        echo ""
    fi

    # Test combined search
    if [ -n "$description" ] && [ "$tags_json" != "[]" ]; then
        print_info "Testing combined search with description and tags"
        if ! run_mcp_command "intelligent_tool_finder" "{\"natural_language_query\": \"$description\", \"tags\": $tags_json}" "Combined search"; then
            print_error "Failed combined search"
        else
            print_success "Combined search completed"
        fi
        echo ""
    fi

    echo ""
    print_success "Service testing completed!"
}


monitor_services() {
    local config_file="${1}"
    local service_name=""

    if [ -n "$config_file" ]; then
        if [ ! -f "$config_file" ]; then
            print_error "Config file not found: $config_file"
            exit 1
        fi

        print_info "Loading config from: $config_file"
        local config_json
        config_json="$(cat "$config_file")"

        # Validate config and extract service name
        local validation_output modified_config
        if ! validation_output=$(validate_config "$config_json"); then
            print_error "Config validation failed"
            echo "$validation_output"  # This contains error message
            exit 1
        fi

        # Parse the two-line output: first line is modified config, second is service name
        modified_config=$(echo "$validation_output" | head -n 1)
        service_name=$(echo "$validation_output" | tail -n 1)

        echo "=== Monitoring Service: $service_name ==="
    else
        echo "=== Monitoring All Services ==="
    fi

    # Check prerequisites
    check_prerequisites

    # Run health check
    if ! run_health_check "$service_name"; then
        exit 1
    fi

    echo ""
    print_success "Monitoring completed!"
}

scan_server_security() {
    local server_url="$1"
    local analyzers="${2:-yara}"
    local api_key="${3:-}"
    local headers="${4:-}"

    if [ -z "$server_url" ]; then
        print_error "Usage: $0 scan <server-url> [analyzers] [api-key] [headers]"
        print_error "Example: $0 scan https://mcp.deepwki.com/mcp"
        print_error "Example: $0 scan https://mcp.deepwki.com/mcp yara,llm"
        print_error "Example: $0 scan https://mcp.deepwki.com/mcp yara,llm \$MCP_SCANNER_LLM_API_KEY"
        print_error "Example: $0 scan https://mcp.deepwki.com/mcp yara '' '{\"X-Authorization\": \"token123\"}'"
        print_error ""
        print_error "Note: For LLM analyzer, set MCP_SCANNER_LLM_API_KEY environment variable"
        print_error "      or pass API key as third argument"
        print_error "Note: For custom headers, pass JSON string as fourth argument"
        exit 1
    fi

    echo "=== Security Scan: $server_url ==="

    # Check if LLM analyzer is requested and API key is available
    if [[ "$analyzers" == *"llm"* ]]; then
        # Check both environment variable and CLI argument
        local key_to_check="${api_key:-$MCP_SCANNER_LLM_API_KEY}"
        if [ -z "$key_to_check" ] || [[ "$key_to_check" == *"your_"* ]] || [[ "$key_to_check" == *"placeholder"* ]]; then
            echo ""
            print_error "LLM analyzer requested but MCP_SCANNER_LLM_API_KEY is not configured"
            print_info "Current value: ${MCP_SCANNER_LLM_API_KEY:-<not set>}"
            print_info ""
            print_info "Options:"
            print_info "  1. Add real API key to .env file: MCP_SCANNER_LLM_API_KEY=sk-..."
            print_info "  2. Set environment variable: export MCP_SCANNER_LLM_API_KEY=sk-..."
            print_info "  3. Pass API key as argument: $0 scan $server_url $analyzers sk-your-key"
            print_info "  4. Use only YARA analyzer: $0 scan $server_url yara"
            return 1
        fi
    fi

    # Build the command as an argv array so untrusted values (server_url,
    # api_key, headers) are passed as data and never re-parsed by a shell.
    local -a cmd=(uv run cli/mcp_security_scanner.py --server-url "$server_url" --analyzers "$analyzers")

    # Add API key if provided
    if [ -n "$api_key" ]; then
        cmd+=(--api-key "$api_key")
    fi

    # Add headers if provided
    if [ -n "$headers" ]; then
        cmd+=(--headers "$headers")
    fi

    print_info "Running security scan..."
    print_info "Analyzers: $analyzers"

    # Run scan and capture exit code
    if (cd "$PROJECT_ROOT" && "${cmd[@]}"); then
        print_success "Security scan completed - Server is SAFE"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 1 ]; then
            print_error "Security scan completed - Server is UNSAFE (has critical or high severity issues)"
        else
            print_error "Security scan failed with error code $exit_code"
        fi
        return $exit_code
    fi
}

show_usage() {
    echo "Usage: $0 {add|delete|monitor|test|scan|add-to-groups|remove-from-groups|create-group|delete-group|list-groups} [args...]"
    echo ""
    echo "Service Commands:"
    echo "  add <config-file> [analyzers] - Add a service using JSON config and verify registration"
    echo "                                  analyzers: yara (default), llm, or yara,llm"
    echo "  delete <service-path> <service-name> - Delete a service by path and name"
    echo "  monitor [config-file]        - Run health check (all services or specific service from config)"
    echo "  test <config-file>           - Test service searchability using intelligent_tool_finder"
    echo "  scan <server-url> [analyzers] [api-key] - Run security scan on MCP server"
    echo "                                            analyzers: yara (default), llm, or yara,llm"
    echo ""
    echo "Server-to-Group Commands:"
    echo "  add-to-groups <server-name> <groups> - Add server to specific scopes groups (comma-separated)"
    echo "  remove-from-groups <server-name> <groups> - Remove server from specific scopes groups (comma-separated)"
    echo ""
    echo "Group Management Commands:"
    echo "  create-group <group-name> [description] - Create a new group in Keycloak"
    echo "  delete-group <group-name>    - Delete a group from Keycloak"
    echo "  list-groups                  - List all groups with synchronization status"
    echo ""
    echo "Config File Requirements:"
    echo "  Required fields: server_name, path, proxy_pass_url"
    echo "  Optional fields: description, tags, num_tools, license,"
    echo "                   auth_provider, auth_scheme, supported_transports, headers, tool_list"
    echo "  Constraints:"
    echo "    - path must start with '/' and be more than just '/'"
    echo "    - proxy_pass_url must start with http:// or https://"
    echo "    - server_name must be non-empty string"
    echo "    - tags must be array of strings"
    echo "    - num_tools must be a non-negative integer"
    echo "    - supported_transports must be array of strings"
    echo "    - headers must be array of objects"
    echo "    - tool_list must be array of objects"
    echo ""
    echo "Examples:"
    echo "  # Service operations"
    echo "  $0 add cli/examples/example-server-config.json           # Add with default YARA analyzer"
    echo "  export MCP_SCANNER_LLM_API_KEY=sk-..."
    echo "  $0 add cli/examples/example-server-config.json yara,llm  # Add with both analyzers"
    echo "  $0 add cli/examples/example-server-config.json llm       # Add with only LLM analyzer"
    echo "  $0 delete /example-server example-server"
    echo "  $0 monitor                                        # All services"
    echo "  $0 monitor cli/examples/example-server-config.json # Specific service"
    echo "  $0 test cli/examples/example-server-config.json    # Test searchability"
    echo ""
    echo "  # Security scanning"
    echo "  $0 scan https://mcp.deepwki.com/mcp              # Security scan with default YARA"
    echo "  export MCP_SCANNER_LLM_API_KEY=sk-..."
    echo "  $0 scan https://mcp.deepwki.com/mcp yara,llm     # Scan with both analyzers (uses env var)"
    echo "  $0 scan https://mcp.deepwki.com/mcp llm sk-...   # Scan with only LLM (pass API key directly)"
    echo "  $0 scan https://mcp.deepwki.com/mcp yara '' '{\"X-Authorization\": \"token\"}' # Scan with custom headers"
    echo ""
    echo "  # Server-to-group operations"
    echo "  $0 add-to-groups example-server 'mcp-servers-restricted/read,mcp-servers-restricted/execute'"
    echo "  $0 remove-from-groups example-server 'mcp-servers-restricted/read,mcp-servers-restricted/execute'"
    echo ""
    echo "  # Group management operations"
    echo "  $0 create-group mcp-servers-finance/read 'Finance team read access'"
    echo "  $0 delete-group mcp-servers-finance/read"
    echo "  $0 list-groups"
}

add_to_groups() {
    local server_name="$1"
    local groups="$2"

    if [ -z "$server_name" ] || [ -z "$groups" ]; then
        print_error "Usage: $0 add-to-groups <server-name> <groups>"
        print_error "Example: $0 add-to-groups example-server 'mcp-servers-restricted/read,mcp-servers-restricted/execute'"
        exit 1
    fi

    echo "=== Adding Server to Scopes Groups: $server_name ==="

    # Check prerequisites
    check_prerequisites

    # Convert comma-separated groups to JSON array format
    local groups_json
    groups_json=$(echo "$groups" | sed 's/,/","/g' | sed 's/^/"/' | sed 's/$/"/')
    groups_json="[$groups_json]"

    print_info "Adding server '$server_name' to groups: $groups"

    # Call the MCP tool
    local response
    if response=$(run_mcp_command "add_server_to_scopes_groups" "{\"server_name\": \"$server_name\", \"group_names\": $groups_json}"); then
        # Check if the response indicates success
        if echo "$response" | grep -q '"success": true'; then
            print_success "Server successfully added to groups"

            # Extract and display details
            local server_path
            server_path=$(echo "$response" | grep -o '"server_path": "[^"]*"' | cut -d'"' -f4)
            if [ -n "$server_path" ]; then
                print_info "Server path: $server_path"
            fi

            print_info "Groups: $groups"
            print_success "Scopes groups updated and auth server reloaded"
        else
            # Extract error message if available
            local error_msg
            error_msg=$(echo "$response" | grep -o '"error": "[^"]*"' | cut -d'"' -f4)
            if [ -n "$error_msg" ]; then
                print_error "Failed to add server to groups: $error_msg"
            else
                print_error "Failed to add server to groups (unknown error)"
                echo "Response: $response"
            fi
            exit 1
        fi
    else
        print_error "Failed to call add_server_to_scopes_groups tool"
        exit 1
    fi

    echo ""
    print_success "Add to groups operation completed!"
}

remove_from_groups() {
    local server_name="$1"
    local groups="$2"

    if [ -z "$server_name" ] || [ -z "$groups" ]; then
        print_error "Usage: $0 remove-from-groups <server-name> <groups>"
        print_error "Example: $0 remove-from-groups example-server 'mcp-servers-restricted/read,mcp-servers-restricted/execute'"
        exit 1
    fi

    echo "=== Removing Server from Scopes Groups: $server_name ==="

    # Check prerequisites
    check_prerequisites

    # Convert comma-separated groups to JSON array format
    local groups_json
    groups_json=$(echo "$groups" | sed 's/,/","/g' | sed 's/^/"/' | sed 's/$/"/')
    groups_json="[$groups_json]"

    print_info "Removing server '$server_name' from groups: $groups"

    # Call the MCP tool
    local response
    if response=$(run_mcp_command "remove_server_from_scopes_groups" "{\"server_name\": \"$server_name\", \"group_names\": $groups_json}"); then
        # Check if the response indicates success
        if echo "$response" | grep -q '"success": true'; then
            print_success "Server successfully removed from groups"

            # Extract and display details
            local server_path
            server_path=$(echo "$response" | grep -o '"server_path": "[^"]*"' | cut -d'"' -f4)
            if [ -n "$server_path" ]; then
                print_info "Server path: $server_path"
            fi

            print_info "Groups: $groups"
            print_success "Scopes groups updated and auth server reloaded"
        else
            # Extract error message if available
            local error_msg
            error_msg=$(echo "$response" | grep -o '"error": "[^"]*"' | cut -d'"' -f4)
            if [ -n "$error_msg" ]; then
                print_error "Failed to remove server from groups: $error_msg"
            else
                print_error "Failed to remove server from groups (unknown error)"
                echo "Response: $response"
            fi
            exit 1
        fi
    else
        print_error "Failed to call remove_server_from_scopes_groups tool"
        exit 1
    fi

    echo ""
    print_success "Remove from groups operation completed!"
}


create_group() {
    local group_name="$1"
    local description="${2:-}"

    if [ -z "$group_name" ]; then
        print_error "Group name is required"
        echo "Usage: $0 create-group <group-name> [description]"
        exit 1
    fi

    echo "=== Creating Group: $group_name ==="

    # Check prerequisites
    check_prerequisites

    # Prepare arguments for create_group MCP tool
    local args="{\"group_name\": \"$group_name\""
    if [ -n "$description" ]; then
        # Escape description for JSON
        local escaped_desc=$(echo "$description" | sed 's/"/\\"/g')
        args="$args, \"description\": \"$escaped_desc\""
    fi
    args="$args}"

    # Call create_group MCP tool
    if ! run_mcp_command "create_group" "$args" "Creating group '$group_name'"; then
        print_error "Failed to create group"
        exit 1
    fi

    echo ""
    print_success "Create group operation completed!"
}


delete_group() {
    local group_name="$1"

    if [ -z "$group_name" ]; then
        print_error "Group name is required"
        echo "Usage: $0 delete-group <group-name>"
        exit 1
    fi

    echo "=== Deleting Group: $group_name ==="

    # Check prerequisites
    check_prerequisites

    # Prepare arguments for delete_group MCP tool
    local args="{\"group_name\": \"$group_name\"}"

    # Call delete_group MCP tool
    if ! run_mcp_command "delete_group" "$args" "Deleting group '$group_name'"; then
        print_error "Failed to delete group"
        exit 1
    fi

    echo ""
    print_success "Delete group operation completed!"
}


list_groups() {
    echo "=== Listing All Groups ==="

    # Check prerequisites
    check_prerequisites

    # Call list_groups MCP tool
    local args="{}"

    print_info "Fetching groups from Keycloak..."

    if output=$(cd "$PROJECT_ROOT" && uv run cli/mcp_client.py --url "${GATEWAY_URL}/mcpgw/mcp" call --tool list_groups --args "$args" 2>&1); then
        print_success "Groups retrieved successfully"
        echo ""
        echo "$output"
    else
        print_error "Failed to list groups"
        echo "$output"
        exit 1
    fi

    echo ""
    print_success "List groups operation completed!"
}


# Main script logic
case "${1:-}" in
    add)
        add_service "$2" "$3"
        ;;
    delete)
        delete_service "$2" "$3"
        ;;
    monitor)
        monitor_services "$2"
        ;;
    test)
        test_service "$2"
        ;;
    scan)
        scan_server_security "$2" "$3" "$4" "$5"
        ;;
    add-to-groups)
        add_to_groups "$2" "$3"
        ;;
    remove-from-groups)
        remove_from_groups "$2" "$3"
        ;;
    create-group)
        create_group "$2" "$3"
        ;;
    delete-group)
        delete_group "$2"
        ;;
    list-groups)
        list_groups
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
