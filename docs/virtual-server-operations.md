# Virtual MCP Server Operations

This document describes operations for managing Virtual MCP Servers. Virtual servers aggregate tools from multiple backend MCP servers into a single unified endpoint.

For the full design and architecture details, see [Virtual MCP Server Design Document](design/virtual-mcp-server.md).

## Video Demo

Watch the video demonstration of Virtual MCP Server creation and management through the web UI:

[![Virtual MCP Server Demo](https://img.shields.io/badge/Watch-Video%20Demo-red?style=for-the-badge&logo=youtube)](https://app.vidcast.io/share/954e6296-f217-4559-8d86-88cec25af763)

[View Video Demo](https://app.vidcast.io/share/954e6296-f217-4559-8d86-88cec25af763)

## Prerequisites

- A valid JWT token (saved to a file, e.g., `.token`)
- Registry URL (e.g., `http://localhost` for local development)

## Available CLI Commands

| Command | Description |
|---------|-------------|
| `vs-create` | Create a virtual MCP server from JSON config |
| `vs-list` | List all virtual MCP servers |
| `vs-get` | Get virtual MCP server details |
| `vs-update` | Update a virtual MCP server |
| `vs-delete` | Delete a virtual MCP server |
| `vs-toggle` | Enable or disable a virtual server |
| `vs-rate` | Rate a virtual MCP server (1-5 stars) |
| `vs-rating` | Get rating information |

## Configuration File Format

Virtual servers are created from a JSON configuration file. Here is an example that combines tools from Context7 (documentation search) and CurrentTime (timezone) servers:

```json
{
  "path": "/virtual/combined-tools",
  "server_name": "Combined Context7 and CurrentTime Tools",
  "description": "Virtual server aggregating documentation search tools from Context7 and timezone tools from CurrentTime server",
  "tool_mappings": [
    {
      "tool_name": "resolve-library-id",
      "backend_server_path": "/context7"
    },
    {
      "tool_name": "query-docs",
      "backend_server_path": "/context7"
    },
    {
      "tool_name": "current_time_by_timezone",
      "alias": "get-current-time",
      "backend_server_path": "/currenttime/"
    }
  ],
  "required_scopes": [],
  "tool_scope_overrides": [],
  "tags": [
    "documentation",
    "time",
    "timezone",
    "libraries",
    "combined"
  ],
  "supported_transports": [
    "streamable-http"
  ],
  "is_enabled": true
}
```

See [cli/examples/virtual-server-combined-example.json](../cli/examples/virtual-server-combined-example.json) for the full example.

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | Virtual server path (e.g., `/virtual/dev-tools`) |
| `server_name` | Yes | Display name for the virtual server |
| `description` | No | Description of the virtual server |
| `tool_mappings` | Yes | Array of tool mappings (at least one required) |
| `required_scopes` | No | Server-level scope requirements |
| `tool_scope_overrides` | No | Per-tool scope overrides |
| `tags` | No | Tags for categorization |
| `supported_transports` | No | Supported transports (default: `["streamable-http"]`) |
| `is_enabled` | No | Whether to enable on creation (default: `true`) |

### Tool Mapping Fields

| Field | Required | Description |
|-------|----------|-------------|
| `tool_name` | Yes | Original tool name on backend server |
| `backend_server_path` | Yes | Backend server path (e.g., `/github`) |
| `alias` | No | Renamed tool name in virtual server |
| `backend_version` | No | Pin to specific backend version |
| `description_override` | No | Override tool description |

## CLI Usage Examples

### Create a Virtual Server

```bash
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-create --config cli/examples/virtual-server-combined-example.json
```

**Example Output:**

```
Virtual server created: /virtual/combined-tools
{
  "message": "Virtual server created successfully",
  "virtual_server": {
    "path": "/virtual/combined-tools",
    "server_name": "Combined Context7 and CurrentTime Tools",
    "description": "Virtual server aggregating documentation search tools from Context7 and timezone tools from CurrentTime server",
    "is_enabled": false,
    "tool_count": 3
  }
}
```

### List Virtual Servers

```bash
# List all virtual servers
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-list

# List only enabled virtual servers
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-list --enabled-only

# Output as JSON
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-list --json
```

### Get Virtual Server Details

```bash
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-get --path /virtual/combined-tools
```

**Example Output:**

```
Virtual MCP Server: /virtual/combined-tools
------------------------------------------------------------
  Name: Combined Context7 and CurrentTime Tools
  Status: enabled
  Description: Virtual server aggregating documentation search tools from Context7 and timezone tools from CurrentTime server
  Rating: 0.0 stars
  Tags: documentation, time, timezone, libraries, combined
  Transports: streamable-http
  Required Scopes: None

  Tool Mappings (3):
    - resolve-library-id
      Backend: /context7
    - query-docs
      Backend: /context7
    - current_time_by_timezone -> get-current-time
      Backend: /currenttime/

  Created: 2026-02-17T13:35:22.803009Z
  Updated: 2026-02-17T13:35:41.075488Z
  Created By: admin
```

### Enable or Disable a Virtual Server

```bash
# Enable
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-toggle --path /virtual/combined-tools --enabled true

# Disable
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-toggle --path /virtual/combined-tools --enabled false
```

### Update a Virtual Server

```bash
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-update --path /virtual/combined-tools --config updated-config.json
```

### Delete a Virtual Server

```bash
# With confirmation prompt
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-delete --path /virtual/combined-tools

# Skip confirmation
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-delete --path /virtual/combined-tools --force
```

### Rate a Virtual Server

```bash
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-rate --path /virtual/combined-tools --rating 5
```

### Get Virtual Server Rating

```bash
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    vs-rating --path /virtual/combined-tools
```

## Scope-Based Access Control

Virtual servers support fine-grained access control through scopes. Virtual servers are configured in scope definitions exactly the same way as regular MCP servers - you simply use the virtual server path (e.g., `/virtual/scoped-tools`) as the server identifier.

For comprehensive documentation on how access control works, see [Virtual MCP Server Access Control](scopes.md#virtual-mcp-server-access-control) in the Fine-Grained Access Control documentation.

See [Scope-Based Access Control Example](../cli/examples/virtual-server-scoped-example.json) for a virtual server configuration with scopes.

### Server-Level Scopes

Use `required_scopes` to require users to have specific scopes to access the virtual server:

```json
{
  "required_scopes": ["virtual-server/access"]
}
```

### Per-Tool Scope Overrides

Use `tool_scope_overrides` to require additional scopes for specific tools:

```json
{
  "tool_scope_overrides": [
    {
      "tool_alias": "sensitive-tool",
      "required_scopes": ["virtual-server/admin"]
    }
  ]
}
```

### E2E Testing Script

An end-to-end test script is provided for testing scope-based access control:

```bash
# Run the E2E test (with automatic cleanup)
./tests/integration/test_virtual_server_scopes_e2e.sh \
    --registry-url http://localhost \
    --token-file .token

# Run without cleanup (saves credentials for UI testing)
./tests/integration/test_virtual_server_scopes_e2e.sh \
    --registry-url http://localhost \
    --token-file .token \
    --no-cleanup

# View saved credentials
cat /tmp/.vs-creds
```

The test script creates:
- A virtual server with scope-based access control
- A user group with matching scopes
- An M2M service account for API testing
- A regular user for UI testing

See [test_virtual_server_scopes_e2e.sh](../tests/integration/test_virtual_server_scopes_e2e.sh) for details.

## Web UI Alternative

All virtual server management operations can also be performed through the web UI. The UI provides a guided wizard for creating virtual servers with:

- Server configuration form
- Tool selection from registered backend servers
- Tool aliasing and scope configuration
- Real-time validation

## Environment Variables

Instead of passing `--registry-url` each time, you can set environment variables:

```bash
export REGISTRY_URL=http://localhost
export TOKEN_FILE=.token

# Then run commands without flags
uv run python api/registry_management.py vs-list
```

## Related Documentation

- [Virtual MCP Server Design Document](design/virtual-mcp-server.md)
- [CLI Reference](cli.md)
- [Server Management](service-management.md)
