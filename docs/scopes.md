# Fine-Grained Access Control System Documentation

> **Note**: While this document discusses Fine-Grained Access Control (FGAC) in the context of Amazon Cognito, the concepts and implementation apply to any Identity Provider (IdP). The same scope-based authorization model can be used with other OAuth2/OIDC providers by adapting the group mapping and token validation mechanisms.

This document provides comprehensive documentation for the fine-grained access control system in the MCP Gateway Registry, explaining how the scope-based authorization model works and how to configure it properly.

## Table of Contents

1. [Overview](#overview)
2. [Scope System Architecture](#scope-system-architecture)
3. [Scope Types and Structure](#scope-types-and-structure)
4. [Methods vs Tools Access Control](#methods-vs-tools-access-control)
5. [Cognito Integration](#cognito-integration)
6. [Scope Validation Logic](#scope-validation-logic)
7. [Configuration Examples](#configuration-examples)
8. [Virtual MCP Server Access Control](#virtual-mcp-server-access-control)
9. [Security Considerations](#security-considerations)
10. [Troubleshooting](#troubleshooting)

## Overview

The MCP Gateway Registry implements a sophisticated fine-grained access control system that provides granular permissions for accessing MCP servers, methods, and tools. The system is built around a scope-based authorization model that:

- Maps Amazon Cognito user groups to MCP server scopes
- Controls access to specific MCP servers, methods, and individual tools
- Supports both user identity mode (OAuth2 PKCE) and agent identity mode (Machine-to-Machine)
- Uses hierarchical scope validation for precise permission control
- Follows the principle of least privilege by default

The access control system is stored in DocumentDB / MongoDB (the `mcp_scopes` collection) and enforced by the validation logic in [`auth_server/server.py`](../auth_server/server.py). Default scopes are seeded at initialization time from the JSON seed files in [`scripts/`](../scripts/) (for example `registry-admins.json` and `federation-service.json`); additional scopes can be created through the scope management API. The legacy `scopes.yml` file was removed in v1.24.8.

## Scope System Architecture

### Core Components

The access control system consists of three main components:

1. **Scope Configuration** (the `mcp_scopes` collection in DocumentDB / MongoDB, seeded from the JSON files in [`scripts/`](../scripts/)): Defines all available scopes and their permissions
2. **Group Mappings**: Maps Amazon Cognito groups to both UI and server scopes
3. **Validation Engine** ([`auth_server/server.py`](../auth_server/server.py)): Enforces access control decisions

### Authentication Flow Integration

The scope system integrates seamlessly with both authentication modes:

- **User Identity Mode**: Users authenticate via OAuth2 PKCE, and their Cognito groups are mapped to scopes
- **Agent Identity Mode**: Agents authenticate via M2M JWT tokens with custom scopes directly assigned

### Relationship with Cognito

The system leverages Amazon Cognito's group membership feature to assign permissions:

1. Users are assigned to Cognito groups (e.g., `mcp-registry-admin`, `mcp-registry-user`)
2. Groups are mapped to scopes via the `group_mappings` configuration
3. Scopes define specific permissions for UI operations and MCP server access
4. The validation engine checks these scopes against requested operations

## Scope Types and Structure

The system defines several types of scopes, each serving different purposes:

### UI Scopes

UI scopes control access to registry management functions through the web interface:

- **`mcp-registry-admin`**: Full administrative access to all registry functions
- **`mcp-registry-user`**: Limited user access to specific servers and operations
- **`mcp-registry-developer`**: Developer access for service registration and management
- **`mcp-registry-operator`**: Operational access for service control without registration rights

#### UI Scope Permissions

Each UI scope defines permissions for specific registry operations:

```yaml
UI-Scopes:
  mcp-registry-admin:
    list_service: [all]           # Can list all services
    register_service: [all]       # Can register any service
    health_check_service: [all]   # Can check health of all services
    toggle_service: [all]         # Can enable/disable all services
    modify_service: [all]         # Can modify all services
```

### Server Scopes

Server scopes control access to MCP servers with read and execute permissions:

- **`mcp-servers-unrestricted/read`**: Read access to all MCP servers and tools
- **`mcp-servers-unrestricted/execute`**: Execute access to all MCP servers and tools
- **`mcp-servers-restricted/read`**: Limited read access to specific servers and tools
- **`mcp-servers-restricted/execute`**: Limited execute access to specific servers and tools

#### Permission Levels

- **Read Permission**: Allows listing tools and reading server information
- **Execute Permission**: Allows calling tools and executing server methods

### Group Mappings

Group mappings connect Cognito groups to both UI and server scopes:

```yaml
group_mappings:
  mcp-registry-admin:
    - mcp-registry-admin                    # UI permissions
    - mcp-servers-unrestricted/read         # Server read access
    - mcp-servers-unrestricted/execute      # Server execute access
  mcp-registry-user:
    - mcp-registry-user                     # Limited UI permissions
    - mcp-servers-restricted/read           # Limited server access
```

> **Important**: All group names (such as `mcp-registry-admin`, `mcp-registry-user`) and scope names (such as `mcp-servers-unrestricted/read`, `mcp-servers-restricted/execute`) are completely customizable by the platform administrator deploying this solution. These names are examples and can be changed to match your organization's naming conventions and security requirements. The same group names must be configured consistently in both your Identity Provider (IdP) and the scope documents stored in DocumentDB (the `mcp_scopes` collection).

## Methods vs Tools Access Control

One of the key features of the access control system is its ability to differentiate between MCP protocol methods and specific tools, providing granular control over what operations users can perform.

### MCP Protocol Methods

Methods are standard MCP protocol operations that all servers support:

- **`initialize`**: Initialize connection with the server
- **`notifications/initialized`**: Handle initialization notifications
- **`ping`**: Health check operation
- **`tools/list`**: List available tools on the server
- **`tools/call`**: Call a specific tool (requires additional tool-level validation)

### Tool-Specific Access Control

Tools are server-specific functions that can be called via the `tools/call` method. The system provides two levels of validation:

1. **Method-Level Validation**: Check if the user can call `tools/call`
2. **Tool-Level Validation**: Check if the user can call the specific tool

#### Validation Logic for `tools/call`

When a user attempts to call a tool via `tools/call`, the system performs enhanced validation:

```python
# For tools/call, check if the specific tool is allowed
if method == 'tools/call' and tool_name:
    if tool_name in allowed_tools:
        # Access granted - user can call this specific tool
        return True
    else:
        # Access denied - user cannot call this tool
        return False
```

#### Example: Tool Access Configuration

```yaml
mcp-servers-restricted/execute:
  - server: currenttime
    methods:
      - initialize
      - notifications/initialized
      - ping
      - tools/list
      - tools/call                    # Can call tools/call method
    tools:
      - current_time_by_timezone      # Can call this specific tool
      - get_config                    # Can call this specific tool
      # Note: Cannot call other tools not listed here
```

### Access Control Scenarios

#### Scenario 1: Method Access Only
User has permission for `tools/list` but not `tools/call`:
- ✅ Can list available tools
- ❌ Cannot execute any tools

#### Scenario 2: Method + Specific Tool Access
User has permission for `tools/call` and specific tools:
- ✅ Can call `current_time_by_timezone`
- ✅ Can call `get_config`
- ❌ Cannot call `advanced_analytics_tool` (not in allowed tools list)

#### Scenario 3: Unrestricted Access
User has unrestricted execute permissions:
- ✅ Can call any method
- ✅ Can call any tool listed in the scope configuration

## Cognito Integration

The access control system integrates deeply with Amazon Cognito for both user and agent authentication modes.

### User Identity Mode Integration

For users authenticating through the web interface:

1. **User Authentication**: Users log in via OAuth2 PKCE flow
2. **Group Membership**: Cognito returns user's group memberships
3. **Scope Mapping**: Groups are mapped to scopes using `group_mappings`
4. **Session Management**: Scopes are stored in session cookies for subsequent requests

### Agent Identity Mode Integration

For agents using their own identity:

1. **M2M Authentication**: Agents authenticate using client credentials flow
2. **Custom Scopes**: Agents are assigned custom scopes directly in Cognito
3. **JWT Token**: Scopes are embedded in JWT tokens
4. **Direct Validation**: Scopes are validated directly without group mapping

### Cognito Configuration Requirements

#### User Pool Setup
- Create user groups matching the scope system (e.g., `mcp-registry-admin`)
- Assign users to appropriate groups
- Configure OAuth2 flows for web application access

#### Resource Server Setup (for M2M)
- Create resource server with identifier (e.g., `mcp-gateway-api`)
- Define custom scopes matching server scope names
- Configure client credentials flow for agent applications

For detailed Cognito setup instructions, see [`docs/cognito.md`](./cognito.md).

## Scope Validation Logic

The scope validation is implemented in the [`validate_server_tool_access()`](../auth_server/server.py) function, which follows a systematic approach to determine access permissions.

### Validation Algorithm

```python
def validate_server_tool_access(server_name: str, method: str, tool_name: str, user_scopes: List[str]) -> bool:
    """
    Validate if the user has access to the specified server method/tool based on scopes.
    
    Returns True if access is allowed, False otherwise
    """
```

### Step-by-Step Validation Process

1. **Input Validation**: Validate server name, method, tool name, and user scopes
2. **Scope Iteration**: Check each user scope for matching permissions
3. **Server Matching**: Find server configurations that match the requested server
4. **Method Validation**: Check if the requested method is allowed
5. **Tool Validation**: For `tools/call`, validate specific tool permissions
6. **Access Decision**: Grant access if any scope allows the operation

### Validation Flow Diagram

```
Request: server_name, method, tool_name, user_scopes
    ↓
For each user_scope:
    ↓
Find scope configuration
    ↓
For each server in scope:
    ↓
Does server name match?
    ↓ (Yes)
Is method in allowed_methods?
    ↓ (Yes)
Is method == 'tools/call'?
    ↓ (Yes)              ↓ (No)
Is tool_name in          Grant Access
allowed_tools?
    ↓ (Yes)    ↓ (No)
Grant Access   Continue to next scope
```

### Access Decision Logic

- **Default Deny**: Access is denied by default if no scope grants permission
- **First Match Wins**: Access is granted as soon as any scope allows the operation
- **Explicit Permission Required**: Both method and tool permissions must be explicitly granted
- **Error Handling**: Access is denied if validation encounters errors

## Configuration Examples

### Example 1: Basic User Setup

Create a basic user with read-only access to specific servers. The examples
below use YAML for readability; the equivalent scope documents are stored in
the DocumentDB `mcp_scopes` collection (one document per scope, keyed by `_id`)
and can be seeded from a JSON file in [`scripts/`](../scripts/) or created via
the scope management API:

```yaml
# Scope document (stored in the mcp_scopes collection)
group_mappings:
  mcp-registry-basic-user:
    - mcp-registry-user
    - mcp-servers-restricted/read

mcp-servers-restricted/read:
  - server: currenttime
    methods:
      - initialize
      - notifications/initialized
      - ping
      - tools/list
    tools:
      - current_time_by_timezone
```

**Cognito Setup:**
1. Create group: `mcp-registry-basic-user`
2. Assign users to this group
3. Users can list and read time tools but cannot execute them

### Example 2: Developer with Service Management

Create a developer role with service registration capabilities:

```yaml
group_mappings:
  mcp-registry-developer:
    - mcp-registry-developer
    - mcp-servers-restricted/read
    - mcp-servers-restricted/execute

UI-Scopes:
  mcp-registry-developer:
    list_service: [all]
    register_service: [all]
    health_check_service: [all]
```

### Example 3: Agent with Specific Tool Access

Configure an agent with access to specific tools:

```yaml
# Agent scope (assigned directly in Cognito resource server)
mcp-servers-restricted/execute:
  - server: currenttime
    methods:
      - initialize
      - notifications/initialized
      - ping
      - tools/list
      - tools/call
    tools:
      - current_time_by_timezone
      - get_config
```

**Cognito Setup:**
1. Create resource server: `mcp-gateway-api`
2. Create custom scope: `mcp-servers-restricted/execute`
3. Assign scope to agent client

### Example 4: Administrative Access

Full administrative access configuration:

```yaml
group_mappings:
  mcp-registry-admin:
    - mcp-registry-admin
    - mcp-servers-unrestricted/read
    - mcp-servers-unrestricted/execute

UI-Scopes:
  mcp-registry-admin:
    list_service: [all]
    register_service: [all]
    health_check_service: [all]
    toggle_service: [all]
    modify_service: [all]
```

## Virtual MCP Server Access Control

Virtual MCP Servers use the same access control model as regular MCP servers. The key difference is that you reference the virtual server by its path (e.g., `/virtual/scoped-tools`) instead of a backend server name.

### How It Works

Virtual servers are treated identically to regular MCP servers in scope definitions:

1. **Server Identification**: Use the virtual server path as the `server` value
2. **Method Control**: Same MCP methods apply (`initialize`, `tools/list`, `tools/call`, etc.)
3. **Tool Control**: You can restrict access to specific tools exposed by the virtual server

### Example: Virtual Server Scope Configuration

```json
{
  "scope_name": "virtual-scoped-tools-users",
  "description": "Users with access to the scoped virtual server",
  "server_access": [
    {
      "server": "/virtual/scoped-tools",
      "methods": ["initialize", "notifications/initialized", "ping", "tools/list", "tools/call"],
      "tools": ["*"]
    },
    {
      "server": "api",
      "methods": ["GET", "POST", "servers", "virtual-servers", "search"],
      "tools": []
    }
  ],
  "group_mappings": ["virtual-scoped-tools-users"],
  "custom_scopes": ["virtual-scoped-tools/access"],
  "create_in_idp": true
}
```

See [virtual-server-scoped-users.json](../cli/examples/virtual-server-scoped-users.json) for the complete example.

### Key Points

| Aspect | Regular MCP Server | Virtual MCP Server |
|--------|-------------------|-------------------|
| Server identifier | Server name (e.g., `currenttime`) | Virtual path (e.g., `/virtual/scoped-tools`) |
| Methods | Standard MCP methods | Same standard MCP methods |
| Tools | Backend server tools | Aggregated tools (possibly aliased) |
| Scope configuration | Identical | Identical |

### Virtual Server-Level Scopes

Virtual servers also support their own `required_scopes` field, which provides an additional layer of access control:

```json
{
  "path": "/virtual/scoped-tools",
  "required_scopes": ["virtual-scoped-tools/access"],
  "tool_scope_overrides": [
    {
      "tool_alias": "sensitive-tool",
      "required_scopes": ["virtual-scoped-tools/admin"]
    }
  ]
}
```

This means access control happens at two levels:
1. **Gateway level**: Defined in the `mcp_scopes` collection (seeded from scope JSON files or created via the scope management API)
2. **Virtual server level**: Defined in the virtual server's `required_scopes`

Both must be satisfied for access to be granted.

### Testing Virtual Server Access Control

An E2E test script is provided for testing scope-based access control with virtual servers:

```bash
./tests/integration/test_virtual_server_scopes_e2e.sh \
    --registry-url http://localhost \
    --token-file .token \
    --no-cleanup
```

See [Virtual Server Operations Guide](virtual-server-operations.md#scope-based-access-control) for more details.

## Security Considerations

### Principle of Least Privilege

The access control system is designed around the principle of least privilege:

- **Default Deny**: All access is denied by default unless explicitly granted
- **Explicit Permissions**: Each permission must be explicitly configured
- **Granular Control**: Permissions can be granted at the method and tool level
- **Scope Separation**: UI and server permissions are managed separately

### Best Practices

#### 1. Group Design
- Create specific groups for different roles (admin, user, developer, operator)
- Avoid overly broad permissions
- Regularly review group memberships

#### 2. Scope Configuration
- Use restricted scopes for most users
- Reserve unrestricted access for administrators only
- Implement tool-level restrictions for sensitive operations

#### 3. Monitoring and Auditing
- Enable detailed logging for access decisions
- Monitor failed access attempts
- Regularly audit scope configurations

#### 4. Production Deployment
- Use separate Cognito user pools for different environments
- Implement proper secret management for client credentials
- Enable MFA for administrative accounts

### Security Boundaries

The system enforces several security boundaries:

- **Authentication Boundary**: Users must authenticate via Cognito
- **Authorization Boundary**: Scopes control what authenticated users can access
- **Server Boundary**: Each server's tools are independently controlled
- **Method Boundary**: Protocol methods and tools have separate permissions

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: User Cannot Access Server

**Symptoms:**
- User receives "Access denied" errors
- Server appears unavailable to user

**Diagnosis:**
1. Check user's Cognito group membership
2. Verify group mapping in the `mcp_scopes` collection (DocumentDB)
3. Confirm server is listed in user's scopes

**Solution:**
```yaml
# Ensure user's group has appropriate server scope
group_mappings:
  user-group-name:
    - mcp-servers-restricted/read  # Add appropriate scope
```

#### Issue 2: Tool Call Fails Despite Method Access

**Symptoms:**
- User can list tools but cannot call specific tools
- `tools/call` method fails with permission error

**Diagnosis:**
1. Verify user has `tools/call` method permission
2. Check if specific tool is listed in allowed tools
3. Confirm tool name matches exactly

**Solution:**
```yaml
mcp-servers-restricted/execute:
  - server: server-name
    methods:
      - tools/call  # Method permission
    tools:
      - specific-tool-name  # Tool permission
```

#### Issue 3: Scope Configuration Not Loading

**Symptoms:**
- All access is allowed (fallback behavior)
- Scope validation logs show "No scopes configuration loaded"

**Diagnosis:**
1. Confirm the `mcp_scopes` collection in DocumentDB is populated (the init
   job seeds it from the scope JSON files in `scripts/`)
2. Check the auth server logs for scope-loading errors at startup
3. Verify the auth server can reach DocumentDB

**Solution:**
```bash
# Confirm the scopes collection is seeded (adjust namespace as needed)
docker exec mcp-mongodb mongosh --quiet --eval \
  'db.getSiblingDB("mcp_registry").mcp_scopes_default.countDocuments({})'

# Re-run the init job to seed default scopes if the collection is empty
python scripts/init-documentdb-indexes.py
```

#### Issue 4: Group Mapping Not Working

**Symptoms:**
- User has correct Cognito group but wrong scopes
- Scope mapping appears incorrect

**Diagnosis:**
1. Verify group name matches exactly in Cognito and the scope documents (the `mcp_scopes` collection)
2. Check for typos in group names
3. Confirm group mapping syntax

**Solution:**
```yaml
# Ensure exact match between Cognito group name and mapping key
group_mappings:
  exact-cognito-group-name:  # Must match Cognito exactly
    - scope-name
```

### Debugging Tools

#### Enable Verbose Logging

The validation function provides detailed logging for troubleshooting:

```python
# Logs show complete validation process
logger.info(f"=== VALIDATE_SERVER_TOOL_ACCESS START ===")
logger.info(f"Requested server: '{server_name}'")
logger.info(f"Requested method: '{method}'")
logger.info(f"Requested tool: '{tool_name}'")
logger.info(f"User scopes: {user_scopes}")
```

#### Test Scope Configuration

Inspect the scope documents stored in DocumentDB to validate the configuration:

```bash
# List all scope documents and their group mappings (adjust namespace as needed)
docker exec mcp-mongodb mongosh --quiet --eval '
db = db.getSiblingDB("mcp_registry");
db.mcp_scopes_default.find({}, {_id: 1, group_mappings: 1}).forEach(function(d) {
  print("Scope: " + d._id + " -> groups: " + JSON.stringify(d.group_mappings));
});
'
```

### Performance Considerations

- **Scope Caching**: Scope configurations are loaded once at startup
- **Validation Efficiency**: Validation stops at first matching scope
- **Memory Usage**: Large scope configurations may impact memory usage
- **Logging Overhead**: Verbose logging can impact performance in production

For production deployments, consider:
- Reducing log verbosity
- Monitoring validation performance
- Optimizing scope configuration structure
- Implementing scope configuration caching strategies

---

This documentation provides a comprehensive guide to understanding and configuring the fine-grained access control system. For additional information about Cognito setup and integration, refer to [`docs/cognito.md`](./cognito.md) and [`docs/auth.md`](./auth.md).