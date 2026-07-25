# Scopes Management

This document describes the scope configuration file format used by the MCP Gateway Registry for fine-grained access control.

## Overview

Scopes define what resources (MCP servers, agents) users can access and what actions they can perform. The registry uses JSON-based scope configuration files that can be loaded during initialization or managed via the CLI.

## Upgrade: skills and agents are now discovery-gated

Skill and agent discovery is gated on a `list_*` UI permission, matching MCP servers. This is a fail-closed change: a non-admin group that does not hold the grant sees **zero** resources of that family, including public ones. If skills or agents suddenly stop appearing for a non-admin user after upgrade, the user's group is missing the discovery scope. Take these actions once, per deployment:

- **Skills discovery (`list_skills`).** This scope is new, so no group has it until you grant it. Run the one-time backfill to grant `list_skills: ["all"]` to the built-in admin group (dry-run by default). The simplest place to run it is inside a running container, where the registry environment is already present:

  ```bash
  # Docker Compose
  docker compose exec registry uv run python scripts/backfill-skill-list-scope.py --apply

  # Kubernetes / EKS
  kubectl exec -it deploy/registry -- uv run python scripts/backfill-skill-list-scope.py --apply
  ```

  To run it outside the deployment (host shell, one-off ECS task, admin pod), pass the connection explicitly; the password and `SECRET_KEY` are read from the environment, never as CLI args:

  ```bash
  DOCUMENTDB_PASSWORD=... SECRET_KEY=... \
    uv run python scripts/backfill-skill-list-scope.py --apply \
    --host <mongo-host> --username admin --direct-connection \
    --auth-server-url <auth-server-url>
  ```

  See `--help` for the full connection-argument list (`--tls`, `--storage-backend documentdb` for Amazon DocumentDB, etc.). Then grant `list_skills` to any non-admin group that should see skills. Admins are unaffected (they bypass the check).

- **Skill mutation (`modify_skill` / `delete_skill` / `toggle_skill`).** A non-admin who **owns** a skill now also needs the matching mutation scope; ownership alone is no longer sufficient. Grant these to whichever non-admin group manages skills. Not covered by the backfill.

- **Agent delete (`delete_agent`).** A non-admin who owns an agent now needs `delete_agent` to delete it (previously ownership alone was enough). Every admin surface already carries this scope, so no data migration is required; grant it to any non-admin group that should delete agents it owns.

- **Agent modify / toggle (`modify_agent` / `toggle_agent`).** These are now enforced against the agent family's own scopes. A non-admin who previously managed agents via the server scopes (`modify_service` / `toggle_service`) must instead be granted `modify_agent` / `toggle_agent`.

The tabs for Servers and custom entity types are hidden when the user lacks the corresponding `list_*` scope. The Skills and Agents tabs stay visible and show an in-page hint explaining that discovery access is admin-managed, so a user whose access changed learns what to ask for rather than seeing the tab disappear.

## Scope Configuration File Format

### Example Files

- `scripts/registry-admins.json` - Bootstrap admin scope loaded during database initialization
- `cli/examples/public-mcp-users.json` - Example scope for users with limited access

### Complete Field Reference

```json
{
  "_id": "scope-name",
  "scope_name": "scope-name",
  "description": "Human-readable description of this scope",
  "group_mappings": ["group-name-1", "group-uuid-2"],
  "server_access": [
    {
      "server": "server-name",
      "methods": ["initialize", "tools/list", "tools/call"],
      "tools": ["tool-name-1", "tool-name-2"]
    },
    {
      "agent": "/agent-path",
      "actions": ["list_agents", "get_agent", "invoke_agent"]
    }
  ],
  "ui_permissions": {
    "list_agents": ["all"],
    "get_agent": ["/specific-agent"],
    "publish_agent": [],
    "list_service": ["all"],
    "toggle_service": ["service-name"]
  },
  "create_in_idp": true
}
```

## Field Descriptions

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_id` | string | Yes | Unique identifier for the scope document in MongoDB. Should match `scope_name`. |
| `scope_name` | string | No | Human-readable scope name. If omitted, `_id` is used. |
| `description` | string | No | Description explaining the purpose of this scope. |
| `group_mappings` | array | Yes | List of IdP group names or IDs that map to this scope. |
| `server_access` | array | Yes | List of MCP server access rules and agent action permissions. |
| `ui_permissions` | object | No | UI-level permissions for the registry web interface. |
| `create_in_idp` | boolean | No | When true, the CLI will create the group in the IdP (Keycloak/Entra). |

### group_mappings Field

The `group_mappings` array contains IdP group identifiers that should be mapped to this scope. When a user authenticates, their IdP groups are matched against these mappings to determine their effective scopes.

**Important for Entra ID:**
- Entra ID uses Group Object IDs (GUIDs), not group names
- You must include the Group Object ID from Azure Portal > Groups > Overview
- Example: `"5f605d68-06bc-4208-b992-bb378eee12c5"`

**For Keycloak:**
- Use the group name as defined in Keycloak
- Example: `"public-mcp-users"`

**Example with both:**
```json
{
  "group_mappings": [
    "public-mcp-users",
    "5f605d68-06bc-4208-b992-bb378eee12c5"
  ]
}
```

This means users in either the Keycloak group `public-mcp-users` OR the Entra ID group with Object ID `5f605d68-06bc-4208-b992-bb378eee12c5` will receive this scope.

### server_access Field

The `server_access` array defines what MCP servers and A2A agents users can access. Each entry is either a **server rule** (`{"server", "methods", "tools"}`) or an **agent rule** (`{"agent", "actions"}`). Both use the same shape: an identifier key plus a verb list. `agent` mirrors `server` (the resource identifier) and `actions` mirrors `methods` (the allowed verbs). MCP scope resolution inspects only `server`-keyed entries and the A2A validator inspects only `agent`-keyed entries, so the two rule types coexist safely in one array.

#### Server Access Rule

```json
{
  "server": "server-name-or-wildcard",
  "methods": ["method-1", "method-2"],
  "tools": ["tool-name-or-wildcard"]
}
```

| Field | Description |
|-------|-------------|
| `server` | Server name or `"*"` for all servers |
| `methods` | List of allowed MCP methods (see below) |
| `tools` | List of allowed tool names or `["*"]` for all tools |

**Standard MCP Methods:**
- `initialize` - Initialize MCP session
- `notifications/initialized` - Session initialized notification
- `ping` - Health check
- `tools/list` - List available tools
- `tools/call` - Execute a tool
- `resources/list` - List available resources
- `resources/templates/list` - List resource templates
- `GET`, `POST`, `PUT`, `DELETE` - HTTP methods for REST API access

**Example - Full MCP access to specific servers:**
```json
{
  "server": "context7",
  "methods": [
    "initialize",
    "notifications/initialized",
    "ping",
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/templates/list"
  ],
  "tools": ["*"]
}
```

**Example - Wildcard access (admin):**
```json
{
  "server": "*",
  "methods": ["all"],
  "tools": ["all"]
}
```

#### Agent Rule

An agent rule defines what operations users can perform on a specific A2A agent (or all agents). It has the same shape as a server rule: the `agent` key is the resource identifier and `actions` is the allowed-verb list.

```json
{
  "agent": "/agent-path",
  "actions": ["list_agents", "get_agent", "invoke_agent"]
}
```

| Field | Description |
|-------|-------------|
| `agent` | Agent path (e.g., `/flight-booking`) or `"*"`/`"all"` for all agents |
| `actions` | List of allowed agent actions, or `["all"]`/`["*"]` for every action |

**Available Agent Actions:**

| Action | Description | API Endpoint |
|--------|-------------|--------------|
| `list_agents` | View agents in listings | `GET /api/agents` |
| `get_agent` | View agent details | `GET /api/agents/{path}` |
| `publish_agent` | Register new agents | `POST /api/agents/register` |
| `modify_agent` | Update existing agents | `PUT /api/agents/{path}` |
| `delete_agent` | Remove agents | `DELETE /api/agents/{path}` |
| `invoke_agent` | Call the agent through the gateway (reverse-proxy mode) | `POST /agent/{path}/` |

To grant access to multiple specific agents, add one rule per agent (each with its own `actions`):

**Example - Limited agent access (two named agents):**
```json
{
  "server_access": [
    {"agent": "/flight-booking", "actions": ["list_agents", "get_agent", "invoke_agent"]},
    {"agent": "/code-reviewer", "actions": ["list_agents", "get_agent"]}
  ]
}
```

**Example - Full agent admin access:**
```json
{
  "agent": "*",
  "actions": ["list_agents", "get_agent", "publish_agent", "modify_agent", "delete_agent", "invoke_agent"]
}
```

> **Do not use the older nested form** `{"agents": {"actions": [{"action": ..., "resources": [...]}]}}`. It is silently dropped by the scope flattener (`_flatten_server_access`) and grants nothing. Use the flat `{"agent", "actions"}` shape shown above.

### ui_permissions Field

UI permissions control what actions users can perform in the web interface and REST API for service/agent management.

```json
{
  "ui_permissions": {
    "permission_name": ["resource-1", "resource-2"]
  }
}
```

**Available UI Permissions:**

| Permission | Description | Applies To |
|------------|-------------|------------|
| `list_agents` | View agents in UI | Agent paths or `"all"` |
| `get_agent` | View agent details | Agent paths or `"all"` |
| `publish_agent` | Register new agents via UI | Agent paths or `"all"` |
| `modify_agent` | Edit agents via UI | Agent paths or `"all"` |
| `delete_agent` | Delete agents via UI | Agent paths or `"all"` |
| `toggle_agent` | Enable/disable agents via UI | Agent paths or `"all"` |
| `list_service` | View MCP servers in UI | Server names or `"all"` |
| `register_service` | Register new MCP servers | Server names or `"all"` |
| `health_check_service` | Run health checks | Server names or `"all"` |
| `toggle_service` | Enable/disable servers | Server names or `"all"` |
| `modify_service` | Edit server configurations | Server names or `"all"` |
| `delete_service` | Delete servers via UI | Server names or `"all"` |
| `list_skills` | View skills in UI | Skill paths or `"all"` |
| `publish_skill` | Register new skills via UI | Skill paths or `"all"` |
| `modify_skill` | Edit skills via UI | Skill paths or `"all"` |
| `delete_skill` | Delete skills via UI | Skill paths or `"all"` |
| `toggle_skill` | Enable/disable skills via UI | Skill paths or `"all"` |
| `list_<type>_entity` | View records of a custom entity type | Record paths or `"all"` |
| `create_<type>_entity` | Create records of a custom entity type | Record paths or `"all"` |
| `modify_<type>_entity` | Edit records of a custom entity type | Record paths or `"all"` |
| `delete_<type>_entity` | Delete records of a custom entity type | Record paths or `"all"` |

### How each operation is authorized

Authorization is uniform across all four asset families (servers, agents, skills, custom entities). An **admin** bypasses every check below and can do anything. For a **non-admin**, each operation resolves as follows:

| Operation | Gate for a non-admin caller |
| --- | --- |
| **Discover** (`list_*`) | scope only (holds the `list_` grant for the resource or `"all"`) |
| **Create** (`register_`/`publish_`/`create_`) | scope only (a non-empty create grant) |
| **Modify** (`modify_*`) | scope **AND** owner |
| **Delete** (`delete_*`) | scope **AND** owner |
| **Toggle** (`toggle_*`) | scope **AND** owner |

**Dual gate (mutations).** Modify, delete, and toggle each require the caller to hold the scope for the specific resource (or `"all"`) **AND** be the resource's owner. Both halves are required: a scope grant alone does not let a non-admin mutate a resource someone else owns, and ownership alone does not let them mutate it without the scope. This is identical across servers, agents, skills, and custom entities.

**Ownership** is the `registered_by` field for servers and agents, and the `owner` field for skills and custom entities (set to the creating user at registration).

**Discovery (`list_*`) is scope-only and fail-closed.** A caller without the `list_` grant for a family sees zero resources of that family, including public ones. Discovery does not depend on ownership.

The per-family scope names for each operation are:

| Operation | Server | Agent | Skill | Custom entity `<type>` |
| --- | --- | --- | --- | --- |
| Discover | `list_service` | `list_agents` | `list_skills` | `list_<type>_entity` |
| Create | `register_service` | `publish_agent` | `publish_skill` | `create_<type>_entity` |
| Modify | `modify_service` | `modify_agent` | `modify_skill` | `modify_<type>_entity` |
| Delete | `delete_service` | `delete_agent` | `delete_skill` | `delete_<type>_entity` |
| Toggle | `toggle_service` | `toggle_agent` | `toggle_skill` | (n/a) |
| Health check | `health_check_service` | (n/a) | (n/a) | (n/a) |

**Example - Read-only access:**
```json
{
  "ui_permissions": {
    "list_service": ["all"],
    "list_agents": ["/flight-booking"],
    "get_agent": ["/flight-booking"]
  }
}
```

**Example - Full admin access:**
```json
{
  "ui_permissions": {
    "list_agents": ["all"],
    "get_agent": ["all"],
    "publish_agent": ["all"],
    "modify_agent": ["all"],
    "delete_agent": ["all"],
    "list_service": ["all"],
    "register_service": ["all"],
    "health_check_service": ["all"],
    "toggle_service": ["all"],
    "modify_service": ["all"]
  }
}
```

## Complete Examples

### Admin Scope (registry-admins.json)

Full access to all servers, agents, and UI functions:

```json
{
  "_id": "registry-admins",
  "group_mappings": ["registry-admins"],
  "server_access": [
    {
      "server": "*",
      "methods": ["all"],
      "tools": ["all"]
    },
    {
      "agent": "*",
      "actions": ["list_agents", "get_agent", "publish_agent", "modify_agent", "delete_agent", "invoke_agent"]
    }
  ],
  "ui_permissions": {
    "list_agents": ["all"],
    "get_agent": ["all"],
    "publish_agent": ["all"],
    "modify_agent": ["all"],
    "delete_agent": ["all"],
    "list_service": ["all"],
    "register_service": ["all"],
    "health_check_service": ["all"],
    "toggle_service": ["all"],
    "modify_service": ["all"]
  }
}
```

### Limited User Scope (public-mcp-users.json)

Access to specific MCP servers and one agent:

```json
{
  "scope_name": "public-mcp-users",
  "description": "Users with access to public MCP servers and flight-booking agent",
  "server_access": [
    {
      "server": "context7",
      "methods": [
        "initialize",
        "notifications/initialized",
        "ping",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/templates/list"
      ],
      "tools": ["*"]
    },
    {
      "server": "api",
      "methods": ["initialize", "GET", "POST", "servers", "agents", "search"],
      "tools": []
    },
    {
      "agent": "/flight-booking",
      "actions": ["list_agents", "get_agent", "invoke_agent"]
    },
    {
      "agent": "/flight-booking-agent",
      "actions": ["list_agents", "get_agent", "invoke_agent"]
    }
  ],
  "group_mappings": [
    "public-mcp-users",
    "5f605d68-06bc-4208-b992-bb378eee12c5"
  ],
  "ui_permissions": {
    "list_service": ["all"],
    "list_agents": ["/flight-booking", "/flight-booking-agent"],
    "get_agent": ["/flight-booking", "/flight-booking-agent"]
  },
  "create_in_idp": true
}
```

## Managing Scopes

### Using the CLI

Import a scope from JSON file:
```bash
uv run python api/registry_management.py \
  --token-file .token \
  --registry-url https://registry.example.com \
  import-group cli/examples/public-mcp-users.json
```

List all scopes:
```bash
uv run python api/registry_management.py \
  --token-file .token \
  --registry-url https://registry.example.com \
  list-groups
```

### Bootstrap Admin Scope

The `registry-admins` scope is automatically loaded during database initialization:
- **Local (MongoDB CE)**: `docker compose up mongodb-init`
- **Production (DocumentDB)**: `./terraform/aws-ecs/scripts/run-documentdb-init.sh`

### Server Path Variations

When defining server access, you may need to include path variations to handle different URL patterns:

```json
{
  "server_access": [
    {"server": "context7", "methods": [...], "tools": ["*"]},
    {"server": "/context7", "methods": [...], "tools": ["*"]},
    {"server": "/context7/", "methods": [...], "tools": ["*"]}
  ]
}
```

This ensures access works regardless of whether the server is accessed as:
- `context7`
- `/context7`
- `/context7/`

## Entra ID Integration

When using Microsoft Entra ID (Azure AD) as the identity provider:

1. **Create a group in Azure Portal:**
   - Navigate to Azure Portal > Azure Active Directory > Groups
   - Create a new Security group
   - Note the Group Object ID (GUID)

2. **Add the Object ID to group_mappings:**
   ```json
   {
     "group_mappings": [
       "my-keycloak-group",
       "12345678-1234-1234-1234-123456789012"
     ]
   }
   ```

3. **Assign users to the Azure AD group:**
   - Users in this group will receive the scope permissions when they authenticate

4. **Configure Entra ID app to include groups in tokens:**
   - In the App Registration, configure the `groups` claim
   - Set `groupMembershipClaims` to `"SecurityGroup"` in the manifest

## Troubleshooting

### User Not Getting Expected Permissions

1. Check group membership in IdP (Keycloak/Entra)
2. Verify `group_mappings` includes the correct group name/ID
3. Check registry logs for scope mapping messages
4. Use the debug endpoint: `GET /api/debug/user-context`

### Scope Not Found

1. Ensure the scope was imported: `list-groups` command
2. Check MongoDB collection: `mcp_scopes_default`
3. Re-run database initialization if bootstrap scope missing

### Entra ID Groups Not Working

1. Verify Group Object ID (not display name) is in `group_mappings`
2. Check that `groupMembershipClaims` is configured in app manifest
3. Verify user is assigned to the group in Azure Portal
4. Check that optional claims include `groups` in ID token
