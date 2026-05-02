# Agent Visibility and Group-Based Access Control

This document explains how the MCP Gateway Registry controls who can see and use agents using two layers: **group scope configs** (admin-managed) and **agent-level allowed_groups** (publisher-managed).

## How Group Scopes Work Today

An admin creates a group scope that defines exactly which agents a group can access. Group scopes can be created through:

- The **UI** (IAM group management page)
- The **CLI** (`registry_management.py scope-create`)
- The **API** directly (see [openapi.json](https://github.com/agentic-community/mcp-gateway-registry/blob/main/api/openapi.json))

The scope is synced to the identity provider (Keycloak, Entra, Cognito, Okta).

### Narrow Scope Example: public-mcp-users

Users in the `public-mcp-users` group can only see the `/flight-booking` agent:

```json
{
  "scope_name": "public-mcp-users",
  "group_mappings": ["public-mcp-users"],
  "ui_permissions": {
    "list_agents": ["/flight-booking"],
    "get_agent": ["/flight-booking"]
  }
}
```

This is a **narrow scope**: the admin explicitly lists which agents the group can access.

### Broad Scope Example: registry-admins

Admin users can see all agents:

```json
{
  "_id": "registry-admins",
  "group_mappings": ["registry-admins"],
  "ui_permissions": {
    "list_agents": ["all"],
    "get_agent": ["all"],
    "publish_agent": ["all"],
    "modify_agent": ["all"],
    "delete_agent": ["all"]
  }
}
```

This is a **broad scope**: `"list_agents": ["all"]` means the group can see every agent in the registry.

## The Problem with Broad Scopes

Broad scopes are convenient for large teams. An admin might configure an `engineering` group with `"list_agents": ["all"]` so engineers can discover and use any agent without filing a request each time.

But what happens when someone publishes a sensitive agent? Say the HR team publishes a `/salary-calculator` agent. With a broad scope, every engineer can see it. The HR team lead does not want that, but they cannot change the group scope config because that requires an admin.

This is where `allowed_groups` comes in.

## What allowed_groups Does

When registering or editing an agent, the publisher can set `visibility: "group-restricted"` and specify `allowed_groups`. This acts as a second filter **on top of** the IAM group scope.

The two layers work as an AND:

1. **IAM scope check**: Is the agent in the user's `accessible_agents` list (from their group scope config)?
2. **allowed_groups check**: If the agent is `group-restricted`, do the user's JWT groups intersect with the agent's `allowed_groups`?

A user must pass **both** checks to see the agent.

## Concrete Scenario

### Setup

An enterprise has three groups configured in the identity provider:

| Group | Scope Type | Agent Access |
|-------|-----------|--------------|
| `engineering` | Broad | `"list_agents": ["all"]` |
| `hr-team` | Broad | `"list_agents": ["all"]` |
| `public-mcp-users` | Narrow | `"list_agents": ["/flight-booking"]` |

The registry has three agents:

| Agent | Visibility | allowed_groups |
|-------|-----------|----------------|
| `/flight-booking` | `public` | `[]` |
| `/code-reviewer` | `public` | `[]` |
| `/salary-calculator` | `group-restricted` | `["hr-team"]` |

### Who Sees What

**Alice (in `engineering` group):**
- `/flight-booking`: IAM scope = `["all"]`, so passes. Visibility = `public`, no group check. **Sees it.**
- `/code-reviewer`: Same logic. **Sees it.**
- `/salary-calculator`: IAM scope = `["all"]`, so passes. But visibility = `group-restricted` and Alice's groups (`engineering`) do not intersect with `allowed_groups` (`hr-team`). **Does NOT see it.**

**Bob (in `hr-team` group):**
- `/flight-booking`: IAM scope = `["all"]`, passes. Visibility = `public`. **Sees it.**
- `/code-reviewer`: Same. **Sees it.**
- `/salary-calculator`: IAM scope = `["all"]`, passes. Visibility = `group-restricted` and Bob's groups (`hr-team`) intersect with `allowed_groups` (`hr-team`). **Sees it.**

**Carol (in `public-mcp-users` group):**
- `/flight-booking`: IAM scope = `["/flight-booking"]`, passes. Visibility = `public`. **Sees it.**
- `/code-reviewer`: IAM scope = `["/flight-booking"]`, does NOT include `/code-reviewer`. **Does NOT see it.** (Filtered at IAM layer, `allowed_groups` is never checked.)
- `/salary-calculator`: IAM scope does NOT include `/salary-calculator`. **Does NOT see it.**

### The Key Takeaway

- For **narrow-scoped groups** like `public-mcp-users`, the IAM scope already controls per-agent access. The `allowed_groups` field has no effect because the IAM layer filters first.
- For **broad-scoped groups** like `engineering`, the IAM scope grants access to everything. The `allowed_groups` field is the publisher's mechanism to restrict visibility within that broad grant, without needing to ask an admin to create a narrower scope.

## When to Use allowed_groups

| Your group scope config | Use allowed_groups? | Why |
|------------------------|---------------------|-----|
| Narrow (`["/agent-a", "/agent-b"]`) | No benefit | IAM already controls per-agent access |
| Broad (`["all"]`) | Yes, for sensitive agents | Lets the publisher restrict who sees their agent |
| Mix of narrow and broad groups | Yes, for agents that broad groups should not all see | Narrows access for broad groups while narrow groups are unaffected (filtered at IAM layer first) |

## Visibility Modes

| Visibility | IAM Check | allowed_groups Check | Who Can See |
|------------|-----------|---------------------|-------------|
| `public` | Must have IAM scope | No | All users with IAM access |
| `group-restricted` | Must have IAM scope | Must be in allowed_groups | Users with IAM access AND in allowed groups |
| `private` | Must have IAM scope | No | Only the agent owner |
| `unlisted` | Must have IAM scope | No | Users with the direct URL |

## API Examples

### Register a Group-Restricted Agent

The HR team lead publishes a salary calculator that only `hr-team` can see:

```bash
curl -s -X POST "https://your-registry/api/agents/register" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Salary Calculator",
    "path": "/salary-calculator",
    "version": "1.0.0",
    "url": "https://example.com/salary-calculator",
    "supportedProtocol": "a2a",
    "description": "Calculate salary projections and tax estimates",
    "visibility": "group-restricted",
    "allowedGroups": ["hr-team"],
    "skills": [
      {
        "id": "calculate-salary",
        "name": "Calculate Salary",
        "description": "Calculate salary projections",
        "tags": ["hr", "finance"],
        "inputSchema": {}
      }
    ]
  }'
```

### Register a Public Agent (No Group Restriction)

A general-purpose agent visible to anyone with IAM access:

```bash
curl -s -X POST "https://your-registry/api/agents/register" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Flight Booking",
    "path": "/flight-booking",
    "version": "1.0.0",
    "url": "https://example.com/flight-booking",
    "supportedProtocol": "a2a",
    "description": "Book flights for business travel",
    "visibility": "public",
    "skills": [
      {
        "id": "book-flight",
        "name": "Book Flight",
        "description": "Search and book flights",
        "tags": ["travel"],
        "inputSchema": {}
      }
    ]
  }'
```

### Update allowed_groups

Expand access to include the `finance-team`:

```bash
curl -s -X PUT "https://your-registry/api/agents/salary-calculator" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Salary Calculator",
    "path": "/salary-calculator",
    "version": "1.0.0",
    "url": "https://example.com/salary-calculator",
    "supportedProtocol": "a2a",
    "description": "Calculate salary projections and tax estimates",
    "visibility": "group-restricted",
    "allowedGroups": ["hr-team", "finance-team"],
    "skills": [
      {
        "id": "calculate-salary",
        "name": "Calculate Salary",
        "description": "Calculate salary projections",
        "tags": ["hr", "finance"],
        "inputSchema": {}
      }
    ]
  }'
```

### List Agents Filtered by allowed_groups

```bash
# Show only agents shared with hr-team
curl -s "https://your-registry/api/agents?allowed_groups=hr-team" \
  -H "Authorization: Bearer $TOKEN" | jq .

# Show agents shared with either hr-team or finance-team
curl -s "https://your-registry/api/agents?allowed_groups=hr-team,finance-team" \
  -H "Authorization: Bearer $TOKEN" | jq .
```

### Group Scope Config for a Broad-Access Team

To create an `engineering` group with broad agent access (where `allowed_groups` becomes useful):

```json
{
  "scope_name": "engineering",
  "description": "Engineering team with broad agent access",
  "group_mappings": ["engineering"],
  "ui_permissions": {
    "list_agents": ["all"],
    "get_agent": ["all"],
    "list_service": ["all"]
  },
  "create_in_idp": true
}
```

Upload via CLI:

```bash
python api/registry_management.py --registry-url https://your-registry \
  --token-file .token-admin import-group --file engineering.json
```

Or via curl:

```bash
curl -s -X POST "https://your-registry/api/servers/groups/import" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d @engineering.json
```

With this config, all engineers see every `public` agent but cannot see `group-restricted` agents unless their group is in the agent's `allowed_groups`.
