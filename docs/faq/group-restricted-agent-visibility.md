# How do I restrict which agents a user can see based on their group?

The registry has two layers of access control for agents. Understanding when each layer applies helps you choose the right approach.

## Quick Answer

**"I want only specific groups to see my agent."**

Set `visibility: "group-restricted"` and `allowedGroups` when registering the agent:

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
    "description": "Calculate salary projections",
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

Only users whose IdP groups include `hr-team` or `finance-team` will see this agent. Admin users always see all agents.

## When Does This Actually Matter?

There are two layers of access control, and `allowed_groups` only adds value depending on how your IAM group scopes are configured:

| Your group scope config | Does allowed_groups help? | Why |
|------------------------|--------------------------|-----|
| **Narrow** (e.g., `"list_agents": ["/flight-booking"]`) | No | IAM already controls per-agent access |
| **Broad** (e.g., `"list_agents": ["all"]`) | Yes | Publisher can restrict who sees their agent without an admin |
| **Mix of narrow and broad** | Yes, for agents that broad groups should not all see | Narrows access for broad groups |

For a full explanation with examples, see [Agent Visibility and Group-Based Access Control](../agent-visibility-and-group-access.md).

## How to Set Up Group-Restricted Access

### Step 1: Make Sure Your Group Has IAM Access

Your group scope config must include agent access. If it uses `"list_agents": ["all"]`, you're set. If it lists specific agents, the agent must be in that list.

### Step 2: Register the Agent as Group-Restricted

**Via API:**

```bash
curl -s -X POST "https://your-registry/api/agents/register" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Internal Finance Agent",
    "path": "/finance-agent",
    "version": "1.0.0",
    "url": "https://finance-agent.internal.example.com",
    "supportedProtocol": "a2a",
    "description": "Agent for internal finance operations",
    "visibility": "group-restricted",
    "allowedGroups": ["finance-team", "finance-admins"],
    "skills": [
      {
        "id": "run-report",
        "name": "Run Report",
        "description": "Run financial reports",
        "tags": ["finance"],
        "inputSchema": {}
      }
    ]
  }'
```

**Via the Web UI:**

The agent registration and edit forms include a Visibility dropdown with the "Group Restricted" option. When selected, an input field appears for specifying the allowed groups.

### Step 3: Update an Existing Agent

```bash
curl -s -X PUT "https://your-registry/api/agents/finance-agent" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Internal Finance Agent",
    "path": "/finance-agent",
    "version": "1.0.0",
    "url": "https://finance-agent.internal.example.com",
    "supportedProtocol": "a2a",
    "description": "Agent for internal finance operations",
    "visibility": "group-restricted",
    "allowedGroups": ["finance-team", "finance-admins", "executive-team"],
    "skills": [
      {
        "id": "run-report",
        "name": "Run Report",
        "description": "Run financial reports",
        "tags": ["finance"],
        "inputSchema": {}
      }
    ]
  }'
```

## Filtering Agents by Visibility or Group

```bash
# List only group-restricted agents
curl -s "https://your-registry/api/agents?visibility=group-restricted" \
  -H "Authorization: Bearer $TOKEN"

# List only agents shared with hr-team
curl -s "https://your-registry/api/agents?allowed_groups=hr-team" \
  -H "Authorization: Bearer $TOKEN"

# List agents shared with either hr-team or finance-team
curl -s "https://your-registry/api/agents?allowed_groups=hr-team,finance-team" \
  -H "Authorization: Bearer $TOKEN"
```

The filter still respects the caller's group membership. A non-admin user filtering by `allowed_groups=hr-team` will only see agents they have both IAM access to and group membership for.

## Visibility Options

| Value | Behavior |
|-------|----------|
| `public` | Visible to all users with IAM access (default) |
| `group-restricted` | Visible only to users with IAM access whose groups overlap with `allowed_groups`. Admins always see all agents. |
| `private` | Visible only to the agent owner and admin users |
| `unlisted` | Visible only to users with the direct URL |

## How Group Matching Works

When a user calls `GET /api/agents`, two checks run in sequence:

1. **IAM scope check**: The user's group scope config determines their `accessible_agents` list. Agents not in this list are filtered out.
2. **allowed_groups check** (only for `group-restricted` agents): The user's IdP groups (from their JWT token) must intersect with the agent's `allowed_groups`. If not, the agent is filtered out.

Admin users bypass both checks and see all agents.

## IdP Independence

The `allowed_groups` field works with any IdP (Keycloak, Entra ID, Cognito, Okta, Auth0) because matching is done against the groups present in the user's JWT token claims. The registry does not call any IdP API to verify group membership.

For Entra ID, the group value is typically the Group Object ID or the group display name, depending on your claims configuration.

## Related Documentation

- [Agent Visibility and Group-Based Access Control](../agent-visibility-and-group-access.md) -- full explanation of the two-layer model with examples
- [Filtering Agents by Tags and Fields](filtering-agents-by-tags-and-fields.md) -- all agent filtering options
- [Restrict Server Visibility by Entra Group](restrict-server-visibility-by-entra-group.md) -- similar setup for MCP servers
- [Registering M2M Clients without IdP Admin Token](registering-m2m-client-without-idp-admin-token.md) -- register M2M client-id-to-group mappings locally
