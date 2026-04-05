# Supported Protocol, Trust Level, and Visibility Fields

## Overview

The Agent Registry now supports registering **any agent** -- not just [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/specification/) protocol agents. A new `supported_protocol` field distinguishes A2A agents from non-A2A agents, while `trust_level` and `visibility` defaults have been updated for consistency across all layers (backend, API, CLI, frontend).

## Supported Protocol Field

The `supported_protocol` field indicates which protocol an agent implements:

| Value   | Description |
|---------|-------------|
| `a2a`   | Agent implements the A2A protocol specification |
| `other` | Agent uses a different protocol (HTTP REST, gRPC, custom, etc.) |

- **Registration API**: `supportedProtocol` is **required** when registering a new agent
- **Agent Card model**: `supported_protocol` defaults to `None` for backward compatibility with existing agents
- **Agent listing**: the field appears in all agent list and detail responses

### Registering via the UI

The registration form includes a **"This is an A2A Protocol Agent"** checkbox. When checked, the agent is registered with `supported_protocol: "a2a"`. When unchecked, it is registered as `"other"`.

The edit dialog also includes a **Supported Protocol** dropdown (A2A / Other) so you can update an existing agent's protocol type.

### Registering via the API

Include the `supportedProtocol` field in your registration request:

```bash
curl -X POST http://localhost/api/agents/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "My Agent",
    "description": "An example agent",
    "url": "https://my-agent.example.com",
    "version": "1.0.0",
    "supportedProtocol": "a2a",
    "tags": ["example"]
  }'
```

### Registering via the CLI

```bash
uv run python api/registry_management.py \
  --registry-url http://localhost \
  --token-file .token \
  agent-register \
  --name "My Agent" \
  --url "https://my-agent.example.com" \
  --supported-protocol a2a \
  --tags "example"
```

## Updated Default Values

### Trust Level

The default `trust_level` has changed from `"unverified"` to `"community"` across all layers:

| Trust Level  | Description |
|-------------|-------------|
| `unverified` | No verification performed |
| `community`  | Community-contributed agent (new default) |
| `verified`   | Verified by registry administrators |
| `trusted`    | Fully trusted agent |

### Visibility

The default `visibility` has changed from `"internal"` to `"public"` across all layers:

| Visibility        | Description |
|-------------------|-------------|
| `public`          | Visible to all users (new default) |
| `group-restricted`| Visible only to members of allowed groups |
| `internal`        | Visible only to the agent owner |

## Backfill Script for Existing Agents

Existing agents in MongoDB that were created before this change will not have the `supported_protocol` field, and may still have the old default values for `trust_level` and `visibility`. A one-time backfill script normalizes these:

```bash
uv run python scripts/backfill_agent_fields.py
```

The script performs three operations:

1. **`supported_protocol`** -- Sets `"other"` on all agents that don't have the field. Agents already registered as A2A are not affected.
2. **`trust_level`** -- Updates agents with `"unverified"` to `"community"` (the new default).
3. **`visibility`** -- Updates agents with `"internal"` to `"public"` (the new default).

### Configuration

The script connects to MongoDB at `localhost:27017` by default. For production deployments (e.g., Amazon DocumentDB), update the `MONGODB_URI` constant in `scripts/backfill_agent_fields.py` before running.

The script is **idempotent** -- running it multiple times has no additional effect. Each operation logs how many documents were modified.

## Agent Card and Server Card Generation Skills

Two Claude Code skills are available to help generate registration cards by analyzing source code:

### Generate Agent Card

Analyzes agent source code (local folder or GitHub URL) and generates an A2A-compliant agent card JSON file. Detects agent name, skills, tools, auth mechanisms, protocol bindings, and streaming support.

```
/generate-agent-card /path/to/agent/folder
/generate-agent-card https://github.com/org/agent-repo
```

See [.claude/skills/generate-agent-card/SKILL.md](../.claude/skills/generate-agent-card/SKILL.md) for details.

### Generate Server Card

Analyzes MCP server source code and generates a registry-compatible server card JSON file. Detects server name, tools, transport type, auth scheme, and deployment URLs.

```
/generate-server-card /path/to/server/folder
/generate-server-card https://github.com/org/server-repo
```

See [.claude/skills/generate-server-card/SKILL.md](../.claude/skills/generate-server-card/SKILL.md) for details.

## Frontend Changes

- The Dashboard now shows an **"A2A Protocol"** badge on agent cards for agents with `supported_protocol: "a2a"`
- Agent details modal shows a clickable A2A card URL for A2A agents
- Trust level and visibility values are read from the API (no longer hardcoded)
- The edit dialog includes dropdowns for Trust Level, Supported Protocol, and Visibility

## API Response Format

The `supported_protocol`, `trust_level`, and `visibility` fields are included in all agent API responses:

```json
{
  "name": "Flight Booking Agent",
  "path": "/flight-booking",
  "supported_protocol": "a2a",
  "trust_level": "community",
  "visibility": "public",
  ...
}
```

Agents that predate this feature will show `"supported_protocol": null` until the backfill script is run.
