# MCP Server Versioning - Operations Guide

This guide covers the operational workflows for managing multiple versions of MCP servers in the gateway registry.

For the technical design and architecture details, see [Server Versioning Design](design/server-versioning.md).

### UI Demo

The following shows the version badges on server cards (both the MCP server-reported version and the user-provided routing version) and the version swap workflow using the Version Selector Modal:

![MCP Server Versioning UI Flow](img/mcp-server-versioning.gif)

---

## Understanding What You See on the Dashboard

### Version Badges on Server Cards

Each server card can display up to two version indicators:

**Routing Version Badge** (e.g., `v2.0.0` with a dropdown arrow):
- This is the user-provided version label that controls which backend receives traffic.
- Only appears when the server has multiple versions registered.
- Clicking the badge opens the Version Selector Modal where you can switch the active version.
- Single-version servers do not show this badge.

**MCP Server Version Badge** (e.g., `srv 2.14.5`):
- This is the software version reported by the running MCP server during health checks.
- This value is determined automatically -- it comes from the `serverInfo.version` field in the MCP `initialize` response.
- A small green dot appears if this version changed within the last 24 hours, indicating the upstream server deployed a new build.
- This badge is informational only and does not affect routing.

These two versions are independent. The routing version is an operational label you control. The MCP server version is a fact about what code is running at the backend URL.

---

## Workflow 1: MCP Server Updates Its Own Version

**Scenario**: An MCP server developer deploys a new build. The server at `https://mcp.context7.com/mcp` starts reporting version `2.14.5` instead of `2.14.4`. No admin action was taken.

**What happens automatically**:

1. The next health check runs against the active version endpoint
2. The health check reads `serverInfo.version` from the MCP `initialize` response
3. The registry detects the version changed from `2.14.4` to `2.14.5`
4. The registry stores:
   - `mcp_server_version`: `2.14.5` (new value)
   - `mcp_server_version_previous`: `2.14.4` (old value)
   - `mcp_server_version_updated_at`: current timestamp
5. A WARNING log message is emitted noting the version change
6. The dashboard card shows `srv 2.14.5` with a green dot indicator

**What the admin sees**:

- The `srv` badge on the server card updates to show the new version
- A green dot appears next to the version for 24 hours
- Hovering over the badge shows: "MCP Server Version: 2.14.5 (previously 2.14.4)"

**No action required**. This is purely informational. The routing configuration does not change. Traffic continues to flow to the same backend URL. The user-provided routing version label (e.g., `v1.0.0`) is unaffected.

---

## Workflow 2: Platform Admin Registers a New Version of a Server

**Scenario**: A platform admin wants to add version `v2.0.0` of Context7 pointing to a new backend URL, while keeping `v1.0.0` active.

### Step 1: Register the New Version

Register the server with the same path but a different version. The registry detects this is a new version of an existing server and creates it as an inactive version document.

**Using the CLI**:
```bash
uv run python -m api.registry_management register \
  --name "Context7 MCP Server" \
  --path /context7 \
  --version v2.0.0 \
  --status beta \
  --proxy-url "https://mcp-v2.context7.com/mcp" \
  --transport streamable-http \
  --tags documentation,search,libraries
```

**Using the API**:
```bash
curl -X POST https://gateway.example.com/api/servers/register \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "server_name": "Context7 MCP Server",
    "path": "/context7",
    "version": "v2.0.0",
    "status": "beta",
    "proxy_pass_url": "https://mcp-v2.context7.com/mcp",
    "supported_transports": ["streamable-http"],
    "tags": ["documentation", "search", "libraries"]
  }'
```

**Using a JSON config file**:
```bash
uv run python -m api.registry_management register --config cli/examples/context7-v2-server-config.json
```

Example config file (`context7-v2-server-config.json`):
```json
{
  "server_name": "Context7 MCP Server",
  "description": "Up-to-date Docs for LLMs and AI code editors (Version 2 - Beta)",
  "path": "/context7",
  "version": "v2.0.0",
  "status": "beta",
  "proxy_pass_url": "https://mcp-v2.context7.com/mcp",
  "supported_transports": ["streamable-http"],
  "tags": ["documentation", "search", "libraries", "packages", "api-reference", "code-examples"]
}
```

### What Happens After Registration

1. The registry detects that `/context7` already exists with version `v1.0.0`
2. A new **inactive** version document is created at `_id: /context7:v2.0.0`
3. The active version document at `_id: /context7` is updated with `/context7:v2.0.0` added to its `other_version_ids` array
4. The nginx configuration is regenerated to include a version map entry for `v2.0.0`
5. Nginx is reloaded

The API response includes `"is_new_version": true` to confirm a version was added rather than a new server created.

### What the Admin Sees on the Dashboard

- The existing Context7 server card now shows a **version badge** (e.g., `v1.0.0` with a dropdown arrow)
- Clicking the badge opens the **Version Selector Modal** showing both versions
- `v1.0.0` is marked as `ACTIVE` (green badge)
- `v2.0.0` is marked as `beta` (blue badge)

### Step 2: Test the New Version

Before promoting `v2.0.0` to active, test it using the `X-MCP-Server-Version` header:

**In an AI coding assistant** (e.g., Roo Code, Claude Desktop):

Add the header to the MCP server configuration:
```json
{
  "mcpServers": {
    "context7": {
      "type": "streamable-http",
      "url": "https://gateway.example.com/context7",
      "headers": {
        "X-MCP-Server-Version": "v2.0.0",
        "X-Authorization": "Bearer <token>"
      }
    }
  }
}
```

**With curl**:
```bash
curl -X POST https://gateway.example.com/context7 \
  -H "X-MCP-Server-Version: v2.0.0" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

Traffic without the header continues to route to `v1.0.0`. Only requests with the explicit header reach `v2.0.0`.

### Step 3: Promote to Active

Once testing is complete, switch the active version:

**Using the Version Selector Modal (UI)**:
1. Click the version badge on the Context7 server card
2. In the modal, click "Set Active" on version `v2.0.0`
3. The modal closes and the card updates

**Using the API**:
```bash
curl -X PUT https://gateway.example.com/api/servers/context7/versions/default \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"version": "v2.0.0"}'
```

**Using the CLI**:
```bash
uv run python -m api.registry_management set-default-version \
  --path /context7 \
  --version v2.0.0
```

### What Happens During the Switch

1. The current active document (`/context7`, version `v1.0.0`) becomes an inactive document at `_id: /context7:v1.0.0`
2. The target inactive document (`/context7:v2.0.0`) becomes the new active document at `_id: /context7`
3. The `other_version_ids` array is updated to reference `/context7:v1.0.0` instead of `/context7:v2.0.0`
4. The search index is re-indexed with `v2.0.0` metadata
5. The nginx configuration is regenerated and reloaded
6. A background health check is triggered for the newly active version
7. The dashboard updates to show `v2.0.0` as active

All traffic without the `X-MCP-Server-Version` header now routes to `v2.0.0`. Clients that still send `X-MCP-Server-Version: v1.0.0` continue to reach the old version.

---

## Workflow 3: Instant Rollback

**Scenario**: After promoting `v2.0.0`, you discover an issue and need to revert to `v1.0.0`.

```bash
# API
curl -X PUT https://gateway.example.com/api/servers/context7/versions/default \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"version": "v1.0.0"}'

# CLI
uv run python -m api.registry_management set-default-version \
  --path /context7 \
  --version v1.0.0
```

The switch is immediate:
- All default traffic reverts to `v1.0.0`
- Clients with `X-MCP-Server-Version: v2.0.0` can still reach `v2.0.0` for debugging
- A health check runs automatically for the restored version

---

## Workflow 4: Deprecate and Remove an Old Version

### Mark as Deprecated

When registering the version, use `--status deprecated`:
```bash
uv run python -m api.registry_management register \
  --name "Context7 MCP Server" \
  --path /context7 \
  --version v1.0.0 \
  --status deprecated \
  --proxy-url "https://mcp.context7.com/mcp" \
  --transport streamable-http
```

The Version Selector Modal shows deprecated versions with an amber badge.

### Remove a Version

```bash
# API
curl -X DELETE https://gateway.example.com/api/servers/context7/versions/v1.0.0 \
  -H "Authorization: Bearer <token>"

# CLI
uv run python -m api.registry_management remove-version \
  --path /context7 \
  --version v1.0.0
```

Constraints:
- You **cannot remove the currently active version**. Switch to a different version first.
- Removing a version deletes its document from the database and removes its nginx map entry.
- Clients sending `X-MCP-Server-Version: v1.0.0` will fall back to the default version after removal.

---

## Workflow 5: List All Versions

```bash
# API
curl https://gateway.example.com/api/servers/context7/versions \
  -H "Authorization: Bearer <token>"

# CLI
uv run python -m api.registry_management list-versions --path /context7
```

Response:
```json
{
  "path": "/context7",
  "default_version": "v2.0.0",
  "versions": [
    {
      "version": "v2.0.0",
      "proxy_pass_url": "https://mcp-v2.context7.com/mcp",
      "status": "stable",
      "is_default": true
    },
    {
      "version": "v1.0.0",
      "proxy_pass_url": "https://mcp.context7.com/mcp",
      "status": "deprecated",
      "is_default": false,
      "sunset_date": "2026-06-01"
    }
  ]
}
```

---

## Workflow 6: Delete a Server (All Versions)

When you delete a server entirely, all version documents are cascade-deleted:

```bash
# CLI
uv run python -m api.registry_management delete --path /context7

# API
curl -X DELETE https://gateway.example.com/api/servers/context7 \
  -H "Authorization: Bearer <token>"
```

This removes:
- The active version document at `/context7`
- All inactive version documents matching `/context7:*`
- The search index entry
- The nginx location block and map entries

---

## How Versioning Affects Search

Only the **active version** of each server appears in search results. Inactive versions are excluded at index time (they are never added to the vector search index), so they do not consume result slots.

When you switch the active version, the search index is automatically re-indexed with the new active version's metadata (name, description, tags, tools). This means search results always reflect the currently active version.

---

## How Versioning Affects Health Checks

Only the **active version** is health-checked. Inactive versions are skipped during the health check cycle. When you switch the active version, a health check for the newly active version is triggered immediately in the background.

---

## Client Configuration for Version Pinning

Clients that need to pin to a specific version add the `X-MCP-Server-Version` header to their requests:

### Claude Desktop / Roo Code / Other MCP Clients

```json
{
  "mcpServers": {
    "context7": {
      "type": "streamable-http",
      "url": "https://gateway.example.com/context7",
      "headers": {
        "X-MCP-Server-Version": "v1.0.0"
      }
    }
  }
}
```

### Programmatic Access

```python
import httpx

response = httpx.post(
    "https://gateway.example.com/context7",
    headers={
        "X-MCP-Server-Version": "v1.0.0",
        "Content-Type": "application/json",
    },
    json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
)
```

### Header Values

| Value | Behavior |
|-------|----------|
| Omitted | Routes to active (default) version |
| `latest` | Routes to active (default) version |
| `v1.0.0` | Routes to version v1.0.0 specifically |
| Unknown value | Falls back to default backend URL |
