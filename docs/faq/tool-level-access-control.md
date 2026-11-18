# How do I let a group call only some of the tools on each MCP server?

Give the group a scope whose `server_access` list has one entry per server, and put the tools you want to allow in that entry's `tools` list. Each server gets its own list, so a group can call two tools on one server, three different tools on another, and every tool on a third.

## Example scope

[`cli/examples/tool-subset-per-server.json`](../../cli/examples/tool-subset-per-server.json) defines a group called `tool-subset-users`:

```json
{
  "scope_name": "tool-subset-users",
  "scope_type": "server_scope",
  "group_mappings": ["tool-subset-users"],
  "server_access": [
    {
      "server": "/realserverfaketools",
      "methods": ["initialize", "notifications/initialized", "ping", "tools/list", "tools/call"],
      "tools": ["quantum_flux_analyzer", "neural_pattern_synthesizer"]
    },
    {
      "server": "/mcpgw",
      "methods": ["initialize", "notifications/initialized", "ping", "tools/list", "tools/call"],
      "tools": ["list_services", "search_registry", "healthcheck"]
    },
    {
      "server": "/currenttime",
      "methods": ["initialize", "notifications/initialized", "ping", "tools/list", "tools/call"],
      "tools": ["*"]
    }
  ],
  "ui_permissions": {
    "list_service": ["/realserverfaketools", "/mcpgw", "/currenttime"]
  },
  "create_in_idp": true
}
```

Members of this group get the following access:

| Server | Tools on the server | Tools the group can call |
| --- | --- | --- |
| `/realserverfaketools` | 6 | `quantum_flux_analyzer`, `neural_pattern_synthesizer` |
| `/mcpgw` | 8 | `list_services`, `search_registry`, `healthcheck` |
| `/currenttime` | 1 | all of them (`"*"`) |
| any other server | | none, because the scope has no entry for it |

## What each field does

- `server` is the server's registered path. The gateway ignores leading and trailing slashes when it compares names, so `/mcpgw`, `mcpgw` and `/mcpgw/` all match the same server, and one entry per server is enough.
- `methods` lists the MCP methods the group may send. A client needs `initialize`, `tools/list` and `tools/call` to find and call tools. Add `resources/*` or `prompts/*` methods only if the group should read those too.
- `tools` lists the tool names the group may call, spelled exactly as the server reports them in `tools/list`. `["*"]` allows every tool on that server, including tools the server adds later. A named list allows only those names, so a tool the server adds later stays blocked until you add it.
- `ui_permissions.list_service` controls which servers the group sees in the registry UI and the `/api/servers` listing. It does not affect what the group can call through the gateway.

## What a member sees

When a member calls `tools/list` on `/realserverfaketools`, the gateway removes the 4 tools the scope does not name, and the client sees 2 tools. A `tools/call` for any other tool returns HTTP 403. Both checks read the same `tools` list, so the tools a client sees are the tools it can call.

The `tools/list` filter is on by default. `MCP_TOOLS_LIST_FILTER_ENABLED=false` turns it off, and then a member sees every tool but can still call only the named ones.

## Import the scope

Import the file with an admin token. Because the file sets `"create_in_idp": true`, the registry also creates a `tool-subset-users` group in your identity provider, and you add users to that group there.

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file .token \
  import-group --file cli/examples/tool-subset-per-server.json
```

The CLI calls `POST /api/servers/groups/import`, and you can call that endpoint directly. It requires an admin bearer token and takes the group file as the JSON body:

```bash
export ACCESS_TOKEN=$(jq -r '.tokens.access_token' .token)

curl -X POST "$REGISTRY_URL/api/servers/groups/import" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d @cli/examples/tool-subset-per-server.json
```

Importing a file whose `scope_name` already exists replaces that group's definition.

To change the tool list later, edit the file and import it again, or edit the group under **Settings > IAM > Groups** in the UI. The auth server reloads scopes on every write, so the change takes effect without a restart.

## Limits

- The gateway checks blocked tools before it reads any scope. A tool that the security scanner or an admin has blocked on a server stays blocked for every group, even one that names it in `tools` or has `["*"]`.
- A scope with a named `tools` list cannot open a stream with a request that has no body. Servers that use streamable HTTP, which includes all three servers above, keep working, because the client sends every call as a POST. A server that supports only the older SSE transport needs `"tools": ["*"]` for now.

## Related

- [Fine-grained access control](../scopes.md#methods-vs-tools-access-control) explains how the gateway checks methods and tools.
- [Scope management](../scopes-mgmt.md) documents every field in a scope file.
- [How do I create a non-admin group that can register servers and run health checks?](read-write-non-admin-group.md)
