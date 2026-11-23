# How do I find out which groups can see or call a tool?

Open the **Tool access** page from the sidebar, or call `GET /api/tool-access`. It lists every tool on every MCP server, OpenAPI MCP server and virtual MCP server, with the groups that can see it in the registry and the groups that can call it through the gateway. The registry works this out from the stored groups with the same rules the gateway enforces, so the answer matches what happens when someone calls the tool.

The listing only reports access. To change it, edit the group under **Settings > IAM > Groups**, or see [How do I let a group call only some of the tools on each MCP server?](tool-level-access-control.md).

## Who can use it

| Caller | What they see |
|--------|---------------|
| Admin | Every tool |
| [Read-only admin](read-only-viewer-group.md#read-only-admin) (`view_all`) | Every tool |
| Owner of a server or virtual server | Nothing by default; the tools on the servers they registered if an admin turns that on |
| Anyone else | Nothing, and no sidebar entry |

Admins and read-only admins can also open the page from the **Who can use these tools** button on a server's card.

## How do I let server owners see their own tools?

Set `TOOL_ACCESS_OWNER_VIEW` to `owned` (the default is `none`):

| Deployment | Setting |
|------------|---------|
| Docker Compose | `TOOL_ACCESS_OWNER_VIEW=owned` in `.env` |
| Terraform (ECS) | `tool_access_owner_view = "owned"` in `terraform.tfvars` |
| Helm | `registry.app.toolAccessOwnerView: owned` |

Restart the registry. An owner who is not an admin then sees the tools on the servers they registered, gets the **Tool access** sidebar entry once they own a server with tools, and gets the card button on those servers. They still see nothing about servers they do not own.

## In the UI

Type in the search box to filter by tool, server or group name as you type. The type filter narrows the list to MCP, OpenAPI or virtual servers. Each row shows:

- **Tool**: the name a client calls. A virtual server's alias shows the backend tool's name under it, and a **Blocked** badge means nobody can call the tool until an admin unblocks it.
- **MCP server**: the server's name and type. Hover over the name for its path, or click it to show only that server's tools.
- **Can see** and **Can call**: the groups that grant the access by themselves. Hover over a group to see the identity provider groups mapped to it, or click it to show everything that group can reach. Admin groups appear as one **Admins** chip.

Some access needs two groups at once. For example, one group may let a member open a session on a server while another grants the tool. When that happens, the cell shows **Combined groups**: open it to see one line per check, and a member of one group from each line has access. Tools on virtual servers often need this, because a call checks the virtual server's grant, the backend server's grant and any groups the virtual server requires.

## From the API

```bash
export REGISTRY_URL="https://registry.example.com"
export ACCESS_TOKEN=$(jq -r '.tokens.access_token' .token)

# First page, 20 tools
curl -sS "$REGISTRY_URL/api/tool-access" -H "Authorization: Bearer $ACCESS_TOKEN" | jq

# Every tool a group can see or call (registry or identity provider group name)
curl -sS "$REGISTRY_URL/api/tool-access?group=docs-readers" -H "Authorization: Bearer $ACCESS_TOKEN" | jq

# One server's tools, one line each
curl -sS "$REGISTRY_URL/api/tool-access?server_path=/cloudflare-docs/" -H "Authorization: Bearer $ACCESS_TOKEN" \
  | jq -r '.rows[] | "\(.tool_name)  see=\(.can_see.alone|join(","))  call=\(.can_call.alone|join(","))"'
```

Other parameters are `q` (search), `server_type` (`mcp`, `openapi` or `virtual`), `include_disabled`, `limit` (1 to 500, default 20) and `offset`. Each row's `can_see` and `can_call` hold `alone` and `together`; the response's `groups` gives each named group's identity provider groups and admin flag.

The same listing from the CLI:

```bash
uv run python api/registry_management.py --registry-url "$REGISTRY_URL" --token-file .token \
  tool-access-list --group docs-readers
```

## Why a change does not show yet

The registry keeps the computed listing for 30 seconds so that searching stays fast. A group or server change can take that long to appear. The page and the API's `generated_at` field show when the data was computed.

## Related

- [Tool access listing](../tool-access-listing.md) explains every check behind each column, and the limits.
- [Scope management](../scopes-mgmt.md) documents the group fields the listing reads.
