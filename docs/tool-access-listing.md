# Tool access listing

The **Tool access** page lists every tool on every MCP server, OpenAPI MCP server and virtual MCP server, with the groups that can see it in the registry and the groups that can call it through the gateway. The same data is available from `GET /api/tool-access` and from the `tool-access-list` command.

The listing grants nothing and changes no access check. To change who can use a tool, edit the group under **Settings > IAM > Groups**, or import the group file again (see [How do I let a group call only some of the tools on each MCP server?](faq/tool-level-access-control.md)).

## Who can see the listing

Group names show how an organization is structured, so the registry limits who sees them:

| Caller | Rows |
|--------|------|
| Admin | Every tool |
| Read-only admin (`view_all`, see [read-only groups](faq/read-only-viewer-group.md#read-only-admin)) | Every tool |
| Owner of a server or virtual server | None by default. With `TOOL_ACCESS_OWNER_VIEW=owned`, the tools on the servers they registered |
| Anyone else | None |

Admins and read-only admins get a **Tool access** entry in the sidebar, and a **Who can use these tools** button on each server card that opens the page filtered to that server.

### Letting owners see their own servers

Out of the box only admins and read-only admins see the listing. To let the owner of a server see which groups can use its tools, set `TOOL_ACCESS_OWNER_VIEW` to `owned`:

| Deployment | Setting |
|------------|---------|
| Docker Compose | `TOOL_ACCESS_OWNER_VIEW=owned` in `.env` |
| Terraform (ECS) | `tool_access_owner_view = "owned"` in `terraform.tfvars` |
| Helm | `registry.app.toolAccessOwnerView: owned` |

Owners then get the sidebar entry once they own a server with tools, and the card button on their own servers. The owner of a server is the user in its `registered_by` field, and of a virtual server the user in `created_by`. A server synced from a peer registry has no local owner, because its `registered_by` names a user of the other registry. The registry refuses to start if the value is anything other than `none` or `owned`.

## Reading a row

| Column | Content |
|--------|---------|
| Tool | The name a client calls. On a virtual server this is the alias, with the backend tool's name under it. A **Blocked** badge means an admin or a security scan blocked the tool, so nobody can call it until it is unblocked. |
| MCP server | The server's name, path and type (MCP, OpenAPI or Virtual), and whether it is disabled. |
| Can see | Groups whose members see the tool in the registry UI, the API and search. |
| Can call | Groups whose members can call the tool through the gateway. |

Each chip is a registry group; hover over it to see the identity provider groups mapped to it. A cell shows up to three groups, then **+N more**. Groups that make their members admins appear as one **Admins** chip; hover over it for the names. Click a group to list everything that group can reach, or a server name to list only that server's tools. The search box filters as you type.

### Alone and together

A user's access combines every group they belong to. Each access column therefore has two parts:

- The chips list the groups that grant the access by themselves.
- **Combined groups** opens the checks a caller must pass when no single group covers them all, one line per check. A user in one group from each line also has access.

For example, a group may let its members open a session on a server (`initialize`) while another group grants the tool. Neither group lets a member call the tool alone, but a member of both can.

The checks behind each column:

| Column | Server type | A caller needs |
|--------|-------------|----------------|
| Can see | MCP, OpenAPI | A `list_service` grant for the server, a `server_access` rule naming its path, and a rule that allows the tool (or admin, or read-only admin) |
| Can see | Virtual | A `list_virtual_server` grant, access to the backend server and its tool, and membership in each group the tool requires |
| Can call | MCP, OpenAPI | `initialize` on the server, and `tools/call` for the tool |
| Can call | Virtual | `initialize` and the alias on the virtual server, `initialize` and the original tool name on the backend server, and membership in every group the virtual server or the tool requires |

A read-only admin group appears under **Can see** and never under **Can call**: `view_all` grants no gateway access.

## How the registry works it out

The registry computes each row from the stored groups every time the page loads, with the same matching code the gateway uses for every call. Nothing is stored, so the listing cannot fall out of step with what the gateway enforces. A `tools` or `methods` value written as a single string is read as a one-item list.

The registry reads the groups, servers and virtual servers in six database queries, whatever their number, and reuses the result for 30 seconds so that search-as-you-type stays fast. A group change can therefore take up to 30 seconds to show; the page shows when its data was computed.

## API

```bash
curl -sS "$REGISTRY_URL/api/tool-access?q=search&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `q` | none | Substring of a tool, backend tool, server name, server path, registry group or identity provider group. Up to 200 characters. |
| `server_path` | none | Exact server path, with or without slashes. |
| `server_type` | none | `mcp`, `openapi` or `virtual`. |
| `group` | none | Exact registry or identity provider group name. |
| `include_disabled` | `true` | Include tools on disabled servers. |
| `limit` | 20 | Rows per page, 1 to 500. |
| `offset` | 0 | Rows to skip. |

The response holds `rows`, `groups`, `total_count`, `limit`, `offset`, `has_next` and `generated_at`. An owner gets an empty list unless `TOOL_ACCESS_OWNER_VIEW` is `owned`. Rows name registry groups only; `groups` gives the identity provider groups and admin flag of each group named on the page. The endpoint accepts a session cookie or a bearer token. Rows never include backend URLs, headers or credentials. A caller who may see no rows gets `200` with an empty list.

## CLI

```bash
uv run python api/registry_management.py --registry-url "$REGISTRY_URL" --token-file .token \
  tool-access-list --query search

uv run python api/registry_management.py --registry-url "$REGISTRY_URL" --token-file .token \
  tool-access-list --server-path /cloudflare-docs/ --json
```

Options: `--query`, `--server-path`, `--server-type`, `--group`, `--exclude-disabled`, `--limit`, `--offset` and `--json`.

## Limits

- The listing covers MCP tools. Agents, skills and custom entities are not included.
- The registry decides "can see" with the same rule as a server list that is searched or filtered. An unfiltered list for a user whose `list_service` grant is `["all"]` can also show a server that no `server_access` rule names, without any of its tools.
- When one group names a server by its display name and another by its path, a member of both sees only the tools of the rule that names the path. The listing treats each group's rule on its own and can show a combination as granting more than that user sees.
