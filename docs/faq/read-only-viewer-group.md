# How do I create a read-only user who can see every server, agent and skill but change nothing?

Create a group whose `ui_permissions` hold only `list_` and `get_` permissions, each set to `["all"]`, then create a user in that group. Members see every MCP server, virtual server, agent and skill in the registry UI and API. Any request to register, edit, enable, disable or delete returns HTTP 403.

A `list_` or `get_` permission never makes a user an admin, even with `"all"`. The registry promotes a user to admin only when a group grants `"all"` on an action that starts with `register_`, `modify_`, `toggle_`, `delete_`, `publish_` or `create_` (see `is_admin_conferring_action` in [registry/auth/privileged_constants.py](../../registry/auth/privileged_constants.py)).

## The group definition

[`cli/examples/read_only_viewer.json`](../../cli/examples/read_only_viewer.json):

```json
{
  "scope_name": "registry-read-only",
  "scope_type": "server_scope",
  "description": "Read-only viewer: sees every MCP server, virtual server, agent and skill in the registry UI and API. Holds no register, modify, toggle, delete, publish or health-check permission, and no gateway access to call tools.",
  "group_mappings": ["registry-read-only"],
  "server_access": [],
  "ui_permissions": {
    "list_service": ["all"],
    "list_virtual_server": ["all"],
    "list_agents": ["all"],
    "get_agent": ["all"],
    "list_skills": ["all"]
  },
  "create_in_idp": true
}
```

| Permission | Lets the user |
| --- | --- |
| `list_service` | see every MCP server and its tools |
| `list_virtual_server` | see every virtual server |
| `list_agents`, `get_agent` | see every agent and open its details |
| `list_skills` | see every skill |

The empty `server_access` list means the user cannot call tools through the gateway. To let them call some tools, add `server_access` entries as shown in [How do I let a group call only some of the tools on each MCP server?](tool-level-access-control.md).

## Step 1: create the group

Both commands need an admin token. Set these first:

```bash
export REGISTRY_URL="https://your-registry"
export ACCESS_TOKEN=$(jq -r '.tokens.access_token' .token)
```

Import the group with the CLI:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file .token \
  import-group --file cli/examples/read_only_viewer.json
```

Or call the API directly:

```bash
curl -X POST "$REGISTRY_URL/api/servers/groups/import" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d @cli/examples/read_only_viewer.json
```

A successful import returns `"idp_created": true`, which means the registry also created the `registry-read-only` group in your identity provider.

## Step 2: create the read-only user

Put the user's initial password in a file that only you can read. `read -rs` asks for it without echoing it or saving it in your shell history:

```bash
umask 077
read -rsp 'Initial password for admin-ro: ' PW && printf '%s' "$PW" > admin-ro.passwd && unset PW
```

Create the user `admin-ro` in the `registry-read-only` group. `jq` reads the password from the file and builds the request body, and `curl` reads that body from standard input, so the password never appears on a command line where `ps` could show it:

```bash
jq -n --rawfile pw admin-ro.passwd '{
  username: "admin-ro",
  email: "admin-ro@example.com",
  firstname: "Admin",
  lastname: "ReadOnly",
  groups: ["registry-read-only"],
  password: ($pw | rtrimstr("\n"))
}' | curl -sS -X POST "$REGISTRY_URL/api/management/iam/users/human" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d @-

rm admin-ro.passwd
```

The response lists the user with `"groups": ["registry-read-only"]` and `"enabled": true`. You can also create the user under **Settings > IAM > Users** in the UI.

The CLI command `user-create-human` takes the password as `--password`, which puts it on the command line. Use the API call above instead, or run `user-create-human` without `--password` and set the password in your identity provider.

## Check the result

Log in as `admin-ro` and call `GET /api/auth/me`. The response shows `"is_admin": false` and only the five permissions from the group. On a local Docker Compose stack, these are the results we got:

| Request | Result |
| --- | --- |
| `GET /api/servers`, `/api/agents`, `/api/skills`, `/api/virtual-servers` | 200, every entry |
| `POST /api/servers/toggle` | 403 |
| `PATCH /api/servers/{path}` | 403 |
| `POST /api/servers/remove` | 403 |
| `POST /api/servers/groups/import` | 403 |

## Limits

- Pages that check for an admin, such as audit logs, IAM settings and the configuration pages, are all or nothing. A group cannot grant read-only access to them, so `admin-ro` does not see them.
- The group leaves out `health_check_service`, because a health check writes: it stores the server's new health status and tool list and rewrites the gateway's nginx configuration. Add `"health_check_service": ["all"]` if you accept that.
- Each custom entity type has its own `list_<type>_entity` permission. Add one per type you want the user to see, for example `"list_dataset_entity": ["all"]`.

## Related

- [Scope management](../scopes-mgmt.md) documents every `ui_permissions` key.
- [How do I create a non-admin group that can register servers and run health checks?](read-write-non-admin-group.md)
- [Why did the Skills or Agents tab go empty?](granting-skill-and-agent-discovery-permissions.md)
