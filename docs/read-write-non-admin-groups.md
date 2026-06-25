# Read-Write Non-Admin Groups: Register and Health-Check Without Toggle, Edit, or Delete

This guide shows how to create two non-admin "read-write" groups, add a user to
each, and verify the access boundary. Both groups can **see servers**, **register
new servers**, and **run health checks**, but they **cannot** toggle a server's
enabled/disabled state, change its lifecycle status, or delete it.

The two groups differ only in what they can see:

| Group | Sees | A server it registers is visible to the group? |
| --- | --- | --- |
| `read-all-register-new` | All servers (`list_service: ["all"]`) | Yes, immediately |
| `read-select-register-new` | Only a select list (currenttime, mcpgw) | No, an admin must add it with `add-to-groups` |

The second case is the interesting one: a user in `read-select-register-new` can
successfully register a new server, but that server will **not** appear in their
listing until an admin explicitly grants the group access to it. This is by design.

## How the access control works

The registry enforces permissions in two layers (see
[registry/api/server_routes.py](../registry/api/server_routes.py) and
[registry/auth/dependencies.py](../registry/auth/dependencies.py)):

1. **API method layer** (`server_access` -> the `api` pseudo-server): gates which
   REST verbs reach the gateway. These groups get `GET` and `POST` (needed to list
   and register), but not `PUT` or `DELETE`.

2. **Fine-grained UI permission layer** (`ui_permissions`): every server route calls
   `user_has_ui_permission_for_service(permission, server_name, ...)`. The mapping is:

   | Capability | Permission key | In these groups? |
   | --- | --- | --- |
   | List/see servers | `list_service` | Yes |
   | Register a new server | `register_service` | Yes |
   | Run a health check | `health_check_service` | Yes |
   | Toggle enabled/disabled | `toggle_service` | **No (omitted)** |
   | Change lifecycle status (PUT/PATCH) | `modify_service` | **No (omitted)** |
   | Delete a server | `delete_service` | **No (omitted)** |

A permission that is not listed defaults to deny, so omitting `toggle_service`,
`modify_service`, and `delete_service` blocks those operations.

### Important: do not use `register_service: ["all"]`

A user is auto-promoted to **admin** if they hold any *mutating* UI permission with
the literal value `"all"`. The mutating prefixes are `register_`, `modify_`,
`toggle_`, `delete_`, `publish_`, `create_` (see `_user_is_admin` and
`_ADMIN_ACTION_PREFIXES` in
[registry/auth/dependencies.py](../registry/auth/dependencies.py)).

Because `register_` is a mutating prefix, writing `register_service: ["all"]` would
flip the user into full admin (settings gear, delete buttons, toggle switches, and
the "Admin Access" badge all appear). To avoid this, these group files use
`register_service: ["*"]` instead. The registration backend only requires
`register_service` to be **non-empty** (it does not require the literal `"all"` -
see [registry/api/server_routes.py](../registry/api/server_routes.py)), so `["*"]`
permits registration without triggering admin promotion.

`list_service` and `health_check_service` are read-only prefixes, so `["all"]` is
safe for them.

### Why a registered server is invisible to `read-select-register-new`

`register_service` only controls whether a user may create a server; it does not
add that server to any group's visible list. Visibility is controlled by
`list_service`, which for this group is a fixed allowlist (`/currenttime`, `/mcpgw`).
Adding a server to a group's `list_service` (and `server_access`) is done by the
`add-to-groups` admin command, which calls `add_server_to_groups` in
[registry/services/scope_service.py](../registry/services/scope_service.py). That
command requires admin privileges, so a non-admin user cannot make their own
newly registered server visible to their group.

## Group definition files

- [cli/examples/read_all_register_new.json](../cli/examples/read_all_register_new.json)
- [cli/examples/read_select_register_new.json](../cli/examples/read_select_register_new.json)

## Prerequisites

Set up environment variables and an admin token. The `--token-file` accepts the
nested token shape produced by the credential provider (an object with a
`tokens.access_token` field).

```bash
cd /home/ubuntu/repos/mcp-gateway-registry

# Registry URL (use your deployment's URL)
export REGISTRY_URL="https://mcpgateway.ddns.net"

# Path to an admin JWT token file (nested {tokens: {access_token: ...}} is supported)
export TOKEN_FILE=".token"
```

## Step 1: Import both group scope configurations

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  import-group --file cli/examples/read_all_register_new.json

uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  import-group --file cli/examples/read_select_register_new.json
```

This writes the scope (server_access + ui_permissions) to DocumentDB. Because both
files set `"create_in_idp": true`, the import also attempts to create the IdP
(Keycloak) group. If the IdP group is not created automatically, create it
explicitly with the next step.

## Step 2: Ensure the IdP groups exist

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  create-group --name read-all-register-new \
  --description "Non-admin: read all, register new" --idp

uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  create-group --name read-select-register-new \
  --description "Non-admin: read select, register new" --idp
```

If a group already exists this returns an error, which is safe to ignore.

## Step 3: Create one user in each group

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  user-create-human \
  --username readall-user \
  --email readall-user@example.com \
  --first-name ReadAll \
  --last-name User \
  --password 'ReadAll#2026' \
  --groups read-all-register-new

uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  user-create-human \
  --username readselect-user \
  --email readselect-user@example.com \
  --first-name ReadSelect \
  --last-name User \
  --password 'ReadSelect#2026' \
  --groups read-select-register-new
```

## Step 4: Verify the group scopes

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  describe-group --name read-select-register-new
```

You should see `list_service` limited to `/currenttime` and `/mcpgw`, and
`register_service` / `health_check_service` set to `all`, with no
`toggle_service`, `modify_service`, or `delete_service` entries.

## Step 5: Demonstrate the visibility boundary

As `readselect-user`, obtain a token and register a new server:

```bash
# Get a user token (interactive login as readselect-user)
uv run python cli/get_user_token.py \
  --username readselect-user \
  --password 'ReadSelect#2026'

# Register a new remote server (succeeds: register_service = all)
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file <readselect-user-token-file> \
  register --config cli/examples/minimal-server-config.json
```

Then list servers as the same user:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file <readselect-user-token-file> \
  list
```

The server the user just registered will **not** appear in the list, because it
is not in the group's `list_service` allowlist. Only `/currenttime` and `/mcpgw`
are visible.

## Step 6: Admin grants visibility (required for read-select)

An admin makes the new server visible to the group with `add-to-groups`:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  add-to-groups --server <new-server-name> --groups read-select-register-new
```

After this, `readselect-user` will see the server in their listing. With the
`read-all-register-new` group this admin step is not needed, because that group's
`list_service` is `all`.

## What these users still cannot do

Both `readall-user` and `readselect-user` will receive HTTP 403 if they attempt to:

- Toggle a server on/off (`toggle` command / `POST /api/servers/toggle`)
- Edit a server or change its lifecycle status (`update-server` / `patch-server`,
  `PUT`/`PATCH /api/servers/{path}`)
- Delete a server (`remove` command / `DELETE /api/servers/{path}`)
