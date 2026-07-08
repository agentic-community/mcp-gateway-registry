# How do I create a non-admin group that can register servers and run health checks but cannot toggle, edit, or delete them?

You can define a "read-write" group whose members can see servers, register new ones, and run health checks, while being blocked from toggling a server's enabled/disabled state, changing its lifecycle status, or deleting it. This is done with a group scope definition, not a code change.

## Quick Answer

Create a group scope JSON that grants the read-write permissions and omits the destructive ones, import it into the registry, and create a matching group of the same name in your IdP. Two ready-made examples ship with the repo:

- [`cli/examples/read_all_register_new.json`](../../cli/examples/read_all_register_new.json): members see **all** servers.
- [`cli/examples/read_select_register_new.json`](../../cli/examples/read_select_register_new.json): members see only a **select** list; a server they register stays invisible to the group until an admin grants access to it.

```bash
export REGISTRY_URL="https://your-registry"
export TOKEN_FILE=".token"   # an admin JWT token file

uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  import-group --file cli/examples/read_select_register_new.json
```

## The two example groups

| Group | Sees | A server it registers is visible to the group? |
| --- | --- | --- |
| `read-all-register-new` | All servers (`list_service: ["all"]`) | Yes, immediately |
| `read-select-register-new` | Only a select list (currenttime, mcpgw) | No, an admin must add it with `add-to-groups` |

The second case is the interesting one: a user in `read-select-register-new` can successfully register a new server, but that server will **not** appear in their listing until an admin explicitly grants the group access to it. This is by design.

## How the access control works

The registry enforces permissions in two layers (see [registry/api/server_routes.py](../../registry/api/server_routes.py) and [registry/auth/dependencies.py](../../registry/auth/dependencies.py)):

1. **API method layer** (`server_access` -> the `api` pseudo-server): gates which REST verbs reach the gateway. These groups get `GET` and `POST` (needed to list and register), but not `PUT` or `DELETE`.

2. **Fine-grained UI permission layer** (`ui_permissions`): every server route calls `user_has_ui_permission_for_service(permission, server_name, ...)`. The mapping is:

| Capability | Permission key | In these groups? |
| --- | --- | --- |
| List/see servers | `list_service` | Yes |
| Register a new server | `register_service` | Yes |
| Run a health check | `health_check_service` | Yes |
| Toggle enabled/disabled | `toggle_service` | **No (omitted)** |
| Change lifecycle status (PUT/PATCH) | `modify_service` | **No (omitted)** |
| Delete a server | `delete_service` | **No (omitted)** |

A permission that is not listed defaults to deny, so omitting `toggle_service`, `modify_service`, and `delete_service` blocks those operations.

## Important: do not use `register_service: ["all"]`

A user is auto-promoted to **admin** if they hold any *mutating* UI permission with the literal value `"all"`. The mutating prefixes are `register_`, `modify_`, `toggle_`, `delete_`, `publish_`, `create_` (see `_user_is_admin` and `_ADMIN_ACTION_PREFIXES` in [registry/auth/dependencies.py](../../registry/auth/dependencies.py)).

Because `register_` is a mutating prefix, writing `register_service: ["all"]` would flip the user into full admin (settings gear, delete buttons, toggle switches, and the "Admin Access" badge all appear). To avoid this, these group files use `register_service: ["*"]` instead. The registration backend only requires `register_service` to be **non-empty** (it does not require the literal `"all"`, see [registry/api/server_routes.py](../../registry/api/server_routes.py)), so `["*"]` permits registration without triggering admin promotion.

`list_service` and `health_check_service` are read-only prefixes, so `["all"]` is safe for them.

## Why a registered server is invisible to `read-select-register-new`

`register_service` only controls whether a user may create a server; it does not add that server to any group's visible list. Visibility is controlled by `list_service`, which for this group is a fixed allowlist (`/currenttime`, `/mcpgw`). Adding a server to a group's `list_service` (and `server_access`) is done by the `add-to-groups` admin command, which calls `add_server_to_groups` in [registry/services/scope_service.py](../../registry/services/scope_service.py). That command requires admin privileges, so a non-admin user cannot make their own newly registered server visible to their group.

## Step 1: Import the group scope configuration

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

This writes the scope (server_access + ui_permissions) to the registry datastore. Because both files set `"create_in_idp": true`, the import also attempts to create the IdP group. If the IdP group is not created automatically, create it explicitly with the next step.

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

## Step 4: Verify the group scope

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  describe-group --name read-select-register-new
```

You should see `list_service` limited to `/currenttime` and `/mcpgw`, `register_service` set to `["*"]`, `health_check_service` set to `["all"]`, and no `toggle_service`, `modify_service`, or `delete_service` entries.

## Step 5: Demonstrate the visibility boundary

As `readselect-user`, obtain a token and register a new server:

```bash
# Get a user token as readselect-user
uv run python cli/get_user_token.py --output readselect.token

# Register a new remote server (succeeds: register_service is granted)
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file readselect.token \
  register --config cli/examples/minimal-server-config.json
```

Then list servers as the same user:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file readselect.token \
  list
```

The server the user just registered will **not** appear in the list, because it is not in the group's `list_service` allowlist. Only `/currenttime` and `/mcpgw` are visible.

## Step 6: Admin grants visibility (required for read-select)

An admin makes the new server visible to the group with `add-to-groups`:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  add-to-groups --server <new-server-name> --groups read-select-register-new
```

After this, `readselect-user` will see the server in their listing. With the `read-all-register-new` group this admin step is not needed, because that group's `list_service` is `all`.

## What these users still cannot do

Both `readall-user` and `readselect-user` will receive HTTP 403 if they attempt to:

- Toggle a server on/off (`toggle` command / `POST /api/servers/toggle`)
- Edit a server or change its lifecycle status (`update-server` / `patch-server`, `PUT`/`PATCH /api/servers/{path}`)
- Delete a server (`remove` command / `DELETE /api/servers/{path}`)

## IdP Independence

The commands above were tested with **Keycloak**, but the concept is the same for any IdP (Entra ID, Cognito, Okta, Auth0). The registry authorizes requests based on the group names present in the user's JWT token claims, matched against the scope's `group_mappings`; it does not depend on a specific IdP.

The portable recipe is:

1. **Create the group scope definition in the registry** (the `import-group` step above). This is the authorization rule: what the group can see and do.
2. **Have a group of the same name in your IdP**, with your users as members. This is the membership and authentication side.

How you create that IdP group and its users is up to you. Depending on your IdP and how it is wired to the registry, you might:

- Use `registry_management.py` as shown (works when the registry has IdP admin credentials, as with the default Keycloak setup).
- Create the group and users **directly in your IdP** (the Entra/Okta/Cognito/Auth0 admin console), then only run the `import-group` step against the registry.
- Create the group through the **Registry Web UI** (IAM group management page).

For Entra ID specifically, the token group claim is usually a Group Object ID (GUID) rather than the display name, so add that GUID to the scope's `group_mappings` (the `import-group` flow does this automatically for Entra). See [Restrict Server Visibility by Entra Group](restrict-server-visibility-by-entra-group.md) for an IdP-specific walkthrough.

## Related Documentation

- [User and Group Management Guide](../../api/USER-GROUP-MANAGEMENT.md) -- the full create-group / create-user workflow
- [How do I restrict which agents a user can see based on their group?](group-restricted-agent-visibility.md) -- the agent-side equivalent
- [How do I restrict which MCP servers a user can see based on their Entra ID group?](restrict-server-visibility-by-entra-group.md) -- IdP-specific server visibility
- [Agent Visibility and Group-Based Access Control](../agent-visibility-and-group-access.md) -- the two-layer access model in depth
