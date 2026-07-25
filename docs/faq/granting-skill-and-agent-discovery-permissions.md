# Why did the Skills or Agents tab go empty, and how do I grant a group access?

After the skill and agent discovery gate landed, discovery of skills and agents is controlled by a `list_` UI permission, the same way MCP servers already worked. This is a fail-closed change: a non-admin user whose group does **not** hold the grant sees **zero** skills or agents, including public ones, and the tab shows a hint like:

> You don't have access to view agents. Agent discovery is managed by your registry administrator. Ask them to grant your group the "list_agents" permission so agents appear here.

Admins are unaffected: they bypass the discovery check and see everything.

## The permissions involved

| Capability | Permission key | Applies to |
| --- | --- | --- |
| See agents in the UI / search | `list_agents` | Agent paths or `"all"` |
| See skills in the UI / search | `list_skills` | Skill paths or `"all"` |

Grant `["all"]` to let the group discover every skill/agent, or a list of specific resource paths to scope discovery to just those. `list_` scopes are read-only, so `["all"]` is safe and does **not** trigger admin auto-promotion (unlike the mutating `modify_`/`delete_`/`toggle_`/`publish_`/`register_`/`create_` prefixes).

There are two situations, described below: creating a **new** group (get the scope right from the start) and **backfilling** groups that already exist.

## New groups: include the discovery scope from the start

When you author a new group, put `list_agents` / `list_skills` in its `ui_permissions` so members can discover skills and agents immediately. Two ways:

- **IAM UI:** Settings > IAM > Groups > **Create Group**, and in the **UI Permissions** editor turn on the **All** toggle (or multi-select specific records) for `list_agents` and `list_skills`. See [IAM Settings UI](../iam-settings-ui.md).
- **CLI:** author the group JSON with the scopes present, then import it:

```json
{
  "scope_name": "my-group",
  "ui_permissions": {
    "list_service": ["all"],
    "list_agents": ["all"],
    "list_skills": ["all"]
  }
}
```

```bash
export REGISTRY_URL="https://your-registry"
export TOKEN_FILE=".token"   # an admin JWT token file

uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  import-group --file my-group.json
```

The shipped example group files ([cli/examples/](../../cli/examples/)) already include `list_skills`, so copying one as a starting point gets this right for free. To scope discovery to specific resources instead of the whole family, list their paths rather than `"all"` (e.g. `"list_agents": ["/flight-booking"]`).

## Backfilling existing groups

Existing groups created before the gate do not hold the new scope. Fix them in two steps.

### Step 1: the built-in admin group (`list_skills` bootstrap)

`list_skills` is brand new, so no group holds it after upgrade, not even the admin group. Run the one-time backfill to grant `list_skills: ["all"]` to `mcp-registry-admin`. It is a dry run by default; add `--apply` to make the change. The simplest place to run it is **inside a running container**, where the registry's own environment is already present:

```bash
# Docker Compose
docker compose exec registry uv run python scripts/backfill-skill-list-scope.py --apply

# Kubernetes / EKS (exec into the registry pod)
kubectl exec -it deploy/registry -- uv run python scripts/backfill-skill-list-scope.py --apply
```

To run it **outside** the deployment (host shell, one-off ECS task, admin pod), the registry service names may not resolve, so pass the connection explicitly. The password and `SECRET_KEY` are read from the environment only, never as CLI args:

```bash
# MongoDB CE reached directly (skip replica-set discovery that advertises
# internal hostnames); password + SECRET_KEY come from the environment
DOCUMENTDB_PASSWORD=... SECRET_KEY=... \
  uv run python scripts/backfill-skill-list-scope.py --apply \
  --host localhost --username admin --direct-connection \
  --auth-server-url http://localhost:8888

# Amazon DocumentDB (TLS + SCRAM-SHA-1 via --storage-backend documentdb)
DOCUMENTDB_PASSWORD=... SECRET_KEY=... \
  uv run python scripts/backfill-skill-list-scope.py --apply \
  --storage-backend documentdb --tls \
  --host docdb.cluster-xxxx.us-east-1.docdb.amazonaws.com --username admin \
  --auth-server-url https://your-auth-server
```

Run `--help` for the full connection-argument list (`--host`, `--port`, `--database`, `--username`, `--auth-source`, `--tls`, `--direct-connection`, `--storage-backend`, `--auth-server-url`). `--auth-server-url` matters when you run outside the deployment: after writing the grant, the script asks the auth-server to reload its scope cache; if that URL is not reachable, the grant is persisted but the running auth-server keeps the old scopes until restarted or reloaded. There is no equivalent backfill for `list_agents`, because that scope already existed before the gate.

### Step 2: every other existing group (audit, then grant)

The backfill only fixes the admin group. For all other groups, first find which ones are missing a discovery scope with the **read-only** audit. It lists every group, flags the gaps, and prints the exact grant command per group. It changes nothing and works against any deployment surface (it only needs the registry URL and an admin token):

```bash
uv run python scripts/audit-discovery-scopes.py \
  --registry-url http://localhost --token-file .token
```

The audit drives `api/registry_management.py` under the hood, so it uses the same auth as the CLI. Use `--scope list_skills` to audit just one scope.

Then grant the scope to each flagged group. The audit prints one of two recipes per group:

- **Most groups:** a describe -> edit -> import round-trip. `import-group` **upserts** (updates the group in place), so you read the current definition, add the discovery key under `ui_permissions`, and re-import:

  ```bash
  uv run python api/registry_management.py \
    --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
    describe-group --name my-group --json > my-group.json
  # add "list_skills": ["all"] (and/or "list_agents") under "ui_permissions", then:
  uv run python api/registry_management.py \
    --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
    import-group --file my-group.json
  ```

- **Groups whose `server_access` contains a reserved wildcard server (`"*"`/`"all"`):** these cannot be re-imported (the import guard refuses wildcard `server_access`), so grant the scope via the IAM UI instead: Settings > IAM > Groups > *group* > UI Permissions > turn on **All** for the missing scope. The audit flags these automatically.

## Related mutation scopes (owner self-service)

Discovery (`list_`) is separate from mutation. If a non-admin can now *see* skills or agents but can no longer edit, delete, or toggle ones they own, grant the group the matching mutation scope too. Mutations are a dual gate: the caller needs the scope for the resource (or `"all"`) **and** must be an admin or the resource's owner.

| Action | Skill scope | Agent scope |
| --- | --- | --- |
| Edit | `modify_skill` | `modify_agent` |
| Delete | `delete_skill` | `delete_agent` |
| Enable/disable | `toggle_skill` | `toggle_agent` |

Grant these the same way as the discovery scopes (IAM UI or `import-group`). Note that a group which previously managed agents via the server scopes (`modify_service` / `toggle_service`) must be switched to the agent scopes (`modify_agent` / `toggle_agent`), which are now what the agent routes enforce.

## Related documentation

- [Scopes Management](../scopes-mgmt.md) - full scope file format, the complete UI-permissions reference table, and upgrade notes.
- [IAM Settings UI](../iam-settings-ui.md) - the visual Groups editor.
- [How do I create a non-admin group that can register servers but not delete them?](read-write-non-admin-group.md) - a worked example of a scoped read-write group.
