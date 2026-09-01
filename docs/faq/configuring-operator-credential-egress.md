# How do I inject a shared Backend Authentication credential on egress?

This FAQ covers `egress_auth_mode=operator_credential`: the gateway injects the server's registered **Backend Authentication** credential on the user proxy hop, after the caller is authorized. Every authorized caller shares one upstream identity. The credential never appears in client Connect configs.

For the mode table and security model, see [egress modes](../design/egress-auth-design.md#the-egress-modes) and [Per-User Egress Credential Vault](../egress-credential-vault.md).

## When to use this mode

Use it for a trusted internal or in-cluster MCP server where:

- the upstream expects a static bearer token or API key
- the gateway is the access-control boundary (scopes / groups at `/validate`)
- you do **not** need the upstream to see a distinct per-user identity

Do **not** use it for hosted third-party MCP servers that should call as the user. GitHub, Slack, Atlassian, and [Datadog MCP](configuring-datadog-mcp-server.md) stay on `oauth_user` or `pat`.

Storing Backend Authentication alone is **not** enough for user tool calls. Health checks and registry tool-listing already decrypt that credential. The `/mcp-proxy/...` hop does not, until this mode is set. That is deliberate: a shared identity on the data path is an operator choice, not an inference from `auth_scheme=bearer`.

## What you need

- `EGRESS_AUTH_ENABLED=true` on the registry and auth-server (same flag as 3LO / PAT / OBO). There is no second deployment flag.
- Registry admin access.
- The server registered with Backend Authentication `bearer` or `api_key` **and** a stored encrypted credential.
- Every registry and auth-server replica on a build that knows `operator_credential` **before** you flip the mode. Mixed-version rollouts vend nothing on old replicas.

## Step 1: Register the server with Backend Authentication

A template is in [`cli/examples/internal-bearer-mcp-server.json`](../../cli/examples/internal-bearer-mcp-server.json). Edit `proxy_pass_url`, then register. Put the real token in `--auth-credential` (or the UI), never in the JSON file:

```bash
uv run python api/registry_management.py \
  --registry-url https://mcpgateway.example.com --token-file .token \
  register --config cli/examples/internal-bearer-mcp-server.json

uv run python api/registry_management.py \
  --registry-url https://mcpgateway.example.com --token-file .token \
  server-update-credential --path /internal-mcp \
  --auth-scheme bearer --credential '<service-token>'
```

For a non-standard header (GitLab `PRIVATE-TOKEN`, Datadog `DD-API-KEY` on an internal wrapper, and so on) use `auth_scheme=api_key` and `--auth-header-name`. Operator mode inherits that header at vend time; there is no separate egress header field.

## Step 2: Enable operator credential egress

```bash
uv run python api/registry_management.py \
  --registry-url https://mcpgateway.example.com --token-file .token \
  egress-configure --path /internal-mcp --mode operator_credential
```

There is no `--provider`, `--client-id`, or `--client-secret`. The equivalent REST body:

```json
{
  "egress_auth_mode": "operator_credential"
}
```

In the UI: **Edit** the server, confirm Backend Authentication is set, then **Egress Auth → Mode → Shared backend credential**.

`400` here means Backend Authentication is `none` or the credential was never stored. Fix that first; do not retry egress-configure.

## Step 3: Enable the server and check Connect config

```bash
uv run python api/registry_management.py \
  --registry-url https://mcpgateway.example.com --token-file .token \
  toggle --path /internal-mcp --enabled true
```

Open **Connect**. The generated IDE config must **not** contain a server `Authorization` / API-key placeholder. The client authenticates to the gateway only; the gateway injects the upstream credential.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Health checks work, user tool calls get upstream `401` | Backend Authentication is set, egress mode is still `none` | Step 2 |
| `400` `operator_credential requires Backend Authentication scheme 'bearer' or 'api_key'` | `auth_scheme` is `none` | Set bearer or api_key and store a credential |
| `400` `requires a stored Backend Authentication credential` | Scheme set, credential never written | `server-update-credential` or the UI credential field |
| `503` `operator credential unavailable` | Stale mode, decrypt failure, or scheme flipped back to `none` | Re-store the credential; fail-closed on purpose |
| Intermittent tokenless upstream requests | Mixed-version rollout | Finish deploying registry + auth-server, then set the mode |
| Connect config still has `Authorization: Bearer YOUR_TOKEN` | Egress mode not saved, or dashboard cache | Re-open Connect after egress-configure; `egress_auth_mode` must not be `none` |
| Want per-user Datadog / GitHub identity | Wrong mode | Use `oauth_user` or `pat`, not `operator_credential` |

## Related documentation

- [How do I register MCP servers that require authentication?](registering-auth-protected-servers.md) — Backend Authentication for health checks
- [How does an admin seed per-user egress PATs?](seeding-per-user-egress-pats-as-admin.md) — per-user static tokens instead of a shared credential
- [How do I configure the Datadog MCP server with per-user egress OAuth?](configuring-datadog-mcp-server.md) — hosted Datadog MCP is 3LO, not this mode
