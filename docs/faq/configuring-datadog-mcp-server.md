# How do I configure the Datadog MCP server with per-user egress OAuth?

This FAQ walks through registering [Datadog's MCP server](https://docs.datadoghq.com/mcp_server/) behind the gateway and wiring it to the **per-user egress credential vault**, so each user connects their own Datadog account once and the gateway injects that user's token on egress. Nobody stores a Datadog credential on their laptop.

Datadog is the reference example of a **public OAuth client**: its authorization server issues clients that have *no* `client_secret` at all. That path requires `custom_token_auth_style: "none"` (added in 1.30.0). The same steps apply to any MCP server whose authorization server advertises `token_endpoint_auth_methods_supported: ["none"]`.

## Why Datadog needs the public-client path

Ask Datadog's authorization server directly:

```bash
curl -s https://mcp.datadoghq.com/.well-known/oauth-authorization-server
```

```json
{
  "issuer": "https://mcp.datadoghq.com/v1/mcp",
  "authorization_endpoint": "https://app.datadoghq.com/oauth2/v1/authorize",
  "token_endpoint": "https://app.datadoghq.com/api/v2/oauth2/token",
  "registration_endpoint": "https://app.datadoghq.com/api/v2/oauth2/register",
  "scopes_supported": ["mcp_all"],
  "code_challenge_methods_supported": ["S256"],
  "pkce_required": true,
  "token_endpoint_auth_methods_supported": ["none"]
}
```

Three facts drive the whole configuration:

- `token_endpoint_auth_methods_supported` is **exactly** `["none"]` — there is no confidential option, so a `client_secret` cannot be supplied even if you wanted to. Set `custom_token_auth_style: "none"`.
- `pkce_required: true` with `S256` — PKCE is the proof of possession that replaces the secret. The gateway always sends it for `custom` providers, so there is nothing to enable.
- The protected-resource metadata (`https://mcp.datadoghq.com/.well-known/oauth-protected-resource`) declares `"resource": "https://mcp.datadoghq.com"`. That bare origin — **not** the endpoint URL — is the [RFC 8707](https://www.rfc-editor.org/rfc/rfc8707) `custom_resource` value.

## What you need before you start

- A running MCP Gateway Registry with the vault enabled (`EGRESS_AUTH_ENABLED=true`) and a public callback base URL (`EGRESS_OAUTH_CALLBACK_BASE_URL`). See [Per-User Egress Credential Vault](../egress-credential-vault.md).
- Registry admin access (the egress configure endpoint is admin-only).
- **Datadog organization admin** access — one step must be done in the Datadog UI (Step 3), and it cannot be worked around.
- A Datadog account on a supported site. Datadog's MCP server is not available on GovCloud (`app.ddog-gov.com`, `us2.ddog-gov.com`).

Throughout, the gateway's callback URL is:

```text
{EGRESS_OAUTH_CALLBACK_BASE_URL}/oauth2/egress/callback
```

The exact value your deployment uses is returned by `GET /api/servers/{path}/egress-auth` as `callback_url` — read it from there rather than assembling it by hand. Examples below use `https://mcpgateway.example.com/oauth2/egress/callback`.

## Step 1: Register the MCP server

A ready-made registration file ships in the repo:

```bash
python api/registry_management.py \
  --registry-url https://mcpgateway.example.com --token-file .token \
  register --config cli/examples/datadog-mcp-server.json
```

> **Argument order matters.** Global flags (`--registry-url`, `--token-file`) come *before* the subcommand. Putting them after `register` fails with `unrecognized arguments`.

Two fields in that file are worth understanding:

- `proxy_pass_url` is `https://mcp.datadoghq.com/v1/mcp` — the current endpoint named by Datadog's protected-resource metadata.
- `append_mcp_path` is `false`, because that URL already ends in the `/mcp` transport segment.

## Step 2: Get a `client_id`

Datadog exposes a [Dynamic Client Registration](https://www.rfc-editor.org/rfc/rfc7591) endpoint that accepts anonymous registration:

```bash
curl -sX POST https://app.datadoghq.com/api/v2/oauth2/register \
  -H 'Content-Type: application/json' \
  -d '{
    "client_name": "MCP Gateway & Registry (egress vault)",
    "redirect_uris": ["https://mcpgateway.example.com/oauth2/egress/callback"],
    "token_endpoint_auth_method": "none",
    "grant_types": ["authorization_code", "refresh_token"],
    "response_types": ["code"]
  }'
```

The response contains a `client_id` and **no** `client_secret` — that is the point.

> **Expect a constant `client_id`.** At the time of writing this endpoint returns the *same* `client_id` regardless of the metadata posted, and merely echoes your `redirect_uris` back. It is a fixed public client behind a DCR-shaped facade, not a per-registration credential. Two consequences: the `client_id` is not a secret and needs no rotation, and the `redirect_uris` you post here are **not** what gets allow-listed — that is Step 3.

## Step 3: Allow-list the gateway callback in Datadog

This is the step that is easy to miss, and skipping it fails late: the user logs into Datadog, approves consent, and only then is the redirect rejected, so the gateway never receives an authorization code.

In the Datadog UI:

1. Go to **Organization Settings → Preferences** (<https://app.datadoghq.com/organization-settings/preferences>).
2. Find **MCP OAuth Redirect URLs**.
3. Add your gateway callback, e.g. `https://mcpgateway.example.com/oauth2/egress/callback`.

While you are there, confirm two related settings:

- **Role permissions** — *Organization Settings → Roles* → the role needs **MCP Read** (and **MCP Write** if you want write tools). Without these, consent succeeds but every tool call fails on permissions.
- **IP allowlist** — if your org enables [IP allowlisting](https://docs.datadoghq.com/account_management/org_settings/ip_allowlist/), remember that requests now originate from the **gateway's** egress IP, not the user's laptop. That is the intended effect of the vault, but the gateway's address must be allowed.

## Step 4: Configure egress auth on the server

```bash
python api/registry_management.py \
  --registry-url https://mcpgateway.example.com --token-file .token \
  egress-configure --path /datadog --mode oauth_user --provider custom \
  --client-id "<client_id from Step 2>" \
  --scopes mcp_all \
  --custom-authorize-url https://app.datadoghq.com/oauth2/v1/authorize \
  --custom-token-url     https://app.datadoghq.com/api/v2/oauth2/token \
  --custom-token-auth-style none \
  --custom-resource      https://mcp.datadoghq.com
```

Note there is **no** `--client-secret`. Passing one is not an error, but it is ignored and not stored: a public client has no request field to put it in.

The equivalent REST call:

```bash
curl -X POST https://mcpgateway.example.com/api/servers/datadog/egress-auth \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "egress_auth_mode": "oauth_user",
    "egress_provider": "custom",
    "client_id": "<client_id from Step 2>",
    "scopes": ["mcp_all"],
    "custom_authorize_url": "https://app.datadoghq.com/oauth2/v1/authorize",
    "custom_token_url": "https://app.datadoghq.com/api/v2/oauth2/token",
    "custom_token_auth_style": "none",
    "custom_resource": "https://mcp.datadoghq.com"
  }'
```

Or in the UI: edit the server, set **Egress Auth → Mode** to *Per-user OAuth (3LO)*, **Provider** to *Custom OIDC*, then set **Token Endpoint Authentication** to *None — public client (PKCE only)*. The Client Secret field disappears when you do, which is the confirmation that the public-client path is active. Fill **Resource (RFC 8707, optional)** with `https://mcp.datadoghq.com`.

Verify what was stored:

```bash
python api/registry_management.py \
  --registry-url https://mcpgateway.example.com --token-file .token \
  egress-config-get --path /datadog
```

`custom_token_auth_style` must echo back as `"none"`. If it comes back `null` or `post_body`, the setting did not persist and consent will fail with `invalid_client`.

## Step 5: Enable the server

Registration does not enable a server — the gateway will not route to it until you toggle it on:

```bash
python api/registry_management.py \
  --registry-url https://mcpgateway.example.com --token-file .token \
  toggle --path /datadog --enabled true
```

## Step 6: Connect your account and verify

Each user connects once:

1. Open the registry UI. The Datadog card now shows the **Link account** (3LO connect) affordance, because the server has per-user egress the current user can reach.
2. Click it, complete the Datadog login and consent, and you are returned to the gateway.
3. Confirm the vaulted connection:

```bash
curl -s https://mcpgateway.example.com/api/egress-auth/connections \
  -H "Authorization: Bearer $USER_TOKEN"
```

```json
[{ "server_path": "/datadog", "provider": "custom", "status": "active", "expires_at": "..." }]
```

`status: "active"` means the code exchange succeeded with `client_id` + PKCE and no secret — the public-client flow end to end. Tool calls now carry that user's Datadog token.

> If you configured the server from the CLI while the dashboard was already open, the connect icon will not appear until you reload the page: the dashboard fetches egress eligibility once and only re-fetches after a connect or disconnect.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `400 'none' is not a valid TokenEndpointAuthStyle` | Registry predates 1.30.0 | Upgrade; the public-client style did not exist before |
| `400 client_id required when custom_token_auth_style is 'none'` | No `client_id` sent | A public client is identified only by `client_id`; supply it |
| `400 client_secret required` | `custom_token_auth_style` was not applied (still `post_body`) | Re-check `egress-config-get`; the style must be `none` |
| Consent completes, then the redirect is rejected | Callback not allow-listed in Datadog | Step 3 — *Organization Settings → Preferences → MCP OAuth Redirect URLs* |
| Token exchange fails with `invalid_client` | Wrong auth style for this provider | See [Choosing `custom_token_auth_style`](../egress-credential-vault.md#choosing-token-auth-style) |
| Connect icon missing on the server card | Stale dashboard, or the user has no access to `/datadog` | Reload; then check `GET /api/egress-auth/available-servers` as that user |
| Consent works but tools return permission errors | Role lacks **MCP Read** / **MCP Write** | Step 3, *Organization Settings → Roles* |
| `num_tools: 0` and no health status | No user has consented yet | Datadog returns `401` to anonymous callers, so discovery stays empty until the first connection |

## Other Datadog sites

The values above are for **US1** (`app.datadoghq.com`). On other sites substitute the regional domain in the authorize URL, token URL, `proxy_pass_url`, and `custom_resource` — for example `app.datadoghq.eu`, `us3.datadoghq.com`, `us5.datadoghq.com`, `ap1.datadoghq.com`. Confirm by fetching the metadata for your site:

```bash
curl -s https://mcp.<your-datadog-site>/.well-known/oauth-authorization-server
```

## Related documentation

- [Per-User Egress Credential Vault](../egress-credential-vault.md) — the full design, API surface, and secret-store setup
- [Choosing `custom_token_auth_style`](../egress-credential-vault.md#choosing-token-auth-style) — the general decision procedure for any provider
- [How do I use a 3LO (per-user OAuth) AgentCore Gateway with the registry?](agentcore-3lo-per-user-oauth.md) — the same flow against a confidential Cognito client
- [How do I register commonly used third-party MCP servers like GitHub, Slack, and Atlassian?](registering-third-party-mcp-servers.md)
- [Datadog MCP Server documentation](https://docs.datadoghq.com/mcp_server/)
