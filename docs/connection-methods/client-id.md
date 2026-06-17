# Connection Method: Pre-Registered Public Client ID

This is one of three ways an AI coding assistant (Cursor, Claude Code, Codex)
can obtain the OAuth `client_id` it needs to log in to a gateway-protected MCP
server. See [the connection methods overview](../ai-coding-assistants-setup.md#how-coding-assistants-connect-three-methods)
for how this compares to Dynamic Client Registration and Client ID Metadata
Documents.

## When to use this method

Use a pre-registered public client when your identity provider (IdP) has
**anonymous Dynamic Client Registration (DCR) disabled** (common in enterprise
Keycloak, and the default posture in many enterprises for governance reasons).
With DCR off, the IDE cannot self-register a client, so the Connect dialog would
otherwise fall back to embedding a static gateway token. This method lets an
operator pre-register one public OAuth client and advertise its id, so the IDE
shows a real login button instead.

This method is a strong fit for an **enterprise registry**: a central IT team or
per-team operator runs the one-time setup, and every user is ready. It avoids the
anonymous-registration exposure and the client sprawl that DCR creates, and it
preserves full per-user auditability (see "Auditability" below).

## How it works

1. An operator registers a **public** OAuth client in the IdP. "Public" means it
   has no secret and uses the authorization-code flow with PKCE.
2. The registry advertises that client id. Configure it two ways:
   - **Registry-wide default:** set `IDE_OAUTH_CLIENT_ID` on the registry. Every
     server's Connect dialog uses it.
   - **Per-server override:** set `oauth_client_id` on an individual server entry.
     This wins over the global default for that server. See
     "[Setting the per-server overrides](#setting-the-per-server-overrides)" below
     for the API/UI fields that write it.
   Resolution happens server-side in `GET /api/servers/{path}/connect-config`
   (`server_info.oauth_client_id` || `settings.ide_oauth_client_id`), so the
   frontend receives a single resolved value.
3. When a client id resolves for a server, the Connect dialog drops the static
   gateway token and emits an OAuth/login config. The user's IDE runs the
   OAuth/PKCE flow in the browser, the user logs in as themselves, and the IDE
   receives a per-user token.

The client id is **public, not a secret** - it is safe to advertise in a Connect
config and to commit. It only identifies the application starting the login, not
the user.

## Fallback behavior when no client id is configured

This method is fully backward-compatible: if neither `IDE_OAUTH_CLIENT_ID` nor a
per-server `oauth_client_id` resolves, the OAuth-login path is skipped entirely
and the Connect dialog falls back to its prior behavior. But the fallback is NOT
uniform - it depends on the selected IDE and on whether the registry's auth
provider is Keycloak. Two important nuances to understand:

1. **The "is DCR available?" decision is provider-name-based, not a real check.**
   The frontend treats `auth_provider === "keycloak"` as "DCR is available"
   (`isDCR`). It does NOT verify that DCR is actually enabled on that Keycloak, and
   it does NOT treat Okta/Auth0/Entra as DCR-capable even though some of them are.
2. **Cursor never participates in the DCR fallback.** Only Claude Code and Codex
   have a DCR branch; Cursor always embeds the static token when no client id is set.

Fallback matrix (no client id configured):

| IDE | Keycloak (`isDCR=true`) | Non-Keycloak (Okta/Entra/Auth0/Cognito) |
| --- | --- | --- |
| Cursor | Embeds the static gateway token | Embeds the static gateway token |
| Claude Code | No token embedded (relies on DCR) | Embeds the static gateway token |
| Codex | Bare command (relies on DCR) | Bearer-token env var (`MCP_AUTH_TOKEN`) |

So your description of "falls back to DCR on Keycloak, embeds the token otherwise"
is accurate for **Claude Code and Codex only**. For **Cursor** the fallback is
always the embedded token, even on Keycloak.

**Caveat worth knowing:** because `isDCR` is just "provider is Keycloak", a
Keycloak deployment with **DCR disabled** (the exact scenario this method exists
for) that does NOT set a client id will have Claude Code/Codex emit a DCR-style
config that cannot actually complete - the user gets neither a working DCR
registration nor an embedded token. Configuring `IDE_OAUTH_CLIENT_ID` is the
correct path in that case; this is precisely why the method exists.

## What the Connect dialog emits per IDE

| IDE | Output |
|-----|--------|
| Cursor (JSON) | `"auth": { "CLIENT_ID": "<id>" }`, gateway token omitted |
| Claude Code | `claude mcp add --transport http --client-id <id> <name> <url>` |
| Codex | `codex mcp add <name> --url <url> --oauth-client-id <id>` |

Other IDEs (Roo Code / Kiro / Goose / VS Code default / curl) keep the existing
static-token behavior; they have no verified fixed-public-client OAuth config
syntax yet.

Example (Cursor):

```json
{
  "mcpServers": {
    "my-server": {
      "url": "https://gateway.example.com/my-server/mcp",
      "auth": { "CLIENT_ID": "<your-public-client-id>" }
    }
  }
}
```

## Access is derived from groups, not from the requested scope

This is the single most important concept, and it confuses people:

- The OAuth **scope** the IDE requests only governs whether the IdP login
  handshake succeeds. Keep it to basic, IdP-universal scopes
  (`openid email profile offline_access`) via `MCP_ADVERTISED_SCOPES`.
- A user's **actual access** is derived from the **`groups` claim** in the token.
  The auth server maps IdP groups to registry scopes (`map_groups_to_scopes`),
  and the registry authorizes against those. So trimming the advertised scope
  list does NOT reduce access.

Consequence: for any IdP, the token MUST carry the user's group membership, and
those groups MUST be mapped to a registry scope (in the `mcp_scopes` collection).
If the token has no groups, the user authenticates successfully but is denied on
every server.

## The callback port: pin it for strict IdPs

The IDE runs a local loopback listener for the OAuth redirect, with a
`redirect_uri` of `http://localhost:<PORT>/callback`. By default the port is
ephemeral (different each attempt). IdPs differ on how they match it:

- **Keycloak** accepts a wildcard redirect URI (`http://localhost/*`), so any
  port works. No port pinning needed.
- **Okta / Entra / Cognito** require the redirect URI to be registered EXACTLY,
  including the port. A rotating port can never match.

For strict IdPs, pin the port with the **`IDE_OAUTH_CALLBACK_PORT`** registry
setting. When set (non-zero), the Connect dialog emits the matching
`--callback-port` in the Claude Code command automatically, so the IDE uses that
fixed port instead of a random one:

```
IDE_OAUTH_CALLBACK_PORT=33418
```

You must register `http://localhost:<PORT>/callback` (the same port) on the
public client in the IdP. With the setting in place, the dialog produces:

```bash
claude mcp add --transport http \
  --client-id <your-public-client-id> \
  --callback-port 33418 \
  my-server https://gateway.example.com/my-server/mcp
```

`0` (default) omits `--callback-port` — correct for Keycloak and DCR flows. The
callback path is always `/callback` and is not configurable.

### IDE support for fixed ports (important)

Only **Claude Code** supports pinning the callback port (`--callback-port`). So:

- **Claude Code:** works with strict IdPs once `IDE_OAUTH_CALLBACK_PORT` is set
  and the matching redirect URI is registered.
- **Codex / Cursor:** have no way to pin the loopback port (Codex's
  `mcp add` has no `--callback-port`; Cursor's JSON config has no port field).
  They use a random port, so OAuth login against a strict IdP (Okta/Entra/
  Cognito) will fail for them. The Connect dialog surfaces a warning in the
  Codex tab when a callback port is configured. These IDEs work fine against
  Keycloak (wildcard redirect) or any DCR-enabled provider.

## Per-IdP setup

The registry ships helper scripts that create the correctly-typed public client
for each IdP. All are env-var driven (secrets via env, never CLI args).

### Keycloak (simplest - 1 step)

```bash
export KEYCLOAK_ADMIN_PASSWORD="<admin pw>"
bash setup/idp/keycloak/setup-ide-public-client.sh
# prints a client id; then set on the registry and restart:
#   IDE_OAUTH_CLIENT_ID=mcp-gateway-ide
#   MCP_ADVERTISED_SCOPES="openid email profile offline_access"
```

Creates `mcp-gateway-ide` as a public authorization-code + PKCE client
(`publicClient=true`, `standardFlowEnabled=true`, PKCE S256), with
**wildcard loopback redirect URIs** (`http://localhost/*`) and a **groups mapper**
so the token carries group membership. This is a different client TYPE than the
`mcp-gateway-m2m` client (which is confidential + client_credentials and cannot
drive an interactive login).

Because of the wildcard redirect and the groups mapper, Keycloak needs no port
pinning and no extra claim configuration. It works in one step.

### Okta (works, but the most involved - 5 steps)

```bash
export OKTA_ORG_URL="https://<your-org>.okta.com"
export OKTA_API_TOKEN="<admin SSWS token>"
bash setup/idp/okta/setup-ide-public-client.sh
```

Creates an OIDC **native** (public) app (`token_endpoint_auth_method=none`,
authorization_code + refresh_token). After that, four more things are required -
Okta is stricter than Keycloak at every layer:

1. **Pin the callback port:** set `IDE_OAUTH_CALLBACK_PORT` on the registry
   (e.g. `56789`) and register exactly `http://localhost:56789/callback` on the
   app. Okta does literal redirect_uri matching (no wildcard, no port-agnostic
   loopback), and the dialog emits the matching `--callback-port` for Claude Code.
2. **Assign the user (or a group) to the app.** Okta grants nothing until the
   user is assigned. Symptom otherwise: `user_not_assigned`.
3. **Add a `groups` claim to the authorization server.** By default an Okta
   access token carries only `openid/email/profile/offline_access` and NO groups,
   so the registry maps the user to no scopes and denies every server. Add a
   claim (claimType ACCESS_TOKEN, valueType GROUPS, regex `.*`) on the custom
   authorization server. THIS IS THE NON-OBVIOUS GOTCHA: login succeeds but every
   server returns 403 until this claim exists.
4. **Map the Okta group to a registry scope** in the `mcp_scopes` collection
   (e.g. Okta group `registry-admins` -> the `registry-admins` scope).

Verified end-to-end: with all five in place, an Okta user in `registry-admins`
connects and is authorized as admin.

### Amazon Cognito (works - validated on ECS)

```bash
export COGNITO_USER_POOL_ID="us-east-1_XXXXXXXXX"
export AWS_REGION="us-east-1"
bash setup/idp/cognito/setup-ide-public-client.sh
```

Creates a Cognito user-pool app client with **no secret** (`--no-generate-secret`),
authorization_code + PKCE. Cognito uses the AWS API, so the script relies on your
AWS credentials/role rather than an admin token. After that:

1. **Pin the callback port:** set `IDE_OAUTH_CALLBACK_PORT` (e.g. `56789`) and the
   script registers `http://localhost:56789/callback` on the client. Cognito
   matches the callback URL literally including the port (no wildcard on
   localhost), so a fixed port is required - the dialog emits the matching
   `--callback-port` for Claude Code.
2. **Advertise Cognito-valid scopes only:** set
   `MCP_ADVERTISED_SCOPES="openid email profile"`. Cognito's hosted UI does NOT
   offer `offline_access` as a selectable OAuth scope, so including it causes
   `invalid_scope`.
3. **Group membership + mapping:** put the user in a Cognito group whose name
   matches a registry scope in `mcp_scopes` (e.g. `registry-admins`). NO claim
   configuration is needed - Cognito includes the `cognito:groups` claim in
   access tokens by default, and the registry's Cognito provider reads it.

One provider-code detail (fixed in this feature): Cognito **access tokens have no
`aud` claim** (they carry `client_id` + `token_use=access`). The Cognito provider
validates the `client_id` against an allowlist (the web client plus
`IDE_OAUTH_CLIENT_ID`) instead of `aud`, so IDE access tokens are accepted.

Verified end-to-end on the ECS/Terraform deployment: a Cognito user in
`registry-admins` logs in via Claude Code (`--callback-port`) and makes
authorized tool calls. Same caveat as Okta: only Claude Code can pin the port,
so Codex/Cursor will fail Cognito's literal callback match.

### Microsoft Entra ID (currently blocked - see note)

```bash
export ENTRA_TENANT_ID="<tenant>"
export ENTRA_GRAPH_CLIENT_ID="<app with Application.ReadWrite.All>"
export ENTRA_GRAPH_CLIENT_SECRET="<secret>"
bash setup/idp/entra/setup-ide-public-client.sh
```

Creates an app registration as a public client (`isFallbackPublicClient=true`,
loopback redirect URIs, `groupMembershipClaims=SecurityGroup`). The client
creation works, BUT the IDE login flow does NOT yet complete on Entra:

Entra binds requested scopes to the `resource` parameter and requires scopes in
the resource-qualified `api://<app-id>/<scope>` form. The PRM currently advertises
bare scopes, so Entra rejects the authorization request with
`AADSTS9010010: The resource parameter ... doesn't match with the requested scopes`.
This is tracked as a known gap and needs the PRM to emit Entra-qualified scopes;
it is not solvable by configuration alone. Until then, Entra IDE login via this
method is not supported.

## Security notes

- The client id is **public** - not a secret. Treat it as an identifier, like a
  mobile app's client id.
- The real risk surface is the public client's **redirect URIs**: scope them to
  loopback only; never register broad external redirect URIs on a public client.
- This method is a security improvement over the static-token fallback: it
  replaces a single long-lived shared bearer token with per-user interactive
  login.

## Auditability

A shared client id does NOT reduce auditability. The client id identifies the
application; each user authenticates in their own browser, so the IdP mints a
per-user token carrying their `sub` / `preferred_username` / `groups`. The
registry's audit log keys on the username (and groups/scopes), not the client id.
Ten thousand users sharing one client id still produce ten thousand individually
attributable identities.

If you want to distinguish which IDE initiated a login (not just which user),
register a separate client per IDE (e.g. `mcp-gateway-cursor`,
`mcp-gateway-claude-code`, `mcp-gateway-codex`, plus an `mcp-gateway-other`
catch-all). User attribution is unaffected either way.

## The `/mcp` suffix (`append_mcp_path`)

The gateway Connect URL normally appends `/mcp`. Some servers (e.g. AWS
Knowledge) serve MCP at the server path itself and break on `/mcp`. Set
`append_mcp_path: false` on that server entry to emit the URL without the suffix;
set `true` to force it; leave unset to auto-detect from `proxy_pass_url`. For an
entirely custom endpoint, use `mcp_endpoint`.

## Setting the per-server overrides

Both per-server overrides (`oauth_client_id` and `append_mcp_path`) are written
through the same surfaces that manage any other server field. Omitting a field leaves
it unset, so existing servers are unaffected (`oauth_client_id` falls back to the
registry-wide `IDE_OAUTH_CLIENT_ID`; `append_mcp_path` auto-detects from
`proxy_pass_url`).

### Registry UI

Set both fields on the server **Register** and **Edit** forms. After saving, the
server's **Connect** dialog reflects the values immediately.

### Register API (JWT Bearer)

Pass them as optional form fields to `POST /api/servers/register`:

```bash
curl -X POST https://registry.example.com/api/servers/register \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "name=AWS Knowledge" \
  -F "description=Root-endpoint MCP server" \
  -F "path=/aws-knowledge" \
  -F "proxy_pass_url=https://knowledge-mcp.example.com" \
  -F "oauth_client_id=mcp-gateway" \
  -F "append_mcp_path=false"
```

### Update API (PUT / PATCH)

`PUT /api/servers/{path}` (full replace) and `PATCH /api/servers/{path}` (partial)
both accept the fields in the JSON body:

```bash
curl -X PATCH https://registry.example.com/api/servers/aws-knowledge \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"oauth_client_id": "mcp-gateway", "append_mcp_path": false}'
```

PATCH merges (only the fields you send change); PUT replaces, so a field omitted from a
PUT body is cleared back to unset.

### Reading back the resolved values

`GET /api/servers/{path}/connect-config` returns the effective `oauth_client_id`
(per-server value, else the registry-wide default) and the stored `append_mcp_path`.
This is the value the frontend Connect dialog consumes.

## Related

- [Connection methods overview](../ai-coding-assistants-setup.md#how-coding-assistants-connect-three-methods)
- [Dynamic Client Registration](dynamic-client-registration.md)
- [Client ID Metadata Documents](client-id-metadata-documents.md)
- [OAuth discovery endpoints](../oauth-discovery-endpoints.md)
- [Keycloak MCP clients](../keycloak-mcp-clients.md)
