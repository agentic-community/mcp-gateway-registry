# Keycloak: MCP Client Guide (DCR + OAuth)

> **Scope of this doc**: How a spec-compliant MCP client (Claude Code,
> Claude.ai connectors, Cursor) authenticates against the gateway via
> Dynamic Client Registration and OAuth when Keycloak is the configured
> identity provider.
>
> **Looking for agent service accounts or operations?** For agent-to-gateway
> M2M authentication using Keycloak service accounts, plus operational
> procedures (backup/restore, agent lifecycle, monitoring), see the
> companion doc: [Keycloak: Agent M2M & Operations Guide](keycloak-agent-m2m.md).

This doc explains how a spec-compliant MCP client (Claude Code, Claude.ai
connectors, Cursor) authenticates against the gateway when Keycloak is the
configured identity provider. It is structured as four levels of progressively
deeper detail: **100** (what), **200** (how, conceptually), **300** (the wire
sequence with messages and payloads), **400** (every Keycloak knob and gateway
code path that participates).

The goal is that a future engineer can read level 100 in 30 seconds, get a
working mental model from level 200, debug their way through level 300, and
make changes with confidence at level 400.

---

## 100 — What this enables

When a user runs a single command on their laptop:

```
claude mcp add --transport http ai-registry-tools https://mcpgateway.ddns.net/airegistry-tools/mcp
```

then opens Claude Code and selects "Authenticate", a browser opens, the user
signs in to Keycloak with their corporate identity, and from that point on:

- Claude Code can invoke MCP tools at the gateway
- Each tool call is gated by the user's Keycloak group memberships, which the
  gateway maps to per-user MCP scopes
- No operator pre-registers an OAuth client; no user pastes a `client_id` or
  `client_secret`

Three IETF specs make this work:

| Spec | What it gives us |
| --- | --- |
| [RFC 9728](https://datatracker.ietf.org/doc/html/rfc9728) | The gateway publishes a Protected Resource Metadata document at `/.well-known/oauth-protected-resource` that points discovery clients at the IdP |
| [RFC 8414](https://datatracker.ietf.org/doc/html/rfc8414) | Keycloak (and the gateway, as a passthrough) publishes Authorization Server Metadata at `/.well-known/oauth-authorization-server` so the client knows where `/authorize`, `/token`, and `/register` live |
| [RFC 7591](https://datatracker.ietf.org/doc/html/rfc7591) | Keycloak accepts a Dynamic Client Registration request from any well-formed MCP client and mints a fresh `client_id` for it |

The gateway implements RFC 9728 + RFC 8414 (PR #1115 / issue #989). Keycloak
implements RFC 7591 + OAuth 2.1 + PKCE.

---

## 200 — How it works conceptually

Three mental milestones in the flow.

### Milestone 1: Discovery

Claude Code knows nothing about the gateway except its base URL. It does:

1. POST any MCP request without a token to `https://mcpgateway.ddns.net/airegistry-tools/mcp`
2. Gateway returns 401 with a `WWW-Authenticate: Bearer realm="mcp", resource_metadata="https://mcpgateway.ddns.net/.well-known/oauth-protected-resource"` header
3. Claude Code GETs the resource_metadata URL, reads the `authorization_servers` field, learns that the IdP is Keycloak's `mcp-gateway` realm
4. Claude Code GETs Keycloak's OIDC discovery doc, learns the URLs of `/register`, `/authorize`, `/token`, and the JWKS

At this point the client knows everything it needs to register itself and
start an OAuth flow.

### Milestone 2: Dynamic Client Registration + Auth Code + PKCE

5. Claude Code POSTs to Keycloak's `registration_endpoint` with a JSON body
   describing itself (name, redirect URIs). Keycloak runs its DCR policies
   (Allowed Client Scopes, Trusted Hosts, Consent Required), then returns a
   fresh `client_id`. No `client_secret` is issued: the client is registered
   as `token_endpoint_auth_method=none` (a public client).
6. Claude Code generates a PKCE code-verifier + code-challenge pair, then
   redirects the browser to Keycloak's `/authorize` endpoint with the
   challenge attached
7. User signs in to Keycloak with their corporate password, sees a consent
   screen ("Grant Access to Claude Code"), clicks Yes
8. Keycloak redirects the browser back to `http://localhost:<port>/callback`
   with an `auth_code` in the URL query string
9. Claude Code's local listener catches the callback, sends the code +
   PKCE-verifier + `client_id` to Keycloak's `/token` endpoint
10. Keycloak validates the verifier against the original challenge and returns
    an `access_token` plus a `refresh_token`

### Milestone 3: First MCP call + per-user authorization

11. Claude Code retries the original MCP request with `Authorization: Bearer
    <access_token>`
12. Gateway's nginx forwards the request through `auth_request /validate` to
    the auth_server
13. auth_server fetches Keycloak's JWKS, verifies the JWT signature, extracts
    the `groups` claim (`["mcp-registry-admin", "registry-admins", ...]`)
14. auth_server calls `map_groups_to_scopes(groups)` against DocumentDB to
    translate IdP groups to registry scopes
15. auth_server checks whether any scope in the result allows access to
    `airegistry-tools/initialize`; if yes, returns 200 to nginx; nginx forwards
    the original request to the airegistry-tools MCP server
16. Tools execute, results return to Claude Code

### What makes the flow Keycloak-specific

Three Keycloak-specific configurations that have to be in place:

- **Groups protocol mapper on the `basic` client-scope**: without this, tokens
  for DCR'd clients have no `groups` claim → step 13 above fails
- **`Allowed Client Scopes` policy includes all realm scopes**: without this,
  DCR rejects registrations referencing any non-OIDC scope → step 5 fails
- **`Trusted Hosts` policy with IP check off + `localhost` allowed**: without
  this, DCR rejects clients on cloud egress IPs → step 5 fails

Both `init-keycloak.sh` (fresh installs) and `upgrade-realm-for-dcr.sh`
(existing installs) now apply these.

---

## 300 — The wire sequence

This is what you'd see if you tcpdump'd the entire flow. Bodies abbreviated
where they don't add information; full schemas at level 400.

```
participant Browser as Browser (laptop)
participant Claude as Claude Code (laptop or EC2)
participant nginx as Gateway nginx
participant auth as Gateway auth_server
participant kc as Keycloak

Note over Claude: User runs /mcp -> Authenticate

Claude->>nginx: POST /airegistry-tools/mcp (no Authorization header)
nginx->>auth: GET /validate (no Authorization)
auth-->>nginx: 401
nginx-->>Claude: HTTP 401\nwww-authenticate: Bearer realm="mcp",\n  resource_metadata="https://mcpgateway.ddns.net/.well-known/oauth-protected-resource"

Claude->>nginx: GET /.well-known/oauth-protected-resource
nginx-->>Claude: 200 {resource, authorization_servers, scopes_supported, ...}

Claude->>nginx: GET /realms/mcp-gateway/.well-known/openid-configuration
nginx->>kc: same path (proxy_pass)
kc-->>nginx: 200 OIDC config (incl. registration_endpoint)
nginx-->>Claude: 200

Note over Claude: Discovery complete. Now register.

Claude->>nginx: POST /realms/mcp-gateway/clients-registrations/openid-connect\n  Body: {client_name, redirect_uris, grant_types, ...}
nginx->>kc: same POST
kc-->>nginx: 201 {client_id, client_id_issued_at, ...}
nginx-->>Claude: 201

Note over Claude: Got client_id. Generate PKCE pair.

Claude->>Browser: open https://mcpgateway.ddns.net/realms/mcp-gateway/protocol/openid-connect/auth?\n  response_type=code&client_id=<dcr_id>&redirect_uri=http://localhost:8765/callback&\n  state=<random>&code_challenge=<S256(verifier)>&code_challenge_method=S256&\n  scope=profile+email+offline_access&prompt=consent

Browser->>nginx: GET /realms/mcp-gateway/protocol/openid-connect/auth?...
nginx->>kc: same
kc-->>Browser: 200 login page

Note over Browser: User enters credentials

Browser->>kc: POST /realms/mcp-gateway/login-actions/authenticate\n  Body: username, password
kc-->>Browser: 302 to consent screen
Browser->>kc: GET consent
kc-->>Browser: 200 "Grant Access to Claude Code?"
Browser->>kc: POST consent (Yes)
kc-->>Browser: 302 http://localhost:8765/callback?code=<auth_code>&state=<random>&iss=...

Browser->>Claude: GET /callback?code=...&state=... (Claude's local listener)

Claude->>nginx: POST /realms/mcp-gateway/protocol/openid-connect/token\n  Body: grant_type=authorization_code, code=<auth_code>,\n  redirect_uri, client_id, code_verifier
nginx->>kc: same POST
kc-->>nginx: 200 {access_token, refresh_token, expires_in, ...}
nginx-->>Claude: 200

Note over Claude: Got token. Retry the MCP call.

Claude->>nginx: POST /airegistry-tools/mcp\n  Authorization: Bearer <access_token>\n  Body: {jsonrpc:2.0, method:initialize, ...}
nginx->>auth: GET /validate (Authorization forwarded)

auth->>kc: GET /protocol/openid-connect/certs (cached)
kc-->>auth: JWKS

Note over auth: 1. Verify JWT signature\n2. Verify issuer matches realm\n3. Extract groups claim\n4. map_groups_to_scopes() in DocumentDB\n5. Check airegistry-tools/initialize allowed?

auth-->>nginx: 200 (X-User, X-Username, X-Scopes headers)
nginx->>nginx: forward to upstream airegistry-tools backend
nginx-->>Claude: 200 {jsonrpc:2.0, result: {tools: [...]}}
```

### Real payloads from the live deployment

#### PRM document (gateway response)

```json
{
  "resource": "https://mcpgateway.ddns.net",
  "authorization_servers": [
    "https://mcpgateway.ddns.net/realms/mcp-gateway"
  ],
  "scopes_supported": [
    "profile",
    "email",
    "offline_access"
  ],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://mcpgateway.ddns.net/docs/oauth"
}
```

`scopes_supported` is intentionally narrow — only Keycloak-recognized OIDC
scopes — so the DCR registration step doesn't get rejected. Per-user
authorization at the gateway uses **groups → scopes** mapping (see level 400),
not these advertised scopes.

#### AS metadata (gateway proxies Keycloak's OIDC config)

Key fields from the response (full doc has ~50 fields):

```json
{
  "issuer": "https://mcpgateway.ddns.net/realms/mcp-gateway",
  "authorization_endpoint": "https://mcpgateway.ddns.net/realms/mcp-gateway/protocol/openid-connect/auth",
  "token_endpoint": "https://mcpgateway.ddns.net/realms/mcp-gateway/protocol/openid-connect/token",
  "registration_endpoint": "https://mcpgateway.ddns.net/realms/mcp-gateway/clients-registrations/openid-connect",
  "jwks_uri": "https://mcpgateway.ddns.net/realms/mcp-gateway/protocol/openid-connect/certs",
  "code_challenge_methods_supported": ["plain", "S256"],
  "grant_types_supported": [
    "authorization_code",
    "refresh_token",
    "client_credentials"
  ],
  "response_types_supported": ["code"]
}
```

The `registration_endpoint` field is the load-bearing one for DCR; without it
Claude Code reports "Incompatible auth server: does not support dynamic client
registration".

#### DCR registration request (Claude Code → Keycloak)

```http
POST /realms/mcp-gateway/clients-registrations/openid-connect HTTP/1.1
Host: mcpgateway.ddns.net
Content-Type: application/json

{
  "client_name": "Claude Code",
  "redirect_uris": ["http://localhost:8765/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "application_type": "web"
}
```

#### DCR registration response (Keycloak)

```http
HTTP/1.1 201 Created
Content-Type: application/json

{
  "client_id": "2cb00b3f-87bd-45bb-bcec-41cca5238790",
  "client_name": "Claude Code",
  "redirect_uris": ["http://localhost:8765/callback"],
  "token_endpoint_auth_method": "none",
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code", "none"],
  "scope": "address phone offline_access microprofile-jwt",
  "subject_type": "public",
  "client_id_issued_at": 1779641813,
  "registration_client_uri": "https://mcpgateway.ddns.net/realms/mcp-gateway/clients-registrations/openid-connect/2cb00b3f-...",
  "registration_access_token": "eyJhbGc..."
}
```

The `client_id` here is a fresh UUID. The `registration_access_token` lets the
client update or delete its own registration later (Claude Code uses it to
re-register when the user runs `/mcp` → Re-authenticate).

#### Authorization request (browser URL)

```
GET https://mcpgateway.ddns.net/realms/mcp-gateway/protocol/openid-connect/auth
  ?response_type=code
  &client_id=2cb00b3f-87bd-45bb-bcec-41cca5238790
  &code_challenge=<base64url(SHA256(verifier))>
  &code_challenge_method=S256
  &redirect_uri=http%3A%2F%2Flocalhost%3A8765%2Fcallback
  &state=<random>
  &scope=profile+email+offline_access
  &prompt=consent
  &resource=https%3A%2F%2Fmcpgateway.ddns.net%2F
```

The `resource` parameter is RFC 8707; Keycloak silently ignores it, but it's
required by the MCP 2025-06-18 spec.

#### Callback URL (Keycloak → browser → Claude Code listener)

```
http://localhost:8765/callback
  ?code=559f4103-d1a3-4503-a2bb-fdc501fd2ef4.<session-state-uuid>.<auth-state-uuid>
  &state=<echoed back>
  &session_state=<keycloak-session-uuid>
  &iss=https%3A%2F%2Fmcpgateway.ddns.net%2Frealms%2Fmcp-gateway
```

#### Token request (Claude Code → Keycloak)

```http
POST /realms/mcp-gateway/protocol/openid-connect/token HTTP/1.1
Host: mcpgateway.ddns.net
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code
&code=<auth_code_from_callback>
&redirect_uri=http%3A%2F%2Flocalhost%3A8765%2Fcallback
&client_id=2cb00b3f-87bd-45bb-bcec-41cca5238790
&code_verifier=<original_pkce_verifier>
```

#### Token response (Keycloak)

```json
{
  "access_token": "eyJhbGc...",
  "expires_in": 300,
  "refresh_expires_in": 1800,
  "refresh_token": "eyJhbGc...",
  "token_type": "Bearer",
  "id_token": "eyJhbGc...",
  "not-before-policy": 0,
  "session_state": "<uuid>",
  "scope": "profile email offline_access basic"
}
```

#### Decoded access_token claims

```json
{
  "exp": 1779642113,
  "iat": 1779641813,
  "auth_time": 1779641810,
  "jti": "<uuid>",
  "iss": "https://mcpgateway.ddns.net/realms/mcp-gateway",
  "aud": "account",
  "sub": "<user-uuid>",
  "typ": "Bearer",
  "azp": "2cb00b3f-87bd-45bb-bcec-41cca5238790",
  "session_state": "<uuid>",
  "acr": "1",
  "allowed-origins": ["http://localhost:8765"],
  "scope": "profile email offline_access basic",
  "email_verified": true,
  "name": "Admin User",
  "groups": ["mcp-registry-admin", "mcp-servers-unrestricted", "registry-admins"],
  "preferred_username": "admin",
  "given_name": "Admin",
  "family_name": "User",
  "email": "admin@example.com"
}
```

The `groups` claim is the load-bearing one for per-user authorization. It
appears here only because the **groups protocol mapper is attached to the
`basic` client-scope** (see level 400). Without that, `groups` is absent and
the gateway falls back to `scope` for authorization, which contains only OIDC
standards and doesn't unlock any registry servers.

#### Validated MCP request (Claude Code → gateway)

```http
POST /airegistry-tools/mcp HTTP/1.1
Host: mcpgateway.ddns.net
Authorization: Bearer eyJhbGc...
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": {"roots": {}, "elicitation": {}},
    "clientInfo": {"name": "claude-code", "version": "2.1.150"}
  }
}
```

---

## 400 — Every detail

### Keycloak realm components that participate in this flow

#### `mcp-gateway` realm itself

Created by `init-keycloak.sh::create_realm()`. Issuer URL is
`https://mcpgateway.ddns.net/realms/mcp-gateway`. Tokens carry this in the
`iss` claim and are validated against it server-side
([auth_server/providers/keycloak.py:118-123](https://github.com/agentic-community/mcp-gateway-registry/blob/main/auth_server/providers/keycloak.py#L118)).

#### Pre-defined clients

- `mcp-gateway-web` — confidential client used by the gateway's own web-UI
  OAuth login flow. Has a `client_secret`. Not relevant to MCP DCR clients.
- `mcp-gateway-m2m` — confidential client for service-to-service M2M tokens.
  Used internally by federation flows.

Both are created by `init-keycloak.sh::create_clients()`. **Neither is used by
Claude Code or any other MCP client.** MCP clients DCR themselves at runtime.

#### DCR'd clients

Created on-the-fly by Claude Code's first POST to
`/clients-registrations/openid-connect`. Each one has:

- A UUID `client_id` (e.g., `2cb00b3f-87bd-45bb-bcec-41cca5238790`)
- `redirect_uris: ["http://localhost:<port>/callback"]`
- `token_endpoint_auth_method: "none"` (public client, no `client_secret`)
- The realm's default-default client-scopes auto-attached, which is just
  `basic` for DCR'd clients
- Optional client-scopes attached: `profile`, `email`, `offline_access`,
  `address`, `phone`, `microprofile-jwt`

Claude Code re-runs DCR every time the user re-authenticates, producing a new
`client_id` each time. Old `client_id`s remain in the realm and accumulate.
Run `bash keycloak/setup/cleanup-stale-dcr-clients.sh` on demand to delete
DCR'd clients with no active sessions; pre-defined clients (mcp-gateway-web,
mcp-gateway-m2m, etc.) are never touched. The script is idempotent and
supports `--dry-run` to preview before deleting. CIMD adoption (Keycloak
26.6+, sub-issue #993) eventually eliminates this entire problem.

#### Client-scopes used in this flow

| Scope | Type for DCR'd clients | Mappers attached | Purpose |
| --- | --- | --- | --- |
| `basic` | default-default | `sub`, `auth_time`, **`groups`** + **`mcp-gateway-audience`** (both added by us) | Always attached. Hosts the groups mapper AND the audience mapper, so every DCR'd-client token carries both `groups` (for per-user authorization) and `aud="mcp-gateway"` (for strict RFC 8707 audience binding). |
| `profile` | optional | `given_name`, `family_name`, `preferred_username`, etc. | Standard OIDC profile claims |
| `email` | optional | `email`, `email_verified` | Standard OIDC email claims |
| `offline_access` | optional | (no mappers) | Allows refresh tokens |

The critical configuration: **the `groups` protocol mapper lives on the `basic`
scope, not on `profile` or `roles`**. Detail at the bottom of this section.

#### Anonymous DCR policies (in `Realm Settings → Client Registration`)

Three policies gate every DCR registration request:

1. **Allowed Client Scopes** (`allowed-client-templates`) — checks that every
   scope name the client requests in its registration is in the allowlist.
   Configured by `configure_dcr_allowed_scopes()` to include all realm scopes.
   Without this, DCR fails for any client that includes registry-internal
   scope names in its request.

2. **Trusted Hosts** (`trusted-hosts`) — gates by client IP and/or redirect
   URI host. Configured by `configure_dcr_trusted_hosts()` with
   `host-sending-registration-request-must-match=false`,
   `client-uris-must-match=true`,
   `trusted-hosts: ["localhost", "127.0.0.1", "claude.ai"]`. So the gateway
   accepts DCR requests from any IP, but only allows redirect URIs that
   resolve to one of those hosts. Three entries cover the realistic clients:
   - `localhost`: Claude Code (CLI) opens its OAuth listener on
     `http://localhost:<port>/callback`
   - `127.0.0.1`: Claude Code variants that send the loopback IP form
     directly instead of `localhost`
   - `claude.ai`: Claude.ai's Custom Connector flow runs DCR from
     Anthropic's cloud and the redirect URI is
     `https://claude.ai/api/mcp/auth_callback`

3. **Consent Required** — forces every DCR'd client to require user consent
   on first OAuth flow. Default-on; unchanged by us. The "Grant Access to
   Claude Code" page the user sees is from this policy.

#### The groups protocol mapper

Configured by `setup_dcr_groups_mapper()`. JSON sent to Keycloak:

```json
{
  "name": "groups",
  "protocol": "openid-connect",
  "protocolMapper": "oidc-group-membership-mapper",
  "consentRequired": false,
  "config": {
    "full.path": "false",
    "id.token.claim": "true",
    "access.token.claim": "true",
    "claim.name": "groups",
    "userinfo.token.claim": "true"
  }
}
```

Attached to the `basic` client-scope. **Why `basic`** specifically: it is the
only scope Keycloak attaches to every client (including DCR'd ones)
automatically. The `roles` scope is realm-default-default, but DCR'd clients
do not actually receive it — Keycloak's behavior is that the realm-default
scope assignment depends on a different code path that doesn't run for DCR.
Putting the mapper on `basic` is the only reliable way to get groups into
DCR'd-client tokens.

`full.path: false` means group names appear as flat strings (`"mcp-admin"`,
not `"/mcp-admin"`).

#### The audience protocol mapper

Configured by `setup_dcr_audience_mapper()`. JSON sent to Keycloak:

```json
{
  "name": "mcp-gateway-audience",
  "protocol": "openid-connect",
  "protocolMapper": "oidc-audience-mapper",
  "consentRequired": false,
  "config": {
    "included.custom.audience": "mcp-gateway",
    "id.token.claim": "false",
    "access.token.claim": "true"
  }
}
```

Also attached to the `basic` client-scope, for the same reason as the groups
mapper: `basic` is the only scope reliably attached to every DCR'd client.

**What this fixes**: by default, Keycloak does NOT include an `aud` claim on
access tokens minted for DCR'd clients. Without an `aud`, the gateway can't
do strict RFC 8707 audience binding — it has to fall back to issuer-only
validation, which is weaker. With this mapper, every token issued by the
realm (whether for pre-defined clients or DCR'd clients) carries
`aud="mcp-gateway"` (alongside Keycloak's default `aud="account"`), which
the gateway's validator strictly requires (see "Token validation" below).

The `included.custom.audience` value `"mcp-gateway"` is the audience name
the gateway accepts. If you change it, you must also update
[auth_server/providers/keycloak.py](https://github.com/agentic-community/mcp-gateway-registry/blob/main/auth_server/providers/keycloak.py)'s
`accepted_audiences` list to match.

### Gateway-side code paths

#### PRM endpoint

[registry/api/wellknown_routes.py::get_oauth_protected_resource()](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/api/wellknown_routes.py)

Reads `settings.registry_url` for the canonical resource URL, calls
`provider.protected_resource_metadata()` (default impl in
[auth_server/providers/base.py](https://github.com/agentic-community/mcp-gateway-registry/blob/main/auth_server/providers/base.py)) which
returns the RFC 9728 doc. Cache-Control: `public, max-age=300`.

#### AS metadata endpoint

[registry/api/wellknown_routes.py::get_oauth_authorization_server()](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/api/wellknown_routes.py)

Calls `provider.authorization_server_metadata()` which fetches Keycloak's
`/.well-known/openid-configuration` (note: hyphen form, not the legacy
underscore form — Keycloak 25+ only serves the hyphen) and rewrites internal
hostnames (`http://keycloak:8080`) to external (`https://mcpgateway.ddns.net`)
so the discovery client can reach the issuer.

#### WWW-Authenticate on 401

Two emission paths:
- nginx `@auth_error` named-location adds the header on auth_request 401s
  (configured in [docker/nginx_rev_proxy_http_and_https.conf](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docker/nginx_rev_proxy_http_and_https.conf))
- ASGI middleware [registry/middleware/mcp_www_authenticate.py](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/middleware/mcp_www_authenticate.py)
  adds the header on FastAPI 401s

The `resource_metadata` URL must equal the PRM `resource` field byte-for-byte
(RFC 9728 §5.1).

#### Token validation

[auth_server/providers/keycloak.py::validate_token()](https://github.com/agentic-community/mcp-gateway-registry/blob/main/auth_server/providers/keycloak.py)

1. Decode JWT header for `kid`
2. Fetch JWKS from `<realm>/protocol/openid-connect/certs` (cached 1h)
3. Verify signature using the matching public key
4. Verify `iss` matches one of the valid issuer URLs (external + internal +
   localhost variants)
5. Verify `aud` is in `[account, mcp-gateway-web, mcp-gateway-m2m, mcp-gateway]`.
   The first three cover the gateway's pre-defined web/M2M clients and
   Keycloak's default user-token audience. The fourth (`mcp-gateway`) is the
   custom audience emitted by the audience protocol mapper attached to the
   `basic` client-scope (see "Client-scopes" above and
   `setup_dcr_audience_mapper` in `init-keycloak.sh`). DCR'd clients receive
   `basic` automatically, so every DCR'd-client token carries
   `aud="mcp-gateway"` and is validated strictly per RFC 8707 — no fallback.

#### Group → scope mapping

[auth_server/server.py::map_groups_to_scopes()](https://github.com/agentic-community/mcp-gateway-registry/blob/main/auth_server/server.py) +
[auth_server/server.py:2099-2106](https://github.com/agentic-community/mcp-gateway-registry/blob/main/auth_server/server.py#L2099)

Reads `validation_result["groups"]`, queries DocumentDB for each group's
`group_mappings`, returns the union of mapped scopes. The result is what gets
checked against the per-server allowlist for the requested MCP method.

For example, the Keycloak `admin` user has groups
`["mcp-registry-admin", "mcp-servers-unrestricted", "registry-admins"]` →
maps to scopes `["mcp-registry-admin", "registry-admins"]` →
allows access to all MCP servers + registry admin operations.

A regular Keycloak user with only `["public-mcp-users"]` →
maps to scope `["public-mcp-users"]` →
allows access only to MCP servers tagged for public use.

### Operator setup

#### Fresh installs

Run `bash keycloak/setup/init-keycloak.sh`. The script:

1. Creates the `mcp-gateway` realm
2. Creates `mcp-gateway-web` and `mcp-gateway-m2m` pre-defined clients
3. Creates the registry-internal scopes (`mcp-registry-admin`, etc.)
4. Creates Keycloak groups + maps them to scopes
5. Creates the realm admin user (password from `INITIAL_ADMIN_PASSWORD`, set temporary)
6. Sets up groups mappers on the pre-defined clients (web-UI flow)
7. **Sets up DCR-specific config (added by PR #1115)**:
   - Groups mapper on `basic` client-scope (covers DCR'd clients, emits `groups` claim)
   - Audience mapper on `basic` client-scope (emits `aud="mcp-gateway"` for strict RFC 8707 binding)
   - Allowed Client Scopes policy widened
   - Trusted Hosts policy relaxed

#### Existing installs

Run `bash keycloak/setup/upgrade-realm-for-dcr.sh`. The script applies steps 7
above and is idempotent — re-running is safe. Use when the realm was created
before PR #1115 landed.

### What is NOT yet production-safe

Captured here so future hardening passes have a checklist:

| Issue | Risk | Fix path |
| --- | --- | --- |
| Anonymous DCR for MCP clients | A caller already inside the network perimeter can mint unlimited DCR records (no escalation; gated by IdP login + per-user scopes), polluting the realm's clients table | nginx `limit_req_zone` on the registration endpoint as a stopgap, OR migrate to CIMD on Keycloak 26.6+ which eliminates DCR entirely |
| Old DCR'd clients accumulate in the realm | DB bloat, no security impact (public clients with no `client_secret` and no privileged scopes) | Run `bash keycloak/setup/cleanup-stale-dcr-clients.sh` (idempotent; `--dry-run` previews). Use `--dry-run` first to see what would be deleted. CIMD adoption (Keycloak 26.6+, sub-issue #993) eventually obviates the issue entirely. |

### Pointers when something breaks

| Symptom | Where to look first |
| --- | --- |
| Claude Code says "Incompatible auth server: does not support DCR" | Gateway's AS metadata endpoint isn't returning `registration_endpoint`. Confirm Keycloak's underlying OIDC config has it, then check `auth_server/providers/keycloak.py::authorization_server_metadata()` |
| Claude Code says "Got new credentials, but ai-registry-tools rejected them on reconnect" | Token exchange succeeded but gateway-side validation failed. Tail `docker logs mcp-gateway-registry-auth-server-1` and look for `Token validation failed:` or `Access denied:` |
| Auth-server log says `Token validation failed: Token is missing the "aud" claim` | The Keycloak audience mapper isn't attached to the `basic` client-scope. Run `bash keycloak/setup/upgrade-realm-for-dcr.sh` to attach it, then have the user re-authenticate to get a fresh token. Existing tokens issued before the mapper was attached will keep failing until they expire (~5min default). |
| Auth-server log says `Access denied ... for user scopes: ['profile', 'email', 'offline_access']` | Token has no `groups` claim. Run `bash keycloak/setup/upgrade-realm-for-dcr.sh` to re-attach the groups mapper. Have the user re-authenticate to get a fresh token |
| DCR returns 403 with `Policy 'Allowed Client Scopes' rejected` | Allowed-scopes policy wasn't widened. Run the upgrade script, OR check the rejected scope name in Keycloak logs |
| DCR returns 403 with `Policy 'Trusted Hosts' rejected` | Either the IP check is still on, or the client's redirect URI host isn't in `trusted-hosts`. Run the upgrade script. The current allowlist is `localhost`, `127.0.0.1`, `claude.ai` — adjust if you're integrating a client whose callback host isn't one of those (e.g. `claude.com`, `cursor.sh`) |
| Browser shows `temporarily_unavailable: authentication_expired` | User's Keycloak session went stale. Have them log out at `https://<gateway>/realms/mcp-gateway/protocol/openid-connect/logout`, then re-authenticate |
| Browser shows `localhost:<port> refused to connect` | Claude Code's listener and the browser are on different machines; the OAuth callback to `localhost:<port>` doesn't reach the listener. Either run Claude Code on the same machine as the browser, or open an SSH tunnel `-L <port>:localhost:<port>` between them |

---

## Cross-references

- PR #1115 / issue #989 — the gateway-side discovery surface
- [docs/oauth-discovery-endpoints.md](oauth-discovery-endpoints.md) — operator-facing PRM/AS-metadata reference
- [.scratchpad/coding-assistant-oauth/discussion-2026-05-24-claude-connector-q-and-a.md](https://github.com/agentic-community/mcp-gateway-registry/blob/main/.scratchpad/coding-assistant-oauth/discussion-2026-05-24-claude-connector-q-and-a.md) — running discussion log including all live-test findings
- [keycloak/setup/init-keycloak.sh](https://github.com/agentic-community/mcp-gateway-registry/blob/main/keycloak/setup/init-keycloak.sh) — fresh-install setup script
- [keycloak/setup/upgrade-realm-for-dcr.sh](https://github.com/agentic-community/mcp-gateway-registry/blob/main/keycloak/setup/upgrade-realm-for-dcr.sh) — standalone upgrade script for existing installs
- [keycloak/setup/cleanup-stale-dcr-clients.sh](https://github.com/agentic-community/mcp-gateway-registry/blob/main/keycloak/setup/cleanup-stale-dcr-clients.sh) — on-demand cleanup of DCR'd clients with no active sessions; supports `--dry-run`
