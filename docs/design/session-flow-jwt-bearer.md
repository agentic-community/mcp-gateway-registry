# JWT / Bearer Token Login Flow

**Audience:** developers and operators who need to understand how the
platform handles non-browser, programmatic access — CLI tools, AI coding
assistants, AI agents, automated pipelines, and federated registries calling
each other. This is the counterpart to
[session-flow-cookie-based.md](session-flow-cookie-based.md) and covers
everything that does NOT use a browser session cookie.

**Related docs:**
- [session-flow-cookie-based.md](session-flow-cookie-based.md) — the
  browser/cookie counterpart.
- [authentication-design.md](authentication-design.md) — broader auth
  architecture (humans, JWTs, M2M).
- [idp-provider-support.md](idp-provider-support.md) — provider-specific
  configuration.
- [registry-auth-architecture.md](../registry-auth-architecture.md) —
  full validation pipeline for incoming requests.

---

## 1. The shape of a programmatic request

```
                            +--------------------+
                            |                    |
                            |  Client (CLI / AI  |
                            |  agent / federated |
                            |  registry / tool)  |
                            +---------+----------+
                                      |
                                      |   Header: X-Authorization: Bearer <token>
                                      |     (or standard Authorization: Bearer)
                                      v
                            +--------------------+
                            |                    |
                            |   nginx (gateway   |
                            |   reverse proxy)   |
                            +---------+----------+
                                      |
            +----- subrequest --------+
            |
            v
    +-------+----------+              |
    |  /validate on    |              |   2. nginx reads X-* response
    |  auth-server     |              |   headers, copies them onto
    +-------+----------+              |   the upstream request as
            |                         |   proxy_set_header X-User, etc.
            |  X-User, X-Username,    |
            |  X-Client-Id, X-Scopes, |
            |  X-Auth-Method, X-Groups|
            +-------------------------+
                                      |
                                      v
                            +--------------------+
                            |                    |
                            |   Registry / MCP   |
                            |   server upstream  |
                            +--------------------+
```

**Key design property:** the registry never validates the token itself. It
trusts the X-* headers nginx injects, because nginx only injects them after a
successful subrequest to auth-server's `/validate`. The trust boundary is
"between auth-server and registry, traveling through nginx" — not "the
client's HTTP request".

---

## 2. Where tokens come from

The platform validates four distinct token kinds. The auth-server's
`/validate` endpoint dispatches based on inspection of the token, not on
configuration.

### 2.1 IdP-issued JWT (the common case)

A user (or workload) obtained an OAuth2 access token from one of the
configured IdPs:

| IdP | Flow | Where the token came from |
|-----|------|---------------------------|
| Cognito | OAuth2 authorization-code or client-credentials | Cognito user pool. |
| Keycloak | OAuth2 authorization-code or client-credentials | Realm token endpoint. |
| Okta | OAuth2 authorization-code or client-credentials | Okta authorization server. |
| Auth0 | OAuth2 authorization-code or client-credentials | Auth0 tenant. |
| Entra ID | OAuth2 authorization-code or client-credentials | Tenant `/oauth2/v2.0/token`. M2M variant uses `roles` claim instead of `groups`. |

Validation: signature checked against the IdP's JWKS, issuer against the
configured allowed-issuer list, audience against the registered client_id,
expiration against current time. Provider-specific code lives under
[auth_server/providers/](../../auth_server/providers) — each provider's
`validate_token` method returns a normalized dict shape.

### 2.2 Self-signed JWT (programmatic API token)

An interactive user clicks "Generate API Token" in the registry UI; the
auth-server mints a JWT signed with HS256 over `SECRET_KEY`. Issuer
`mcp-auth-server`, audience `mcp-registry`. These are used by CLI tools and
AI coding assistants on behalf of the human user — same scopes as the
issuing user.

Validation: HS256 signature against `SECRET_KEY`, issuer match, audience
match, `token_use=access`, expiration. See
[auth_server/providers/entra.py:226-296](../../auth_server/providers/entra.py#L226-L296)
(present on every provider — the auth-server checks self-signed first,
falls back to provider validation if the issuer does not match).

`SECRET_KEY` is required at process startup (see #1042) — there is no
"development-secret-key" fallback.

### 2.3 Static registry API key

Operator-configured shared secret intended for trusted internal tooling
(monitoring scripts, CI runners). Two configuration knobs:

- `REGISTRY_API_TOKEN` — single-key form, simplest setup.
- `REGISTRY_API_KEYS` — multi-key JSON form, each key has its own name +
  scopes (issue #779). Allows revoking one key without affecting others.

Activated by `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true`. If enabled but no
keys are set, static-token auth is **disabled with a warning** rather than
failing open. See
[auth_server/server.py:129-149](../../auth_server/server.py#L129-L149).

### 2.4 Federation static token

Used when one registry calls another in a federation deployment.
`FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true` + `FEDERATION_STATIC_TOKEN=<32+ char value>`.
Minimum length enforced (`MIN_FEDERATION_TOKEN_LENGTH`). The validating
registry recognizes the token and synthesizes a `federation-peer` identity
with federation-specific scopes. See
[auth_server/server.py:376-396](../../auth_server/server.py#L376-L396) and
[auth_server/server.py:1846-1850](../../auth_server/server.py#L1846-L1850).

---

## 3. The `/validate` endpoint

`GET /validate` on auth-server is the single entry point for token
validation. nginx invokes it as an `auth_request` subrequest before
forwarding the upstream request to the registry.

### 3.1 Headers it reads

| Header | Purpose |
|--------|---------|
| `X-Authorization` (preferred) or `Authorization` | The gateway ingress bearer token. `X-Authorization` is the canonical custom header; `Authorization` is accepted too. Both are **ingress-only** and stripped on the egress hop (issue #1266) — neither is forwarded to an upstream (except the internal `airegistry-tools` relay). |
| `Cookie` | If a session cookie is present and no Authorization header is set, the cookie path is used (rare; mostly browser SSE). |
| `X-User-Pool-Id` | Cognito-specific — required to validate Cognito tokens against the right user pool. |
| `X-Client-Id` | Cognito-specific — used to set the audience constraint. |
| `X-Region` | Cognito-specific — defaults to `us-east-1`. |
| `X-Original-URL` | The URL nginx is about to proxy to. Used for scope-vs-server access checks. |
| `X-Body` | Optional captured request body. Used by tool-level access control (#1026) to inspect JSON-RPC method arguments. |
| `Mcp-Session-Id` | Optional MCP session id — passed through to the audit logger. |
| `X-Request-ID` | Optional client-supplied request id — passed through to logs. |

### 3.2 Decision flow

```
+----------------------+
| /validate request    |
+----------+-----------+
           |
           v
+----------+-----------+
| Network-trusted?     |--- yes -->  Synthesize identity from network context.
| (env: trusted_ips)   |             Set X-* response headers.
+----------+-----------+
           | no
           v
+----------+-----------+
| Federation static    |
| token match?         |--- yes -->  Synthesize "federation-peer" identity.
+----------+-----------+
           | no
           v
+----------+-----------+
| Registry static API  |
| key match?           |--- yes -->  Use key's configured scopes/name.
+----------+-----------+
           | no
           v
+----------+-----------+
| JWT (self-signed     |
| OR IdP-issued)?      |--- yes -->  Provider-specific validation.
+----------+-----------+
           | no
           v
        401 Unauthorized
```

On any successful match, auth-server writes the X-* headers documented in
section 4 below and returns 200. nginx's `auth_request_set` directives copy
those response headers into shell variables, then `proxy_set_header` injects
them onto the upstream request.

### 3.3 Failure semantics

- 401 — token missing, signature invalid, issuer not allowed, audience
  mismatch, expired, or `token_use != access`.
- 403 — token valid, but the requested URL is not in the user's allowed-
  servers / allowed-tools set.
- 500 — only for unexpected internal errors (JWKS fetch failure, etc.).

The registry never sees a 401/403 directly from auth-server — nginx
intercepts the subrequest result and returns the appropriate status to the
client. The registry only ever sees requests where `/validate` returned 200.

---

## 4. The header surface auth-server -> nginx -> registry

These are the canonical X-* headers carried from auth-server's `/validate`
response onto the upstream request:

| Header | Type | Set by | Read by | Meaning |
|--------|------|--------|---------|---------|
| `X-User` | string | auth-server | registry (legacy alias) | Username. Set on every successful validation. |
| `X-Username` | string | auth-server | registry | Username. The canonical name; new code uses this. Both are set so old upstreams keep working. |
| `X-Client-Id` | string | auth-server | registry | OAuth client_id (or `federation-static` / static-key name for non-OAuth tokens). |
| `X-Scopes` | string (space-separated) | auth-server | registry | Pre-computed scopes for this principal. Format: `"scope1 scope2 scope3"`. Empty string is allowed. |
| `X-Auth-Method` | string | auth-server | registry | One of: `cognito`, `keycloak`, `okta`, `auth0`, `entra`, `self_signed`, `network-trusted`, `federation-static`, or the static-key's configured method. Used by audit logging and by the unified user-context derivation. |
| `X-Groups` | string (space-separated) | auth-server | registry | IdP groups (for OAuth tokens) or roles (for Entra M2M). Empty if the principal has no groups. |
| `X-Server-Name` | string | auth-server | registry | The server the request was validated for (extracted from `X-Original-URL`). |
| `X-Tool-Name` | string | auth-server | registry | The tool name when the URL targets a specific tool (used by #1026 tool-level access control). |

The registry consumes these in `nginx_proxied_auth`
([registry/auth/dependencies.py:562-668](../../registry/auth/dependencies.py#L562-L668)).
Both the cookie path and the header path eventually call `_derive_user_context`,
the single source of truth for "is this user an admin and what can they see?"

### 4.1 Fallback when X-Auth-Method is absent

If `X-Auth-Method` is missing on a header-auth request (a misconfigured
nginx, or a future provider whose `/validate` forgot to set it), the
registry **logs a warning and defaults to `keycloak`** — see
[registry/auth/dependencies.py:626-637](../../registry/auth/dependencies.py#L626-L637).
This used to be silent; #1055 added the warning. The default is preserved for
backward compatibility but operators are expected to fix the upstream
config.

### 4.2 Why X-Groups, not synthesized from X-Scopes

Before #1042 the header path would synthesize groups from scope strings if
`X-Groups` was missing. That heuristic produced different admin verdicts
between the cookie path and the header path for the same user (#933). It's
gone. Today, if `X-Groups` is missing, groups are simply empty and admin
status is derived purely from the auth-server-computed scopes in `X-Scopes`.

---

## 5. The `_derive_user_context` contract

Both auth paths normalize their inputs to the same tuple
`(username, groups, scopes, auth_method, provider, client_id?)` and call
`_derive_user_context`
([registry/auth/dependencies.py:453-518](../../registry/auth/dependencies.py#L453-L518)).
The output is a dict used everywhere downstream:

```python
{
    "username": "alice@example.com",
    "client_id": "abc-1234-...",
    "groups": ["devs", "admins"],
    "scopes": ["mcp-servers-unrestricted/read", ...],
    "auth_method": "entra",       # or self_signed, federation-static, ...
    "provider": "entra",
    "session_id": "...or None",
    "accessible_servers": ["server-a", "server-b"],
    "accessible_tools": {...},
    "accessible_services": [...],
    "accessible_agents": [...],
    "ui_permissions": {...},
    "can_modify_servers": True,
    "is_admin": True,             # mutating UI scope with "all" -> admin
}
```

Same input -> same output, regardless of how the user authenticated. This is
the property the cookie + header symmetry guarantees.

Special case: `auth_method == "federation-static"` short-circuits to a
no-access context. Federation static tokens use a separate routing path
(peer-to-peer federation API), not registry-side scope derivation.

---

## 6. Provider-specific notes

### 6.1 Entra ID

- M2M tokens use the `roles` claim instead of `groups`. The provider
  detects this and substitutes
  ([auth_server/providers/entra.py:199-203](../../auth_server/providers/entra.py#L199-L203)).
- Two valid issuers per token: `https://login.microsoftonline.com/<tenant>/v2.0`
  (v2.0 endpoint) and `https://sts.windows.net/<tenant>/` (v1.0/M2M).
  The validator accepts either.
- Audience is checked against both `<client_id>` and `api://<client_id>`
  to handle Entra's two audience formats.
- Group-overage handling (browser flow only — irrelevant here) is covered in
  the cookie-flow doc.

### 6.2 Cognito

- Requires `X-User-Pool-Id` and `X-Client-Id` headers from nginx.
- Validates against the user pool's JWKS endpoint.
- `cognito:groups` claim populates `X-Groups`.

### 6.3 Keycloak / Okta / Auth0

- Standard OAuth2/OIDC validation against the provider's JWKS.
- Audience and issuer constraints from the per-provider config.
- Auth0 requires a Rule/Action in the tenant to inject the groups claim
  (default claim name `https://mcp-gateway/groups`, configurable).

### 6.4 Self-signed (programmatic API tokens)

- HS256 over `SECRET_KEY`. No JWKS fetch, no network call.
- Validated **before** any IdP — auth-server inspects the token's `iss`
  claim; if it equals `mcp-auth-server`, self-signed validation runs and
  IdP validation is skipped.

### 6.5 Static keys

- Constant-time string comparison against the configured value(s).
- Identity synthesized from the key's name and configured scopes.

---

## 7. Audit logging

Every `/validate` call emits an audit-log event when the request targets an
MCP server:

```json
{
  "event": "mcp_access",
  "request_id": "...",
  "mcp_session_id": "...",
  "username": "alice@example.com",
  "auth_method": "entra",
  "client_id": "...",
  "server_name": "server-a",
  "tool_name": "search_logs",
  "duration_ms": 12,
  "outcome": "allowed" | "denied"
}
```

Stored in `audit_logs_<namespace>` with TTL. Operator export procedures will
land under [docs/operations/](../operations/) per
[issue #1056](https://github.com/agentic-community/mcp-gateway-registry/issues/1056).

---

## 8. Differences from the cookie flow

| Aspect | Cookie flow | JWT/Bearer flow |
|--------|-------------|-----------------|
| Where state lives | Server-side in `oauth_sessions_<ns>` collection. | Stateless — every request carries the full token. |
| What the client transmits | One short signed cookie. | The full token (typically 800-2000 bytes for OAuth tokens, less for self-signed). |
| Validation cost per request | One indexed `find_one()` against MongoDB. | One JWKS-cached signature check (no DB read). |
| Logout | Delete the server-side record + clear the cookie. | No logout — token expires on its own. To revoke before expiry, use the IdP's revocation endpoint or rotate the static-key value. |
| CSRF concerns | Yes (mitigated by `SameSite=Lax` + CSRF token bound to `session_id`). | No (no browser session = no CSRF surface). |
| Affected by `SECRET_KEY` rotation | Yes — invalidates all sessions. | Self-signed JWTs invalidated. IdP-issued JWTs unaffected (signed with IdP keys). |
| Multi-replica coordination | Shared `SECRET_KEY` across replicas; session record visible to all. | Stateless — every replica independently validates. |
| Threat model for token leak | Server-side delete closes the replay window. | Token replayable until `exp`. Mitigation: short TTLs + IdP revocation. |

---

## 9. Common questions

**Q: Why two header names — `X-User` and `X-Username` — for the same value?**
A: Legacy. `X-User` is the original; `X-Username` is the canonical name new
code reads. Both are set for backward compatibility with older upstreams.
The registry's `nginx_proxied_auth` reads `X-Username or X-User`.

**Q: Why does `/validate` accept both `X-Authorization` and `Authorization`?**
A: Both are accepted as the gateway ingress token (`/validate` checks
`X-Authorization` first, then `Authorization`). `X-Authorization` is the
canonical custom header; `Authorization` is accepted for clients that cannot
set custom headers.

> **Ingress vs egress (issue #1266).** All client-sent auth headers
> (`Authorization`, `X-Authorization`, `Cookie`) are **ingress** credentials:
> they authenticate the caller to the gateway and are **stripped on the egress
> hop** — they are never forwarded to an upstream MCP server. An upstream that
> needs a credential gets it from the egress vault (per-user OAuth/3LO, PAT),
> not by relaying a client header. The one exception is the gateway's built-in,
> same-trust-domain registry-tools server (`airegistry-tools`), a hardcoded
> internal constant that receives the relayed `Authorization` (never
> `X-Authorization`/`Cookie`). So the historical "downstream server reuses
> `Authorization`" pattern is no longer served by client-header relay for
> external servers — use the vault.

**Q: Can a client present both a cookie and a Bearer token?**
A: Yes. The header path takes precedence — `nginx_proxied_auth` checks
headers first, falls back to cookies only if no `X-User`/`X-Username` is
present. See
[registry/auth/dependencies.py:611-668](../../registry/auth/dependencies.py#L611-L668).

**Q: Does the JWT flow ever touch the `oauth_sessions_<ns>` collection?**
A: No. Tokens are stateless; that collection is only for browser sessions.

**Q: What if auth-server is temporarily down?**
A: nginx's `auth_request` returns 502/504 to the client. The registry never
sees the request. This is by design — it's a fail-closed posture.

**Q: How are scopes resolved on every request — does that hit MongoDB?**
A: Yes. `_derive_user_context` calls `resolve_scope_access(scopes)`, which
walks the scopes-config repo. The result is cached per-request via the
shared scope_repo factory. Hot paths could be cached longer if needed
(observable via `registry_session_store_resolve_total` for the cookie path
and per-server access metrics for the header path).
