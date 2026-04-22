# Registry API Authentication

This page is the single source of truth for how callers authenticate against the **Registry API** (`/api/*`, `/v0.1/*`) — the HTTP surface used by the UI, the `registry_management.py` CLI, and any script or service that talks to the registry.

**Scope clarification.** This document covers the **Registry API** only. The **MCP Gateway** surface (`/<server>/tools/list`, `/<server>/messages`, etc.) always requires full IdP authentication and is governed by `scopes.yml` / `mcp_scope_default`. MCP gateway authn/authz is described in [auth.md](auth.md) and [scopes.md](scopes.md).

## Table of contents

1. [The big picture](#the-big-picture)
2. [Accepted credentials today](#accepted-credentials-today)
3. [Static API token (`REGISTRY_API_TOKEN`)](#static-api-token-registry_api_token)
4. [Session cookie (browser UI)](#session-cookie-browser-ui)
5. [IdP-issued JWT (Okta / Entra / Cognito / Keycloak)](#idp-issued-jwt)
6. [UI-issued self-signed JWT](#ui-issued-self-signed-jwt)
7. [Coexistence rules (who wins when)](#coexistence-rules)
8. [Roadmap: near-term improvements](#roadmap-near-term-improvements)
   - [#779 — multiple static API keys with per-key groups](#779--multi-key-static-tokens)
   - [#826 — external user access tokens (service-on-behalf-of-user)](#826--external-user-access-tokens)
9. [Common operator tasks](#common-operator-tasks)
10. [FAQ](#faq)
11. [References](#references)

## The big picture

Every call to a Registry API endpoint passes through the **auth server's `/validate` endpoint** before reaching the registry application. The auth server decides, for each incoming request, whether the caller is authenticated and what identity to stamp on the request.

```
Client                 nginx                 auth_server:/validate              registry
  │                      │                          │                              │
  │── GET /api/... ─────▶│                          │                              │
  │  (cookie or Bearer)  │                          │                              │
  │                      │── auth_request ─────────▶│                              │
  │                      │                          │── 200 + X-Auth-Method,       │
  │                      │                          │           X-Scopes, ...      │
  │                      │                          │   OR 401/403                 │
  │                      │◀─────────────────────────│                              │
  │                      │                          │                              │
  │                      │── proxy_pass ────────────────────────────────────────▶ │
  │                      │   (with X-Auth-Method and other identity headers)      │
  │                      │                                                         │
  │◀─────────────────────│◀────────────────── response ───────────────────────────│
```

The registry reads `X-Auth-Method` and related headers to decide what the caller can do. It does **not** re-validate the credential — the auth server has the only say on identity.

## Accepted credentials today

On a Registry API path the auth server checks credentials in this order (as of [issue #871](https://github.com/agentic-community/mcp-gateway-registry/issues/871)):

| # | Credential | Enabled by | `X-Auth-Method` | Notes |
|---|---|---|---|---|
| 1 | Session cookie (`mcp_gateway_session=...`) | Always | `oauth2` / IdP-specific | UI browser flow. Short-circuits everything else. |
| 2 | Federation static token | `FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true` and the request path is `/api/federation/*` or `/api/peers/*` | `federation-static` | Peer-to-peer federation only. Narrow scope. |
| 3 | Registry static token (`REGISTRY_API_TOKEN`) | `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true` | `network-trusted` | See the section below. |
| 4 | IdP-issued JWT (Okta RS256, Entra, Cognito, Keycloak) | Always | `oauth2` (or IdP-specific) | Full per-user identity with groups from the ID token at login time. |
| 5 | UI-issued self-signed JWT (HS256) | Always | `self-signed` | Tokens minted by the **Get JWT Token** sidebar button or `POST /api/tokens/generate`. |
| — | No credential | — | — | 401 returned. |

**Before [issue #871](https://github.com/agentic-community/mcp-gateway-registry/issues/871)**, turning on the registry static token made it the **only** accepted Bearer credential on `/api/*`. IdP and self-signed JWTs were rejected with 401/403 before reaching their validation blocks. After #871, a mismatched or missing bearer on the static-token path **falls through** to the JWT validators instead of terminating. This is what lets mixed-mode deployments (machine callers + per-user callers) share the same registry.

## Static API token (`REGISTRY_API_TOKEN`)

A single shared secret, validated with `hmac.compare_digest` and mapped to a hard-coded "network-trusted" identity.

### Configuration

| Variable | Type | Default | Notes |
|---|---|---|---|
| `REGISTRY_STATIC_TOKEN_AUTH_ENABLED` | bool | `false` | When `true`, the static token is accepted on Registry API paths. |
| `REGISTRY_API_TOKEN` | str | empty | The shared secret. Must be non-empty for the flag to take effect. |

If `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true` but `REGISTRY_API_TOKEN` is empty, the auth server logs an error and **silently disables** the feature at startup.

### Generate a token

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Treat the result like a password: rotate periodically, never commit to git, store in a secrets manager for production.

### Deployment

**Docker Compose** — add to your `.env`:

```bash
REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true
REGISTRY_API_TOKEN=your-generated-token
```

**AWS ECS (terraform)** — add to `terraform.tfvars`:

```hcl
registry_static_token_auth_enabled = true
registry_api_token                 = "your-generated-token"
```

Or pass via environment variable to avoid committing the value to a file:

```bash
export TF_VAR_registry_api_token="your-generated-token"
```

**Helm** — set `registry.app.registryStaticTokenAuthEnabled=true` and `registry.app.registryApiToken=<value>` in the umbrella chart values.

### Usage

```bash
curl -sS -H "Authorization: Bearer $REGISTRY_API_TOKEN" \
  "$REGISTRY_URL/api/servers"
```

Via CLI:

```bash
echo -n "$REGISTRY_API_TOKEN" > /tmp/static-token
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file /tmp/static-token \
  list
```

### Identity granted by the static token

When the static token matches, the auth server returns:

```json
{
  "valid": true,
  "username": "network-user",
  "client_id": "network-trusted",
  "method": "network-trusted",
  "groups": ["mcp-registry-admin"],
  "scopes": [
    "mcp-servers-unrestricted/read",
    "mcp-servers-unrestricted/execute"
  ]
}
```

Downstream the registry **hard-codes full admin access** whenever `X-Auth-Method == "network-trusted"` ([registry/auth/dependencies.py:620-637](../registry/auth/dependencies.py)) — it does **not** look up the `mcp-servers-unrestricted/*` scopes in MongoDB. So:

- You do **not** need to seed scope documents for `mcp-servers-unrestricted/*` for this to work.
- Anyone holding `REGISTRY_API_TOKEN` is effectively a registry admin. Protect the secret accordingly.

### Where the static token does NOT work

- **MCP gateway paths** (`/<server>/tools/list` etc.) always require IdP auth. The static token is ignored there.
- **Paths outside `/api/*` and `/v0.1/*`** (e.g. health endpoints, audit endpoints behind other prefixes) follow their own rules.

## Session cookie (browser UI)

When a browser user logs in through the UI, the response sets a `mcp_gateway_session=...` cookie. On subsequent calls to `/api/*`, the auth server detects the cookie and short-circuits to session validation — **no static-token check runs**. This is the browser's primary auth path and is unaffected by any of the issues on this page.

## IdP-issued JWT

Tokens issued by your configured IdP (`AUTH_PROVIDER=okta|entra|cognito|keycloak|...`) are validated by the provider-specific `validate_token` implementation. Groups are extracted from the token's `groups` claim (or equivalent). These tokens work on `/api/*` **regardless** of whether static-token mode is on, as of #871.

## UI-issued self-signed JWT

The auth server's sidebar **Get JWT Token** button produces an HS256 JWT signed with the registry's own secret. These tokens carry the user's groups baked in at mint time and are validated by `_validate_self_signed_token`. They work on `/api/*` just like IdP JWTs.

## Coexistence rules

Starting with [#871](https://github.com/agentic-community/mcp-gateway-registry/issues/871), the registry-static-token block is **non-terminal**:

1. If the request has a valid session cookie → session auth wins.
2. Else if the path is a federation path and the federation static token matches → `federation-static`.
3. Else if the path is a Registry API path AND static-token mode is on AND the bearer matches `REGISTRY_API_TOKEN` → `network-trusted`.
4. Else fall through to IdP JWT / self-signed JWT validation.
5. Else 401.

**Behavior change since #871**: a bearer that matches neither the static token nor any valid JWT now returns **401** from the JWT block, where it previously returned **403 "Invalid API token"** from the static-token block. No legitimate caller is broken by this — only one that was already sending an invalid credential.

## Roadmap: near-term improvements

The current registry API auth model has two known gaps. Both are tracked and sequenced on top of #871.

### #779 — multi-key static tokens

Tracked at [issue #779](https://github.com/agentic-community/mcp-gateway-registry/issues/779).

**Problem.** Today there is exactly one `REGISTRY_API_TOKEN`, and it grants hard-coded full admin access. Every script that needs any Registry API access gets the same superuser privileges — a read-only monitoring script is indistinguishable from a deployment pipeline that writes server configs.

**Proposed solution.** Replace the single token with a map of keys, each with its own set of groups that flow through the normal `group_mappings` → scopes resolution:

```env
REGISTRY_API_KEYS='{
  "monitoring-script":  { "key": "<token-1>", "groups": ["mcp-readonly"] },
  "deploy-pipeline":    { "key": "<token-2>", "groups": ["mcp-registry-admin"] },
  "koda-integration":   { "key": "<token-3>", "groups": ["koda_users"] }
}'
```

**How it composes with #871.** #871 extracts a `_check_registry_static_token(bearer_token) -> dict | None` helper. #779 swaps that helper's body from a single `hmac.compare_digest` to a map iteration. The `/validate` control flow (including the fall-through to JWT) does not change.

**Benefits.**
- Per-key identity in audit logs (which script made this call?).
- Principle of least privilege for automation (read-only scripts don't write).
- Legacy `REGISTRY_API_TOKEN` is kept as a shorthand for single-entry deployments (back-compat).

**Status.** Design pending. Will land on a branch after #871 ships.

### #826 — external user access tokens

Tracked at [issue #826](https://github.com/agentic-community/mcp-gateway-registry/issues/826).

**Problem.** An external application ("Frontend App") that has its own IdP integration and wants to call the registry API **on behalf of a user** cannot do so today:

- The token was issued for the external app, not the registry, so the `aud`/`cid` claim won't match the registry's own client ID.
- Okta's org authorization server puts groups in the **ID token**, not the **access token**, so the access token arrives with empty groups.
- There's no groups-resolution path for external user tokens today (the M2M enrichment via `idp_m2m_clients` is for client-credentials M2M, not user access tokens).

Result: external user tokens get zero scopes and are effectively denied.

**Proposed solutions (two options).**

**Option A — userinfo group enrichment.** After validating the external user's access token's signature against JWKS, call the IdP's `/userinfo` endpoint with that token to retrieve groups. Cache with a short TTL. Requires a new config of **trusted client IDs** (whose tokens are accepted despite audience mismatch).

- Pros: minimal change on the external app side; groups stay fresh; OIDC-standard approach.
- Cons: runtime dependency on IdP `/userinfo` for every unique token; subject to IdP rate limits on cache miss.

**Option B — token exchange endpoint.** The external app exchanges its ID+access tokens for a **registry-minted self-signed JWT** via a new `POST /oauth2/token-exchange` endpoint. Subsequent API calls use the self-signed token, validated locally with no IdP roundtrip.

- Pros: no runtime IdP dependency; proper `aud: "mcp-registry"` on the minted token; delegation visible via `source_client_id` claim.
- Cons: external app must implement the exchange + token caching; new endpoint is additional attack surface.

**How it composes with #871.** Both options rely on the fall-through behavior #871 introduces — without it, external tokens would be rejected by the static-token block before ever reaching JWT validation (Option A) or `_validate_self_signed_token` (Option B). #871 does not ship either solution; it just makes them possible.

**Status.** Design pending. Expected to land after #779. Solution A is the recommended first cut.

## Common operator tasks

### Enable static-token mode

```bash
# .env
REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true
REGISTRY_API_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# then:
docker compose restart auth-server registry
```

### Rotate the static token

1. Generate a new token with the `secrets.token_urlsafe` command above.
2. Update `REGISTRY_API_TOKEN` in your deployment config.
3. Restart the auth server.
4. Update all clients that use the token (CI/CD pipelines, scripts).

During the rotation window clients will see 401/403 until they switch over. There is no atomic rotation today; multi-key support (#779) will make it possible to run old and new keys in parallel during the cutover.

### Disable static-token mode

Set `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=false`. Session cookies and IdP JWTs keep working unchanged. Any client relying on the static token will start getting 401.

### Verify the System Config UI

The current values appear on the **Settings → Authentication** page in the web UI. `REGISTRY_API_TOKEN` is masked. The field registry is defined in [registry/api/config_routes.py:75-76](../registry/api/config_routes.py).

## FAQ

See the dedicated FAQ page: [Registry API Authentication FAQ](faq/registry-api-auth-faq.md).

## References

- Issue #871: [feat: allow JWT/session auth to coexist with static token auth](https://github.com/agentic-community/mcp-gateway-registry/issues/871)
- Issue #779: [feat: Support multiple static API keys with per-key group/scope assignments](https://github.com/agentic-community/mcp-gateway-registry/issues/779)
- Issue #826: [feat: Support External User Access Tokens (Service-to-Service on Behalf of Users)](https://github.com/agentic-community/mcp-gateway-registry/issues/826)
- Auth server entry point: [`auth_server/server.py`](../auth_server/server.py) — `/validate` endpoint
- Registry auth handoff: [`registry/auth/dependencies.py`](../registry/auth/dependencies.py) — consumes `X-Auth-Method` header
- Scope configuration format: [`scopes.md`](scopes.md)
- General authentication overview: [`auth.md`](auth.md)
