# Registry API Authentication

This page is the single source of truth for how callers authenticate against the **Registry API** (`/api/*`, `/v0.1/*`) — the HTTP surface used by the UI, the `registry_management.py` CLI, and any script or service that talks to the registry.

**Scope clarification.** This document covers the **Registry API** only. The **MCP Gateway** surface (`/<server>/tools/list`, `/<server>/messages`, etc.) always requires full IdP authentication and is governed by `scopes.yml` / `mcp_scope_default`. MCP gateway authn/authz is described in [auth.md](auth.md) and [scopes.md](scopes.md).

## Table of contents

1. [The big picture](#the-big-picture)
2. [Accepted credentials today](#accepted-credentials-today)
3. [Static API token (`REGISTRY_API_TOKEN`)](#static-api-token-registry_api_token)
4. [Multi-key static tokens (`REGISTRY_API_KEYS`)](#multi-key-static-tokens-registry_api_keys)
5. [Session cookie (browser UI)](#session-cookie-browser-ui)
6. [IdP-issued JWT (Okta / Entra / Cognito / Keycloak)](#idp-issued-jwt)
7. [UI-issued self-signed JWT](#ui-issued-self-signed-jwt)
8. [Coexistence rules (who wins when)](#coexistence-rules)
9. [Threat model for static tokens](#threat-model-for-static-tokens)
10. [Roadmap: near-term improvements](#roadmap-near-term-improvements)
    - [#826 — external user access tokens (service-on-behalf-of-user)](#826--external-user-access-tokens)
11. [Common operator tasks](#common-operator-tasks)
12. [FAQ](#faq)
13. [References](#references)

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
| 3 | Registry static token(s) (`REGISTRY_API_TOKEN` and/or `REGISTRY_API_KEYS`) | `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true` | `network-trusted` | Single legacy key or multiple per-key scoped keys. See sections below. |
| 4 | IdP-issued JWT (Okta RS256, Entra, Cognito, Keycloak) | Always | `oauth2` (or IdP-specific) | Full per-user identity with groups from the ID token at login time. |
| 5 | UI-issued self-signed JWT (HS256) | Always | `self-signed` | Tokens minted by the **Get JWT Token** sidebar button or `POST /api/tokens/generate`. |
| — | No credential | — | — | 401 returned. |

**Before [issue #871](https://github.com/agentic-community/mcp-gateway-registry/issues/871)**, turning on the registry static token made it the **only** accepted Bearer credential on `/api/*`. IdP and self-signed JWTs were rejected with 401/403 before reaching their validation blocks. After #871, a mismatched or missing bearer on the static-token path **falls through** to the JWT validators instead of terminating. This is what lets mixed-mode deployments (machine callers + per-user callers) share the same registry.

## Static API token (`REGISTRY_API_TOKEN`)

A single shared secret (the "legacy" key), validated with `hmac.compare_digest` and mapped to a full-admin identity. This is the simplest setup and is backwards-compatible with all previous releases.

### Configuration

| Variable | Type | Default | Notes |
|---|---|---|---|
| `REGISTRY_STATIC_TOKEN_AUTH_ENABLED` | bool | `false` | When `true`, static tokens are accepted on Registry API paths. |
| `REGISTRY_API_TOKEN` | str | empty | The shared secret. At least one of `REGISTRY_API_TOKEN` or `REGISTRY_API_KEYS` must be set for the flag to take effect. |

If `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true` but neither `REGISTRY_API_TOKEN` nor `REGISTRY_API_KEYS` is set, the auth server logs an error and disables the feature at startup.

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

### Identity granted by the legacy static token

When `REGISTRY_API_TOKEN` matches, the auth server returns the legacy admin identity:

```json
{
  "valid": true,
  "username": "network-user",
  "client_id": "network-trusted",
  "method": "network-trusted",
  "groups": ["mcp-registry-admin"],
  "scopes": ["mcp-registry-admin", "mcp-servers-unrestricted/read", "mcp-servers-unrestricted/execute"]
}
```

The `mcp-registry-admin` scope (a UI scope name) ensures the registry resolves this caller as a full admin through the standard permissions path. Anyone holding `REGISTRY_API_TOKEN` is effectively a registry admin. Protect the secret accordingly.

### Where static tokens do NOT work

- **MCP gateway paths** (`/<server>/tools/list` etc.) always require IdP auth. Static tokens are ignored there.
- **Paths outside `/api/*` and `/v0.1/*`** (e.g. health endpoints, audit endpoints behind other prefixes) follow their own rules.

## Multi-key static tokens (`REGISTRY_API_KEYS`)

*Added in [issue #779](https://github.com/agentic-community/mcp-gateway-registry/issues/779).*

Multiple static API keys, each with its own name and groups. Each key's groups flow through the standard `group_mappings` to scopes resolution, so a read-only key gets read-only permissions and an admin key gets admin permissions.

### Configuration

| Variable | Type | Default | Notes |
|---|---|---|---|
| `REGISTRY_API_KEYS` | JSON string | empty | Map of named keys. Format below. |

`REGISTRY_API_KEYS` is only consulted when `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true`. If both `REGISTRY_API_TOKEN` and `REGISTRY_API_KEYS` are set, they are merged: the legacy token becomes an implicit entry named `legacy` with `groups=["mcp-registry-admin"]`.

### Format

```env
REGISTRY_API_KEYS='{"monitoring":{"key":"<token-1>","groups":["mcp-readonly"]},"deploy":{"key":"<token-2>","groups":["mcp-registry-admin"]}}'
```

Rules:
- **name**: must match `^[a-z0-9][a-z0-9_-]{0,63}$` (log-safe identifier)
- **key**: minimum 32 characters (use `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`)
- **groups**: non-empty list of group names from your `scopes.yml` / `mcp_scope_default` group_mappings
- Reserved names: `legacy`, `network-user`, `network-trusted` cannot be used
- Key values must be unique across all entries
- On any parse or validation error, the feature is disabled entirely (fail-closed)

### How scopes are resolved

At startup, the auth server calls `map_groups_to_scopes(entry.groups)` for each entry to resolve groups into scopes using the same pipeline as IdP/JWT auth. The resolved scopes are cached in memory. When an operator imports or modifies group_mappings (e.g., via `registry_management.py import-group`), the registry triggers an auth server scope reload that also rebuilds the static token map, so changes propagate without a restart.

### Identity for multi-key matches

When a named key matches, the auth server returns:

```json
{
  "valid": true,
  "username": "monitoring",
  "client_id": "monitoring",
  "method": "network-trusted",
  "groups": ["mcp-readonly"],
  "scopes": ["mcp-readonly/read"]
}
```

The key **name** becomes the `username` and `client_id`, which appear in audit logs. This is how operators can answer "which consumer made this call."

### Registry-side authorization

The registry no longer hard-codes admin access for `network-trusted` callers. Instead, it resolves permissions from the scopes returned by the auth server, just like any other auth method. A key with `groups=["mcp-readonly"]` will NOT be able to delete servers, register agents, or perform other admin actions.

### Example: read-only monitoring key

1. Ensure your `scopes.yml` has a group like `mcp-readonly` mapped to read-only scopes.
2. Generate a key: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
3. Add to your config:

```bash
REGISTRY_API_KEYS='{"monitoring":{"key":"YOUR_GENERATED_KEY","groups":["mcp-readonly"]}}'
```

4. Use it:

```bash
curl -sS -H "Authorization: Bearer YOUR_GENERATED_KEY" "$REGISTRY_URL/api/servers"
```

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
3. Else if the path is a Registry API path AND static-token mode is on AND the bearer matches any entry in `_STATIC_TOKEN_MAP` (legacy `REGISTRY_API_TOKEN` or any `REGISTRY_API_KEYS` entry) → `network-trusted`.
4. Else fall through to IdP JWT / self-signed JWT validation.
5. Else 401.

**Behavior change since #871**: a bearer that matches neither the static token nor any valid JWT now returns **401** from the JWT block, where it previously returned **403 "Invalid API token"** from the static-token block. No legitimate caller is broken by this — only one that was already sending an invalid credential.

## Threat model for static tokens

`REGISTRY_API_KEYS` is a sensitive secret. An attacker who obtains the raw JSON value gains access equivalent to the most privileged key in the map. Specifically:

- Any entry whose groups include `mcp-registry-admin` (or any group that maps to admin UI scopes) is equivalent to full admin compromise.
- Read-only keys limit the blast radius to data exfiltration (listing servers, reading configs) but cannot mutate.
- Key names appear in audit logs, so a compromised key is identifiable after the fact.

Mitigations:
- Store `REGISTRY_API_KEYS` in a secrets manager (AWS Secrets Manager, Vault, etc.), never in plaintext config files.
- Terraform variables use `sensitive = true`; Helm renders the value into a Kubernetes Secret.
- Rotate keys by adding a new key, migrating clients, then removing the old key. Restart the auth server after each config change.
- Consider using the `existingSecret` Helm pattern to pull from an External Secrets Operator rather than templating the value.

## Roadmap: near-term improvements

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

**Status.** Design pending. Solution A is the recommended first cut.

## Common operator tasks

### Enable static-token mode

```bash
# .env
REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true
REGISTRY_API_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# then:
docker compose restart auth-server registry
```

### Rotate a static token

**Legacy single-key (`REGISTRY_API_TOKEN`):**

1. Generate a new token with the `secrets.token_urlsafe` command above.
2. Update `REGISTRY_API_TOKEN` in your deployment config.
3. Restart the auth server.
4. Update all clients that use the token (CI/CD pipelines, scripts).

**Multi-key (`REGISTRY_API_KEYS`) zero-downtime rotation:**

1. Add a new entry (e.g. `deploy-v2`) with a fresh key to the JSON map.
2. Restart the auth server. Both old and new keys now work.
3. Migrate clients to the new key.
4. Remove the old entry from the JSON map.
5. Restart the auth server again.

This overlap-rotation pattern avoids any window where clients see 401.

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
