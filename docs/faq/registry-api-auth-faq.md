# Registry API Authentication FAQ

Common questions about authenticating against the Registry API (`/api/*`, `/v0.1/*`). For the full authentication model, see [Registry API Authentication](../registry-api-auth.md).

## Can I use an IdP token and the static token on the same deployment?

Yes, as of [#871](https://github.com/agentic-community/mcp-gateway-registry/issues/871). When `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true`, the static token is accepted as an **additional** credential, not an exclusive gate. Valid Okta / Entra / Cognito / Keycloak JWTs, UI-issued self-signed JWTs, and session cookies all continue to work on `/api/*`. A bearer that doesn't match the static token falls through to the JWT validation pipeline.

Before #871, turning on static-token mode silently broke every non-static-token caller on `/api/*` with a 401/403 before JWT validation ran.

## Do I need to seed MongoDB with `mcp-servers-unrestricted/*` scope docs for the static token to work?

No. The registry **hard-codes** full admin access when the auth server sets `X-Auth-Method == "network-trusted"` (see [registry/auth/dependencies.py:620-637](../../registry/auth/dependencies.py)). It does not look up `mcp-servers-unrestricted/read` or `mcp-servers-unrestricted/execute` in Mongo on the static-token path. The scopes returned in the `/validate` response on that path are informational (they appear in audit logs) but are not used for authorization decisions at the registry layer.

## Can I give a static token read-only access?

Not with today's single-token model — anyone holding `REGISTRY_API_TOKEN` is effectively a full registry admin. This is tracked in [issue #779](https://github.com/agentic-community/mcp-gateway-registry/issues/779), which proposes replacing the single token with a map of keys, each with its own set of groups that flow through normal `group_mappings` → scopes resolution. Until #779 ships, use per-user IdP tokens with narrow scopes (or a session cookie) for read-only automation.

## Why does the static token not work on `/<server>/tools/list`?

By design. The static token is only accepted on Registry API paths (`/api/*`, `/v0.1/*`). **MCP gateway tool invocations always require full IdP authentication** regardless of static-token settings. This is a deliberate boundary, not a bug: the static token grants admin-level access on registry metadata endpoints, but tool invocations — which can have real-world side effects — stay gated behind per-user identity and scopes from the IdP.

A curl with `-H "Authorization: Bearer $REGISTRY_API_TOKEN"` against an MCP gateway path will currently return a 500 wrapping the JWT validation failure. That's a pre-existing error-code bug (separate from #871), not a sign the call should have succeeded.

## What status code does a fully invalid bearer get?

Since [#871](https://github.com/agentic-community/mcp-gateway-registry/issues/871): **401** from the JWT block (detail: `"Missing or invalid Authorization header. Expected: Bearer <token> or valid session cookie"`).

Before #871: **403** from the static-token block (detail: `"Invalid API token"`).

No caller with a valid credential is affected by this change. The status-code shift only applies to bearers that were going to be rejected anyway.

## Is my UI-issued JWT usable against `/api/*`?

Yes, since #871. Before the fix, the **Get JWT Token** sidebar in the UI produced valid HS256 JWTs that were nonetheless rejected on `/api/*` when static-token mode was on. After #871 they flow through the same `_validate_self_signed_token` path as any other UI-issued token, regardless of whether static-token mode is on.

## How do I rotate `REGISTRY_API_TOKEN` without downtime?

Not possible today — there is a cutover window during which old clients are rejected while new clients have yet to pick up the new value. Mitigations:

- Roll out the new token value to clients first, then flip the server value.
- Or accept a brief 401/403 window and notify callers.

The clean fix is [#779](https://github.com/agentic-community/mcp-gateway-registry/issues/779) — multiple keys can coexist, so rotation becomes: add new key, move clients over, remove old key.

## Where do I see the current values in the UI?

The **Settings → Authentication** page shows:

| Field | Label | Behavior |
|---|---|---|
| `registry_static_token_auth_enabled` | Static Token Auth Enabled | Displayed as `true` / `false` |
| `registry_api_token` | Registry API Token | Masked |
| `m2m_direct_registration_enabled` | M2M Direct Registration Enabled | Displayed as `true` / `false` (from [#851](https://github.com/agentic-community/mcp-gateway-registry/issues/851)) |

The field registry is defined in [registry/api/config_routes.py](../../registry/api/config_routes.py).

## What's the roadmap?

Three improvements, landing in order on top of each other:

1. **[#871](https://github.com/agentic-community/mcp-gateway-registry/issues/871) — coexistence** (shipped): static token and JWT auth work together on `/api/*`.
2. **[#779](https://github.com/agentic-community/mcp-gateway-registry/issues/779) — multi-key static tokens**: replaces the single `REGISTRY_API_TOKEN` with a `REGISTRY_API_KEYS` map, each key carrying its own groups. Lets operators give scripts the minimum privilege they need.
3. **[#826](https://github.com/agentic-community/mcp-gateway-registry/issues/826) — external user access tokens**: lets a frontend application that has its own IdP integration call the Registry API on behalf of a logged-in user, either via `/userinfo` group enrichment (Solution A) or a new token-exchange endpoint (Solution B).

See the [full design in Registry API Authentication](../registry-api-auth.md#roadmap-near-term-improvements).

## Related FAQs

- [How do I register an M2M client and assign it groups without an IdP Admin API token?](registering-m2m-client-without-idp-admin-token.md)
- [Can I use an Entra ID token to call the registry API instead of the UI-generated token?](use-entra-token-for-registry-api.md)
- [How do I register and manage MCP servers that require authentication?](registering-auth-protected-servers.md)
