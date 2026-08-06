# How do I generate an MCP access token that lasts longer than 8 hours?

The token minted by the **Generate Token** page (and `POST /api/tokens/generate`) is a self-signed gateway JWT that works as an MCP access token. Its **8 hours is only the default**, not the ceiling: the shipped maximum is **24 hours**, and both the default and the maximum are operator-configurable (from [#1477](https://github.com/agentic-community/mcp-gateway-registry/issues/1477)).

## Short answer

**For a user going through the UI, changing the config parameters is the only way to get a longer lifetime.** The UI exposes only fixed presets (1h / 8h / 24h) and cannot request an arbitrary value, and the token API is session-authenticated (not a simple Bearer `curl`), so there is no self-service "just ask for N hours" path. An operator sets the lifetime with two `.env` parameters (restart the registry and auth-server after changing them):

- `MCP_TOKEN_DEFAULT_TTL_HOURS` — the lifetime every UI entry point mints (default `8`).
- `MCP_TOKEN_MAX_TTL_HOURS` — the ceiling on any request (default `24`, hard-capped at 168h / 7 days).

To go **beyond 24h**, raising `MCP_TOKEN_MAX_TTL_HOURS` is the **only** way. The rest of this FAQ explains the details.

So there are two different things you might want, and they need different actions.

## I want a token with a lifetime other than the presets (e.g. 12 hours)

**From the UI, you can only pick the built-in presets.** The **Generate Token** page's **Expires In** dropdown offers exactly **1 hour / 8 hours / 24 hours** — there is no 12-hour option and no free-form hours field. The sidebar **Get JWT Token** button and a server's **MCP Configuration** dialog don't let you choose at all; they always mint at the **configured default** lifetime. So for anything other than 1h / 8h / 24h, the UI alone will not do it.

To get an arbitrary lifetime (say 12h) you have two options:

1. **Change the default (simplest, recommended).** Set `MCP_TOKEN_DEFAULT_TTL_HOURS=12` (see the next section) and restart. Every token the sidebar button, the MCP Configuration dialog, and the "omit expires_in_hours" path mint is then 12h.
2. **Call the API with a custom `expires_in_hours`.** The endpoint accepts any integer from `1` to the configured maximum (24 by default) in the JSON body — but see the auth note below, it is not a simple Bearer `curl`.

> **`POST /api/tokens/generate` is a browser/session endpoint, not a Bearer-token API.** It is authenticated by your logged-in **session cookie** (`enhanced_auth`), so a plain `curl` with only an `Authorization: Bearer <token>` header is rejected with `401 Authentication required` — you cannot mint a new token from an existing bearer token. Drive it from the UI (which carries the session cookie and, for browsers, the CSRF token), or from a script that first performs the OAuth login and reuses the resulting session cookie. A request whose `expires_in_hours` is not an integer in `1`..max is rejected with `400`.

## I want to change the default, or allow tokens longer than 24 hours

These are policy settings, applied at the **auth-server + registry** (both consume them). Two parameters:

| Parameter (`.env`) | Terraform | Helm | Default | Controls |
|--------------------|-----------|------|---------|----------|
| `MCP_TOKEN_DEFAULT_TTL_HOURS` | `mcp_token_default_ttl_hours` | `mcpTokenDefaultTtlHours` | `8` | Lifetime used when a caller omits `expires_in_hours`. |
| `MCP_TOKEN_MAX_TTL_HOURS` | `mcp_token_max_ttl_hours` | `mcpTokenMaxTtlHours` | `24` | Hard cap; a larger requested value is clamped/rejected. |

- To make omitted-lifetime tokens default to 12h: set `MCP_TOKEN_DEFAULT_TTL_HOURS=12`.
- To allow requests beyond 24h (for example 72h): set `MCP_TOKEN_MAX_TTL_HOURS=72`.

For docker-compose, add the variable(s) to your `.env` and restart the auth-server and registry. For Terraform/Helm, set the corresponding variable/value. See the full cross-surface mapping in [`docs/unified-parameter-reference.md`](../unified-parameter-reference.md).

## Why can't I set an unlimited lifetime?

`MCP_TOKEN_MAX_TTL_HOURS` is itself bounded by a **hardcoded absolute ceiling of 168 hours (7 days)** — configuring a higher value is clamped down (with a warning), and any value below 1 is floored to 1.

This is deliberate. These tokens are **self-signed bearer tokens with no revocation path**: there is no introspection or denylist, so a leaked long-lived token stays valid for its full lifetime, and the only kill switch is rotating `SECRET_KEY`, which invalidates *every* token at once (and every backend credential encrypted with it). A very long TTL turns any single leaked token into a long-lived liability.

If you need longer-lived credentials **with proper revocation**, use an **IdP-issued token** (Keycloak, Cognito, Entra, Okta, Auth0) instead. Its lifespan is set at the IdP, it supports real revocation, and the gateway validates it directly. See the identity-provider sections of the [Configuration Reference](../configuration.md).

## Why don't I see the Generate Token page?

It is in the left sidebar as **Generate Token**, gated by the `token-generation` permission. If it is missing for your user, your group lacks that scope rather than the page being absent — grant `token-generation` to the user's group. See [Access Control and Visibility](index.md#access-control-and-visibility).

## Note: this is not `session_max_age_seconds`

`SESSION_MAX_AGE_SECONDS` controls only the **registry browser session cookie** (and its CSRF token), not the MCP access token. Changing it does not affect how long a generated token is valid. The two defaults both being 8h is a coincidence.

## Related FAQs

- [Registry API Authentication FAQ (static token, IdP JWT, coexistence)](registry-api-auth-faq.md)
- [Can I use an Entra ID token to call the registry API instead of the UI-generated token?](use-entra-token-for-registry-api.md)
- [How do I get my AI coding assistant to work with this registry?](connect-ai-coding-assistant.md)
