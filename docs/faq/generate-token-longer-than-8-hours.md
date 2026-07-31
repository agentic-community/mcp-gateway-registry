# How do I generate an MCP access token that lasts longer than 8 hours?

The token minted by the **Generate Token** page (and `POST /api/tokens/generate`) is a self-signed gateway JWT that works as an MCP access token. Its **8 hours is only the default**, not the ceiling: the shipped maximum is **24 hours**, and both the default and the maximum are operator-configurable (from [#1477](https://github.com/agentic-community/mcp-gateway-registry/issues/1477)).

So there are two different things you might want, and they need different actions.

## I just want one token that lasts, say, 12 hours

Any value up to the configured maximum (24h by default) needs **no config change at all** — you simply request it.

**From the UI:** open **Generate Token** in the left sidebar, pick the lifetime from the **Expires In** dropdown, and generate.

**From the API:** pass `expires_in_hours` in the request body.

```bash
export REG=http://localhost   # or your deployment URL

curl -sS -X POST "$REG/api/tokens/generate" \
  -H "Authorization: Bearer $(cat .token)" \
  -H "Content-Type: application/json" \
  -d '{"expires_in_hours": 12}'
```

The value must be an integer between `1` and the configured maximum; a request above the maximum is rejected with `400`.

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
