# Identity Model — EnforceAI Gateway

## IdentityContext Definition
```
IdentityContext {
    user_id: string,
    agent_id: string,
    provider: "oidc" | "gateway-token" | "api-key",
    scopes: string[],
    user_roles?: string[],
    metadata?: Record<string, any>
}
```

## Rules
- Constructed once per request
- Agent identity NEVER comes from IdP
- Authorization uses agent scopes only

## Supported Credential Sources
- OIDC JWT
- Gateway Token
- API Key

## Credential Transport (Decision)
- Canonical: `Authorization: Bearer <token>` for both OIDC and gateway tokens.
- Fallback: `X-Gateway-Token: <token>` (gateway tokens only).
- The gateway must reject requests that include more than one credential source to avoid ambiguity.

## Enforcement Point (Decision)
- Request-path identity resolution and FGAC enforcement runs in the stateful enforcement point (`auth_server`) behind Nginx `auth_request`.

## Agent Identity Source (Decision)
- `agent_id` is required for any request that reaches MCP server routing / FGAC.
- Source of `agent_id`:
  - `gateway-token`: from the token claim `agent_id`.
  - `api-key`: from the API key record (API keys are agent-bound).
  - `oidc`: from the `X-Agent-Id` request header, validated against the gateway-managed agent registry for the authenticated `user_id`.

## Tenancy (Decision)
- Phase 1 tenancy boundary is `user_id` (no explicit `tenant_id/org_id` in the identity model or storage schemas).
- A future `tenant_id` may be introduced later as an additive migration if required for delegated admin or org-level policy.
