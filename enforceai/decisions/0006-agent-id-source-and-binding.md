# Decision 0006 — Agent ID Source and Binding
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI authorization is agent-scoped and requires `IdentityContext { user_id, agent_id, scopes }`.
Agent identity must be gateway-managed (not derived from IdP claims).
We must define how `agent_id` is supplied for each authentication mode.

## Decision
- `agent_id` is required for any request that reaches MCP server routing / FGAC enforcement.
- Source of `agent_id` by provider:
  - `gateway-token`: `agent_id` is embedded as a JWT claim.
  - `api-key`: API keys are agent-bound and resolve to `{user_id, agent_id}`.
  - `oidc`: callers must send `X-Agent-Id: <agent_id>`; the enforcement point validates ownership under the authenticated `user_id`.

## Consequences
- OIDC-authenticated MCP calls require an additional header.
- Eliminates implicit “default agent” behavior and keeps auditing unambiguous.
