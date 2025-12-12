# Decision 0011 — Revocation Strategy
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI supports gateway tokens, API keys, and an agent registry. We require the ability to rapidly disable access after compromise and to revoke individual credentials without relying solely on expiration.
The enforcement path must remain streamlined but must not fail open on revocation checks.

## Decision
Adopt a layered, agent-scoped revocation model:
- Agent kill switch: `agent.revoked` denies all access for that agent (applies to gateway tokens and API keys).
- Token-level revocation: deny if JWT `jti` is present in the token revocation table.
- Bulk token revocation per agent: maintain `agent.tokens_valid_after` and deny if `token.iat < tokens_valid_after`.
- Default-deny if revocation/agent registry state cannot be read on the enforcement path.

## Consequences
- Supports rapid response: revoke a token, revoke all tokens for an agent, or revoke an agent entirely.
- Requires checking agent and revocation state on the request path (SQLite in Phase 1; cacheable later).
- Demands careful operational discipline to avoid denying traffic due to persistence outages (expected and preferred over fail-open).
