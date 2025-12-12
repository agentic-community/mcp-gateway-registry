# Decision 0008 — Effective Scope Source for Gateway Tokens
*Status: Accepted*  
*Date: 2025-12-12*

## Context
Gateway tokens must embed scopes, but EnforceAI requires that agent permissions are gateway-managed and can be tightened without relying on token rotation.
We also want least-privilege delegation where a token can be narrower than its agent.

## Decision
- For `gateway-token` requests, compute effective scopes as:
  - `effective_scopes = token.scopes ∩ agent.scopes`
- Token scopes must never elevate beyond current agent registry scopes.
- If a token contains scopes not present on the agent, treat it as a reduction (log the mismatch) rather than rejecting, to allow immediate scope tightening.

## Consequences
- Requires reading agent scopes from the registry on the request path (SQLite in Phase 1; cacheable later).
- Enables least-privilege tokens while preserving centralized control over agent permissions.
