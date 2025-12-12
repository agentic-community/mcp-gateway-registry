# Decision 0014 — Backward Compatibility Scope
*Status: Accepted*  
*Date: 2025-12-12*

## Context
The upstream MCP Gateway Registry includes provider-specific authentication modes (Cognito/Keycloak/Entra) and a legacy HS256 token vending token type.
EnforceAI requires a unified identity layer with generic OIDC, RS256 gateway tokens, API keys, and strict agent binding.

## Decision
- EnforceAI does not require backward compatibility with upstream provider-specific modes or legacy token types.
- Only EnforceAI authentication modes are supported:
  - OIDC (generic, multi-issuer)
  - Gateway tokens (RS256)
  - API keys
  - Mixed mode via IdentityResolver

## Consequences
- Cleaner implementation and fewer security foot-guns.
- Existing upstream flows may stop working until migrated to EnforceAI modes.
