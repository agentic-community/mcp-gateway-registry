# Decision 0013 — Enterprise Policy Catalog Source (Phase 1)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI uses an enterprise-controlled scope catalog to define what each scope allows at the MCP server/method/tool level.
The upstream repo already includes a scope schema and an enforcement evaluator wired into the request path.

## Decision
- Phase 1 reuses `auth_server/scopes.yml` as the authoritative enterprise policy catalog.
- Agent scopes reference scope names defined in this catalog.
- IdP group mappings remain optional/legacy and must not override gateway-managed agent scopes.

## Consequences
- Minimizes policy-system churn and leverages existing enforcement logic.
- Catalog file currently includes multiple concerns (UI scopes, group mappings, server scopes); a later cleanup may split catalog vs mappings for clarity.
