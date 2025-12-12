# Decision 0004 — Enforcement Point Service Boundary
*Status: Accepted*  
*Date: 2025-12-12*

## Context
The upstream MCP Gateway Registry already implements gateway authorization using Nginx `auth_request` to an `auth_server` `/validate` endpoint.
EnforceAI requires additional identity modes (generic OIDC, gateway-issued tokens, API keys), an agent registry, and revocation checks, while keeping request-path validation streamlined.

## Decision
- Extend the existing `auth_server` into the EnforceAI stateful enforcement point.
- Keep Nginx `auth_request` as the integration mechanism for enforcing identity + FGAC before proxying to MCP servers.
- Treat `registry` as optional for management/UI flows (agent CRUD, token issuance UI/CLI), but not required for the enforcement path.

## Consequences
- Minimizes architectural churn and preserves the current gateway wiring.
- Concentrates security-critical logic in one service; requires clear internal modularization (resolver, providers, stores) and strong tests.
- Supports Phase 1 single-instance deployment cleanly, with a path to later scale-out by moving SQLite to Postgres and optionally adding caching.
