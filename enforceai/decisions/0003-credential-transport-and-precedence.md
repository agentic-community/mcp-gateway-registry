# Decision 0003 — Credential Transport and Precedence
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI supports mixed authentication modes (OIDC, gateway tokens, API keys). Clients include standard HTTP tooling and MCP clients/coding assistants.
We need a clear, compatible way to transmit credentials and a strict rule set to prevent ambiguous multi-credential requests.

## Decision
- Canonical transport for token-based auth is `Authorization: Bearer <token>` for both OIDC JWTs and gateway-issued tokens.
- `X-Gateway-Token: <token>` is accepted as a fallback for gateway tokens only.
- The gateway must reject requests that include more than one credential source (e.g., `Authorization` plus `X-Gateway-Token`, or `X-API-Key` plus any token header).

## Consequences
- Maximizes compatibility with standard HTTP tooling and proxy stacks.
- Keeps the public client contract simple while still supporting constrained clients.
- Prevents “credential confusion” bugs by default-denying ambiguous requests.
