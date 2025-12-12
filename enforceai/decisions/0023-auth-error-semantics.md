# Decision 0023 — Auth/Error Semantics
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI supports multiple authentication modes and must provide predictable error behavior for MCP clients.
The enforcement point must fail closed but should distinguish retryable internal failures from credential and policy errors.

## Decision
- `401 Unauthorized` for missing/invalid credentials:
  - missing credential header
  - malformed token
  - invalid signature
  - expired token
  - API key not found/mismatch
- `403 Forbidden` for authenticated-but-not-authorized:
  - FGAC policy denies access
  - revoked agent/token/API key
  - invalid/missing agent binding (e.g., missing `X-Agent-Id` for OIDC MCP access)
- `503 Service Unavailable` for internal enforcement dependency failures:
  - cannot read agent registry or revocation state
  - cannot load/parse required verification keys
  - configuration/state prevents a safe authorization decision
  - Requests are denied (fail closed), but status indicates retry.

## Consequences
- Clients can distinguish retryable failures (503) from credential problems (401) and permission denials (403).
- Requires consistent mapping of internal exceptions to these statuses.
