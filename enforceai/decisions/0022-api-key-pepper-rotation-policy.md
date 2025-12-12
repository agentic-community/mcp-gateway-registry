# Decision 0022 — API Key Pepper Rotation Policy (Phase 1)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
API key verification uses a peppered HMAC hash:
`secret_hash = HMAC-SHA256(API_KEY_PEPPER, secret)`.
Rotating the pepper invalidates existing hashes unless pepper versioning is implemented.

## Decision
- Phase 1 does not support pepper rotation/versioning.
- Rotating `API_KEY_PEPPER` is a breaking change for existing API keys unless a future pepper-versioning scheme is added.

## Consequences
- Simplifies Phase 1 implementation and operations.
- If pepper compromise occurs, remediation requires revoking/reissuing API keys (and optionally agent kill switches) rather than seamless pepper rotation.
