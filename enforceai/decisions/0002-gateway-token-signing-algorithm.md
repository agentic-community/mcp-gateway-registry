# Decision 0002 — Gateway Token Signing Algorithm
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI requires gateway-issued tokens for non-OIDC clients (coding assistants, headless agents, CI/CD).
These tokens must be verifiable on the enforcement path and support future key rotation and eventual scale-out.
Compatibility with standard JWT tooling is a primary requirement.

## Decision
- Use `RS256` for gateway-issued tokens.
- Include a `kid` header to support key rotation.
- Keep validation streamlined by verifying with locally loaded public keys (no network calls on the request path).

## Consequences
- Token minting is heavier than HMAC-based signing but acceptable for expected issuance rates.
- Public-key distribution is safe: verifiers do not gain minting capability.
- Rotation is straightforward: publish/add a new public key + switch signing key, keep old keys for verification until tokens expire.
