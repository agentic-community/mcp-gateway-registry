# Decision 0010 — Gateway Token Key Management and Rotation
*Status: Accepted*  
*Date: 2025-12-12*

## Context
Gateway tokens use RS256. Validation must be streamlined on the request path and keys must be manageable and rotatable with `kid`.
Phase 1 is a single-instance deployment but must remain portable to later multi-instance deployments.

## Decision
- Canonical key delivery mechanism is mounted secret files:
  - Private key: a PEM file used for signing.
  - Public keys: one PEM file per `kid` to support rotation and verification.
- Validation remains local:
  - Load and parse public keys at startup; cache in memory.
  - Select verification key by JWT header `kid`.
- Rotation is restart-based in Phase 1:
  - Add a new keypair, switch active `kid`, restart.
  - Keep old public keys available until all previously issued tokens expire.
- Future (optional): publish a JWKS endpoint for public keys to support additional verifiers; do not introduce request-path network dependencies.

## Consequences
- Strong secret hygiene and compatibility with container/K8s secret patterns.
- Rotation is operationally simple in Phase 1.
- Supports later scale-out without changing token format or verification approach.
