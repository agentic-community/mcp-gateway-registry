# Decision 0005 — Generic OIDC (Multi-Issuer Support)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI requires “any OIDC IdP” support. Real deployments often need to accept more than one issuer over time (IdP migrations, parallel cutovers, multi-tenant).
We also require a streamlined validation path (no per-request network dependencies beyond JWKS refresh).

## Decision
- Implement generic OIDC validation using a multi-issuer configuration map keyed by JWT `iss`.
- The issuer map may contain a single issuer in Phase 1 (map-of-one), but the structure is retained to avoid later redesign.
- Validation is local:
  - Select issuer config by unverified `iss` (fail closed if unknown).
  - Validate signature using cached JWKS for that issuer.
  - Validate `iss`, `aud`, `exp`, `iat`.
- Claim mapping for scopes/roles may be configured globally and/or per issuer.

## Consequences
- Slightly more config/parsing complexity than single-issuer mode.
- Significantly simpler future IdP cutovers and multi-issuer deployments.
