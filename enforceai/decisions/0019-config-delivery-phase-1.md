# Decision 0019 — Config Delivery (Phase 1)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI requires several structured configuration inputs (e.g., `OIDC_ISSUERS`) and multiple secrets (RS256 private key, API key pepper).
Phase 1 is single-instance and should minimize additional configuration systems while remaining deployable via Docker/K8s patterns.

## Decision
- Phase 1 configuration is environment-variable driven.
- Secrets are provided via mounted secret files and referenced through path environment variables (avoid placing private key material or peppers directly in env vars).

## Consequences
- Simple operational model aligned with container deployments.
- Large structured env vars (JSON) require careful formatting; sample snippets should be maintained in docs.
