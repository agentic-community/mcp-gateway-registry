# Decision 0012 — Gateway Token Lifetime Policy
*Status: Accepted*  
*Date: 2025-12-12*

## Context
Gateway tokens are used by coding assistants and headless agents where frequent re-issuance is operationally expensive.
EnforceAI includes layered revocation controls to rapidly disable compromised tokens or agents.

## Decision
- Phase 1 uses long-lived gateway tokens (PAT-style).
- Tokens must always include `exp` (no non-expiring tokens).
- Maximum lifetime target is up to 365 days, with shorter lifetimes recommended for higher-risk agents.

## Consequences
- Lower operational churn for clients.
- Requires strong issuance governance, monitoring, and reliable revocation workflows to mitigate longer compromise windows.
- Key rotation must retain old public keys for verification until the longest-lived tokens expire.
