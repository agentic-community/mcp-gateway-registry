# Decision 0021 — Identifier Canonicalization (`user_id`, `agent_id`)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI uses `user_id` and `agent_id` as critical identifiers across tokens, headers, persistence, and audit logs.
Generic OIDC supports multiple issuers, so identifiers must be collision-resistant across issuers.

## Decision
- `user_id` is issuer-namespaced from OIDC:
  - `user_id = "<iss>|<sub>"`
- `agent_id` is a UUIDv4 string as the canonical identifier used for binding and authorization.
- Agents may have an optional human-friendly alias/name for display, but authorization always uses `agent_id`.

## Consequences
- Avoids collisions across multiple OIDC issuers.
- Keeps agent identifiers stable even if display names change.
