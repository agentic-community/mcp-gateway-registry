# Decision 0017 — Tenancy Boundary (Phase 1)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI Phase 1 focuses on single-instance enforcement with self-service per user. The locked identity model defines `user_id` and `agent_id` as critical terms.
We must decide whether to introduce explicit multi-tenancy (`tenant_id/org_id`) now or later.

## Decision
- Phase 1 tenancy boundary is `user_id`.
- Do not introduce an explicit `tenant_id/org_id` field in `IdentityContext` or persistence schemas in Phase 1.
- If needed later, add `tenant_id` as an additive migration and extend authorization/admin models accordingly.

## Consequences
- Keeps Phase 1 schemas and enforcement logic simple.
- Introduces potential migration work later if org-level admin/policy becomes required.
