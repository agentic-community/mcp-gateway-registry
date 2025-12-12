# Decision 0020 — Audit Retention Policy (SQLite)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
Phase 1 persists audit events locally in SQLite. Without retention controls, the audit table will grow unbounded and become an operational risk.
Retention controls must not impact request-path latency.

## Decision
- Use a hybrid, configurable retention policy:
  - Time-based retention (e.g., `ENFORCEAI_AUDIT_RETENTION_DAYS`)
  - Size-based cap (e.g., `ENFORCEAI_AUDIT_MAX_DB_BYTES`)
- Cleanup is performed out of band (manual CLI or scheduled job), not on the request path.

## Consequences
- Prevents unbounded growth while allowing long retention windows when disk permits.
- Requires operational configuration and periodic cleanup execution.
