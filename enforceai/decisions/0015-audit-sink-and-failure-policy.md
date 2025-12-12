# Decision 0015 — Audit Sink and Failure Policy (Phase 1)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI requires audit logs for every authorization decision. Phase 1 is a single-instance deployment with a local SQLite database, and the enforcement path must remain streamlined.
We need a durable local audit history without introducing external dependencies.

## Decision
- Dual-sink audit in Phase 1:
  - Emit structured JSON audit events to stdout.
  - Persist audit events to a local SQLite audit table for investigation and retention.
- Pragmatic failure policy:
  - Do not fail requests solely due to audit persistence failure.
  - Emit a high-severity log event indicating audit persistence failure to prompt remediation.

## Consequences
- Provides local durability while remaining compatible with centralized log aggregation later.
- Requires a retention/cleanup policy for the SQLite audit table (time/size-based) and configurable thresholds.
- Accepts the risk of missing persisted audit events during storage failures in exchange for availability.
