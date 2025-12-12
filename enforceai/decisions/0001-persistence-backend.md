# Decision 0001 — Persistence Backend (Phase 1)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI introduces stateful identity data that is not present in the base MCP Gateway Registry:
- Agent registry (agents per user, scopes, allowed tools, revocation)
- API key records
- Gateway token revocation state

The current repo is largely stateless in the authentication hot path (`/validate`), and early development will run a single gateway instance.

## Decision
- Phase 1 will use a local SQLite database file as the source of truth for agents, API keys, and token revocations.
- The persistence layer will be designed for later migration to Postgres:
  - ORM + migrations from day one, even on SQLite.
  - Postgres-compatible schema and query patterns.
  - All reads/writes behind small store interfaces to keep enforcement storage-agnostic.
- Redis is not required in Phase 1, but may be added later as a read-through cache if `/validate` latency demands it, with explicit invalidation on revocation.

## Consequences
- Fast local iteration and simple deployment for a single instance.
- No guarantees for multi-replica consistency in Phase 1.
- Later move to Postgres is expected to be configuration + migrations + one-time data export/import, not a rewrite.
