# Agent Registry — EnforceAI Gateway

## Agent Model
```
Agent {
    agent_id: string,
    user_id: string,
    scopes: string[],
    allowed_tools?: string[],
    metadata?: Record<string, any>,
    alias?: string,
    created_at: timestamp,
    tokens_valid_after?: timestamp,
    revoked: boolean
}
```

## Rules
- Every agent belongs to exactly one user
- Scopes must be assigned explicitly
- Revoked agents cannot authenticate
- Token revocation supports a bulk revoke mechanism using `tokens_valid_after` (deny if `token.iat < tokens_valid_after`).
- If `allowed_tools` is set, it is an additional restriction layer for both tool execution and tool visibility (see `enforceai/architecture/fgac_model.md`).

## Persistence Strategy

### Phase 1 (single instance)
- Source of truth is a local SQLite database file owned by the gateway.
- Stores agent records, agent scopes, API keys, and token revocations needed for enforcement.
- Assumes one running gateway instance; no cross-node coordination required in this phase.

### Portability Requirements
- All persistence access goes through small store interfaces (e.g., AgentStore, ApiKeyStore, RevocationStore) so auth/FGAC code is storage-agnostic.
- Use an ORM plus migrations even on SQLite, and keep schemas Postgres-safe (no SQLite-only features or types).
- Avoid relying on SQLite locking or filesystem semantics in the enforcement hot path.

### Future Scale-Out
- Planned migration target is Postgres as the primary store.
- Redis may be added later as a read-through cache for `/validate` lookups, with explicit invalidation on revocation.

## Service Boundary
- The agent registry is enforced by the gateway's stateful enforcement point (`auth_server`) on the request path.
- Management surfaces (UI/CLI) may live in `registry`, but enforcement must not depend on the UI being available.

## Required CRUD APIs
- POST /agents
- GET /agents/:user
- PATCH /agents/:agent_id
- DELETE /agents/:agent_id
- POST /agents/:agent_id/revoke

## Management Surface (Decision)
- Phase 1 is CLI-first for agent and credential lifecycle management.
- Management is self-service per user:
  - Users can create/manage/revoke their own agents and credentials.
  - Cross-user operations (if needed) are reserved for an explicit admin path later.
