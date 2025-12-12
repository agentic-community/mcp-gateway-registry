# Decision 0009 — API Key Model (Phase 1)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI supports an API-key-only authentication mode. API keys must remain compatible with the agent-scoped authorization model:
- API keys authenticate an agent owned by a user (`IdentityContext { user_id, agent_id, scopes }`).
- API keys must not elevate permissions beyond the current agent registry scopes.
- Validation must be streamlined on the request path.

## Decision
- API keys are agent-bound and resolve to `{user_id, agent_id}`.
- Transport: `X-API-Key: eak_<key_id>.<secret>` (prefix + id + secret).
- Storage: store only a verifier hash, not the raw secret:
  - `secret_hash = HMAC-SHA256(API_KEY_PEPPER, secret)`
  - `API_KEY_PEPPER` is stored as a secret (env var or mounted secret file), never committed.
- Optional per-key scope restriction is supported:
  - `effective_scopes = api_key.scopes ∩ agent.scopes`
  - If `api_key.scopes` is unset/empty, treat as no extra restriction and use `agent.scopes`.
- Persistence: stored in the Phase 1 SQLite database alongside agents and revocations (portable to Postgres later).

## Consequences
- Database compromise does not directly reveal API keys without the pepper.
- Supports least-privilege API keys without requiring additional agents.
- Requires a secure pepper management and rotation plan.

## Future: Task-Scoped Permissions
Later EnforceAI may add task-level authorization (e.g., `task_id`-scoped actions/tools). This does not change the API key format or hashed-at-rest approach.
When introduced, task constraints should be modeled as an additional restriction dimension (e.g., credential/agent constraints metadata) and enforced alongside scope intersection.
