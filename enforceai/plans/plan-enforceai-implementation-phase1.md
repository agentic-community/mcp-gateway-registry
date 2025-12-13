# EnforceAI Gateway — High-Level Implementation Plan
*Created: 2025-12-12*

This plan implements the EnforceAI identity + enforcement layer on top of this repo with strict stage checkpoints and test gates.

## Locked Decisions (inputs)
- Enforcement point: extend `auth_server` behind Nginx `auth_request`.
- Persistence: single-instance SQLite now, portable to Postgres later (ORM + migrations + store interfaces).
- Auth modes only: generic OIDC (multi-issuer), gateway tokens (RS256), API keys (mixed via IdentityResolver); no backward compatibility.
- Canonical credential transport: `Authorization: Bearer`; allow `X-Gateway-Token` fallback; reject multi-credential requests.
- Agent binding: `agent_id` required for MCP access; from token claim, API key record, or `X-Agent-Id` for OIDC.
- Policy catalog: reuse `auth_server/scopes.yml`.
- Effective scopes: `token.scopes ∩ agent.scopes` and `api_key.scopes ∩ agent.scopes` (or agent scopes when key scopes unset).
- Revocation: layered (agent kill switch, `jti` table, bulk revoke via `agent.tokens_valid_after`).
- Tokens: RS256 with mounted key files; restart-based rotation; PAT-style up to 365 days, `exp` required.
- Audit: stdout JSON + SQLite; pragmatic persistence failure; hybrid configurable retention; cleanup out of band.
- Identifiers: `user_id = "<iss>|<sub>"`, `agent_id` UUIDv4 (optional alias for UX).
- Tool visibility: filter `tools/list` to only callable tools under effective authorization.
- Error semantics: `401` invalid/missing creds, `403` authorized-but-denied/binding failures, `503` internal enforcement dependency failure (deny but signal retry).

## Stage 0 — Foundation / Guardrails
**Goal**: ensure repo is ready for safe iterative implementation.

**Deliverables**
- Create EnforceAI code layout within `auth_server/` (resolver, providers, stores, models), keeping `server.py` as a thin orchestrator.
- Add a minimal “EnforceAI configuration” doc snippet (env vars and secret file paths) without secrets.
- Add testing scaffolding for `auth_server` components (new pytest modules, fixtures for temp SQLite and RSA keys).

**Checkpoint (must pass)**
- New module skeletons are importable.
- `make test-unit` passes.
- `uv run python -m py_compile` (or equivalent) passes for new/modified Python files.
- `enforceai/session_state/latest.md` updated at end of work session.

---

## Stage 1 — Data Layer (SQLite + Migrations + Store Interfaces)
**Goal**: introduce portable persistence for agents, API keys, revocations, audit.

**Deliverables**
- SQLite schema (via migrations) for:
  - `agents` (includes `user_id`, `agent_id`, scopes, allowed_tools, alias, revoked, tokens_valid_after, metadata)
  - `api_keys` (key_id, secret_hash, user_id, agent_id, optional scopes, revoked, expires_at, created_at, last_used_at)
  - `token_revocations` (jti, user_id, agent_id, revoked_at, expires_at, reason)
  - `audit_events` (append-only; indexed by time/user_id/agent_id)
- Store interfaces (storage-agnostic):
  - `AgentStore`, `ApiKeyStore`, `RevocationStore`, `AuditStore`
- Local-only DB path config (`ENFORCEAI_DB_PATH`) and safe initialization.

**Checkpoint (must pass)**
- Migration up/down works on a fresh DB.
- Unit tests for each store: CRUD, revoke flows, bulk revoke via `tokens_valid_after`, and failure modes.
- Coverage for new stores meets repo threshold (80% overall; add focused tests).
- `make test-unit` passes.

---

## Stage 2 — Cryptography + Token Primitives (RS256)
**Goal**: implement key loading, token mint/verify, and claims validation.

**Deliverables**
- Keyring loader:
  - private key path + public keys dir + active `kid`
  - cached parsed keys; reject missing/invalid key material as startup/config error
- Gateway token mint/verify:
  - claims: `iss`, `sub` (`user_id`), `agent_id`, `scopes`, `iat`, `exp`, `jti`
  - header: `kid`
  - enforce `exp`, `iat` sanity (clock skew tolerance), and issuer match

**Checkpoint (must pass)**
- Unit tests:
  - valid token verifies
  - wrong `kid` / wrong key / tampered token fails with `401`
  - expired token fails with `401`
  - malformed token fails with `401`
- No secrets leak in logs (test asserts logs do not contain private key material or full tokens).
- `make test-unit` passes.

---

## Stage 3 — Generic OIDC Provider (Multi-Issuer, Local JWKS Cache)
**Goal**: validate OIDC JWTs generically without provider-specific code.

**Deliverables**
- `OIDC_ISSUERS` map parsing + validation.
- Local JWKS fetch/cache per issuer (refresh on TTL; never fetch on every request).
- Claim mapping defaults + per-issuer overrides:
  - scopes: `scp` → `scope` → `permissions`
  - roles: `roles` → `groups` → `permissions` (audit-only)
- `user_id` derivation: `"<iss>|<sub>"`

**Checkpoint (must pass)**
- Unit tests for multi-issuer selection:
  - unknown issuer -> `401`
  - wrong audience -> `401`
  - JWKS rotation scenario (cache refresh) -> succeeds after refresh
- Unit tests for claim parsing across shapes (string vs list).
- `make test-unit` passes.

---

## Stage 4 — IdentityResolver + IdentityContext Assembly
**Goal**: unify OIDC, gateway tokens, and API keys into a single IdentityContext.

**Deliverables**
- Resolver enforces:
  - exactly one credential source (reject ambiguous) -> `401`
  - OIDC MCP access requires `X-Agent-Id` -> `403` if missing/invalid
- Providers:
  - `GatewayTokenProvider` (RS256 verify + revocation + agent binding)
  - `OidcProvider` (JWT validate + agent binding via `X-Agent-Id`)
  - `ApiKeyProvider` (agent-bound key verify + optional scope restriction)
- IdentityContext output is consistent:
  - provider, user_id, agent_id, effective scopes, metadata

**Checkpoint (must pass)**
- Unit tests covering:
  - each auth mode happy-path
  - all error semantics (401/403/503) mapped per decision
  - multi-credential rejection
  - revocation deny paths (agent revoked, token revoked, tokens_valid_after)
- `make test-unit` passes.

---

## Stage 5 — FGAC Enforcement + Tool Visibility Filtering
**Goal**: enforce policy catalog and avoid broken discovery workflows.

**Deliverables**
- Authorization checks use:
  - effective scopes against `auth_server/scopes.yml`
  - `allowed_tools` as additional restriction
- `tools/list` response filtering:
  - only list tools the caller can execute under effective authorization
  - return empty tool list when none callable
- Ensure deny paths return correct status codes and audit events.

**Checkpoint (must pass)**
- Unit tests for scope evaluation integration:
  - tools/call allowed/denied across servers/tools
  - tools/list filtering matches what tools/call would allow
- Integration tests (FastAPI TestClient or docker-based) that simulate:
  - an MCP server exposing multiple tools
  - caller sees only allowed tools and can call only those
- `make test-unit` and `make test-integration` pass.

---

## Stage 6 — Management APIs + CLI (Self-Service)
**Goal**: enable users to manage their own agents and credentials without UI dependency.

**Deliverables**
- Management endpoints (served by enforcement point) for:
  - agent CRUD + revoke
  - mint gateway token for an agent (token scopes <= agent scopes)
  - create/revoke API keys (optional scope restriction)
  - audit retention cleanup (out of band)
- CLI commands (Phase 1):
  - login/auth instructions
  - agent create/list/update/revoke
  - token mint/revoke-all (via tokens_valid_after bump)
  - api-key create/revoke

**Checkpoint (must pass)**
- Integration tests:
  - self-service ownership enforcement (cannot modify another user’s agent)
  - token mint respects scope intersection and revocation
  - api-key create/verify/revoke works end-to-end
- `make test-fast` passes locally.

---

## Stage 7 — Audit + Retention + Hardening
**Goal**: make enforcement operable and safe in enterprise-like conditions.

**Deliverables**
- Audit event model and emission:
  - stdout JSON always
  - SQLite persistence best-effort (pragmatic failure policy)
- Configurable retention thresholds + cleanup command:
  - `ENFORCEAI_AUDIT_RETENTION_DAYS`
  - `ENFORCEAI_AUDIT_MAX_DB_BYTES`
- Performance guardrails:
  - ensure JWKS and key parsing are cached
  - ensure DB queries on request path are indexed

**Checkpoint (must pass)**
- Load tests (lightweight) show no request-path network dependency and stable latency under reasonable local load.
- Tests:
  - audit persistence failure does not deny request, but emits high-severity log
  - cleanup deletes expected rows and leaves recent rows intact
- `make test` passes before declaring the stage complete.

---

## Robust Testing Strategy (Summary)
**Unit tests (fast, deterministic)**
- Stores (SQLite), including failure modes.
- Token mint/verify (RS256), claim validation, `kid` selection.
- OIDC validation with multi-issuer and JWKS caching behavior (mocked HTTP).
- Resolver edge cases and error semantics mapping.
- FGAC evaluation and tools/list filtering logic.

**Integration tests (ASGI-level)**
- `auth_server` endpoints with TestClient:
  - validate flow for each auth mode
  - management endpoints (self-service constraints)
- Tool discovery + execution:
  - mock MCP server provides tools; verify tools/list filtering + tools/call enforcement.

**End-to-end (docker-compose, optional per stage)**
- Bring up stack; validate:
  - Nginx auth_request wiring
  - real MCP client flows for tools/list and tools/call
  - revocation and scope tightening behavior

**Security regression tests**
- Assert no secrets are logged (tokens, API key secrets, private keys).
- Fail-closed behavior when persistence is unavailable (expect `503`).

**Quality gates**
- `make test-unit` for every PR-sized change.
- `make test-fast` at the end of each stage.
- `make test` before “stage complete”.

