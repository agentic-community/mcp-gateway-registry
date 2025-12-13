# Stage 1 — Data Layer (SQLite + Migrations + Store Interfaces)
*Created: 2025-12-13*

## Goal
Introduce portable persistence for EnforceAI (agents, API keys, token revocations, audit) with a minimal migration system and storage-agnostic store interfaces.

## Non-Goals (Stage 1)
- No request-path integration into `auth_server/server.py`
- No OIDC/JWKS work
- No gateway token signing/verification
- No API key hashing/verification logic (Stage 1 stores accept pre-hashed secrets)
- No background retention cleanup scheduling (only primitives and tests)

## Inputs (Locked Decisions)
- Enforcement point is `auth_server` behind Nginx `auth_request`
- Phase 1 persistence is local SQLite with storage-agnostic interfaces (migration target: Postgres later)
- Tenancy boundary is `user_id` (no org/tenant id)
- Identifiers: `user_id = "<iss>|<sub>"`, `agent_id` is UUIDv4 string
- Error semantics: 401/403/503 per `enforceai/architecture/architecture_lock.md`

## Proposed Code Layout (within `auth_server/enforceai/`)
- `db/`
  - `connection.py` (SQLite connection helpers; pragmas; context manager)
  - `migrations.py` (migration runner; `schema_migrations` table)
  - `migrations/sql/` (`0001_*.up.sql`, `0001_*.down.sql`, etc.)
- `models/`
  - `agent.py`, `api_key.py`, `audit.py`, `revocation.py` (Pydantic models; JSON serialization helpers)
- `stores/`
  - `interfaces.py` (Protocols/ABCs for `AgentStore`, `ApiKeyStore`, `RevocationStore`, `AuditStore`)
  - `sqlite/` (SQLite implementations)

## Test Strategy (must pass for every phase)
Per-phase (single-run) gate:
- `uv run python -m py_compile <changed_files...>`
- `uv run pytest -q -o addopts='' tests/unit/enforceai`

End-of-stage (Stage 1 final phase) gate:
- `make test` (full suite, including coverage)

Use Stage 0.3 fixtures:
- `enforceai_sqlite_db_path` (temp SQLite file)
- `tmp_path` for isolated per-test storage

## Phase 1.1 — Migration System + Baseline Schema
**Scope (single run)**
- Implement a tiny migration runner (no new dependencies):
  - discovers ordered SQL pairs: `NNNN_name.up.sql` and `NNNN_name.down.sql`
  - tracks applied migrations in `schema_migrations(version TEXT PRIMARY KEY, applied_at TEXT)`
  - supports `upgrade_to_latest()` and `downgrade_one()` (or `downgrade_to(version)`)
- Add baseline migration(s) creating tables (empty-but-real schema; indexes included):
  - `agents`, `api_keys`, `token_revocations`, `audit_events`

**Tests to add**
- `tests/unit/enforceai/test_migrations.py`
  - fresh DB upgrades to latest
  - upgrade is idempotent (run twice, no change/error)
  - downgrade then upgrade restores schema

**Exit criteria**
- Migrations apply/rollback on a fresh DB in tests
- No network access; only filesystem + SQLite
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

## Phase 1.2 — Agent Store Interface + SQLite Implementation
**Scope (single run)**
- Define `AgentRecord` (Pydantic) and `AgentStore` interface:
  - create agent (user_id, agent_id, scopes, optional allowed_tools, alias, metadata)
  - get agent by agent_id
  - list agents by user_id
  - update agent fields (scopes, allowed_tools, alias, metadata)
  - revoke agent (kill switch)
  - bump `tokens_valid_after` (bulk revoke primitive)
- Implement SQLite store:
  - JSON fields stored as TEXT (JSON)
  - indexes: `(user_id)`, `(user_id, revoked_at)`, `(agent_id)`

**Tests to add**
- `tests/unit/enforceai/test_agent_store_sqlite.py`
  - CRUD happy path
  - list-by-user isolation
  - revoke updates state and persists
  - tokens_valid_after bump persists and is monotonic
  - invalid inputs fail fast (UUIDv4 agent_id, canonical user_id)

**Exit criteria**
- AgentStore behavior fully covered by unit tests
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

## Phase 1.3 — API Key Store Interface + SQLite Implementation
**Scope (single run)**
- Define `ApiKeyRecord` (Pydantic) and `ApiKeyStore` interface:
  - create key record (key_id, secret_hash, user_id, agent_id, optional scopes, expires_at)
  - get by key_id
  - revoke key (set revoked_at)
  - update last_used_at
  - list keys by user_id and/or agent_id (for management)
- Implement SQLite store with indexes:
  - `(user_id)`, `(agent_id)`, `(key_id)`, `(expires_at)`

**Tests to add**
- `tests/unit/enforceai/test_api_key_store_sqlite.py`
  - create/get/revoke
  - scope field optional behavior
  - last_used_at update does not expose secrets in logs

**Exit criteria**
- API key persistence primitives are in place; no secret verification logic yet
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

## Phase 1.4 — Token Revocation Store Interface + SQLite Implementation
**Scope (single run)**
- Define `TokenRevocationRecord` and `RevocationStore` interface:
  - revoke `jti` with optional reason and expires_at
  - check if `jti` is revoked (must be fast; indexed)
  - list by agent_id for debugging (optional)
- Add optional cleanup primitive (delete expired revocations) as a store method (no scheduling).

**Tests to add**
- `tests/unit/enforceai/test_revocation_store_sqlite.py`
  - revoke/check
  - expired revocations cleanup behavior (if implemented)

**Exit criteria**
- Revocation store works and is isolated/deterministic in tests
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

## Phase 1.5 — Audit Store Interface + SQLite Implementation
**Scope (single run)**
- Define `AuditEventRecord` and `AuditStore` interface:
  - append audit event
  - query recent events by user_id/agent_id/time window (for later retention + debugging)
- Ensure schema supports efficient time queries (index on timestamp).

**Tests to add**
- `tests/unit/enforceai/test_audit_store_sqlite.py`
  - append/read basic behavior
  - does not print sensitive fields (use `caplog` / `capsys`)

**Exit criteria**
- Audit persistence primitives are in place; retention policies are configured in Stage 0.2 settings and enforced later
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

## Phase 1.6 — Store Factory + End-to-End Data Layer Smoke Test
**Scope (single run)**
- Add a minimal factory/helper to open the DB and provide store instances:
  - applies migrations on startup (explicit call, not hidden side effects in imports)
  - enforces SQLite pragmas appropriate for single-node reliability
- Add a single smoke test that wires everything together on a fresh DB.

**Tests to add**
- `tests/unit/enforceai/test_data_layer_smoke.py`
  - upgrade DB
  - create agent
  - create api key record bound to agent
  - revoke jti
  - append audit event
  - re-open DB and re-read (persistence check)

**Exit criteria**
- `uv run pytest -q -o addopts='' tests/unit/enforceai` passes
- `make test` passes (Stage 1 completion gate)
- `enforceai/session_state/latest.md` updated with completion status and next stage pointer
