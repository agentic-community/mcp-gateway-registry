# Stage 0 Implementation Plan — Foundation and Guardrails
*Created: 2025-12-12*

This document breaks **Stage 0** into small, single-run phases. Each phase has an explicit scope, required tests, and a strict checkpoint before moving to the next phase.

## Stage 0 Objectives
- Establish a clean EnforceAI module layout inside `auth_server/` without implementing business logic.
- Create core internal contracts (types/models/errors) that all later stages build on.
- Introduce configuration parsing/validation scaffolding (env-driven; secrets via mounted files).
- Set up unit-test scaffolding and fixtures so later stages can be implemented test-first.

## Non-Goals (Stage 0)
- No SQLite schema/migrations (Stage 1).
- No RS256 token mint/verify (Stage 2).
- No OIDC JWKS fetching/validation (Stage 3).
- No IdentityResolver wiring into `/validate` (Stage 4).
- No management endpoints or CLI (Stage 6).

## Global Rules (apply to all phases)
- No request-path behavior change in `auth_server/server.py` during Stage 0.
- No network calls in unit tests.
- After any Python code change:
  - Run `uv run python -m py_compile <changed files>` (or equivalent local compile check).
  - Run `make test-unit` (or at minimum `pytest -q` for the new tests introduced in the phase).
- Update `enforceai/session_state/latest.md` at the end of each completed phase:
  - mark phase complete
  - record what was added/changed
  - note the next phase to run

---

## Phase 0.1 — EnforceAI Package Skeleton + Core Contracts
**Goal**: make the EnforceAI codebase “real” (importable, typed, testable) with zero runtime integration.

### Scope (single run)
- Create `auth_server/enforceai/` module layout (namespace package or package subdir).
- Add **core contracts only**:
  - `IdentityContext` model (per locked EnforceAI architecture)
  - canonical identifiers (`user_id = "<iss>|<sub>"`, `agent_id` UUIDv4)
  - exception types to represent `401/403/503` semantics (no enforcement logic)
  - logging helpers that guarantee secret-safe output (masking utilities)

### Deliverables
- New modules created (example layout; adjust names if needed):
  - `auth_server/enforceai/identity.py` (IdentityContext model)
  - `auth_server/enforceai/errors.py` (typed errors with status mapping)
  - `auth_server/enforceai/logging.py` (safe masking helpers)
  - `auth_server/enforceai/__init__.py` (minimal exports; avoid heavy imports)

### Tests (must add in this phase)
- `tests/unit/enforceai/test_imports.py`
  - imports all new modules (ensures packaging works under pytest)
- `tests/unit/enforceai/test_identity_context_model.py`
  - validates required fields and serialization behavior (no runtime logic)
- `tests/unit/enforceai/test_error_semantics.py`
  - asserts errors map to the correct HTTP status class (`401/403/503`)

### Checkpoint (must pass before Phase 0.2)
- `uv run python -m py_compile` on all changed/new Python files passes.
- `make test-unit` passes (or `pytest -q tests/unit/enforceai` if running a focused subset).
- No changes to `auth_server/server.py` request behavior.

---

## Phase 0.2 — Configuration Scaffolding (Env-Driven, Validated)
**Goal**: parse and validate EnforceAI configuration early so later stages can rely on it.

### Scope (single run)
- Add a single “source of truth” settings object for EnforceAI configuration.
- Only parse + validate; do not perform network operations or side effects.
- Configuration must align to decisions already documented:
  - `OIDC_ISSUERS` JSON map
  - key paths (`*_PATH`, `*_DIR`, `GATEWAY_ACTIVE_KID`)
  - DB path (`ENFORCEAI_DB_PATH`)
  - audit retention thresholds (`ENFORCEAI_AUDIT_RETENTION_DAYS`, `ENFORCEAI_AUDIT_MAX_DB_BYTES`)
  - error semantics remain unchanged (this phase only provides settings)

### Deliverables
- `auth_server/enforceai/config.py`
  - settings model (prefer Pydantic for validation)
  - strict validation errors with actionable messages (do not log secrets)
- Small helper(s) for parsing structured env vars (e.g., JSON parsing with error context).

### Tests (must add in this phase)
- `tests/unit/enforceai/test_config_parsing.py`
  - valid `OIDC_ISSUERS` parses (map-of-one and map-of-two)
  - invalid JSON fails with a clear error
  - missing required vars fails with a clear error
- `tests/unit/enforceai/test_config_validation.py`
  - invalid key-path combinations (e.g., missing public keys dir) are rejected
  - invalid retention values rejected (negative days/bytes)

### Checkpoint (must pass before Phase 0.3)
- `uv run python -m py_compile` passes for changed files.
- `make test-unit` passes.
- Config parsing tests do not hit the network and do not read real secrets.

---

## Phase 0.3 — Test Fixtures for Later Stages (RSA + Temp SQLite + Env Helpers)
**Goal**: make later work fast by adding reusable fixtures now.

### Scope (single run)
- Provide fixtures to support upcoming stages:
  - RSA keypair generation for RS256 (in-memory; no external tools)
  - temporary directory fixtures for key file layout
  - temporary SQLite file fixture (no schema yet; just file + connection helpers)
  - environment patch helpers for deterministic tests

### Deliverables
- `tests/conftest.py` additions or `tests/fixtures/enforceai_fixtures.py` (prefer isolated fixture module).
- Fixtures should avoid global side effects and be safe for parallel test runs.

### Tests (must add in this phase)
- `tests/unit/enforceai/test_test_fixtures.py`
  - validates fixtures create expected artifacts (keys written, temp dirs)
  - ensures private key material is never printed/logged by default

### Checkpoint (must pass before Stage 0 completion)
- `uv run python -m py_compile` passes for changed files.
- `make test-unit` passes.
- New fixtures are documented inline (docstrings) and used by at least one test.

---

## Stage 0 Completion Criteria (strict)
Stage 0 is complete only when:
- Phases 0.1–0.3 checkpoints have all passed.
- EnforceAI modules are importable and validated by unit tests.
- Configuration parsing/validation is covered by unit tests.
- Fixtures for RSA keys and temp SQLite are available for Stage 1/2 work.
- `enforceai/session_state/latest.md` reflects Stage 0 completion and points to the next stage/phase to run.

## Suggested “end of stage” test gate
- `make test-unit`
- Optional: `make test-fast` (if it remains quick in your environment)

