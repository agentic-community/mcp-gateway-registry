# Stage 6 — Management APIs + CLI (Self-Service)
*Created: 2025-12-14*

## Goal
Enable users to manage their own agents and credentials without depending on the registry UI:
- Agent lifecycle (create/list/get/update/revoke)
- API keys (create/list/revoke; optional scope restriction)
- Gateway tokens (mint; revoke by `jti`; revoke-all via `tokens_valid_after`)

All management operations must enforce strict ownership (`user_id`) and preserve Stage 4/5 error semantics:
- `401` missing/invalid credentials
- `403` authenticated-but-denied (ownership mismatch, invalid binding, revoked agent/credential)
- `503` internal dependency failures (deny but signal retry)

## Non-Goals (Stage 6)
- No new UI (Stage 6 is CLI-first; UI is optional later)should add any more unit/e2e/integration tests at this point?
- No background retention/cleanup jobs (Stage 7)
- No external network calls in unit/integration tests (OIDC JWKS must be mocked)
- No scope-catalog redesign (reuse `auth_server/scopes.yml`; only validate against it)

## Inputs (Locked Decisions)
- Enforcement point is `auth_server` (FastAPI) behind Nginx `auth_request` for MCP traffic.
- IdentityContext is agent-scoped and produced by Stage 4 resolver:
  - OIDC requires `X-Agent-Id`
  - gateway token contains `agent_id`
  - api-key is agent-bound
- Management endpoints may require “any valid identity”, but must never allow cross-user operations.

## Proposed Code Layout (within `auth_server/enforceai/`)
- `management/`
  - `models.py` (request/response models for management operations)
  - `service.py` (pure management operations; ownership validation; store orchestration)
- `api/`
  - `management_routes.py` (FastAPI router mounted into `auth_server/server.py`)
- `cli/` (repo `cli/`)
  - `enforceai_cli.py` (argparse CLI that calls management APIs via httpx)

## Testing Policy (required for every phase)
Each phase must add/extend tests and finish with:
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `.venv/bin/python -m pytest` (full suite + coverage gate pass)

Integration tests must remain local-only (no external network); for OIDC:
- Inject a fake JWKS fetcher into `JWKSCache` (as done in Stage 5.6).

---

## Phase 6.1 — Management Service Layer (Domain + Validation)
**Goal**: implement all management operations as deterministic, testable Python functions (no HTTP yet).

### Scope (single run)
- Add `auth_server/enforceai/management/service.py` and implement:
  - `create_agent(user_id, scopes, allowed_tools, alias, metadata)`
  - `list_agents(user_id)`
  - `get_agent(user_id, agent_id)`
  - `update_agent(user_id, agent_id, scopes?, allowed_tools?, alias?, metadata?)`
  - `revoke_agent(user_id, agent_id)`
  - `create_api_key(user_id, agent_id, scopes?, expires_at?)` returning `(key_id, secret, api_key_value)`
  - `list_api_keys(user_id, agent_id)`
  - `revoke_api_key(user_id, key_id)`
  - `mint_gateway_token(user_id, agent_id, scopes, ttl_seconds?, expires_at?)` (scopes must be subset of agent scopes)
  - `revoke_token_jti(user_id, agent_id, jti)` and `revoke_all_tokens(user_id, agent_id)` (bump `tokens_valid_after`)
- Validate requested scopes against the loaded `ScopeCatalog` (typo-proofing).
- Ensure secrets are never logged or returned except as intended (API key secret once, token string once).

### Tests to add
- `tests/unit/enforceai/test_management_service.py`
  - agent CRUD + ownership enforcement
  - scope validation against a synthetic catalog
  - api-key create/revoke semantics and “secret returned once” behavior
  - gateway token mint scope-subset rule
  - revoke-all bumps `tokens_valid_after`
  - dependency failure mapping (store exceptions bubble to `DependencyUnavailableError`)

### Exit criteria
- `.venv/bin/python -m pytest` (pass)

---

## Phase 6.2 — Management API Routes in `auth_server`
**Goal**: expose service-layer operations as FastAPI endpoints (no Nginx involvement).

### Scope (single run)
- Add an `APIRouter` (e.g., `auth_server/enforceai/api/management_routes.py`) and mount it in `auth_server/server.py`:
  - `GET /enforceai/agents`
  - `POST /enforceai/agents`
  - `GET /enforceai/agents/{agent_id}`
  - `PATCH /enforceai/agents/{agent_id}`
  - `POST /enforceai/agents/{agent_id}/revoke`
  - `POST /enforceai/agents/{agent_id}/tokens/revoke-all`
  - `POST /enforceai/agents/{agent_id}/api-keys`
  - `GET /enforceai/agents/{agent_id}/api-keys`
  - `POST /enforceai/api-keys/{key_id}/revoke`
  - `POST /enforceai/agents/{agent_id}/tokens/mint`
  - `POST /enforceai/tokens/revoke` (accept `jti` + `agent_id` or accept a gateway token and extract `jti`)
- Use Stage 4 resolver to authenticate callers (same credentials as the gateway).
- Enforce ownership by comparing `IdentityContext.user_id` to stored `agent.user_id` / `api_key.user_id`.
- Emit best-effort audit events for management actions (reuse Stage 5 audit sink; failures must not flip allow/deny).

### Tests to add
- `tests/integration/test_enforceai_management_routes.py`
  - full happy-path flows for each auth mode (OIDC, gateway-token, api-key) using local TestClient + mocked JWKS
  - deny paths:
    - cross-user access attempts (`403`)
    - revoked agent cannot mint tokens/create keys (`403`)
    - invalid scopes in create/update (`400` or mapped `403` as decided; document and test)
  - dependency failure path (`503`) by forcing store errors

### Exit criteria
- `.venv/bin/python -m pytest` (pass)

---

## Phase 6.3 — CLI (HTTP, Argparse, Local-Only Tests)
**Goal**: provide a CLI that calls Stage 6.2 endpoints and can be used without any UI.

### Scope (single run)
- Add `cli/enforceai_cli.py` (argparse) implementing commands:
  - `agents list|create|get|update|revoke`
  - `api-keys create|list|revoke`
  - `tokens mint|revoke|revoke-all`
- Support credential injection via args and env vars (no interactive login):
  - `--authorization` (or `ENFORCEAI_AUTHORIZATION`)
  - `--x-agent-id` (or `ENFORCEAI_X_AGENT_ID`) for OIDC
  - `--x-gateway-token` / `--x-api-key` (mutually exclusive)
  - `--base-url` (or `ENFORCEAI_AUTH_SERVER_URL`, default `http://localhost:8888`)
- Make output stable (JSON by default; `--pretty` for pretty JSON).

### Tests to add
- `tests/unit/cli/test_enforceai_cli_args.py`
  - argument parsing (mutual exclusion, required fields)
  - header construction does not leak secrets in logs/errors
- `tests/integration/test_enforceai_cli_roundtrip.py`
  - run CLI code against an in-process ASGI app via `httpx.ASGITransport` (no network)
  - validate at least one happy-path flow (create agent -> mint token -> revoke-all)

### Exit criteria
- `.venv/bin/python -m pytest` (pass)

---

## Phase 6.4 — Hardening + Documentation
**Goal**: finish Stage 6 with secure defaults and operator-grade docs.

### Scope (single run)
- Add docs to `enforceai/instructions/` describing:
  - required env vars (DB path, pepper path, key paths, OIDC issuers)
  - CLI examples for common operations
  - recommended operational model for “bootstrap” (requires at least one existing agent credential to manage agents)
- Add regression tests for:
  - “secret returned once” behavior (API key secret)
  - audit emission best-effort (audit failure does not block management requests)

### Exit criteria (Stage 6 completion gate)
- `.venv/bin/python -m pytest` (full suite + coverage gate pass)
- `enforceai/session_state/latest.md` updated with Stage 6 completion summary and next stage pointer

