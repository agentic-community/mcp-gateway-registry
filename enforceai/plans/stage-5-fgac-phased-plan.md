# Stage 5 — FGAC Enforcement + Tool Visibility Filtering
*Created: 2025-12-14*

## Goal
Wire the Stage 4 identity layer into the gateway request path and enforce agent-scoped FGAC using the enterprise scope catalog (`auth_server/scopes.yml`), including correct tool visibility filtering (`tools/list`) and execution enforcement (`tools/call`).

## Status
- Phase 5.1 complete (scope catalog loader + caching)
- Phase 5.2 complete (FGAC evaluator + decision model)
- Phase 5.3 complete (request context wiring + caching)
- Phase 5.4 complete (gateway `tools/list` filtering via Nginx Lua using `X-Allowed-Tools`)
- Phase 5.5 complete (gateway `tools/call` enforcement in auth_server `/validate` + best-effort audit hook)
- Phase 5.6 complete (multi-auth integration scenarios + integration tests)

## Implementation Notes (as-built)
- Enforcement point: `auth_server/server.py` `/validate` (behind Nginx `auth_request`).
- `tools/list` filtering: Nginx `body_filter_by_lua` filters JSON-RPC responses using `X-Allowed-Tools` returned by `/validate`.
- `tools/call` enforcement: `/validate` denies `403` when the `(server, tool)` pair is not authorized under FGAC; maps internal dependency failures to `503`.
- Best-effort auditing: `/validate` emits allow/deny events to stdout and attempts to persist via EnforceAI SQLite `AuditStore`; audit failures do not flip allow/deny outcomes.

## References (code + tests)
- FGAC catalog + evaluator: `auth_server/enforceai/fgac/catalog.py`, `auth_server/enforceai/fgac/evaluate.py`
- Request context wiring: `auth_server/enforceai/auth/dependency.py`
- Gateway filtering wiring: `registry/core/nginx_service.py`, `docker/registry-entrypoint.sh` (Lua scripts written at container start)
- Tool-call enforcement + audit: `auth_server/server.py`
- Unit tests: `tests/unit/enforceai/test_scope_catalog.py`, `tests/unit/enforceai/test_fgac_evaluate.py`, `tests/unit/enforceai/test_enforceai_dependency_wiring.py`, `tests/unit/enforceai/test_tools_call_enforcement.py`

## Non-Goals (Stage 5)
- No new management APIs/CLI for agents/keys/tokens (Stage 6)
- No retention/cleanup jobs (Stage 7)
- No external network calls in unit tests (JWKS fetches must be mocked)
- No redesign of the scope catalog schema (reuse `auth_server/scopes.yml` as-is)

## Inputs (Locked Decisions)
- IdentityContext is constructed once per request (Stage 4 resolver).
- Authorization is agent-scoped and derived from `IdentityContext.scopes` plus optional `agent.allowed_tools` restriction.
- IdP roles/groups/scopes are audit metadata only and must not elevate permissions.
- Error semantics:
  - `401` missing/invalid credentials (identity failure)
  - `403` authenticated-but-denied (FGAC deny, binding failures, revoked agent/credential)
  - `503` internal dependency failures (deny but signal retry)

## Proposed Code Layout (within `auth_server/enforceai/`)
- `fgac/`
  - `catalog.py` (load/parse `auth_server/scopes.yml`, validate, cache)
  - `evaluate.py` (decision engine: allow/deny + reason)
  - `models.py` (typed decision outputs + request descriptors)
- `auth/`
  - `resolver.py` (Stage 4.5)
  - `dependency.py` (FastAPI dependency: build identity once per request; Stage 5)

If upstream already has usable scope evaluation primitives (e.g., in `registry/services/access_control_service.py`), prefer reusing them rather than duplicating logic; adapt them to consume `IdentityContext`.

---

## Phase 5.1 — Scope Catalog Loader + Cached Validation
**Goal**: deterministically load and validate `auth_server/scopes.yml` into an in-memory catalog for fast per-request checks.

### Scope (single run)
- Implement `ScopeCatalog` loader:
  - Reads `auth_server/scopes.yml` from a configured path (default to repo path).
  - Validates structure and required fields (fail fast on startup/load).
  - Caches parsed catalog (avoid re-parsing per request).
- Provide a small, typed API to query “what scopes authorize what actions” based on the existing schema.

### Tests to add
- `tests/unit/enforceai/test_scope_catalog.py`
  - loads the real `auth_server/scopes.yml` successfully
  - rejects malformed YAML/invalid schema (use temp file fixture)
  - cache behavior (load once, re-use)

### Exit criteria
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai -k scope_catalog` (pass)
- `.venv/bin/python -m pytest` (full suite pass)

---

## Phase 5.2 — FGAC Evaluation Engine (Scopes + allowed_tools)
**Goal**: implement a single decision function used by both `tools/list` and `tools/call`.

### Scope (single run)
- Implement `evaluate_tool_visibility(...)` and `evaluate_tool_call(...)`:
  - Input: `IdentityContext`, server/tool descriptors, and the scope catalog.
  - Enforce `allowed_tools` (if set) as an additional restriction layer.
  - Return a typed `Decision` object (allow/deny + reason code).
- Ensure the same predicate drives both:
  - “listed tools” must be a subset of “callable tools”.

### Tests to add
- `tests/unit/enforceai/test_fgac_evaluate.py`
  - scope allow/deny matrix using a small synthetic catalog
  - allowed_tools restriction (deny even if scopes allow)
  - consistency property: `tools/list` visibility implies `tools/call` allow for same identity

### Exit criteria
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai -k fgac` (pass)
- `.venv/bin/python -m pytest` (full suite pass)

---

## Phase 5.3 — Request-Path Wiring (IdentityContext + Catalog + Evaluator)
**Goal**: wire identity + FGAC into the gateway’s FastAPI request path without duplicating work per request.

### Scope (single run)
- Add a FastAPI dependency (or middleware) that:
  - extracts credentials (Stage 4.1),
  - resolves `IdentityContext` (Stage 4.5),
  - attaches it to `request.state` (or returns it via dependency injection),
  - loads `ScopeCatalog` once (startup or cached loader).
- Ensure error mapping is consistent and stable (`401/403/503`).

### Tests to add
- `tests/unit/enforceai/test_enforceai_dependency_wiring.py`
  - uses FastAPI TestClient (local-only) to validate:
    - missing creds → `401`
    - missing/invalid `X-Agent-Id` (OIDC) → `403`
    - dependency failure (store read error) → `503`

### Exit criteria
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai -k dependency` (pass)
- `.venv/bin/python -m pytest` (full suite pass)

---

## Phase 5.4 — Tool Visibility Filtering (`tools/list`)
**Goal**: filter `tools/list` so callers only see tools they can execute under FGAC.

### Scope (single run)
- Identify the gateway route that serves tool discovery (upstream may proxy MCP servers).
- Apply filtering using the Phase 5.2 decision engine:
  - remove disallowed tools
  - return an empty list when nothing is callable (not an error)
- Ensure filtering does not break schema compatibility for clients.

### Tests to add
- `tests/unit/agents/test_visibility_filtering_enforceai.py` (or extend existing `tests/unit/agents/test_visibility_filtering.py`)
  - verifies filtered response matches expected allowed tools
  - verifies empty list behavior
- `tests/integration/test_enforceai_tools_list_filtering.py`
  - local FastAPI TestClient integration test that hits the real route path

### Exit criteria
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `make test-unit` (pass)
- `.venv/bin/python -m pytest` (full suite pass)

---

## Phase 5.5 — Tool Execution Enforcement (`tools/call`) + Audit Hook (best-effort)
**Goal**: enforce FGAC on tool execution and record allow/deny events.

### Scope (single run)
- Identify the gateway route that executes tools.
- Before proxying/executing:
  - evaluate FGAC decision for `(server, tool)` and deny with `403` if not allowed
  - map internal failures to `503` (deny but retryable)
- Add best-effort audit emission (stdout + SQLite via existing `AuditStore`), ensuring audit failures do not deny a request (Stage 7 will harden retention/cleanup).

### Tests to add
- `tests/unit/enforceai/test_tools_call_enforcement.py`
  - allow path (authorized)
  - deny path (403)
  - dependency failure path (503)
  - audit failure does not flip allow/deny outcome
- `tests/integration/test_enforceai_tools_call_enforcement.py`
  - local integration test covering list→call consistency

### Exit criteria
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `make test-unit` and `make test-integration` (pass)
- `.venv/bin/python -m pytest` (full suite pass)

---

## Phase 5.6 — Multi-Auth Integration Scenarios (OIDC + Gateway Token + API Key)
**Goal**: ensure Stage 5 behavior is identical across all auth modes and that mixed-mode routing works end-to-end.

### Scope (single run)
- Add integration tests that hit the real FastAPI routes (local-only) and validate:
  - OIDC + `X-Agent-Id` binding + tool visibility/call enforcement
  - gateway-token binding + revocation behavior affecting tool calls
  - api-key binding + optional scope restriction affecting tool calls
  - mixed-mode bearer routing by `iss` (gateway issuer vs configured OIDC issuer)

### Tests to add
- `tests/integration/test_enforceai_stage5_roundtrip.py`
  - uses mocked JWKS fetcher (no network)
  - uses temp SQLite stores for agents/keys/revocations/audit
  - covers `tools/list` and `tools/call` for each auth mode

### Exit criteria (Stage 5 completion gate)
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `make test-unit` and `make test-integration` (pass)
- `.venv/bin/python -m pytest` (full suite + coverage gate pass)
- `enforceai/session_state/latest.md` updated with Stage 5 completion and pointer to Stage 6
