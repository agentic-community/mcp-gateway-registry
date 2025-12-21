# Plan
*Created: 2025-12-20*

Implement end-to-end gateway-terminated upstream OAuth for MCP servers: users register a server with OAuth requirements, complete a browser consent flow as the currently logged-in user, and the UI manages upstream tokens seamlessly (no token pasting, no token display).

## Requirements
- Add MCP servers with `upstream_auth.type` in `{oauth2, oidc, provider-oauth}` and `provider`/`credential_binding` set from the UI.
- Initiate OAuth consent as the logged-in user (browser session), not via manually pasted EnforceAI credentials.
- Store upstream tokens server-side encrypted-at-rest; UI shows only metadata (provider, scopes, expiry, status).
- Support Connect (start) / Callback (finish) / Disconnect flows with stable error semantics.
- Work offline in tests: no real providers, no external network.

## Scope
- In:
  - Server create/edit supports upstream OAuth requirements (already in place).
  - “Connect” UX for OAuth servers uses browser redirects (authorization-code + PKCE).
  - Token lifecycle handled by gateway: refresh on-demand, revoke/disconnect.
  - Unified UI session for calling `/enforceai/*` without pasting long-lived secrets (token vending).
- Out:
  - mTLS upstream auth.
  - Long-term Vault/KMS provider-secret backends beyond env/file interfaces (interfaces only).
  - Multi-tenant org boundaries (single-tenant assumed).

## Key files and entry points
- Auth server OAuth flow and routes:
  - `auth_server/enforceai/api/management_routes.py`
  - `auth_server/enforceai/models/upstream_oauth.py`
  - `auth_server/enforceai/upstream/oauth_flow.py`
  - `auth_server/enforceai/stores/sqlite/upstream_oauth_state_store.py`
- Registry session + UI token vending (new):
  - `registry/api/auth_routes.py` (or dedicated `registry/api/enforceai_session_routes.py`)
  - `registry/auth/*` (existing session verification)
- Frontend OAuth UX:
  - `frontend/src/features/credentials/UpstreamCredentialModal.tsx`
  - `frontend/src/api/enforceai.ts`
  - `frontend/src/router.tsx` (new callback route)
  - `frontend/src/contexts/AuthContext.tsx` (or new EnforceAI session context)
- Tests:
  - `tests/integration/test_enforceai_upstream_oauth_flow.py`
  - `tests/integration/test_enforceai_management_routes.py`
  - `frontend/src/test/mocks/handlers.ts`
  - `frontend/src/features/credentials/__tests__/*`

## Data model / API changes
- OAuth state storage must include:
  - server_path, provider, credential_type, credential_binding, user_id (+ optional agent_id)
  - PKCE code_verifier (+ nonce for OIDC)
  - `ui_return_url` (same-origin) for final redirect back into the SPA
- Canonical server-scoped OAuth endpoints (UI-facing):
  - `POST /enforceai/upstream/servers/{server_path}/oauth/start`
  - `GET  /enforceai/upstream/servers/{server_path}/oauth/callback`
  - `POST /enforceai/upstream/servers/{server_path}/oauth/disconnect`
- Unified UI session:
  - Registry provides `POST /api/auth/enforceai/token` (name TBD) to vend a short-lived EnforceAI management token for the logged-in cookie session (CSRF protected).
  - Auth server verifies that token and produces `user_id` for management context.

## Phased implementation (each phase: single agent run + test gate)

### Phase 1 — Server-scoped OAuth endpoints + browser-safe callback
Goal: make the OAuth flow work end-to-end with redirects and encrypted token storage (even if start is still authenticated via existing EnforceAI management auth).

Deliverables
- Add `/enforceai/upstream/servers/{server_path}/oauth/start|callback|disconnect` endpoints in `auth_server/enforceai/api/management_routes.py` that:
  - create state + return `authorization_url` (PKCE)
  - complete callback without relying on custom headers
  - store tokens via upstream credential store
  - redirect to `ui_return_url` after completion
- Extend OAuth state store/model to persist `ui_return_url` with allowlisted same-origin validation.
- Keep existing `/enforceai/upstream/oauth/*` endpoints temporarily as compatibility wrappers (optional).

Tests and validation (must pass)
- Unit:
  - state record validation (binding, server_path canonicalization)
  - `ui_return_url` validation (no open redirects)
  - PKCE and state consume invariants
- Integration (offline):
  - Extend `tests/integration/test_enforceai_upstream_oauth_flow.py` to cover:
    - start -> callback (no auth headers) -> credential stored
    - callback redirects to UI return URL and does not leak tokens
    - disconnect revokes credential
- Commands:
  - `make test-unit`
  - `make test-integration -k upstream_oauth`

### Phase 2 — Frontend: Connect/Callback/Disconnect UX aligned to server-scoped endpoints
Goal: user can click “Connect” for an OAuth server and complete the flow entirely from the UI, with correct status updates.

Deliverables
- Update `frontend/src/api/enforceai.ts` to call the server-scoped OAuth endpoints (and match request/response shapes).
- Add SPA callback route and page:
  - `GET /credentials/upstream/oauth/callback` shows success/failure and navigates back to the relevant server detail/credential modal.
- Update `UpstreamCredentialModal.tsx` to:
  - send `ui_return_url`
  - handle “Connected” state by refetching metadata

Tests and validation (must pass)
- Frontend unit tests:
  - Connect button calls start endpoint and sets `window.location.href`
  - Callback page renders success/failure and triggers a refresh
  - Disconnect calls API and updates status
- Commands:
  - `cd frontend && npm run typecheck`
  - `cd frontend && npm test`

### Phase 3 — Seamless UI auth: Registry session -> short-lived EnforceAI token vending
Goal: UI can call `/enforceai/*` as the logged-in user without pasting long-lived EnforceAI tokens/API keys.

Deliverables
- Registry endpoint (CSRF-protected, cookie-auth):
  - `POST /api/auth/enforceai/token` (name TBD) returns `{access_token, expires_at}`.
  - Token is short-lived and audience-limited to EnforceAI management APIs.
- Auth server accepts the vended token:
  - verification keys/secret configured via env/file
  - maps to `user_id` for management context
- Frontend automatically acquires and uses the vended token for EnforceAI API calls (in-memory only; no localStorage).

Tests and validation (must pass)
- Backend unit tests:
  - token issuance claims (exp/aud/sub) and CSRF enforcement
  - auth server verification and context derivation
- Integration:
  - cookie-auth UI session can successfully call a read-only EnforceAI endpoint (`GET /enforceai/agents`) without manual credentials
- Commands:
  - `make test-unit`
  - `make test-integration -k enforceai`
  - `cd frontend && npm test`

### Phase 4 — Harden: derive/validate OAuth parameters from server `upstream_auth`
Goal: prevent UI/backend mismatches and ensure the server’s declared upstream auth is enforced.

Deliverables
- On OAuth start:
  - load/validate server’s `upstream_auth` and reject if the server does not require OAuth
  - derive provider/credential_type/binding from server config (or validate the client-provided values match)
- Optional: support `user+agent` binding in UI with agent selector.

Tests and validation (must pass)
- Unit tests for mismatch handling (server says `api-key` but start requests `oauth2`, etc.)
- Integration test that start fails for non-OAuth servers and succeeds for OAuth servers.
- Commands:
  - `make test-unit`
  - `make test-integration -k upstream_oauth`

### Phase 5 — End-to-end regression: server registration -> connect -> proxy uses token
Goal: prove the actual gateway proxy uses stored upstream tokens (and refresh works on-demand).

Deliverables
- Integration test scenario:
  - register server with upstream OAuth
  - connect flow stores tokens
  - proxy a request and assert upstream sees injected Authorization bearer
  - simulate expiry and assert refresh path (using stub provider)

Tests and validation (must pass)
- `make test-integration`
- Optional: `make test-e2e` if the repo’s e2e suite covers browser redirects; otherwise keep coverage in integration tests.

## Risks and edge cases
- Open redirect risk via `ui_return_url`: must enforce same-origin allowlist and ignore unsafe URLs.
- Provider callback cannot preserve headers: callback must be state-driven and not require bearer headers.
- CSRF: token vending endpoint must require CSRF; OAuth start should also be CSRF-protected if cookie-auth is used.
- Cookies/SameSite: redirect flows must work with current cookie settings; avoid third-party cookie reliance.
- Logging: ensure no tokens appear in logs; redact any token exchange errors.

## Open questions (max 2)
- Final token-vending contract: exact endpoint name and signing model (HS256 shared secret vs RS256 with published registry public keys).
- Whether `credential_binding=user+agent` is required in Phase 1 or can be deferred to Phase 4.

