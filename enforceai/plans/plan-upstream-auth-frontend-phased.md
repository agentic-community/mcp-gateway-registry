# Plan: Upstream MCP Auth (Frontend, Phased)
*Created: 2025-12-17*

## Assumptions (Confirmed)

- Backend phases are implemented first (this plan depends on backend APIs existing).
- Agents/clients authenticate only to the gateway; upstream credentials are never required in client configs.
- Injection architecture: `auth_server` `/validate` computes upstream injection; Nginx applies injection to upstream requests.
- stdio/pipes appear as HTTP/WS behind the gateway (bridge), so UI still treats upstream auth as header-based.

## Objective

Make upstream authentication requirements visible and manageable in the UI, aligned to:
- `enforceai/AUTHENTICATION.md`
- `enforceai/mcp_upstream_auth_requirements.md`
- Backend management APIs introduced in `enforceai/plans/plan-upstream-auth-backend-phased.md`

Each phase below is scoped to be completable in a single agent run and must be fully tested before marking complete.

## Dev/Local Test Harness (Applies to All Phases)

Frontend phases must be testable locally without any real upstream secrets or external provider dependencies.

### Required mocking and fixtures

- Use MSW fixtures to model:
  - servers with `upstream_auth` requirements
  - per-principal credential statuses (missing/configured/expired/revoked)
  - OAuth connect start/disconnect flows (simulated)
- Never store or snapshot real secrets in tests; use placeholders only.

### Required validation commands

- Typecheck: `cd frontend && npm run typecheck`
- Tests: `cd frontend && npm test`

### Local run expectations (per phase)

Each phase must extend MSW handlers and fixtures so the UI can be run locally against mocked upstream-credential APIs without requiring a running backend.

## Phase 1 — Surface Upstream Auth Requirements in Server Catalog

### Scope
- Extend server types in the React app to include upstream auth fields returned by `GET /api/servers` (or a dedicated upstream-auth endpoint).
- Display per-server “Upstream Auth” badge in:
  - server cards (dashboard list)
  - server details (where present)
- Display per-server credential status for the current principal:
  - `Configured`, `Missing`, `Expired`, `Revoked` (based on backend-provided status)

### Deliverables
- Update server model types used by the dashboard list (e.g., `frontend/src/hooks/useServerStats.ts`, `frontend/src/components/ServerCard.tsx`).
- UI badge component (shared) for auth type + status.
- MSW mocks updated to include upstream auth fields.

### Tests (must pass)
- `cd frontend && npm run typecheck` (required)
- `cd frontend && npm test` (required)
- Add/extend unit tests verifying badge rendering for each auth type and status.

## Phase 2 — Admin Egress Allowlist UI (SSRF Controls)

### Scope
- Add an admin-only UI to manage the gateway egress allowlist (strict allowlist policy):
  - list existing allowlist entries
  - create/update/delete entries
  - optional TTL (`expires_at`) support in the UI
  - clear warnings about SSRF risk and what the allowlist does
- Integrate with backend admin endpoints for allowlist management.
- Add UI feedback and validation:
  - prevent obvious invalid hostnames
  - show expiration state for expiring entries

### Deliverables
- New route under an admin section (or a new “Network Policy” page).
- API client functions and MSW handlers for allowlist CRUD.

### Tests (must pass)
- `cd frontend && npm run typecheck` (required)
- `cd frontend && npm test` (required)
- Add tests for:
  - admin-only access gating in the UI
  - CRUD happy path
  - TTL field rendering and expired indication

## Phase 3 — “Upstream Credentials” Section (List + Status Filters)

### Scope
- Add a new navigation section: “Upstream Credentials”.
- Implement list view:
  - show all servers requiring upstream credentials
  - filter by status (missing/configured/expired)
  - link to per-server configuration
- Implement empty/loading/error states with actionable guidance.

### Deliverables
- New route(s) in `frontend/src/router.tsx` and nav updates.
- New feature module under `frontend/src/features/upstream-credentials/`.
- API client functions for upstream credential status/list (backed by `/enforceai/upstream/*` endpoints).

### Tests (must pass)
- `cd frontend && npm run typecheck` (required)
- `cd frontend && npm test` (required)
- Add tests for list rendering, filtering, and error/empty states.

### Local verification (manual)

- Run `cd frontend && npm run dev` and verify:
  - list view renders with MSW fixtures
  - status filters behave as expected

## Phase 4 — Per-Server Credential Configuration: API Key + JWT (Static)

### Scope
For servers with upstream auth types:
- `api-key`: UI to set/rotate/revoke key (secret shown once)
- `jwt` (static): UI to set/revoke token (secret shown once)

Requirements:
- Secrets never shown again after creation.
- Provide “Copy as Header” and “Download” helpers (mirroring existing EnforceAI credential UX patterns).
- Show status and last-updated metadata.

### Deliverables
- Per-server configuration page and modals.
- Reusable “SecretOnce” component pattern (acknowledgement gating close).
- Update MSW handlers for new endpoints and add fixtures.

### Tests (must pass)
- `cd frontend && npm run typecheck` (required)
- `cd frontend && npm test` (required)
- Add tests for:
  - create success → shows secret once → cannot close without acknowledgement
  - revoke updates status

### Local verification (manual)

- Verify “secret once” behavior:
  - secret visible only in the post-create screen
  - navigating away and returning does not show the secret again (only metadata)

## Phase 5 — OAuth Connect/Disconnect UX (OAuth2/OIDC/Provider OAuth)

### Scope
- Add “Connect” flows for OAuth-based upstream auth:
  - Start flow calls backend “start” endpoint, redirects to provider authorization URL.
  - Callback/success page confirms connection and returns user to originating server page.
  - Error page explains reason and recovery steps.
- Add “Disconnect” action for connected providers.
- Show token metadata (provider, scopes, expiry) but never show token values.

### Deliverables
- New pages/components for OAuth redirect handling (distinct from registry login flow).
- Deep-link support (return to server config page after success).
- MSW mocks for start/disconnect/status.

### Tests (must pass)
- `cd frontend && npm run typecheck` (required)
- `cd frontend && npm test` (required)
- Add tests for:
  - connect button triggers start call and redirect
  - disconnect updates status and disables upstream calls
  - success/error pages render correct messages

### Local verification (manual)

- With MSW fixtures:
  - Connect triggers a simulated redirect
  - Success page returns user to the server configuration page

## Phase 6 — Update MCP Config Modal (Gateway-Terminated Messaging)

### Scope
- Update the MCP config generator modal so it:
  - includes only gateway ingress auth placeholders
  - never suggests adding upstream credentials to client config
  - links to “Upstream Credentials” UI when a server requires upstream auth
  - displays a clear message: “Upstream auth is handled by the gateway”

### Tests (must pass)
- `cd frontend && npm run typecheck` (required)
- `cd frontend && npm test` (required)
- Add tests verifying generated config never contains `X-API-Key` or upstream `Authorization` placeholders for upstream tokens.

### Local verification (manual)

- Open the MCP config modal for a server requiring upstream auth and confirm it:
  - instructs the user to configure upstream credentials in “Upstream Credentials”
  - does not include any upstream secrets in the client config

## Future (Not Implemented in This Plan): mTLS UI

mTLS management UI is explicitly deferred until backend mTLS support exists.
