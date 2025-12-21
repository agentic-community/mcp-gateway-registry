# Plan: Upstream OAuth Provider Registry (Admin-Managed)
*Created: 2025-12-21*

## Goal
Enable an admin-managed registry of upstream OAuth providers (authorization/token endpoints + client credentials) so admins can configure OAuth clients via the UI and users can connect per-user upstream OAuth tokens without any token pasting.

This plan adds:
- Encrypted-at-rest provider storage in the EnforceAI DB (write-only secrets).
- Admin APIs for provider CRUD (CSRF + audit).
- Runtime wiring so upstream OAuth flows use provider configs from the DB.
- Frontend admin UI to manage providers.
- Robust unit + integration + frontend test coverage (offline; no real provider network calls).

## Non-Goals
- Supporting per-server unique OAuth clients as the primary model. (Multiple servers may reference the same provider id; providers are keyed by `provider_id`.)
- Long-term Vault/KMS integration details beyond interfaces. (DB registry remains the Phase 1 concrete backend; external secret backends are optional.)

## Preconditions
- EnforceAI is enabled and has a KEK configured (`ENFORCEAI_DB_PATH` and `ENFORCEAI_UPSTREAM_KEK_PATH` are non-empty and valid).
- Server records reference the provider id via `server.upstream_auth.provider`.

## Phase Completion Gate (Applies to Every Phase)
Do not mark a phase complete unless the full test suites run cleanly:
- Backend (required): `make test`
- Frontend (required): `cd frontend && npm run typecheck && npm test`

## Phase 1 — Data Layer: provider model + encrypted store
**Outcome:** Provider configs are stored in SQLite with encrypted client secrets and safe read shapes.

Deliverables
- Add schema + migration:
  - Table `upstream_oauth_providers` keyed by `provider_id` (string).
  - Store `authorization_endpoint`, `token_endpoint`, `client_id`, `default_scopes`, `extra_authorize_params`, `created_at`, `updated_at`.
  - Store encrypted `client_secret` payload (envelope with nonce/tag) using the upstream KEK.
- Add models:
  - `UpstreamOAuthProviderRecord` (read model; no secret).
  - `UpstreamOAuthProviderCreate/Update` request models (accept secret).
  - `UpstreamOAuthProviderPublic` response model with `secret_present: bool`.
- Add store interface + SQLite implementation:
  - `create_provider`, `get_provider`, `list_providers`, `update_provider`, `delete_provider`.
  - Secrets are write-only: no method returns decrypted secret except an internal `get_provider_secret_for_runtime()` used only by OAuth flows.

Tests (must pass)
- Unit:
  - Model validation (provider_id normalization rules; endpoint URL validation).
  - Encryption-at-rest (ciphertext differs from plaintext; decrypt roundtrip works).
  - Write-only invariants (public models never contain secrets).
- Commands (incremental + full gate):
  - `.venv/bin/python -m py_compile auth_server/enforceai/models/upstream_oauth_provider.py auth_server/enforceai/stores/sqlite/upstream_oauth_provider_store.py`
  - `make test-unit -k upstream_oauth_provider`
  - `make test`
  - `cd frontend && npm run typecheck && npm test`

## Phase 2 — Auth Server: admin CRUD APIs for providers
**Outcome:** Admins can manage provider configs via `/enforceai/admin/*` endpoints; secrets are never returned.

Deliverables
- Add admin endpoints (CSRF-protected, admin-only):
  - `GET /enforceai/admin/upstream-oauth-providers`
  - `POST /enforceai/admin/upstream-oauth-providers`
  - `GET /enforceai/admin/upstream-oauth-providers/{provider_id}`
  - `PUT /enforceai/admin/upstream-oauth-providers/{provider_id}`
  - `DELETE /enforceai/admin/upstream-oauth-providers/{provider_id}`
- Enforce:
  - Admin authorization required (same rules as other `/enforceai/admin/*`).
  - CSRF required on POST/PUT/DELETE.
  - Audit events for create/update/delete (no secret values in logs).
- Define deletion semantics:
  - Default: reject deletion if referenced by any server in the registry catalog.
  - Optional: allow `?force=true` if explicitly required (must be audited).

Tests (must pass)
- Integration:
  - Non-admin cannot list/create/update/delete.
  - Missing CSRF rejected.
  - Responses never include `client_secret`.
  - Delete fails when provider is referenced by a server.
- Commands (incremental + full gate):
  - `make test-integration -k upstream_oauth_providers`
  - `make test-unit`
  - `make test`
  - `cd frontend && npm run typecheck && npm test`

## Phase 3 — Runtime wiring: use DB provider registry for OAuth flows
**Outcome:** OAuth connect/refresh uses provider config from DB; env-based config remains fallback.

Deliverables
- Update upstream OAuth start/callback/refresh to resolve provider config in this order:
  1. Provider registry store (DB) by `provider_id`
  2. Fallback to `ENFORCEAI_UPSTREAM_OAUTH_PROVIDERS` (dev/bootstrap) if DB missing
- Add stable error semantics when provider is missing:
  - `400` for invalid input
  - `424`/`409` only when appropriate
  - include `X-EnforceAI-Error-Code=UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED` on relevant failure paths
- Ensure token refresh uses the same provider secret source.

Tests (must pass)
- Integration (offline):
  - Create provider via admin API
  - Register server pointing to that provider id
  - Start OAuth -> callback -> stored credential
  - Validate/proxy path refresh works (using in-process stub provider; no network)
- Commands (incremental + full gate):
  - `make test-integration -k upstream_oauth`
  - `make test-unit`
  - `make test`
  - `cd frontend && npm run typecheck && npm test`

## Phase 4 — Frontend: Admin UI for provider management
**Outcome:** Admins can create/update/delete provider configs in the SPA; secrets are never displayed.

Deliverables
- Add `Admin → Upstream OAuth Providers` page:
  - List providers (provider_id, endpoints, client_id, scopes, secret_present).
  - Create provider modal (client secret input; write-only).
  - Edit provider modal (optionally rotate secret).
  - Delete provider with confirmation (shows “in use” errors).
- Add frontend API functions for the new endpoints.
- Ensure `/enforceai/*` calls use the existing EnforceAI UI token vending and CSRF behaviors (no localStorage).
- Update MSW handlers and unit tests.

Tests (must pass)
- Frontend unit tests:
  - List renders from API
  - Create sends payload, does not store/display secret after creation
  - Update rotates secret
  - Delete handles “referenced by server” failure
- Commands (incremental + full gate):
  - `cd frontend && npm run typecheck`
  - `cd frontend && npm test`
  - `make test`

## Phase 5 — Server registration UX hardening + end-to-end regression
**Outcome:** Server registration/edit flow and OAuth connect flow are consistent and validated end-to-end.

Deliverables
- Server register/edit UI:
  - Replace free-form provider input with a dropdown sourced from provider registry (admin-only route); keep a manual override only if explicitly needed.
  - Validate `provider` required when upstream auth type is OAuth.
- Backend validation:
  - When registering/updating a server with OAuth upstream auth, validate referenced provider exists (admin-only enforcement).
- Add end-to-end regressions:
  - Backend integration test that rejects server registration referencing unknown provider.
  - Frontend test that register modal requires provider selection for OAuth types.

Tests (must pass)
- Incremental:
  - `make test-unit`
  - `make test-integration`
  - `cd frontend && npm run typecheck && npm test`
- Full gate:
  - `make test`

## Final validation (after Phase 5)
- `make test`
- Optional: `cd frontend && npm run build`
