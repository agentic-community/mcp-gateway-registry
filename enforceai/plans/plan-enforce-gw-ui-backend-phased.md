# Enforce Gateway UI — Backend Implementation Plan (Phased)
*Created: 2025-12-15*

This plan implements the backend contract in `docs/enforceai-ui-backend-changes.md` to support a completely new Enforce Gateway UI.

Primary constraints:
- Each phase is scoped to be finishable in a single agent run.
- Each phase adds/updates tests for the phase’s scope.
- The full test suite must pass at the end of every phase (`make test`).

## References (inputs)
- Requirements: `docs/enforceai-ui-requirements.md`
- Backend checklist/contract: `docs/enforceai-ui-backend-changes.md`
- Current EnforceAI architecture context: `enforceai/instructions/ENFORCEAI_CONTEXT.md`

## Current state (important context)
As of 2025-12-15, this repo already has:
- A registry session cookie (`mcp_gateway_session`) signed via `itsdangerous`.
- OAuth2 login handled by `auth_server`, which sets a registry-compatible session cookie.
- `auth_server /validate` already supports session-cookie authentication (in addition to bearer credentials) for Nginx `auth_request`.
- EnforceAI management APIs under `/enforceai/*` (currently authenticated via bearer token / API key / gateway token; OIDC requires `X-Agent-Id`).
- Several registry “internal/operator” endpoints still protected by HTTP Basic.

## Global guardrails (apply to every phase)
- Do not change EnforceAI identity/enforcement semantics (those are locked).
- Keep browser auth cookie-based (no Basic Auth and no long-lived secrets in the browser).
- Keep changes backwards-compatible where needed until the migration phase explicitly removes legacy paths.
- For every phase, update `enforceai/session_state/latest.md` with:
  - what changed
  - which tests were run
  - what’s next

## Target end-state (Phase 1 UI backend contract)
At the end of all phases:
- One browser session cookie authorizes both `registry` (`/api/*`) and `auth_server` (`/enforceai/*`).
- CSRF is enforced for all state-changing endpoints used by the UI when authenticated via cookies.
- Local username/password users exist (multiple users, admin + non-admin), and OIDC admin is driven by IdP group/role `enforceai-admin`.
- Admin break-glass credential exists (gateway token and/or API key), is auditable, and revocable.
- EnforceAI has admin cross-user APIs (users directory + cross-user agent/credential operations).
- Registry internal/operator endpoints are migrated off HTTP Basic to unified admin auth.

---

# Phase 1 — Session Contract + “Me” Parity (no behavior change yet)
**Goal**: define a single, explicit session schema and make both services expose the same “who am I” view, without breaking existing auth flows.

## Deliverables
- Add a versioned session model (Pydantic) and helpers:
  - canonical `user_id` formats:
    - OIDC: `"<iss>|<sub>"`
    - local password: `"local|<username>"`
  - fields at minimum: `auth_method`, `user_id`, `email`, `username`, `roles`/`groups`, `expires_at`, `session_id` (even if session invalidation lands later).
- Registry:
  - Ensure `/api/auth/me` returns (at minimum) `{user_id, email, username, auth_method, is_admin, roles/groups}` derived from the session.
  - Ensure the registry session decode path supports “new schema” and legacy cookies.
- Auth server:
  - Ensure session-cookie validation in `auth_server/server.py` can parse the same “new schema” and legacy cookies.
  - Ensure the OAuth2 callback populates the new session fields (at least `user_id`, `email`, `groups`).
- Document the session schema (short contract section) in the plan or a dedicated doc snippet referenced by both services.

## Tests to add/update (must remain deterministic)
- Unit tests:
  - Add tests for session encode/decode roundtrips (new schema + legacy compatibility).
  - Update existing auth unit tests that assume the old cookie payload shape.
- Integration tests:
  - Add/extend a test for `/api/auth/me` returning canonical identity fields.

## Required test gate
- `uv run python -m py_compile` for changed Python files.
- `make test` (must pass).

---

# Phase 2 — CSRF Foundation for Cookie-Authenticated SPA Calls
**Goal**: add consistent CSRF enforcement for all UI-driven state changes when the request is authenticated via cookies.

## Scope decisions (locked for this phase)
- Standardize on: `GET /api/auth/csrf` returning a token; UI sends `X-CSRF-Token` on all non-GET.
- Only enforce CSRF when the request is authenticated via the session cookie.
  - If a request uses non-cookie credentials (e.g., API key, bearer token), CSRF checks must not block it.

## Deliverables
- Registry:
  - Add `GET /api/auth/csrf` (token mint).
  - Add CSRF validation dependency/middleware and apply it to all state-changing `/api/*` routes reachable from the UI.
- Auth server:
  - Add equivalent CSRF validation for `/enforceai/*` state-changing routes when cookie-authenticated.
- Nginx (if needed for header/cookie forwarding):
  - Ensure `X-CSRF-Token` is forwarded to upstreams for `/api/*` and `/enforceai/*`.

## Tests to add/update
- Unit tests:
  - CSRF: missing/invalid token rejects POST/PUT/PATCH/DELETE with `403` (cookie-auth).
  - CSRF: bearer/API-key authenticated requests continue working without CSRF header.
- Integration tests:
  - Update existing registry integration tests that POST without CSRF headers (add CSRF header setup).
  - Update EnforceAI management integration tests only if they move to cookie-auth in later phases (keep them token-based for now).

## Required test gate
- `uv run python -m py_compile` for changed Python files.
- `make test` (must pass).

---

# Phase 3 — EnforceAI DB: Users Directory + Local Password Users (data + stores)
**Goal**: add a persistent user directory and local user store (multi-user, roles), without yet changing the browser login flow.

## Deliverables
- DB schema + migrations (EnforceAI SQLite):
  - `users` table with: `user_id`, `auth_method`, `username`, `email`, `role` (`admin|user`), timestamps, `disabled_at`, `last_login_at`.
  - indexes for search by `email` and `username`.
- Store interfaces + SQLite implementations:
  - `UserStore` (create, update, disable, get by user_id, query by email/username, upsert-last-seen).
  - Password hashing utilities (Argon2id preferred; acceptable fallback to bcrypt if already in deps).
- Minimal service layer methods for:
  - create local user (admin-only)
  - disable user (admin-only)
  - reset/rotate password (admin-only)
  - upsert “first-seen” OIDC user record (user_id + email)

## Tests to add/update
- Unit tests:
  - migration up/down for the new tables.
  - `UserStore` CRUD/search semantics, uniqueness, disabled behavior.
  - password hash/verify correctness and non-leak logging checks.
- Integration tests:
  - add a data-layer integration test that initializes DB and can create/query a user record.

## Required test gate
- `uv run python -m py_compile` for changed Python files.
- `make test` (must pass).

---

# Phase 4 — Unified Session Issuer + Server-Side Session Invalidation (logout is real)
**Goal**: move from “self-contained signed cookie only” to a revocable session model that both services can validate.

## Deliverables
- Session storage:
  - add `sessions` table in EnforceAI DB:
    - `session_id`, `user_id`, `auth_method`, `created_at`, `expires_at`, `last_seen_at`, `revoked_at`, `revoked_reason`.
  - add store interface + sqlite impl: `SessionStore` (create, touch, revoke, lookup).
- Cookie payload changes:
  - cookie must include `session_id` and enough identity data for the UI (`user_id`, `email`, `username`, `roles/groups`, expiry).
  - both `registry` and `auth_server` must validate:
    - signature + expiry
    - session exists and is not revoked
- Logout semantics:
  - registry logout and auth_server logout revoke the session server-side (not only cookie deletion).
  - ensure logout works for both OIDC sessions and local password sessions.

## Tests to add/update
- Unit tests:
  - `SessionStore` create/touch/revoke.
  - cookie validation denies revoked sessions.
- Integration tests:
  - login simulation (or direct store + cookie creation) then logout then verify subsequent request is `401`.

## Required test gate
- `uv run python -m py_compile` for changed Python files.
- `make test` (must pass).

---

# Phase 5 — EnforceAI: Cookie-Session Management Auth + Admin Role Enforcement
**Goal**: EnforceAI management endpoints become usable from the UI without pasting tokens, while keeping existing token-based access for CLI/automation.

## Deliverables
- Add a “management auth” dependency for `/enforceai/*` that supports:
  - cookie session (primary for browser UI)
  - existing header-based credentials (OIDC bearer + X-Agent-Id, gateway token, API key) for non-browser clients
- OIDC admin role:
  - Treat `enforceai-admin` as the admin role/group (configurable mapping per issuer).
  - Persist admin-ness in the session payload and enforce it for admin endpoints.
- User directory population:
  - On cookie-session creation (OIDC login), upsert `{user_id, email, last_seen_at}` into the users table.
  - On local password login, upsert `{user_id="local|<username>", username, email, last_seen_at}`.

## Tests to add/update
- Integration tests:
  - `/enforceai/*` works with cookie session (no Authorization header).
  - Admin-only endpoints reject non-admin cookie sessions (`403`).
  - Existing token-based flows still work (no regression to `tests/integration/test_enforceai_management_routes.py`).

## Required test gate
- `uv run python -m py_compile` for changed Python files.
- `make test` (must pass).

---

# Phase 6 — EnforceAI Admin APIs (Cross-User)
**Goal**: implement the admin surface required by the UI (users directory + cross-user agent/credential operations) with auditable outcomes.

## Deliverables
- Admin user directory endpoints:
  - `GET /enforceai/admin/users?query=<email_or_username>`
  - `GET /enforceai/admin/users/{user_id}`
- Admin cross-user operations:
  - list/create/revoke agents for a `target_user_id`
  - create/revoke API keys for a `target_user_id`
  - revoke gateway tokens by `jti` (and revoke-all via `tokens_valid_after`) for a `target_user_id`
- Audit requirements:
  - Every admin action emits an audit event with `actor_user_id`, `target_user_id`, `action`, `outcome`, and `X-Request-Id`.

## Tests to add/update
- Unit tests:
  - management service methods enforce admin gating and “actor vs target” semantics.
- Integration tests:
  - happy path: admin can act on another user.
  - denial: non-admin cannot use admin routes.
  - audit event persistence path is exercised (best-effort DB write).

## Required test gate
- `uv run python -m py_compile` for changed Python files.
- `make test` (must pass).

---

# Phase 7 — Registry Admin Auth Migration (Remove HTTP Basic) + Break-Glass
**Goal**: remove all UI-facing HTTP Basic auth and replace it with unified admin auth; add break-glass credentials for emergency/automation.

## Deliverables
- Registry:
  - Replace HTTP Basic checks on internal/operator endpoints with unified admin authorization derived from the session/break-glass credential.
  - Ensure state-changing endpoints are CSRF-protected for cookie sessions (should already be true from Phase 2).
- Auth server:
  - Replace HTTP Basic on admin-only endpoints (e.g., scopes reload) with unified admin authorization.
- Break-glass:
  - Add an admin credential type (gateway token and/or API key) that grants admin capabilities:
    - auditable and revocable (store + revocation)
    - safe defaults (short-lived tokens)
  - Ensure the credential is not required for normal browser UI flows (OIDC/password + session is primary).

## Tests to add/update
- Integration tests:
  - registry internal endpoints deny non-admin cookie sessions.
  - admin cookie session allows.
  - break-glass credential allows without cookie session.
  - legacy Basic auth is rejected (or removed) depending on the migration choice.

## Required test gate
- `uv run python -m py_compile` for changed Python files.
- `bash -n` for any modified shell scripts (e.g., nginx entrypoint).
- `make test` (must pass).

---

## Post-plan validation (after Phase 7)
This is not a separate phase; it’s a final verification checklist:
- Bring up the compose stack and confirm `docker/nginx_rev_proxy_*.conf` exposes `/enforceai/*` to the browser.
- Verify cookie + CSRF behavior with a minimal UI smoke script (curl or a tiny Playwright check) without storing tokens.
- Confirm no long-lived secrets are returned by “me/config” endpoints.
