# Enforce Gateway UI — Backend Changes Required
*Created: 2025-12-15*

## Purpose
Capture the backend changes required to support the new Enforce Gateway UI requirements (`docs/enforceai-ui-requirements.md`). This document is an implementation checklist/contract for backend work (not UI work).

## Scope
Applies to:
- `registry` service (`/api/*`, login/session, registry management)
- `auth_server` service (`/enforceai/*`, EnforceAI identity/management)
- Nginx gateway wiring where needed (session/header forwarding, CSRF, etc.)

## High-Level Outcomes Required
1. Single unified browser session across Registry + EnforceAI (same-site cookie session).
2. Multiple username/password users with admin and non-admin roles.
3. OIDC admin role granted by IdP group/role `enforceai-admin`.
4. Admin break-glass credential supported (gateway token and/or API key) for emergency/automation.
5. Cross-user admin APIs for both Registry and EnforceAI (no HTTP Basic anywhere in the new UI).
6. Admin “Users directory” searchable by email/username (not by canonical UUID-style identifiers).

## 1) Unified Session Across `registry` and `auth_server`
### 1.1 Shared session model (cookie)
Backend must support a single browser session that authorizes:
- Registry endpoints under `/api/*`
- EnforceAI endpoints under `/enforceai/*`

Minimum requirements:
- A shared cookie name and schema for session payload.
- A shared signing/verification strategy:
  - Either both services share a signing secret (not ideal operationally), or
  - A single session issuer (preferred) and both services validate via that issuer.

Session payload must include at least:
- `auth_method`: `oidc` | `password`
- `user_id` (canonical):
  - OIDC: `"<iss>|<sub>"`
  - password users: `"local|<username>"` (decision)
- `email` (for OIDC; primary admin display identifier)
- `username` (for password users; may also exist for OIDC as preferred_username)
- `is_admin` or `roles`/`groups` containing `enforceai-admin`
- Expiry timestamps and session id for invalidation

Cookie security:
- `HttpOnly`, `Secure`, `SameSite=Lax` (or `Strict` if feasible), path `/`.
- Idle timeout + absolute timeout.
- Logout invalidates session server-side (not cookie deletion only).

### 1.2 CSRF protection (cookies + SPA)
All state-changing endpoints reachable from the UI must be CSRF-protected.

Required backend support (choose one approach and standardize):
- CSRF token endpoint: `GET /api/auth/csrf` returns a token; SPA sends `X-CSRF-Token` on all non-GET.
- Or double-submit cookie strategy.

The backend must reject state-changing requests missing/invalid CSRF token.

### 1.3 Session-to-EnforceAI identity mapping
`auth_server` must accept the session identity and construct an EnforceAI identity context for management/admin endpoints.

Key requirements:
- Management endpoints should not require users to paste bearer tokens in the UI.
- If OIDC mode still requires an agent binding (`X-Agent-Id`) for some EnforceAI actions, the backend must support a “current agent context” header that the UI can set once per session.
- Admin endpoints must not require `X-Agent-Id` binding for cross-user operations (admin operates by `target_user_id`).

## 2) Username/Password Users (Multiple Users + Roles)
Current state is single `admin_user`/`admin_password` style validation; this must be replaced.

Backend must add:
- Persistent local user store (likely in the EnforceAI SQLite DB for Phase 1 simplicity).
- Fields:
  - `username` (unique)
  - `email` (mandatory)
  - `password_hash` (Argon2id preferred; bcrypt acceptable)
  - `role`: `admin` | `user` (at minimum)
  - `created_at`, `updated_at`, `last_login_at`, `disabled_at` (recommended)
- Login flow:
  - Rate limiting + backoff/lockout
  - Audit logging for success/failure (no secrets)
- Management:
  - Admin-only provisioning (no self-service signup)
  - Admin can create/disable users and rotate/reset passwords
  - Non-admin cannot create users

Canonical identity:
- password user `user_id` must be `local|<username>` to conform to the existing `'<iss>|<sub>'` validation pattern.

## 3) Admin Authorization (OIDC + Break-Glass)
### 3.1 OIDC-admin (primary)
Backend must grant admin access when the authenticated user is in IdP group/role:
- `enforceai-admin`

Requirements:
- OIDC claim mapping must be configurable per issuer (which claim contains groups/roles).
- Admin status must be recorded in the session payload (`is_admin` or `groups`).

### 3.2 Admin break-glass credential (secondary)
Backend must support a dedicated admin credential usable without browser OIDC:
- Both gateway token and API key, granting admin capabilities.

Phase 1 requirements:
- Admin credential must be auditable and revocable.
- Admin credential must not require UI to store long-lived secrets persistently.

Preferred default usage:
- Use an admin **gateway token** as the primary break-glass mechanism (short-lived, revocable by `jti`, standard bearer transport).
- Use an admin **API key** for automation/integration scenarios where API keys are operationally preferred, with strict rotation and revocation practices.

Future improvement (explicitly not required immediately, but must be supported later):
- Restrict break-glass credentials by IP allowlist / “reason required” / short TTL / time-box enablement.

## 4) EnforceAI Admin APIs (Cross-User)
Current `/enforceai/*` endpoints are ownership-scoped by `user_id`. Add an admin API surface that allows cross-user operations.

### 4.1 User directory APIs (admin)
Backend must provide a searchable “Users” directory for admins.

Minimum endpoints (example shapes; names can vary):
- `GET /enforceai/admin/users?query=<email_or_username>`
  - returns list of users with: `user_id`, `email`, `username`, `auth_method`, `last_seen_at`
- `GET /enforceai/admin/users/{user_id}`
  - returns a single user record + summaries (agent counts, etc.)

Population rules:
- On first-seen login for OIDC users, upsert `{iss, sub, user_id, email, last_seen_at}`.
- On password login, upsert `{username, user_id="local|<username>", last_seen_at}`.

### 4.2 Cross-user EnforceAI management endpoints (admin)
Required Phase 1 operations (cross-user):
- View user’s EnforceAI agents
- Create agent for a user
- Revoke agent for a user
- Revoke API key for a user
- Revoke gateway token by `jti` for a user

These actions must:
- Require admin authorization (OIDC-admin session or break-glass admin credential).
- Be fully audited (stdout + DB audit store) with `X-Request-Id`.
- Avoid requiring `X-Agent-Id` binding (admin operates on a `target_user_id`).
- Use “admin acting on target” semantics (no impersonation/act-as in Phase 1).

## 5) Registry Admin Auth Migration (Remove HTTP Basic)
Registry currently has internal/operator endpoints protected by HTTP Basic.

Backend must:
- Migrate those endpoints to unified admin auth (same session/admin identity).
- Remove any UI dependency on Basic Auth credentials.
- Ensure admin operations are auditable (who did what) and CSRF protected.

## 6) Registry + EnforceAI Auth Parity for SPA
Some APIs are currently cookie-session oriented, some are JWT-bearer oriented.

Backend must provide one consistent story for SPA calls:
- Prefer cookie-session authentication for browser calls (same-site).
- If JWT-bearer endpoints must remain, provide a server-side token vending mechanism that:
  - issues short-lived UI tokens tied to the session
  - never exposes long-lived secrets to the browser

## 7) Audit and Observability for Admin Operations
Backend must ensure that all admin actions (cross-user changes) produce audit events including:
- actor identity (`actor_user_id`, and whether actor is admin)
- target identity (`target_user_id`, target `agent_id`/`key_id`/`jti`)
- action name (stable string)
- outcome (allow/deny)
- request id (`X-Request-Id`)
- reason when provided (future: “reason required”)

## 8) Data Layer Changes (Phase 1: SQLite)
Given Phase 1 uses SQLite:
- Extend the EnforceAI DB schema to include:
  - `users` (directory + local password users)
  - optional `user_aliases` or indexed email/username fields
- Ensure migrations exist (upgrade/downgrade where supported).

## Questions / Clarifications Needed
No open questions; the Phase 1 requirements above reflect current decisions.
