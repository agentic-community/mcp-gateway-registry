# Enforce Gateway UI Requirements (Phase 1)
*Created: 2025-12-15*

## Purpose
Define a complete, UI-focused requirements specification for a new Enforce Gateway (EnforceAI) web UI. This document is requirements-only (no implementation).

Related backend checklist: `docs/enforceai-ui-backend-changes.md`.

## Scope
This UI must fully replace the existing UI in `frontend/` (the old UI will be deprecated) and make the full gateway experience usable via the new UI.

This includes making EnforceAI Phase 1 functionality usable without the CLI:
- EnforceAI agent lifecycle management (create/list/get/update/revoke)
- Credential management (API keys, gateway tokens, token revocation, revoke-all)
- Visibility into FGAC behavior (scopes catalog + tool visibility concepts)
- Operator guidance (audit retention and operational troubleshooting)

And it includes replacing the existing Registry UI capabilities exposed under `/api/*`:
- MCP server management (list/register/edit/toggle/refresh/details/tools)
- A2A agent management (register/list/get/update/delete/toggle/discovery/semantic discovery)
- Token generation UI (`/api/tokens/generate`) for programmatic access where applicable
- Login/logout and “me” identity display

This UI is expected to interact with:
- EnforceAI management API served by `auth_server` under `/enforceai/*`
- Registry APIs served under `/api/*`

## Non-Goals (Phase 1 UI)
- No changes to enforcement semantics (`401/403/503`, fail-closed).
- No background jobs inside the UI (audit cleanup remains out-of-band).
- No requirement to edit server-side environment variables or secret files from the UI.
- Initial bootstrap flows (creating the first credential) can be assumed out-of-band initially.

## Terminology (avoid ambiguity)
This repo contains multiple concepts named “agent”.

- **EnforceAI Agent (identity/enforcement agent)**: A gateway-managed record used for authentication binding and FGAC (`agent_id`, `scopes`, optional `allowed_tools`). Managed via `auth_server` `/enforceai/*`.
- **A2A Agent (agent card / registry agent)**: An agent definition registered in the Registry (`/api/agents/*`).
- **MCP Server**: A registered server entry in the Registry (`/api/servers`, `/api/register`, etc.).

This UI requirements doc includes both EnforceAI and Registry capabilities.

## Personas
- **Developer**: Needs an agent + credential to use the gateway from tools like VS Code / Cursor / Claude Desktop.
- **Security Admin (self-service)**: Manages EnforceAI agents and credentials for their own `user_id` (scopes, `allowed_tools`, revocation).
- **Operator/Admin (cross-user)**: Cross-user administration across `user_id`s (support and incident response), plus system operational visibility.
- **Auditor (read-only)**: Reviews enforcement decisions and management actions (may be satisfied via log shipping in Phase 1, but the UI should support it when APIs exist).

## Information Architecture (Navigation)
Minimum navigation (left nav or tabs):
1. **Overview**
2. **Registry: Servers**
3. **Registry: A2A Agents**
4. **EnforceAI: Agents**
5. **EnforceAI: Credentials**
6. **Scopes and Policy**
7. **Tools (Discovery)**
8. **Audit**
9. **Admin**
10. **Settings**
11. **Help**

## Deployment Model (Standalone SPA)
Must:
- The UI is a standalone SPA (separately deployed from the backend services).
- The UI must allow configuring base URLs for the services it calls:
  - Registry base URL (for `/api/*`)
  - Auth Server base URL (for `/enforceai/*` and auth flows)

Should:
- Support per-environment config (dev/stage/prod) via build-time env vars, plus an in-app override for local testing.

## Authentication and Session Requirements
The UI must support authenticating to EnforceAI management APIs using exactly one credential source per request:
- OIDC: `Authorization: Bearer <oidc_jwt>` and `X-Agent-Id: <uuidv4>`
- Gateway token: `Authorization: Bearer <gateway_token>` or `X-Gateway-Token: <gateway_token>`
- API key: `X-API-Key: eak_<key_id>.<secret>`

The UI must also support the existing Registry UI authentication patterns:
- OIDC login (OAuth provider selection)
- Username/password login (non-IdP)

### Enterprise Security Defaults (best practice)
Given the UI is deployed on the same domain as the backends and must use a single session:
- Prefer a server-managed session cookie for browser authentication:
  - `HttpOnly`, `Secure` (HTTPS-only), `SameSite=Lax` (or `Strict` if compatible)
  - short idle timeout + absolute timeout
  - session invalidation on logout and credential rotation
- Enforce CSRF protection for all state-changing operations when using cookies:
  - double-submit cookie or CSRF token endpoint + `X-CSRF-Token` header
- Do not store long-lived tokens in browser storage (`localStorage`).
- Use token-based auth (gateway token / API key) primarily for non-browser clients and break-glass scenarios.
- For username/password auth:
  - store password verifiers with a modern KDF (Argon2id preferred; bcrypt acceptable)
  - rate-limit login attempts and implement lockouts/backoff
  - support admin-initiated password reset/rotation

### Unified Session (single session across Registry + EnforceAI)
Must:
- A single authenticated browser session must authorize both:
  - Registry operations (`/api/*`)
  - EnforceAI operations (`/enforceai/*`)
- The UI must not require users to paste/manage additional credentials after login in the normal case.
- If EnforceAI still requires an agent binding (`agent_id`) for OIDC-based calls, the UI must support selecting a “current agent context” once per session.

### Session Model
- The UI must allow configuring a **Management API Base URL** (default `http://localhost:8888`).
- The UI must allow configuring a **Registry API Base URL** (default `http://localhost`).
- The UI must allow the user to choose **credential type** (OIDC bearer, gateway token, API key).
- The UI must support storing session state in-browser (at minimum, in-memory; optionally `sessionStorage`) with a “Clear session” action.
- The UI must never write credentials to logs.
- The UI must never persist credentials in long-lived storage by default (no `localStorage` unless explicitly enabled with a clear warning).

### Request Correlation
- The UI should generate a client-side request id per API call and send `X-Request-Id` to help correlate audit events.

### Error Handling (must be consistent)
- `401`: show “Authentication failed” with actionable next steps (missing header, malformed token/key, unknown issuer, expired).
- `403`: show “Forbidden” with the specific scenario when safe (ownership mismatch, revoked agent/key/token, missing/invalid `X-Agent-Id`).
- `503`: show “Enforcement dependency unavailable” with retry guidance and operator pointers (DB unavailable, keyring misconfigured).

## Feature Requirements

### 1) Overview
**Goal**: immediate confirmation that the UI can reach the management API and that the current credential is valid.

Must:
- Show configured base URL and credential type (without revealing secrets).
- Provide “Test connection” that performs a safe read operation (recommended: `GET /enforceai/agents`) and reports:
  - success/failure
  - HTTP status and error message
  - elapsed time
- If the server indicates management is not enabled (for example, missing `ENFORCEAI_DB_PATH` resulting in route not mounted), show an explicit message that EnforceAI management must be enabled on the Auth Server.
- Provide quick links to:
  - Create agent
  - Mint token
  - Create API key
  - Revoke all tokens for an agent

Must (Registry):
- Show a “Registry reachable” indicator (recommended: `GET /api/servers`).
- Show summary counts:
  - number of MCP servers
  - number of A2A agents

Should:
- Summarize counts: number of agents; number of revoked agents; number of API keys (requires iterating agents unless a dedicated endpoint exists).

### 2) Registry: Servers
**Goal**: fully replace the existing dashboard server management UX.

Must support these Registry actions (existing backend endpoints):
- List servers: `GET /api/servers`
- Register server: `POST /api/register`
- Edit server: `POST /api/edit/{service_path:path}`
- Toggle server: `POST /api/toggle/{service_path:path}`
- Refresh server health/tools: `POST /api/refresh/{service_path:path}`
- Server details: `GET /api/server_details/{service_path:path}`
- Server tools: `GET /api/tools/{service_path:path}`

Must:
- Provide search/filter by name/path/tags and enabled/disabled.
- Provide create/edit forms with validation aligned to server fields (name/path/proxy_pass_url/tags/description/etc).
- Provide clear permission gating in the UI (disable actions the user cannot perform; show “why”).

Should:
- Provide bulk operations (toggle multiple servers, refresh multiple servers) if backend supports it later.

### 3) Registry: A2A Agents
**Goal**: fully replace existing A2A agent management and discovery UX.

Must support these Registry actions (existing backend endpoints):
- Register agent: `POST /api/agents/register`
- List agents: `GET /api/agents`
- Get agent: `GET /api/agents/{path:path}`
- Update agent: `PUT /api/agents/{path:path}`
- Delete agent: `DELETE /api/agents/{path:path}`
- Toggle agent: `POST /api/agents/{path:path}/toggle`
- Discovery: `POST /api/agents/discover`
- Semantic discovery: `POST /api/agents/discover/semantic`

Must:
- Provide forms for agent registration/update aligned to the API schema (agent card fields, skills, tags, visibility).
- Provide list/search/filter (query, enabled_only, visibility).
- Provide enable/disable control and status display.

### 4) EnforceAI: Agents (identity/enforcement)
**APIs**:
- `GET /enforceai/agents`
- `POST /enforceai/agents`
- `GET /enforceai/agents/{agent_id}`
- `PATCH /enforceai/agents/{agent_id}`
- `POST /enforceai/agents/{agent_id}/revoke`
- `POST /enforceai/agents/{agent_id}/tokens/revoke-all`

#### 4.1 Agent List
Must:
- Display a table of agents owned by the authenticated `user_id`.
- Columns (minimum):
  - `alias`
  - `agent_id`
  - `scopes` (count + expandable list)
  - `allowed_tools` (count + expandable list)
  - `revoked_at`
  - `tokens_valid_after`
  - `updated_at`
- Provide actions per row:
  - View details
  - Edit
  - Revoke agent
  - Revoke all tokens
- Provide search/filter:
  - by alias, agent_id
  - filter revoked vs active

Should:
- Provide a “Copy agent_id” action.
- Display warnings for:
  - agents with empty scopes (effectively deny all)
  - agents with `allowed_tools` set but empty (effectively deny all tools)

#### 4.2 Agent Create
Must:
- Form fields:
  - `alias` (optional)
  - `scopes` (required, non-empty list)
  - `allowed_tools` (optional list)
  - `metadata` (optional JSON object)
- Validate `scopes` against the server-side scope catalog (surface “Unknown scopes” errors cleanly).
- Provide a “Save” action that calls `POST /enforceai/agents`.
- After create, show the created record and highlight the generated `agent_id`.

Should:
- Provide a scope picker UI sourced from the scope catalog view (see “Scopes and Policy”).
- Provide an “allowed tools picker” sourced from tool discovery (see “Tools (Discovery)”).

#### 4.3 Agent Update
Must:
- Allow editing:
  - `scopes`
  - `allowed_tools`
  - `alias`
  - `metadata`
- Use `PATCH /enforceai/agents/{agent_id}`.
- Show diff-like confirmation before applying changes (especially when removing scopes).

#### 4.4 Agent Revoke
Must:
- Confirm destructive action (revocation cannot be undone in Phase 1).
- Call `POST /enforceai/agents/{agent_id}/revoke`.
- After revoke, reflect `revoked_at` and disable token/key creation actions for that agent.

#### 4.5 Revoke All Tokens (bulk)
Must:
- Confirm action and explain effect: denies any token with `iat < tokens_valid_after`.
- Call `POST /enforceai/agents/{agent_id}/tokens/revoke-all`.
- Display the updated `tokens_valid_after` value.

### 5) EnforceAI: Credentials

#### 5.1 API Keys
**APIs**:
- `POST /enforceai/agents/{agent_id}/api-keys`
- `GET /enforceai/agents/{agent_id}/api-keys`
- `POST /enforceai/api-keys/{key_id}/revoke`

Must:
- Provide per-agent API key list with:
  - `key_id`
  - `scopes` (optional)
  - `expires_at`
  - `revoked_at`
  - `created_at`
  - `last_used_at`
- Provide API key creation form:
  - optional `scopes` (must not elevate beyond agent scopes; show subset guidance)
  - optional `expires_at`
- On creation, show the returned secret materials exactly once:
  - `api_key_value` (full `eak_<key_id>.<secret>` value)
  - `key_id`
  - `secret`
- Force an explicit acknowledgement step: “I have copied/stored this secret” before allowing navigation away.
- Provide revoke action per key with confirmation.

Should:
- Offer “Copy api_key_value” and “Download as .txt” (download is local-only).
- Offer a short “How to use” snippet (header name and format) with copy button.
- Offer “Export agent credentials instructions” that never includes secrets, only header names and where to paste values.

#### 5.2 Gateway Tokens (minting)
**API**:
- `POST /enforceai/agents/{agent_id}/tokens/mint`

Must:
- Provide mint form:
  - target `agent_id`
  - `scopes` (required; must be subset of agent scopes; show failures clearly)
  - expiry selection:
    - either `ttl_seconds` (positive integer) or `expires_at` (datetime)
    - must enforce mutual exclusivity
- On success, show the token exactly once and strongly warn the user to store it securely.
- Provide a “Decode (local only)” view that shows header and claims without sending the token back to the server (optional but recommended).

Should:
- Offer presets for TTL (1 hour, 1 day, 7 days, 30 days, custom).
- Offer “Copy as Authorization header” helper (produces `Authorization: Bearer <token>` locally).

#### 5.3 Token Revocation (by `jti` or by pasting token)
**API**:
- `POST /enforceai/tokens/revoke`

Must:
- Support revoking:
  - by providing `agent_id` + `jti`, or
  - by pasting a `gateway_token` (server extracts claims)
- Provide optional `reason`.
- Show the returned `TokenRevocationRecord`:
  - `jti`, `agent_id`, `revoked_at`, optional `expires_at`, optional `reason`
- Prevent accidental leakage:
- do not persist the pasted token
- do not include token in error messages

### 6) Scopes and Policy
**Goal**: make scope assignment understandable and safe.

Must:
- Display the scope catalog content (from the configured server-side catalog path; if no API exists, this section must clearly state it cannot be loaded from the server yet).
- For each scope, show:
  - scope name
  - allowed servers and methods
  - tool restrictions (including `all`/`*` semantics)
- Provide search and filter across scope names and server names.

Should:
- Provide a “What does this scope allow?” explainer that maps to the gateway actions:
  - `tools/list` visibility (filtered)
  - `tools/call` execution

### 7) Tools (Discovery)
**Goal**: help users correctly configure `allowed_tools` and understand what an agent can call.

Must:
- List available MCP servers and their tools from the Registry APIs:
  - `GET /api/servers`
  - `GET /api/tools/{service_path:path}`
- Provide per-server tool lists (tool names).

Should:
- Provide an “Allowed tools builder” that:
  - lets the user select tool names and applies them to an agent’s `allowed_tools`
  - warns when the same tool name appears in multiple servers (because `allowed_tools` is currently tool-name-only)
- Provide an “Effective access preview” for a selected agent:
  - visible tools per server (as approximated by scope catalog + tool list)
  - explicit disclaimer that final enforcement is server-side

### 8) Audit
Phase 1 audit sources:
- stdout JSON lines with `event_type="enforceai_audit"`
- SQLite `audit_events` persisted best-effort at `ENFORCEAI_DB_PATH`

Must:
- Provide guidance on how to find audit events (for Docker Compose and local runs).
- Provide filtering guidance by `X-Request-Id` correlation.
- Provide “Common audit actions” glossary:
  - `tools/list`, `tools/call`
  - `management/agents/*`, `management/api-keys/*`, `management/tokens/*`

Should (if a future API is added):
- Provide an audit event viewer with filtering by time, agent_id, outcome, action, request_id.

### 9) Admin
**Goal**: enable cross-user/operator administration with strong safety controls.

#### Admin Authorization (Phase 1)
Must support both:
- **OIDC-admin (human admins)**: admin access granted based on IdP group/role membership (least privilege, centralized offboarding).
- **Admin break-glass credential (automation/emergency)**: a dedicated EnforceAI credential (gateway token and/or API key) with admin privileges, short TTLs where possible, strict audit, and clear operational controls.

Must:
- Provide a distinct “admin mode” UX that is visibly different (to reduce accidental cross-user actions).
- Require explicit selection of a target `user_id` context before showing cross-user resources (agents/keys/tokens/audit).
- Require confirmation dialogs for destructive operations (revoke agent/key/token, revoke-all).
- Provide auditability for admin actions (surface `X-Request-Id` usage and ensure admin actions produce management audit events).
- Provide core cross-user operations (Phase 1):
  - View users (searchable directory)
  - Create EnforceAI agent for a selected user
  - Revoke EnforceAI agent for a selected user
  - Revoke EnforceAI API key for a selected user
  - Revoke EnforceAI gateway token (`jti`) for a selected user
  - Delete A2A agent records (Registry) as needed for incident response

Should:
- Provide guardrails for irreversible operations:
  - “type the agent_id/path to confirm”
  - “reason required” for admin revocations (captured in audit details)
- Provide “break-glass” mode for emergency revocations with extra confirmation and prominent labeling.

Should:
- Support “act-as” vs “direct admin action” distinction (display “acting as user X” vs “admin action on user X”).
- Provide rate-limited bulk actions (if/when backend supports it), with previews and exports.

Note: cross-user admin operations will likely require additional backend API surface beyond the current self-service `/enforceai/*` endpoints (ownership is currently enforced by `user_id`).

#### User Directory (Admin View)
Goal: admins must be able to view/search users in a human-comprehensible way (UUID-only identifiers are insufficient).

Must:
- Provide an admin “Users” view with:
  - search by username/email/display name (partial match)
  - display canonical `user_id` as a secondary field (copyable, but not primary)
  - show `auth_method`/issuer/provider when available
- The system must maintain a minimal “user directory” data source that supports the UI:
  - For OIDC users: capture `iss`, `sub`, and a preferred human identifier (email and/or preferred_username) on first-seen and update `last_seen_at`.
  - For username/password users: capture `username`, assign `user_id = "local|<username>"`, and update `last_seen_at`.
  - Store must support stable lookup from human identifier → canonical `user_id`.

### 10) Settings
Must:
- Base URL configuration for management API.
- Credential type selection and secure entry UI.
- OIDC mode support for specifying `X-Agent-Id` header value for management calls.
- “Clear session” action.

Should:
- Provide environment/configuration checklist (read-only) for operators:
  - `ENFORCEAI_DB_PATH`, `ENFORCEAI_AUTH_PROVIDER`, `ENFORCEAI_SCOPES_CATALOG_PATH`
  - token keyring paths when gateway tokens are enabled
  - pepper path when API keys are enabled

### 11) Help
Must:
- Link to:
  - `docs/enforceai-setup-guide.md`
  - `enforceai/instructions/ENFORCEAI_MANAGEMENT.md`
  - `enforceai/instructions/ENFORCEAI_AUDIT_RETENTION.md`
- Provide a troubleshooting decision tree for `401/403/503`.

## Security Requirements
Must:
- Never log or persist secrets by default (tokens, API key secrets, pasted gateway tokens).
- Mask secrets on screen by default with explicit “reveal” actions.
- Ensure copy-to-clipboard is deliberate and does not auto-copy.
- Prevent accidental secret exposure in URLs (no query-string credentials).
- Provide clear warnings on “show once” materials (API key secret and minted token).

Should:
- Provide a configurable “safe mode” that disables token reveal and only allows “copy” (for screen-sharing).

## Accessibility and UX Requirements
Must:
- Keyboard navigable forms and dialogs.
- Screen-reader compatible labels for all inputs.
- Clear empty states and error states.

Should:
- Responsive layout (desktop-first, works on tablet).

## Decisions (confirmed)
1. The new UI replaces the existing UI; the old UI will be deprecated.
2. The UI supports OIDC and non-IdP username/password login.
3. The UI is a standalone SPA.
4. Initial bootstrap can be assumed out-of-band initially.
5. The UI must include a cross-user/operator admin surface.
6. The UI is hosted on the same domain as the backends (same-site).
7. The UI uses a single session across Registry and EnforceAI surfaces.
8. Admin authorization supports both OIDC-admin and a dedicated admin break-glass credential.
9. For EnforceAI agents, revoke (soft-delete) is sufficient for Phase 1.
10. Admin “user directory” primary display identifier is `email` (OIDC claim).
11. Registry internal/operator actions must be migrated to unified admin auth (no HTTP Basic in the new UI).
12. Admin role is granted by OIDC group/role `enforceai-admin`.
13. Username/password auth supports multiple users with admin and non-admin roles.
14. Username/password users are admin-provisioned only (no self-service signup).
15. Username/password users must have a mandatory `email`.
16. Admin operations use “admin acting on a target user” semantics (no impersonation in Phase 1).
17. Admin break-glass supports both gateway tokens and API keys (gateway token preferred default; API key for automation scenarios).

## Future Improvements
1. Add additional restrictions for admin break-glass credentials (recommended):
   - IP allowlist, explicit “reason required”, short TTL, or time-based enablement

## Open Issues (Backend Dependencies)
These are not UI design questions, but requirements-impacting backend work that must exist for the UI to be fully functional.

1. **Single unified session across services**: `auth_server` EnforceAI endpoints currently authenticate via bearer/API-key style credentials; to meet “single session”, `auth_server` must accept the same browser session identity used by the Registry (or both must trust a shared session issuer).
2. **Cross-user admin APIs**: current EnforceAI management endpoints are ownership-scoped by `user_id`; cross-user create/revoke/delete requires new admin-only endpoints and an admin permission model.
3. **User directory source of truth**: add a minimal user directory store (likely in the EnforceAI DB) populated on first-seen from OIDC (`email`) and from username/password logins (`username`), so Admin → Users is searchable and comprehensible.
4. **Registry operator endpoints**: Registry internal/operator actions currently protected by HTTP Basic must be migrated to unified admin auth and audited.
5. **Auth parity for UI flows**: some Registry APIs are JWT-bearer oriented today; for a cookie-session UI, either (a) add cookie-auth equivalents, or (b) vend short-lived UI-scoped tokens server-side without exposing long-lived secrets to the browser.
6. **Local user management**: add a persistent user store for username/password identities (multiple users, admin vs non-admin role), with secure password hashing, resets, and audit logging.

## Appendix B: Registry Internal Admin (HTTP Basic) vs Unified Admin Auth
The Registry currently exposes some “internal/operator” endpoints protected by HTTP Basic Auth (admin username/password in a header).

### Meaning
- **Keep HTTP Basic (as-is):**
  - UI would prompt for an operator username/password (or accept a pre-configured operator secret) specifically for these endpoints.
  - This is separate from OIDC-admin and EnforceAI-admin credentials.
- **Migrate to unified admin auth (preferred):**
  - Replace/augment those endpoints so they accept the same admin identity used everywhere else (OIDC-admin session and/or EnforceAI admin break-glass credential).
  - UI uses the same session/credential model consistently across Admin features.

### Pros/Cons
- Keep HTTP Basic:
  - Pros: minimal backend changes; fast path if you only need a few operator endpoints.
  - Cons: shared secret handling; weaker audit identity (often “admin”); poor least-privilege; harder rotation; encourages credential reuse; separate auth path in UI; more CSRF/UX pitfalls.
- Migrate to unified admin auth:
  - Pros: consistent enterprise posture; centralized access control (IdP groups); better auditing (who did what); supports fine-grained admin permissions; easier user lifecycle/offboarding; aligns with “single session”.
  - Cons: requires backend work to refactor/extend endpoints; requires defining admin scopes/permissions; needs careful testing to avoid breaking operator workflows.

### Phase 1 Requirement
- The new UI must not require or accept HTTP Basic credentials for admin/operator operations.
- Any Registry admin/operator operations needed in the UI must be available via unified admin auth (OIDC-admin session and/or admin break-glass credential).


## Appendix A: Management API Contract (Phase 1)
This section is included so UI work can map fields and validation rules directly to the implemented API.

### Endpoints
- Agents:
  - `GET /enforceai/agents`
  - `POST /enforceai/agents`
  - `GET /enforceai/agents/{agent_id}`
  - `PATCH /enforceai/agents/{agent_id}`
  - `POST /enforceai/agents/{agent_id}/revoke`
  - `POST /enforceai/agents/{agent_id}/tokens/revoke-all`
- API keys:
  - `POST /enforceai/agents/{agent_id}/api-keys`
  - `GET /enforceai/agents/{agent_id}/api-keys`
  - `POST /enforceai/api-keys/{key_id}/revoke`
- Tokens:
  - `POST /enforceai/agents/{agent_id}/tokens/mint`
  - `POST /enforceai/tokens/revoke`

### Models (response bodies)
- `AgentRecord`
  - `user_id: string` (canonical)
  - `agent_id: string` (UUIDv4)
  - `scopes: string[]`
  - `allowed_tools: string[] | null`
  - `alias: string | null`
  - `metadata: object | null`
  - `revoked_at: string(datetime) | null`
  - `tokens_valid_after: string(datetime) | null`
  - `created_at: string(datetime)`
  - `updated_at: string(datetime)`
- `ApiKeySummary`
  - `key_id: string`
  - `user_id: string`
  - `agent_id: string`
  - `scopes: string[] | null`
  - `expires_at: string(datetime) | null`
  - `revoked_at: string(datetime) | null`
  - `created_at: string(datetime)`
  - `last_used_at: string(datetime) | null`
- `CreateApiKeyResponse` (secret returned once)
  - `key_id: string`
  - `secret: string`
  - `api_key_value: string` (full `eak_<key_id>.<secret>`)
- `MintTokenResponse` (token returned once)
  - `token: string`
- `TokenRevocationRecord`
  - `jti: string`
  - `user_id: string`
  - `agent_id: string`
  - `revoked_at: string(datetime)`
  - `expires_at: string(datetime) | null`
  - `reason: string | null`

### Models (request bodies)
- `CreateAgentRequest`
  - `scopes: string[]` (required, non-empty)
  - `allowed_tools: string[] | null`
  - `alias: string | null`
  - `metadata: object | null`
- `UpdateAgentRequest`
  - `scopes: string[] | null`
  - `allowed_tools: string[] | null`
  - `alias: string | null`
  - `metadata: object | null`
- `CreateApiKeyRequest`
  - `scopes: string[] | null`
  - `expires_at: string(datetime) | null`
- `MintTokenRequest`
  - `scopes: string[]` (required, non-empty)
  - `ttl_seconds: number | null` (mutually exclusive with `expires_at`)
  - `expires_at: string(datetime) | null` (mutually exclusive with `ttl_seconds`)
- `RevokeTokenRequest`
  - Option A: `gateway_token: string` and optional `reason`
  - Option B: `agent_id: string` + `jti: string` and optional `reason`
