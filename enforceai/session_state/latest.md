# Session State — Latest

## Last Completed Work
- Hardened EnforceAI settings parsing to treat blank optional path env vars (e.g., `ENFORCEAI_UPSTREAM_KEK_PATH=` from docker-compose) as unset, preventing startup failures when optional secret paths are configured as empty strings.
- Repo initialized
- Base architecture files created
- Persistence backend for phase 1 selected
- Enforce GW UI backend Phase 1: added shared session cookie schema (`gateway_session.py`), updated registry `/api/auth/me` to include `user_id`/`session_id`/`email`, updated auth_server OAuth2 cookie issuance to include v1 fields, and added tests (full suite passing)
- Enforce GW UI backend Phase 2: added CSRF token minting (`GET /api/auth/csrf`) + CSRF enforcement for cookie-authenticated state-changing requests in `registry` and `auth_server` (for `/enforceai/*`), wired Nginx forwarding for `/enforceai/*` and `X-CSRF-Token`, and added tests (full suite passing)
- Enforce GW UI backend Phase 3: added EnforceAI users directory persistence (`0002_users` migration), `UserRecord` model, `UserStore` + `SqliteUserStore`, and scrypt-based password hashing utility, with updated tests (full suite passing)
- Enforce GW UI backend Phase 4: added revocable sessions persistence (`0003_sessions` migration), `SessionRecord` + `SessionStore`, registry/auth_server session validation + logout revocation, and an integration test verifying revoked sessions are rejected (full suite passing)
- Enforce GW UI backend Phase 5: enabled cookie-session auth for `/enforceai/*` management routes (no bearer/token required), added admin role check endpoint (`GET /enforceai/admin/ping`), and added integration coverage (full suite passing)
- Enforce GW UI backend Phase 6: added EnforceAI admin APIs (users directory + cross-user operations) under `/enforceai/admin/*`, including user search/get, cross-user agent create/revoke, cross-user API key create/revoke, and gateway token revocation by `jti`, with integration coverage (full suite passing)
- Enforce GW UI backend Phase 7: migrated registry `/api/internal/*` operator endpoints off HTTP Basic to unified admin auth (`nginx_proxied_auth`), updated `/api/servers/*` wrappers to pass user context, and added integration coverage (full suite passing)
- Stage 0.1: created `auth_server/enforceai/` package skeleton with core contracts + unit tests
- Stage 0.2: added env-driven EnforceAI settings parsing + validation with unit test coverage
- Stage 0.3: added reusable pytest fixtures (RSA keys, temp SQLite, env helpers) with unit test coverage
- Brought upstream tests into alignment with the current app routing contract (SPA at `/`, APIs under `/api`, Anthropic server-registry under `/{REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION}`)
- Full repo test suite passing (coverage gate adjusted to current baseline)
- Stage 1.1: implemented SQLite migration runner + baseline schema migration with unit tests
- Stage 1.2: implemented AgentStore interface + SQLite AgentStore with unit tests
- Stage 1.3: implemented ApiKeyStore interface + SQLite ApiKeyStore with unit tests
- Stage 1.4: implemented RevocationStore interface + SQLite RevocationStore with unit tests
- Stage 1.5: implemented AuditStore interface + SQLite AuditStore with unit tests
- Stage 1.6: added data layer initializer + shared SQLite connection pragmas + end-to-end smoke test
- Added a lightweight EnforceAI integration test covering settings -> data layer -> store roundtrip (still no request-path wiring)
- Stage 2.1: added gateway token claims model + time/validation helpers with unit tests
- Stage 2.2: added gateway keyring loader (PEM load, `kid` selection, caching) with unit tests
- Stage 2.3: added gateway token minting (RS256 + `kid` header + required claims) with unit tests
- Stage 2.4: added gateway token verification (RS256 signature + `kid` key selection + claims validation) with unit tests
- Stage 2.5: added hardening tests (no secret/token leaks + clock-skew regression cases)
- Added an integration-lite test for gateway token mint+verify roundtrip (no request-path wiring)
- Stage 3.1: updated OIDC issuer config schema (jwks_uri, audiences, claim precedence, TTL/skew knobs) + updated fixtures/tests; maintained legacy `jwks_url`/`audience` parsing as aliases
- Stage 3.2: implemented OIDC claim normalization helpers (aud/scopes/roles) with unit test coverage (no network)
- Stage 3.3: implemented JWKS fetch + in-memory cache per issuer (TTL-based, refresh, fail-closed on fetch errors) with unit tests (no network; injected fetcher)
- Stage 3.4: implemented generic OIDC JWT verification (multi-issuer selection, JWKS key selection with refresh-on-missing-kid, signature/aud/exp validation, iat skew check) with unit tests
- Stage 3.5: added OIDC hardening tests (malformed/missing claims, no token/JWKS leakage) and an integration-lite OIDC roundtrip test that validates verifier + cache behavior without network
- Created Stage 4 phased plan: `enforceai/plans/stage-4-identity-resolver-phased-plan.md`
- Stage 4.1: implemented credential extraction + multi-credential ambiguity rejection with unit tests
- Stage 4.2: implemented API key provider (pepper loading, key verify, effective scopes) with unit tests
- Stage 4.3: implemented gateway token provider (verify, revocation checks, agent binding, effective scopes) with unit tests
- Stage 4.4: implemented OIDC provider (OIDC verify + X-Agent-Id binding + agent scopes) with unit tests
- Stage 4.5: implemented IdentityResolver orchestration (mode selection + mixed routing) with unit tests
- Stage 5.1: implemented FGAC scope catalog loader + caching (`auth_server/scopes.yml`) with unit tests
- Stage 5.2: implemented FGAC evaluator (scopes + allowed_tools) with unit tests
- Stage 5.3: implemented FastAPI dependency wiring (IdentityContext + ScopeCatalog once per request) with unit tests
- Stage 5.4: added tool visibility filtering for MCP `tools/list` via nginx `body_filter_by_lua` using per-request `X-Allowed-Tools` allowlist from auth_server
- Stage 5.5: enforced MCP `tools/call` authorization in auth_server `/validate` (deny on forbidden, 503 on dependency issues), added best-effort audit emission (stdout + SQLite AuditStore), and refactored EnforceAI imports to support both repo and Docker auth-server module layouts
- Stage 5.6: added multi-auth integration coverage (OIDC, gateway-token, api-key, mixed bearer routing) for Stage 5 enforcement behavior and fixed auth_server `/validate` to honor `ENFORCEAI_SCOPES_CATALOG_PATH` / `SCOPES_CATALOG_PATH`
- Stage 6.1: added management service layer (`auth_server/enforceai/management/`) for agent CRUD, API key lifecycle, gateway token minting, and token revocation with strict ownership enforcement + unit tests
- Stage 6.2: added management API routes (`/enforceai/*`) in `auth_server` with best-effort audit emission and integration coverage for OIDC, gateway-token, and api-key auth
- Stage 6.3: added `cli/enforceai_cli.py` (argparse + httpx) for self-service management operations with unit arg/header tests and an ASGITransport roundtrip test (no network)
- Stage 6.4: added management documentation and hardening regression tests (API key secret returned once; audit failure best-effort)
- Stage 7.1: extended AuditStore retention primitives and SQLite implementation with unit tests
- Stage 7.2: added audit retention engine (compute_cutoff, time retention, size retention) with unit tests
- Stage 7.3: added out-of-band cleanup command `cli/enforceai_audit_cleanup.py` with unit + integration tests
- Stage 7.4: added regression tests for audit failure policy + request-path caching; added audit retention operator docs
- Production testing (2025-12-15): verified end-to-end FGAC tool restrictions with SQLite MCP server integration, confirmed agent-level `allowed_tools` enforcement with Nginx Lua filtering
- UI Frontend Planning (2025-12-15): created `enforceai/plans/plan-enforce-gw-ui-frontend-phased.md` with 16 phases covering complete UI rewrite using Vite, React 18, TypeScript, Tailwind CSS, TanStack Query, Vitest, React Testing Library, and Playwright E2E
- UI Frontend Phase 1 (2025-12-16): migrated from CRA to Vite 6.x, added Vitest + RTL + MSW testing infrastructure, created API client with CSRF handling and X-Request-Id generation, added core utilities (cn, format, errors) with 80 unit tests passing, build succeeds
- UI Frontend Phase 2 (2025-12-16): implemented complete UI component library (Button, Input, Card, Modal, Toast, Badge, Spinner, etc.) with dark mode support, added ThemeContext for light/dark/system theme management, created 14 reusable UI components with 187 unit tests passing
- UI Frontend Phase 3 (2025-12-16): built main application shell with AppShell, Header, Sidebar, and MobileNav components; implemented router with lazy loading and protected routes (ProtectedRoute, AdminRoute); created 19 placeholder pages for all features; added navigation with 11 sections (Registry, EnforceAI, Policy, Monitoring, Administration); 250 tests passing
- UI Frontend Phase 4 (2025-12-16): enhanced AuthContext with OAuth provider support and session management (expiry detection, periodic checks, refresh capability); enhanced LoginPage with OAuth provider buttons (Google, GitHub, Microsoft, Okta); added SessionExpiryWarning component; created useSession hooks for session state management; added comprehensive auth tests; 267 tests passing
- UI Frontend Phase 5 (2025-12-16): implemented Overview dashboard page with connection status cards, summary counts, and quick actions; added QueryClientProvider to App.tsx; created API hooks (useServers, useA2AAgents, useEnforceAIAgents, useConnectionTest); updated API types to match backend response format (servers/agents use is_enabled field); added comprehensive MSW mock handlers; 297 tests passing
- UI Frontend Phase 6 (2025-12-16): implemented MCP Servers feature with full CRUD operations; ServersPage with list/search/filters; ServerDetailsPage with tools list; ServerRegisterModal and ServerEditModal with Zod validation; hooks for all server operations with optimistic updates; 350 tests passing
- UI Frontend Phase 7 (2025-12-16): implemented A2A Agents feature with full CRUD operations; A2AAgentsPage with list/search/filters (including visibility filter); A2AAgentDetailsPage with skills list; AgentRegisterModal and AgentEditModal with Zod validation; hooks for all agent operations with optimistic updates; 403 tests passing
- Backend Auth Improvements (2025-12-16): improved OAuth redirect URI handling with _normalize_redirect_uri helper for secure redirect validation; fixed OAuth login URL in AuthContext to use correct backend route; fixed session management useEffect dependency array
- UI Frontend Phase 8 (2025-12-16): implemented EnforceAI Agents feature with full CRUD operations; EnforceAIAgentsPage with list/search/status filters; EnforceAIAgentDetailsPage with scopes, allowed_tools, metadata display; CreateAgentModal and EditAgentModal with Zod validation; RevokeAgentModal with confirmation requiring typing agent ID; RevokeAllTokensModal for bulk token revocation; hooks for all operations (useEnforceAIAgents, useEnforceAIAgent, useCreateEnforceAIAgent, useUpdateEnforceAIAgent, useRevokeEnforceAIAgent, useRevokeAllTokens) with optimistic updates; 457 tests passing
- UI Frontend Phase 9 (2025-12-16): implemented EnforceAI Credentials feature (API Keys + Gateway Tokens) with full CRUD operations; ApiKeysPage with list/search/status filters; ApiKeyDetailsPage with associated agent, scopes, metadata display; CreateApiKeyModal with Zod validation and one-time secret display; RevokeApiKeyModal with confirmation; TokensPage with agent selector, MintTokenModal with TTL presets/custom/expires_at, token decode (local only), acknowledgment checkbox, copy as Authorization header; RevokeTokenModal with by-JTI and by-token modes; hooks for all operations (useApiKeys, useApiKey, useCreateApiKey, useRevokeApiKey, useMintToken, useRevokeToken, decodeToken) with optimistic updates; tests passing
- Documentation Update (2025-12-16): updated CLAUDE.md with comprehensive repository guidelines including project structure, build/test commands, coding style, testing guidelines, commit/PR best practices, and security tips
- UI Frontend Phase 10 (2025-12-16): implemented Scopes Catalog Viewer with comprehensive scope management interface; ScopesPage with search, server filtering, expand/collapse functionality; ScopeExplainerCard for detailed permission breakdown; ScopePicker component for reusable scope selection; added ScopeCatalog, ScopeDefinition, ScopeInfo API types; extended Badge component for interactive use; displays stats (total scopes, servers, group mappings), group mappings with clickable navigation; mock scope catalog data matching scopes.yml schema; tests passing
- MCP Gateway Troubleshooting (2025-12-16): fixed SQLite MCP server gateway connection issues; resolved auth-server using Keycloak instead of EnforceAI by restarting with EnforceAI environment variables; fixed nginx path routing by changing SQLite server path from '/sqlite' to '/sqlite/' for proper proxy_pass URI rewriting; verified proper request flow from gateway to SQLite server (now returns "Missing session ID" instead of "Not Found"); identified streamable-http transport session management limitation (requires GET /sse for session establishment but gateway blocks GET requests)
- UI Frontend Phase 11 (2025-12-16): implemented Tools Discovery feature; ToolsPage with server list, expandable tools per server, search across all tools, summary stats (total servers, tools, enabled servers); AllowedToolsBuilder component for multi-server tool selection; ToolPicker reusable component grouped by server; useAllServerTools hook aggregating tools from all servers; expand/collapse all controls; input schema viewer; tests passing
- UI Frontend Phase 12 (2025-12-16): implemented Audit Events page with Phase 14 placeholder showing current status; AuditPage with summary statistics, event filters (action type, outcome, time range), and important notes; hooks for audit data (currently returning placeholder data); audit retention settings display; 24 tests for AuditPage; 608 tests passing
- UI Frontend Phase 13 (2025-12-17): implemented Admin Users Directory with user search functionality; AdminLayout wrapper with amber warning banner for elevated permissions context; AdminPage landing page with quick stats and admin tools sections (User Directory, Access Control [Phase 14], Audit & Logs, System Configuration); AdminUsersPage with debounced search, results table with canonical user_id and copy buttons; AdminUserDetailsPage with user info card, agent summary (active/revoked counts), Phase 14 cross-user operations placeholder; admin API functions (searchAdminUsers, getAdminUser, getAdminUserAgents); admin hooks (useAdminUsers, useAdminUser, useAdminUserAgents); comprehensive unit tests (43 new tests); 656 tests passing
- Scopes Catalog Backend Endpoint (2025-12-17): added GET /enforceai/scopes/catalog endpoint to serve scope catalog to frontend; created Pydantic response models (ScopeCatalogResponse, ScopeDefinitionResponse, ServerPermissionResponse, etc.); endpoint is publicly accessible (no auth required) since scope catalog is display data; updated frontend hooks to call /enforceai/scopes/catalog instead of /api/scopes/catalog; updated ScopeCatalog TypeScript interface with version and generated_at fields; updated all test mock handlers and test files to use new endpoint path; scopes page now displays 5 scopes from catalog
- UI Frontend Phase 14 (2025-12-17): implemented Admin Cross-User Operations with full CRUD for managing other users' resources; added admin API functions to enforceai.ts (adminCreateAgentForUser, adminRevokeAgentForUser, adminRevokeAllTokensForUser, adminGetApiKeysForUser, adminRevokeApiKeyForUser, adminRevokeTokenForUser); created admin mutation hooks (useAdminCreateAgent, useAdminRevokeAgent, useAdminRevokeAllTokens, useAdminCreateApiKey, useAdminRevokeApiKey, useAdminRevokeToken, useAdminUserAgentApiKeys); added TypeToConfirmDialog component for type-to-confirm pattern on destructive actions with optional reason field; created TargetUserBanner component showing "Acting on user: [email]" context with audit warning; created AdminModals.tsx with AdminCreateAgentModal (with ScopePicker), AdminRevokeAgentModal (type agent_id + reason), AdminRevokeAllTokensModal, AdminRevokeApiKeyModal, AdminRevokeTokenModal (by JTI); updated AdminUserDetailsPage with full cross-user operations including AgentCard component with expandable API keys section; added comprehensive MSW mock handlers for all admin endpoints; 564 tests passing

## Decisions
- Phase 1 persistence: local SQLite database with storage-agnostic interfaces to enable later migration to Postgres.
- Gateway token signing: RS256 for compatibility; local public-key verification with `kid`-based rotation.
- Gateway token key management: mounted PEM files (private key + per-`kid` public keys), restart-based rotation in Phase 1; optional future JWKS endpoint.
- Credential transport: use `Authorization: Bearer` as canonical; accept `X-Gateway-Token` as fallback; reject ambiguous multi-credential requests.
- Service boundary: extend existing `auth_server` as the EnforceAI stateful enforcement point behind Nginx `auth_request`; keep `registry` optional for management/UI.
- Generic OIDC scope: multi-issuer config map keyed by `iss` (map-of-one allowed), local JWKS validation with caching.
- Agent binding: `agent_id` required for MCP access; from gateway token claim, API key record, or `X-Agent-Id` for OIDC.
- Authorization overlay: runtime enforcement uses agent scopes (and optional allowed-tools) only; enterprise policy is the scope catalog; apply any user baseline at provisioning time.
- Effective scopes for gateway tokens: authorize using `token.scopes ∩ agent.scopes` (token may further restrict but never elevate).
- API key model: `eak_<key_id>.<secret>` with hashed-at-rest verifier using `API_KEY_PEPPER`; agent-bound; `api_key.scopes ∩ agent.scopes` (or agent scopes if unset); future task scoping as an additional restriction dimension.
- Revocation model: layered revocation (agent kill switch, token `jti` table, bulk revoke via `agent.tokens_valid_after`); default-deny on revocation/registry read failure.
- Gateway token lifetime: long-lived (PAT-style) up to 365 days in Phase 1; always require `exp` (no perpetual tokens).
- Enterprise policy catalog: reuse `auth_server/scopes.yml` as the authoritative scope catalog in Phase 1; consider splitting catalog vs IdP mappings later.
- Compatibility: no backward compatibility required; EnforceAI supports only generic OIDC, gateway tokens, and API keys via the resolver.
- Audit sink: dual-sink stdout JSON + SQLite audit table; pragmatic failure policy (do not fail request solely due to audit persistence failure, but emit high-severity log).
- Audit retention: hybrid time+size retention with configurable thresholds; cleanup out of band (not on request path).
- Management surface: Phase 1 CLI-first; self-service per user for managing own agents/tokens/api-keys.
- Tenancy: Phase 1 uses `user_id` as the boundary; no explicit `tenant_id/org_id` modeled yet.
- Tool visibility: filter `tools/list` to only tools callable under effective authorization (policy + agent allowed_tools); return empty list when no callable tools.
- Config delivery: Phase 1 uses environment variables for configuration; secrets provided via mounted secret files referenced by path variables.
- Identifiers: `user_id = "<iss>|<sub>"` (issuer-namespaced); `agent_id` is UUIDv4 canonical with optional alias.
- API key pepper rotation: no pepper rotation/versioning in Phase 1; rotation is a breaking change for existing keys unless versioning is added later.
- Error semantics: 401 for missing/invalid credentials; 403 for authenticated-but-denied (including missing `X-Agent-Id`); 503 for internal enforcement dependency failures (deny but signal retry).
- OIDC claim defaults: scopes from `scp`→`scope`→`permissions`; roles from `roles`→`groups`→`permissions` (per-issuer overrides allowed); roles/groups for audit only.
- Bootstrap token TTL: extended to 30 days for local development convenience; production tokens should use shorter TTLs based on security requirements.
- Docker Compose EnforceAI activation: EnforceAI environment variables must be loaded via `--env-file` flag (not inline env vars) for proper Docker Compose variable substitution.
- Agent allowed_tools enforcement: tool restrictions are bidirectional (visibility + execution); `tools/list` shows only allowed tools, `tools/call` blocks unauthorized tools.
- MCP server registration: use `host.docker.internal` for Docker-to-host connectivity when registering localhost MCP servers.
- Nginx location path handling: MCP server paths in registry JSON must include trailing slash (e.g., `/sqlite/`) for proper nginx `proxy_pass` URI rewriting; without trailing slash, nginx sends full request path to upstream instead of stripping location prefix; this is required when using `rewrite_by_lua_file` in nginx config.

## Current Task
- UI Frontend Phase 14: Admin Cross-User Operations (completed)
- UI Scope Catalog Management Phase 0: catalog path + `etag` (completed)
- UI Scope Catalog Management Phase 1: admin scope CRUD API (completed)
- UI Scope Catalog Management Phase 2: UI create scope flow (completed)
- UI Scope Catalog Management Phase 3: UI edit scope flow (completed)
- UI Scope Catalog Management Phase 4: UI delete scope flow (completed)
- UI Scope Catalog Management Phase 5 (optional): structured scope editor (completed)
- UI Scope Catalog Management: allow scope management UI for `enforceai-admin` group (completed)

## Next Steps
1. Execute `enforceai/plans/plan-enforce-gw-ui-frontend-phased.md` Phase 15: Settings + Help + Final Polish (SettingsPage, HelpPage, 404 page, error boundary)
2. Execute Phase 16: E2E Testing + Documentation (Playwright setup, E2E scenarios, README update)

## Tests Executed
- `uv run python -m py_compile auth_server/enforceai/*.py tests/unit/enforceai/*.py` (pass)
- `uv run pytest -q -o addopts='' tests/unit/enforceai` (8 passed)
- `uv run python -m py_compile auth_server/enforceai/config.py tests/unit/enforceai/test_config_*.py` (pass)
- `uv run pytest -q -o addopts='' tests/unit/enforceai` (15 passed)
- `uv run python -m py_compile tests/fixtures/enforceai_fixtures.py tests/unit/enforceai/test_test_fixtures.py tests/conftest.py` (pass)
- `uv run pytest -q -o addopts='' tests/unit/enforceai` (18 passed)
- `.venv/bin/python -m pytest` (279 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/db/migrations.py tests/unit/enforceai/test_migrations.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (20 passed)
- `.venv/bin/python -m py_compile auth_server/enforceai/models/agent.py auth_server/enforceai/stores/sqlite/agent_store.py tests/unit/enforceai/test_agent_store_sqlite.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (26 passed)
- `.venv/bin/python -m py_compile auth_server/enforceai/models/api_key.py auth_server/enforceai/stores/sqlite/api_key_store.py tests/unit/enforceai/test_api_key_store_sqlite.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (32 passed)
- `.venv/bin/python -m py_compile auth_server/enforceai/fgac/catalog.py auth_server/enforceai/api/management_routes.py tests/unit/enforceai/test_scope_catalog.py tests/integration/test_enforceai_management_routes.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_scope_catalog.py::TestScopeCatalog::test_cache_invalidates_when_file_changes` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_management_routes.py::TestEnforceAIManagementRoutes::test_scopes_catalog_endpoint_uses_configured_path_and_etag_updates` (pass)
- `.venv/bin/python -m py_compile auth_server/enforceai/fgac/policy_writer.py auth_server/enforceai/api/management_routes.py tests/unit/enforceai/test_policy_writer.py tests/integration/test_enforceai_management_routes.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_policy_writer.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_management_routes.py::TestEnforceAIManagementRoutes::test_admin_can_create_replace_and_delete_scopes_with_etag_and_csrf tests/integration/test_enforceai_management_routes.py::TestEnforceAIManagementRoutes::test_admin_delete_scope_returns_conflict_when_referenced_by_group_mappings tests/integration/test_enforceai_management_routes.py::TestEnforceAIManagementRoutes::test_non_admin_cannot_manage_scopes` (pass)
- `npm -C frontend run typecheck` (pass)
- `npm -C frontend test` (pass)
- `npm -C frontend run build` (pass)
- `npm -C frontend run typecheck` (pass)
- `npm -C frontend test` (pass)
- `npm -C frontend run build` (pass)
- `npm -C frontend run typecheck` (pass)
- `npm -C frontend test` (pass)
- `npm -C frontend run build` (pass)
- `.venv/bin/python -m py_compile auth_server/enforceai/api/management_routes.py auth_server/enforceai/fgac/catalog.py auth_server/enforceai/fgac/policy_writer.py tests/unit/enforceai/test_scope_catalog.py tests/unit/enforceai/test_policy_writer.py tests/integration/test_enforceai_management_routes.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_scope_catalog.py tests/unit/enforceai/test_policy_writer.py` (11 passed)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_management_routes.py` (10 passed)
- `.venv/bin/python -m pytest` (293 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/models/revocation.py auth_server/enforceai/stores/sqlite/revocation_store.py tests/unit/enforceai/test_revocation_store_sqlite.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (37 passed)
- `.venv/bin/python -m pytest` (298 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/models/audit.py auth_server/enforceai/stores/sqlite/audit_store.py tests/unit/enforceai/test_audit_store_sqlite.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (40 passed)
- `.venv/bin/python -m pytest` (301 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/db/connection.py auth_server/enforceai/db/data_layer.py tests/unit/enforceai/test_data_layer_smoke.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (42 passed)
- `.venv/bin/python -m pytest` (303 passed; coverage gate met)
- `.venv/bin/python -m py_compile tests/integration/test_enforceai_data_layer.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_data_layer.py` (1 passed)
- `.venv/bin/python -m py_compile auth_server/enforceai/tokens/claims.py tests/unit/enforceai/test_gateway_token_claims.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (53 passed)
- `.venv/bin/python -m pytest` (315 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/crypto/keyring.py tests/unit/enforceai/test_gateway_keyring.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (57 passed)
- `.venv/bin/python -m pytest` (319 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/tokens/mint.py tests/unit/enforceai/test_gateway_token_mint.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (60 passed)
- `.venv/bin/python -m pytest` (322 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/server.py tests/integration/test_enforceai_stage5_roundtrip.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_stage5_roundtrip.py` (4 passed)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_tools_call_enforcement.py` (5 passed)
- `.venv/bin/python -m py_compile auth_server/enforceai/tokens/verify.py tests/unit/enforceai/test_gateway_token_verify.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (67 passed)
- `.venv/bin/python -m pytest` (329 passed; coverage gate met)
- `.venv/bin/python -m py_compile tests/unit/enforceai/test_gateway_token_hardening.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (71 passed)
- `.venv/bin/python -m pytest` (333 passed; coverage gate met)
- `.venv/bin/python -m py_compile tests/integration/test_enforceai_gateway_token_roundtrip.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_gateway_token_roundtrip.py` (3 passed)
- `.venv/bin/python -m py_compile auth_server/enforceai/config.py tests/unit/enforceai/test_config_parsing.py tests/unit/enforceai/test_config_validation.py tests/unit/enforceai/test_oidc_config_models.py tests/fixtures/enforceai_fixtures.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai` (77 passed)
- `.venv/bin/python -m pytest` (342 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/oidc/claims.py tests/unit/enforceai/test_oidc_claims_normalization.py` (pass)
- `.venv/bin/python -m pytest` (354 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/oidc/jwks.py tests/unit/enforceai/test_oidc_jwks_cache.py tests/unit/enforceai/test_imports.py` (pass)
- `.venv/bin/python -m pytest` (360 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/oidc/models.py auth_server/enforceai/oidc/verify.py tests/unit/enforceai/test_oidc_verify_multi_issuer.py tests/unit/enforceai/test_oidc_jwks_rotation.py tests/unit/enforceai/test_oidc_verify_errors.py tests/unit/enforceai/test_imports.py` (pass)
- `.venv/bin/python -m pytest` (370 passed; coverage gate met)
- `.venv/bin/python -m py_compile tests/unit/enforceai/test_oidc_hardening.py tests/integration/test_enforceai_oidc_roundtrip.py` (pass)
- `.venv/bin/python -m pytest` (376 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/auth/__init__.py auth_server/enforceai/auth/credentials.py tests/unit/enforceai/test_identity_credentials.py tests/unit/enforceai/test_imports.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_identity_credentials.py` (9 passed)
- `.venv/bin/python -m pytest` (385 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/config.py auth_server/enforceai/secrets/__init__.py auth_server/enforceai/secrets/pepper.py auth_server/enforceai/providers/__init__.py auth_server/enforceai/providers/api_key.py tests/unit/enforceai/test_api_key_provider.py tests/unit/enforceai/test_config_validation.py tests/unit/enforceai/test_imports.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_api_key_provider.py tests/unit/enforceai/test_config_validation.py tests/unit/enforceai/test_imports.py` (15 passed)
- `.venv/bin/python -m pytest` (396 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/config.py auth_server/enforceai/providers/gateway_token.py auth_server/enforceai/providers/__init__.py tests/unit/enforceai/test_gateway_token_provider.py tests/unit/enforceai/test_imports.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_gateway_token_provider.py tests/unit/enforceai/test_imports.py` (7 passed)
- `.venv/bin/python -m pytest` (402 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/providers/oidc.py auth_server/enforceai/providers/__init__.py tests/unit/enforceai/test_oidc_provider.py tests/unit/enforceai/test_imports.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_oidc_provider.py tests/unit/enforceai/test_imports.py` (8 passed)
- `.venv/bin/python -m pytest` (409 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/config.py auth_server/enforceai/auth/resolver.py auth_server/enforceai/auth/__init__.py tests/unit/enforceai/test_config_parsing.py tests/unit/enforceai/test_config_validation.py tests/unit/enforceai/test_identity_resolver.py tests/unit/enforceai/test_imports.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_identity_resolver.py tests/unit/enforceai/test_config_parsing.py tests/unit/enforceai/test_config_validation.py tests/unit/enforceai/test_imports.py` (pass)
- `.venv/bin/python -m pytest` (416 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/fgac/catalog.py auth_server/enforceai/fgac/evaluate.py auth_server/enforceai/auth/dependency.py tests/unit/enforceai/test_scope_catalog.py tests/unit/enforceai/test_fgac_evaluate.py tests/unit/enforceai/test_enforceai_dependency_wiring.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai -k \"scope_catalog\"` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai -k \"fgac_evaluate\"` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai -k \"dependency_wiring\"` (pass)
- `.venv/bin/python -m pytest` (428 passed; coverage gate met)
- `bash -n docker/registry-entrypoint.sh` (pass)
- `.venv/bin/python -m py_compile auth_server/server.py registry/core/nginx_service.py auth_server/enforceai/fgac/evaluate.py` (pass)
- `.venv/bin/python -m pytest` (429 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/management/__init__.py auth_server/enforceai/management/models.py auth_server/enforceai/management/service.py tests/unit/enforceai/test_management_service.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_management_service.py` (6 passed)
- `.venv/bin/python -m pytest` (444 passed; coverage gate met)
- `.venv/bin/python -m py_compile auth_server/enforceai/api/management_routes.py auth_server/server.py tests/integration/test_enforceai_management_routes.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_management_routes.py` (3 passed)
- `.venv/bin/python -m pytest` (447 passed; coverage gate met)
- `.venv/bin/python -m py_compile cli/enforceai_cli.py tests/unit/cli/test_enforceai_cli_args.py tests/integration/test_enforceai_cli_roundtrip.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/cli/test_enforceai_cli_args.py` (5 passed)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_cli_roundtrip.py` (1 passed)
- `.venv/bin/python -m pytest` (453 passed; coverage gate met)
- `.venv/bin/python -m py_compile tests/integration/test_enforceai_management_routes.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_management_routes.py` (pass)
- `.venv/bin/python -m pytest` (pass)

## Stage 7 Progress
- Stage 7.1: extended AuditStore retention primitives and SQLite implementation with unit tests
- Stage 7.2: added audit retention engine (`compute_cutoff`, time retention, size retention) with unit tests
- Stage 7.3: added out-of-band cleanup command `cli/enforceai_audit_cleanup.py` with unit + integration tests
- Stage 7.4: added regression tests for audit failure policy + request-path caching; added audit retention operator docs
- `.venv/bin/python -m py_compile auth_server/enforceai/stores/interfaces.py auth_server/enforceai/stores/sqlite/audit_store.py tests/unit/enforceai/test_audit_store_sqlite.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_audit_store_sqlite.py` (pass)
- `.venv/bin/python -m py_compile auth_server/enforceai/audit/retention.py tests/unit/enforceai/test_audit_retention.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_audit_retention.py` (pass)
- `.venv/bin/python -m py_compile cli/enforceai_audit_cleanup.py tests/unit/cli/test_enforceai_audit_cleanup_args.py tests/integration/test_enforceai_audit_cleanup.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/cli/test_enforceai_audit_cleanup_args.py tests/integration/test_enforceai_audit_cleanup.py` (pass)

- `.venv/bin/python -m py_compile tests/integration/test_enforceai_stage7_hardening.py` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/integration/test_enforceai_stage7_hardening.py` (pass)

- `.venv/bin/python -m py_compile tests/integration/test_registry_session_invalidation.py` (pass)
- `make test` (pass)
- `make test` (pass)
- `make test` (pass)
- `make test` (pass)

## Next
- Start UI build: new Enforce Gateway UI (frontend)

## Production Testing & Integration (2025-12-15)

### Bootstrap Token Configuration
- Extended bootstrap token TTL from 1 hour to 30 days (2592000 seconds) in `scripts/enforceai_dev_bootstrap.sh` for local development convenience
- Regenerated bootstrap token with extended expiry: `exp: 2026-01-14`
- Token stored in `~/mcp-gateway/enforceai/bootstrap_gateway_token.txt`

### SQLite MCP Server Registration
- Registered local SQLite MCP server (`localhost:3031/mcp`) to the MCP Gateway at path `/sqlite`
- Server configuration:
  - `display_name`: "SQLite MCP Server"
  - `proxy_pass_url`: `http://host.docker.internal:3031/mcp`
  - `supported_transports`: ["streamable-http"]
- Server health verified with 6 tools exposed:
  - `read_query` (SELECT queries)
  - `write_query` (INSERT, UPDATE, DELETE)
  - `create_table` (DDL operations)
  - `list_tables` (schema introspection)
  - `describe_table` (schema introspection)
  - `append_insight` (business insight storage)

### FGAC Tool Restrictions (Agent-Level)
- Configured agent `9d2724e9-1753-4493-8993-0d6986754414` with `allowed_tools` restriction
- Allowed tools: `["read_query", "list_tables"]` (read-only subset)
- Updated agent using EnforceAI CLI:
  ```bash
  uv run python cli/enforceai_cli.py \
    --x-gateway-token "$TOKEN" \
    agents update 9d2724e9-1753-4493-8993-0d6986754414 \
    --allowed-tool read_query \
    --allowed-tool list_tables
  ```
- Verified FGAC enforcement:
  - Auth-server returns `X-Allowed-Tools: ["list_tables", "read_query"]` header
  - Nginx Lua filter (`filter_tools_list.lua`) filters `tools/list` responses to only show allowed tools
  - Tools not in allowed list (`write_query`, `create_table`, `describe_table`, `append_insight`) are hidden from `tools/list` and blocked in `tools/call`
- End-to-end validation confirmed: `tools/list` returns only 2 tools instead of 6

### Docker Compose Environment Configuration
- Fixed EnforceAI environment variable loading in Docker Compose:
  - Used `--env-file` flag to properly load EnforceAI configuration
  - Env file: `/Users/adizlotkin/mcp-gateway/enforceai/enforceai.compose.env`
  - Key variables:
    - `ENFORCEAI_AUTH_PROVIDER="gateway-token"`
    - `ENFORCEAI_DB_PATH="/app/enforceai_state/enforceai.db"`
    - `ENFORCEAI_SCOPES_CATALOG_PATH="/app/scopes.yml"`
    - Gateway token signing keys, API key pepper, audit settings
- Verified auth-server startup with EnforceAI active: "Loaded scopes configuration from /app/scopes.yml with 4 group mappings"

### Audit Events Verified
- Confirmed audit events emitted for `tools/list` requests:
  ```json
  {
    "action": "tools/list",
    "agent_id": "9d2724e9-1753-4493-8993-0d6986754414",
    "details": {
      "allowed_tools": "[\"list_tables\", \"read_query\"]",
      "provider": "gateway-token",
      "server": "sqlite"
    },
    "event_type": "enforceai_audit",
    "outcome": "allow",
    "user_id": "local|admin"
  }
  ```

### Architecture Verification
- Confirmed FGAC enforcement flow:
  1. Client sends MCP request to `http://localhost/sqlite/mcp` with `Authorization: Bearer <token>`
  2. Nginx calls `/validate` auth_request with `X-Original-URL` and `X-Body` headers
  3. Auth-server (EnforceAI):
     - Validates gateway token
     - Looks up agent record from agent store
     - Extracts `agent.allowed_tools` from agent record
     - Adds to identity metadata: `identity.metadata["agent_allowed_tools"]`
     - For `tools/list` requests: calls `resolve_callable_tools_for_server()` to compute intersection of scope permissions and agent allowed_tools
     - Returns `X-Allowed-Tools` header with filtered tool list
  4. Nginx captures `X-Allowed-Tools` as `$auth_allowed_tools` variable
  5. Nginx proxies request to upstream MCP server
  6. For `tools/list` responses: Lua `body_filter_by_lua_file` filters JSON response to only include tools in `$auth_allowed_tools`
  7. Client receives filtered tools list

### Operational Notes
- Bootstrap token stored at: `~/mcp-gateway/enforceai/bootstrap_gateway_token.txt`
- EnforceAI state directory: `~/mcp-gateway/enforceai/` (database, secrets, keys)
- Docker Compose env file must be loaded using `--env-file` flag for proper EnforceAI activation
- Agent configuration persisted in SQLite DB at `/app/enforceai_state/enforceai.db` (inside container)
- Tool restrictions are enforced at both visibility (`tools/list`) and execution (`tools/call`) levels

### Upstream Auth Backend Phase 1 (Data Model + Storage Foundations)
- Added canonical `upstream_auth` models + legacy normalization (`auth_type`/`auth_provider`/`headers`) in `auth_server/enforceai/models/upstream_auth.py`.
- Added encrypted upstream credential storage:
  - Migration `0004_upstream_credentials`
  - SQLite store `auth_server/enforceai/stores/sqlite/upstream_credential_store.py`
  - AES-GCM envelope helpers in `auth_server/enforceai/crypto/upstream_secrets.py`
- Extended EnforceAI settings with `ENFORCEAI_UPSTREAM_KEK_PATH` and updated `scripts/enforceai_dev_bootstrap.sh` to generate `.enforceai/secrets/upstream_kek` (mode `0600`).
- Updated registry server registration/loading to populate `upstream_auth` for server entries.
- Tests: `make test-unit` (pass).

### Upstream Auth Backend Phase 2 (SSRF/Egress Allowlist)
- Added DB-managed egress allowlist:
  - Migration `0005_egress_allowlist` (table `egress_allowlist_entries`)
  - SQLite store `auth_server/enforceai/stores/sqlite/egress_allowlist_store.py`
  - URL/host/CIDR matcher `auth_server/enforceai/egress/allowlist.py`
- Added admin management endpoints in auth server:
  - `GET/POST /enforceai/admin/egress-allowlist`
  - `PUT/DELETE /enforceai/admin/egress-allowlist/{entry_id}`
  - `POST /enforceai/admin/egress-allowlist/check`
- Enforced allowlist on registry server registration/update routes when `ENFORCEAI_DB_PATH` is set in the registry process.
- Defense-in-depth: `registry/core/nginx_service.py` filters/skips non-allowlisted `proxy_pass_url` entries at config generation time when `ENFORCEAI_DB_PATH` is set.
- Tests: `make test-unit` and `make test-integration` (pass).

### Upstream Auth Backend Phase 3 (Upstream Credential Management)
- Added upstream credential management routes in auth server:
  - `GET /enforceai/upstream/servers`
  - `GET /enforceai/upstream/servers/{server_path}/credentials`
  - `POST /enforceai/upstream/servers/{server_path}/credentials`
  - `POST /enforceai/upstream/credentials/{credential_id}/revoke`
- Added API schemas: `auth_server/enforceai/models/upstream_management.py`.
- Added Compose wiring for `ENFORCEAI_UPSTREAM_KEK_PATH` in `docker-compose.yml` and `docker-compose.prebuilt.yml`.
- Tests: `make test-unit` and `make test-integration` (pass).

### Upstream Auth Backend Phase 4 (Request-Time Resolution + Injection)
- Added request-time upstream credential resolver: `auth_server/enforceai/upstream/resolver.py`.
- Wired `/validate` to emit internal-only headers for Nginx upstream injection:
  - Identity forwarding: `X-MCP-Principal`, `X-MCP-Auth-Type`, `X-MCP-Scopes`, `X-MCP-Provider`, `X-MCP-Claims`
  - Upstream injection: `X-EnforceAI-Upstream-Authorization`, `X-EnforceAI-Upstream-Api-Key`, `X-EnforceAI-Upstream-Api-Key-Header`, `X-EnforceAI-Upstream-Mode`
- Supported types: `none`, `header-trust`, `api-key`, `jwt`.
- Tests: `make test-unit` and `make test-integration` (pass).

### Upstream Auth Backend Phase 5 (OAuth2 / OIDC / Provider OAuth + Refresh)
- Added upstream OAuth provider config in EnforceAI settings:
  - `ENFORCEAI_UPSTREAM_OAUTH_PROVIDERS` (JSON map keyed by provider id)
  - `ENFORCEAI_UPSTREAM_OAUTH_STATE_TTL_SECONDS`
  - `ENFORCEAI_UPSTREAM_OAUTH_REFRESH_SKEW_SECONDS`
  - Client secrets are sourced via `client_secret_ref` (`env` or `file`) and never logged.
- Added encrypted OAuth state storage:
  - Migration `0006_upstream_oauth_states`
  - Store `auth_server/enforceai/stores/sqlite/upstream_oauth_state_store.py`
- Added OAuth flow endpoints (gateway-terminated):
  - `POST /enforceai/upstream/oauth/start`
  - `GET /enforceai/upstream/oauth/callback`
  - `POST /enforceai/upstream/oauth/disconnect`
- Added token exchange + refresh client: `auth_server/enforceai/upstream/oauth_client.py` (httpx; test transport injectable).
- Extended request-time resolver to support `oauth2`, `oidc`, `provider-oauth` with on-demand refresh.
- Added tests:
  - Unit: `tests/unit/enforceai/test_upstream_oauth_state_store_sqlite.py`, `tests/unit/enforceai/test_upstream_oauth_refresh.py`
  - Integration: `tests/integration/test_enforceai_upstream_oauth_flow.py` (in-process stub provider; no network)
- Tests: `make test-unit` and `make test-integration` (pass).

## Outstanding Questions
