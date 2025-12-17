# Plan: Upstream MCP Auth (Backend, Phased)
*Created: 2025-12-17*

## Assumptions (Confirmed)

- Agents/clients authenticate **only** to the gateway.
- All upstream authentication and upstream credential management is **gateway-terminated**.
- Frontend work starts only after required backend phases land.
- Injection architecture: `auth_server` `/validate` computes upstream injection; Nginx applies injection to upstream requests.
- stdio/pipes are supported via an HTTP/WS bridge behind the gateway (no native Nginx stdio).
- mTLS is not a priority (explicitly deferred to a future plan).
- Token refresh strategy (Phase 1): on-demand refresh only (no background refresher).
- SSRF/egress policy: strict allowlist, DB-managed with optional TTL, admin-gated and audited.
- Tenancy (Phase 1): single-tenant (no canonical `tenant_id/org_id`).

## Objective

Implement gateway-managed upstream authentication for MCP servers as described in:
- `enforceai/AUTHENTICATION.md`
- `enforceai/mcp_upstream_auth_requirements.md`

Each phase below is scoped to be completable in a single agent run and must be fully tested before marking complete.

## Dev/Local Test Harness (Applies to All Phases)

Each backend phase must keep dev/local testing fully offline-capable and repeatable.

### Required local components

- **Encrypted SQLite** state for recoverable upstream secrets:
  - secrets encrypted with a gitignored KEK file under `.enforceai/` (mode `0600`)
- **Stub OAuth/OIDC server** for tests (no real providers, no internet)
- **Upstream test servers** that assert required injected auth (api-key, bearer, jwt, header-trust, mtls)

### Required validation commands

- Unit tests: `make test-unit`
- Integration tests: `make test-integration` (use local stubs/servers only)

### Local run expectations (per phase)

Each phase must include (or update) local fixtures so a developer can validate end-to-end without external dependencies:

- A stub OAuth/OIDC provider used by integration tests.
- A small set of upstream “auth assertion” servers used by integration tests.
- A documented one-command reset for state under `.enforceai/`.

## Phase 1 — Data Model + Storage Foundations

### Scope
- Define canonical server-side representation of upstream auth requirements:
  - Add a normalized `upstream_auth` structure for server configs.
  - Provide backwards-compatible mapping from existing `auth_type` / `auth_provider` / `headers` into `upstream_auth`.
- Add DB schema for upstream credentials:
  - Credential types: `api-key`, `oauth2`, `oidc`, `provider-oauth`, `jwt` (static), `mtls`, `header-trust` (no secret).
  - Binding keys: `service`, `user`, `agent`, `user+agent`.
  - Status metadata: created/updated, expires_at, revoked_at, last_used_at, provider, scopes, token_type.
- Add encryption-at-rest envelope for secret fields (minimal dev-safe implementation):
  - One symmetric master key configured via env/file.
  - Encrypt only secret payloads; store metadata in plaintext.
- Add store interface(s) + SQLite store implementation(s) under `auth_server/enforceai/stores/`.

### Deliverables
- New Pydantic models in `auth_server/enforceai/models/` for upstream auth config + credentials.
- New SQLite migration(s) in `auth_server/enforceai/db/migrations/`.
- New store protocol(s) in `auth_server/enforceai/stores/interfaces.py` and implementation(s) in `auth_server/enforceai/stores/sqlite/`.
- Documentation update (minimal): reference new schema in `enforceai/mcp_upstream_auth_requirements.md`.

### Tests (must pass)
- `make test-unit` (required)
- Add unit tests for:
  - schema validation
  - encryption roundtrip + redaction
  - store CRUD (metadata vs secret handling)

### Local verification (manual)

- Run `./scripts/enforceai_dev_bootstrap.sh --force` (or equivalent) and verify:
  - `.enforceai/` contains a KEK file for upstream secret encryption (mode `0600`)
  - the DB schema includes upstream credential tables

## Phase 2 — SSRF/Egress Allowlist (DB-Managed) + Enforcement

### Scope
- Add a DB-managed egress allowlist with optional TTL:
  - entries include `hostname`/`domain_pattern` and/or `ip_cidr` (choose a minimal safe subset first; prefer hostname/domain patterns)
  - optional `expires_at`
  - audit events on create/update/delete
- Add admin-gated management endpoints:
  - list/add/update/delete allowlist entries
  - optional: “dry-run check” endpoint that validates a candidate `proxy_pass_url` against the allowlist and returns a reason
- Enforce allowlist:
  - on server registration/update (reject disallowed `proxy_pass_url`)
  - on proxy-time as defense-in-depth (reject if runtime resolution differs; protect against DNS rebinding and redirects)

### Deliverables
- New SQLite migration(s) for allowlist tables + indexes.
- New admin routes under EnforceAI management surface.
- Shared validator used by server registration/update code paths.

### Tests (must pass)
- `make test-unit` (required)
- `make test-integration` (required)
- Unit tests for:
  - allowlist match semantics (exact host, domain suffix, CIDR if supported)
  - TTL behavior (expired entries not accepted)
  - admin-only access enforcement
- Integration test for:
  - server registration rejects disallowed `proxy_pass_url`

## Phase 3 — Management API for Upstream Credentials

### Scope
- Add management endpoints (EnforceAI-protected) for upstream credentials:
  - `GET /enforceai/upstream/servers` (list servers requiring upstream credentials + current principal status)
  - `GET /enforceai/upstream/servers/{server_id}/credentials` (metadata list)
  - `POST /enforceai/upstream/servers/{server_id}/credentials` (create/set; secret returned only once if applicable)
  - `POST /enforceai/upstream/credentials/{credential_id}/revoke`
  - Optional in this phase: `POST /enforceai/upstream/credentials/{credential_id}/rotate` (if rotation semantics are simple)
- Implement ownership/authorization checks:
  - enforce binding rules (`service` vs `user` vs `agent` etc.)
  - scope-gate high-risk types (e.g., JWT mint config) behind an admin-only scope if required.
- Ensure responses never return stored secrets after creation.
- Emit audit events for create/revoke/list operations.

### Deliverables
- New routes under `auth_server/enforceai/api/` (or a new `upstream_routes.py`) and service layer under `auth_server/enforceai/management/`.
- New API schemas in `auth_server/enforceai/models/` for request/response.
- Updated OpenAPI.

### Tests (must pass)
- `make test-unit` (required)
- Add unit tests for:
  - authz: owner vs non-owner
  - secret non-disclosure
  - validation errors for unsupported combinations

### Local verification (manual)

- Use `curl` or the CLI (if added) to:
  - create an upstream `api-key` credential and confirm it is returned once
  - list credentials and confirm the secret is not returned

## Phase 4 — Request-Time Upstream Credential Resolution + Header Injection (HTTP/SSE)

### Scope
- Implement request-time resolver that, for a proxied MCP request:
  - identifies target server from request context (`X-Original-URL` / path)
  - reads `upstream_auth` for the server
  - identifies the principal (from EnforceAI IdentityContext)
  - resolves the correct active credential (if required)
  - returns injection instructions to the proxy layer
- Integrate with Nginx `auth_request` flow:
  - Extend `/validate` response headers to include upstream injection headers (never exposed to clients).
  - Inject into upstream proxy request via Nginx `proxy_set_header` (generated config and/or static template).
- Implement support (in this phase) for:
  - `none`
  - `api-key` (header injection)
  - `jwt` (static bearer injection)
  - `header-trust` (identity headers only; no upstream secret)
- Lock error semantics for “missing required upstream credential”:
  - Return `424 Failed Dependency` with `error_code=UPSTREAM_CREDENTIALS_REQUIRED`.
- Ensure inbound spoofing protection:
  - strip inbound `X-MCP-*` headers and any reserved upstream-injection headers.

### Deliverables
- Resolver/service module (suggested): `auth_server/enforceai/upstream/` (new package)
- `/validate` sets stable, internal-only headers for upstream injection
- Nginx config generator updated to forward those headers to upstream MCP servers

### Tests (must pass)
- `make test-unit` (required)
- `make test-integration` (required)
- Integration tests with a fake upstream server container/app verifying:
  - correct injected `Authorization`/`X-API-Key`
  - correct injected `X-MCP-*` identity headers
  - correct error code when credential missing
  - streaming endpoint behavior not broken (SSE/streamable HTTP)

### Local verification (manual)

- Configure one upstream test server with `upstream_auth.type=api-key`, create the credential, then call the gateway MCP endpoint with only gateway auth and verify upstream acceptance.

## Phase 5 — OAuth2 / OIDC / Provider OAuth (Gateway-Terminated) + Refresh

### Scope
- Add gateway-terminated OAuth flows for upstream servers requiring OAuth:
  - Start flow endpoint returning authorization URL (per server/provider config)
  - Callback endpoint storing tokens bound to the principal+server
  - Disconnect endpoint revoking and removing tokens where supported
- Add refresh support:
  - On-demand refresh when expired/near-expiry during request-time resolution
  - Optional: background refresh task later; in this phase on-demand is sufficient
- Inject upstream access token on proxy requests.

### Provider client secrets (required)

- OAuth provider client credentials (client id/secret) MUST be sourced from an external secrets backend (Vault/KMS) where available.
- Dev/local fallback MUST support env/file configuration (no external dependency required for tests).

### Deliverables
- OAuth provider configuration model for upstream usage (distinct from registry UI login providers)
- Token exchange implementation + secure token storage
- Resolver support for `oauth2`, `oidc`, `provider-oauth`

### Tests (must pass)
- `make test-unit` (required)
- `make test-integration` (required)
- Unit tests for:
  - state binding and CSRF properties
  - token refresh behavior and error handling
- Integration tests using a local stub OAuth/OIDC server (no external network) to validate:
  - end-to-end connect → store → inject
  - refresh on expiry

### Local verification (manual)

- Start the stub OAuth/OIDC provider and complete a connect flow through the gateway UI/callback endpoints, then proxy an MCP request to verify injection.

## Phase 6 — WebSocket and stdio Coverage (If In-Scope for This Repo)

### Scope
- WebSocket: ensure ingress auth + upstream auth injection works on WS upgrade and upstream connection establishment.
- stdio/pipes: implement and test the HTTP/WS bridge strategy (gateway remains header-based to the bridge).

### Tests (must pass)
- `make test-unit` (required)
- `make test-integration` (required)
- Add transport-specific tests as applicable.

## Future (Not Implemented in This Plan): mTLS

mTLS upstream support is explicitly deferred. When prioritized, add a new phased plan covering:

- secure storage and materialization of client cert + private key
- Nginx client-certificate presentation mechanics
- rotation and reload strategy
- integration tests with a local TLS upstream requiring client certs
