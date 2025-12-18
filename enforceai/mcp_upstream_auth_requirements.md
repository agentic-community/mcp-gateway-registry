# Upstream MCP Authentication Support Requirements (Gateway-Terminated)
*Created: 2025-12-17*

## Background

`enforceai/AUTHENTICATION.md` defines common authentication methods used by MCP servers and an architecture where the client authenticates only to the gateway while the gateway authenticates to upstream MCP servers on the client’s behalf.

This document specifies the functional and non-functional requirements, code changes, UI/UX components, and test coverage needed to implement full upstream authentication support in this repository.

## Core Clarification (Non-Negotiable)

- The agent/client MUST authenticate only to the gateway.
- The agent/client MUST NOT authenticate directly to upstream MCP servers.
- All upstream connections and upstream credentials MUST be managed by the gateway (gateway-terminated authentication).
- Client-visible MCP configurations MUST NOT require upstream credentials.

## Goals

1. Support all upstream authentication categories defined in `enforceai/AUTHENTICATION.md`:
   - No authentication
   - API keys
   - OAuth 2.x bearer tokens
   - OAuth + OpenID Connect (OIDC)
   - Provider-delegated OAuth (GitHub/Google/Slack-style providers)
   - JWT bearer tokens
   - Gateway-terminated header trust model
   - Mutual TLS (mTLS) is explicitly deferred (future).
2. Make upstream authentication requirements visible and actionable in the UI.
3. Ensure upstream credentials are managed by the gateway (not required in client configs).
4. Provide a consistent identity context contract forwarded to upstream servers regardless of upstream auth method.
5. Add comprehensive unit and integration test coverage for credential resolution, injection, and transport behavior.

## Non-Goals (for initial implementation, unless explicitly required)

- Implementing agent-side OAuth flows or requiring agents to manage upstream tokens.
- Standardizing upstream token formats beyond what’s required to satisfy upstream servers.
- Long-term secrets/vault integration implementation details (this document defines interfaces and requirements; concrete vault backends can be staged).

## Definitions

- **Ingress auth**: Client (agent/UI/CLI) authenticates to the gateway/auth server.
- **Upstream auth**: Gateway authenticates to the upstream MCP server when proxying a request.
- **Gateway-terminated**: Client does not send upstream credentials; gateway resolves and injects them.
- **Client-supplied upstream auth (passthrough)**: explicitly not supported; the edge must strip and reject attempts to supply upstream credentials from clients.

## Current State (Key Observations)

- Server registry data supports `auth_type`, `auth_provider`, and optional `headers` on server entries (registration endpoints accept these fields).
- The React server dashboard consumes `GET /api/servers` which currently does not include `auth_type` / `auth_provider`, so upstream auth is not visible in the UI.
- The “MCP config” modal in the React UI assumes a single bearer token header and does not model per-server upstream auth requirements.
- The `.well-known/mcp-servers` endpoint includes an `authentication` block derived from `auth_type`, but the UI does not consume it.
- Existing OAuth tooling (`credentials-provider/`, `.oauth-tokens/`, legacy token refresh service) primarily supports token generation and refresh for client-side configuration rather than gateway-managed upstream auth.
- EnforceAI management UI exists for ingress credentials (API keys and gateway tokens) and is not currently aligned to upstream server authentication needs.

## Target State Overview

The system must support:

1. **Declarative per-server upstream auth requirements** (what the upstream server expects).
2. **Per-principal credential management** (how the gateway authenticates to the upstream server on behalf of a principal), including lifecycle and rotation.
3. **Credential resolution + injection** on every proxied MCP request across supported transports.
4. **UI/UX flows** to view requirements, connect provider OAuth, upload or create API keys, configure JWTs, and manage upstream credentials (mTLS is future).
5. **Testing** that proves correctness for each auth method and transport.

## Requirements by Capability

### A. Canonical Identity Context Forwarding (Applies to all upstream modes)

#### Functional requirements

1. The gateway MUST forward a canonical identity context to upstream MCP servers in a stable format:
   - `X-MCP-Principal`: canonical principal identifier (e.g., `user:<user_id>` or `agent:<agent_id>`).
   - `X-MCP-Auth-Type`: the ingress authentication mechanism used (e.g., `oidc`, `gateway-token`, `api-key`, `mixed`).
   - `X-MCP-Scopes`: effective scopes for the request (space-delimited string).
   - `X-MCP-Provider`: optional when an upstream provider is involved (e.g., `github`).
   - `X-MCP-Claims`: optional, structured claims (JSON) with strict allowlist to prevent sensitive leakage.
2. The gateway MUST support configuration to map/alias headers for upstream servers that require different naming conventions.
3. The gateway MUST support transport-specific identity propagation:
   - HTTP/SSE/streamable HTTP: headers.
   - WebSocket: initial handshake headers and/or subprotocol metadata.
   - stdio/pipes: environment variables and/or structured handshake metadata (see Transport section).

#### Security requirements

1. The gateway MUST treat identity context headers as “trusted internal” and MUST NOT allow clients to spoof them (strip inbound `X-MCP-*` headers at the edge).
2. The gateway MUST ensure least-privilege disclosure:
   - Default: no raw tokens or secrets in identity headers.
   - Claims are allowlisted, size-limited, and redacted.

#### Implementation changes (expected)

- Nginx and/or registry proxy layer strips inbound `X-MCP-*` headers and re-injects computed values.
- Introduce a shared library/model for “IdentityContext-to-Headers” formatting.

### B. Server Registry Model Extensions (Declarative upstream auth requirements)

#### Functional requirements

1. Server entries MUST include a structured `upstream_auth` object (or an equivalent normalized form) with at least:
   - `mode`: `gateway-managed` | `none`
   - `type`: `none` | `api-key` | `oauth2` | `oidc` | `provider-oauth` | `jwt` | `mtls` | `header-trust`
   - `provider`: optional (e.g., `github`, `google`, `slack`, `custom`)
   - `credential_binding`: `service` | `user` | `agent` | `user+agent` (defines the lookup key for credentials)
   - `injection`: header/query/cookie/handshake/env metadata describing where to inject (default header-based)
   - `authorization_model`: optional (expected scopes/claims, mapping rules)
2. Existing fields (`auth_type`, `auth_provider`, `headers`) MUST be mapped to this new structure for backward compatibility and migration.
3. Validation MUST occur on server registration and update APIs:
   - Reject unsupported combinations.
   - Reject `upstream_auth.type=mtls` until mTLS support is implemented (future).
   - Enforce size limits and safe defaults.

#### UI requirements

1. The server catalog UI MUST display upstream auth requirements for each server:
   - Badge: `Upstream: None`, `Upstream: API Key`, `Upstream: OAuth2`, etc.
   - Provider label where applicable.
2. Server details UI MUST show:
   - Required auth type and injection mechanism.
   - Credential binding (service vs per-user/per-agent).
   - Credential status for the current principal (configured, missing, expired, revoked).

#### Implementation changes (expected)

- Extend server JSON schema and registration routes (`registry/api/server_routes.py`) to accept `upstream_auth` (JSON form field and API JSON variant).
- Update `GET /api/servers` to return `upstream_auth` (and/or `auth_type` and `auth_provider`) so React can display it.
- Update `.well-known/mcp-servers` to reflect the normalized `upstream_auth` contract.

### C. Upstream Credential Store (Gateway-managed secrets)

#### Functional requirements

1. The system MUST provide storage for upstream credentials with:
   - Ownership / binding per `credential_binding` rules.
   - Per-server scoping.
   - Audit trails for create/update/revoke/use (best-effort for “use”, sampled as needed).
2. The store MUST support these credential types:
   - API key secret
   - OAuth2 access + refresh token (plus metadata: expiry, scopes, provider)
   - OIDC tokens (ID token optionally; access/refresh as needed)
   - Provider OAuth tokens (same as OAuth2 but keyed by provider)
   - JWT bearer token material:
     - Either stored as a static token, or
     - Stored as a signing configuration enabling minting per request (issuer, audience, key id)
   - mTLS materials: Future (client certificate and private key, plus optional CA bundle / SNI)
3. The store MUST support lifecycle operations:
   - Create / set
   - Rotate (create new and mark active)
   - Revoke / disable
   - Expiry tracking and health signals

#### Security requirements

1. Secrets MUST NOT be stored in plaintext without an explicit encryption-at-rest mechanism.
2. Secret access MUST be permissioned:
   - Only the proxying component can read decrypted secrets.
   - Management surfaces can create/update/revoke but not read back secrets except on creation when required.
3. Implement logging redaction to avoid token/secret disclosure.

#### Implementation changes (expected)

- Add DB tables (SQLite in Phase 1) and migrations for upstream credentials.
- Add encryption envelope mechanism:
  - Minimum: single master key from env/file for local/dev.
  - Roadmap: pluggable KMS/HSM/Vault integration.

Phase 1 concrete implementation notes:
- SQLite migration: `auth_server/enforceai/db/migrations/sql/0004_upstream_credentials.up.sql` (table: `upstream_credentials`).
- Dev/local KEK file: `ENFORCEAI_UPSTREAM_KEK_PATH` (generated by `scripts/enforceai_dev_bootstrap.sh` under `.enforceai/secrets/upstream_kek` with mode `0600`).

### D. Credential Resolution Engine (Per request)

#### Functional requirements

1. For every proxied MCP request, the system MUST:
   - Identify the target server (by request path mapping).
   - Load server `upstream_auth` requirements.
   - Determine the principal and effective scopes from ingress identity.
   - Resolve an upstream credential according to the server’s binding rules.
   - Inject upstream credential into the upstream request (or fail with actionable error).
2. The resolver MUST support precedence rules:
   - If `mode=gateway-managed`: never accept client-supplied upstream secrets.
3. The resolver MUST provide stable error semantics surfaced to clients and UI:
   - `401` for missing gateway authentication (ingress).
   - `403` for forbidden (FGAC).
   - `424 Failed Dependency` for missing required upstream credentials, with `error_code=UPSTREAM_CREDENTIALS_REQUIRED`.
   - `409` for misconfiguration states (e.g., multiple active credentials where only one allowed).

#### Implementation changes (expected)

- Implement a shared “UpstreamAuthResolver” service (location TBD: `registry/services/` or `auth_server/enforceai/`).
- Ensure the proxy layer can access the resolver and the decrypted credential material at request time.

### E. Upstream Auth Injection (Per auth type)

This section defines minimum injection behaviors for each authentication type.

## Phase 2 Notes (SSRF/Egress Allowlist)
- SQLite migration: `auth_server/enforceai/db/migrations/sql/0005_egress_allowlist.up.sql` (table: `egress_allowlist_entries`).
- Admin management endpoints (auth_server):
  - `GET /enforceai/admin/egress-allowlist`
  - `POST /enforceai/admin/egress-allowlist`
  - `PUT /enforceai/admin/egress-allowlist/{entry_id}`
  - `DELETE /enforceai/admin/egress-allowlist/{entry_id}`
  - `POST /enforceai/admin/egress-allowlist/check`
- Registry registration/update enforcement is enabled when the registry process has `ENFORCEAI_DB_PATH` set to the same EnforceAI SQLite DB used by the auth server.
- Defense-in-depth: Nginx config generation will skip/omit servers whose `proxy_pass_url` is not allowlisted when `ENFORCEAI_DB_PATH` is set.

## Phase 3 Notes (Upstream Credential Management API)
- Management endpoints (auth_server):
  - `GET /enforceai/upstream/servers` (currently lists servers with configured upstream credentials for the caller)
  - `GET /enforceai/upstream/servers/{server_path}/credentials` (metadata only)
  - `POST /enforceai/upstream/servers/{server_path}/credentials` (create/set; returns `secret_payload` only at creation time)
  - `POST /enforceai/upstream/credentials/{credential_id}/revoke`
- Ownership rules:
  - `service` binding requires EnforceAI admin
  - `user` binding is scoped to the caller’s `user_id`
  - `agent` / `user+agent` bindings require the agent to be owned by the caller (admin may set agent-bound creds)

#### E1. No authentication (`type=none`)

- Proxy forwards request with identity context headers only.
- Must be supported for all transports.

#### E2. API keys (`type=api-key`)

Functional requirements:
- Configurable injection:
  - Header (default): `X-API-Key: <secret>` or configured header name.
  - Query parameter (optional): `?api_key=<secret>` (discouraged; must be opt-in).
- Credential binding supports:
  - Service-bound keys (recommended)
  - Principal-bound keys (supported)

#### E3. OAuth2 bearer tokens (`type=oauth2`)

Functional requirements:
- Gateway must store refresh tokens when available and refresh access tokens on-demand during request-time resolution (Phase 1).
- Injection default:
  - `Authorization: Bearer <access_token>` to upstream.
- Scopes and provider metadata must be tracked.

#### E4. OIDC (`type=oidc`)

Functional requirements:
- Support upstream servers that expect:
  - Access token as bearer, or
  - ID token as bearer, or
  - Both via separate headers (configurable).
- Claims forwarding must be allowlisted.

#### E5. Provider-delegated OAuth (`type=provider-oauth`)

Functional requirements:
- Supports per-provider OAuth configuration:
  - Authorization and token endpoints
  - Client id/secret
  - PKCE support
  - Offline access/refresh token support
  - Provider-specific required headers (e.g., tenant/workspace ids) as typed metadata
- Gateway must support user consent flows from the UI, tied to the authenticated principal.
- Token refresh strategy in Phase 1 is on-demand during request-time resolution (no background refresh worker).

#### E6. JWT bearer tokens (`type=jwt`)

Functional requirements:
Support at least two modes:
1. **Static JWT**: store a bearer token and inject as `Authorization: Bearer <jwt>`.
2. **Minted JWT**: gateway mints a JWT per request using configured signing keys and templates (claims mapping).

Security requirements:
- Strict control over claims templates to avoid privilege escalation.
- Key management and rotation requirements for signing keys.

#### E7. Mutual TLS (`type=mtls`)

Status: Future (not part of the initial implementation plan)

Functional requirements:
- Support upstream mTLS at least for service-bound credentials:
  - Per-server client cert + private key + optional CA bundle.
  - Optional SNI override.
- Must work for HTTP-based upstream transports; for WebSocket, ensure TLS settings apply.

Implementation requirements:
- Nginx proxy config must support `proxy_ssl_certificate` / `proxy_ssl_certificate_key` per server location, or the proxying component must implement equivalent TLS client configuration.

#### E8. Header trust model (`type=header-trust`)

Functional requirements:
- Upstream servers never authenticate requests; they rely on gateway-injected identity context.
- The gateway MUST:
  - Enforce FGAC before proxying.
  - Strip any inbound identity headers from clients.
  - Inject canonical `X-MCP-*` headers.

### F. Transport Support Requirements

#### F1. Streamable HTTP / SSE

- Must preserve streaming semantics for MCP methods.
- Must support header injection without breaking chunked encoding.
- Must support request-body based authorization decisions already in place (if applicable).

#### F2. WebSocket

- Must support:
  - Ingress auth via headers/cookies during upgrade.
  - Upstream auth injection either in handshake headers or via a gateway-side connection to the upstream that includes credentials.
- Provide integration tests with a local WebSocket upstream requiring a bearer token or API key.

#### F3. stdio / pipes

Decision for Phase 1: stdio/pipes are supported via a compatibility adapter (HTTP/WS bridge) that exposes stdio servers behind the gateway.

Requirements:
- The gateway MUST treat bridged stdio servers the same as any other upstream HTTP/WS server for auth purposes (header-based injection).
- Native subprocess management for stdio servers is Future work and out of scope for the initial implementation plan.

### L. SSRF / Egress Controls (Required)

Because upstream server URLs (`proxy_pass_url`) can be attacker-controlled via registration workflows, SSRF defenses are mandatory.

Requirements (Phase 1):
- Enforce a strict egress allowlist for upstream targets.
- Allowlist is DB-managed, admin-gated, audited, and supports optional TTL (`expires_at`).
- Registration/update MUST reject any `proxy_pass_url` that is not allowlisted.
- Proxy-time MUST re-check and fail closed as defense-in-depth (protect against DNS rebinding and redirects).

### G. Management APIs for Upstream Credentials

#### Functional requirements

Add REST endpoints for managing upstream credentials, separated from ingress credentials:

1. Per-server credential CRUD:
   - Create/set credential (secret accepted, returned only once if needed)
   - List credentials (metadata only, redacted secrets)
   - Rotate/revoke credential
2. OAuth/provider OAuth flows:
   - Start flow: returns authorization URL (with state binding to principal and server/provider)
   - Callback handler: stores tokens, marks credential active
   - Disconnect: revoke and delete stored tokens (where provider supports revocation)
3. Status endpoints:
   - Credential status for current principal and server (missing/valid/expired/revoked)
   - Optional “test connection” operation (calls upstream health endpoint if available)

#### Security requirements

- Endpoints must be protected by EnforceAI identity resolver and ownership checks.
- Secrets must never be returned after creation.
- CSRF protections for UI-based flows.

### H. UI/UX Requirements

#### H1. Server Catalog and Details

1. Server cards must show:
   - Upstream auth requirement badge.
   - Credential status indicator for the current principal (configured/required).
2. Server details view must provide:
   - A “Connect” / “Configure” action appropriate to `upstream_auth.type`.
   - Clear explanation of what is required and what is stored by the gateway.
   - A “Test upstream auth” action (where supported).

#### H2. Credentials Area (Upstream)

Add a new “Upstream Credentials” section distinct from EnforceAI ingress credentials:

- List servers requiring upstream credentials.
- Filter by status (missing/expired/configured).
- Per-server management page:
  - API key: set/rotate/revoke
  - OAuth/provider OAuth: connect/disconnect, show scopes and expiry
  - JWT: upload static token or configure minted token templates (admin-gated)
  - mTLS: Future (only when backend mTLS is implemented)

#### H3. OAuth Consent UX

- Provide in-browser redirect flow with:
  - Server/provider selection
  - State parameter binding to user session and server id
  - Clear success/failure pages
  - Audit event emission

#### H4. MCP Client Config Modal

The generated MCP client configuration shown in the UI must be updated to reflect gateway-terminated upstream auth:

- Always include only the gateway ingress credential placeholder in the client config (never upstream tokens/keys/certs).
- Clearly distinguish between:
  - Gateway auth header (client → gateway)
  - Gateway-managed upstream auth (gateway → upstream), which is configured in the gateway UI/management surface

### I. Observability and Audit

#### Requirements

1. Emit audit events for:
   - Upstream credential create/rotate/revoke
   - OAuth connect/disconnect and token refresh outcomes
   - Upstream credential resolution failures during proxying
2. Metrics:
   - Upstream auth method distribution per server
   - Token refresh success/failure
   - Proxy failures by error code (`UPSTREAM_CREDENTIALS_REQUIRED`, `UPSTREAM_TOKEN_EXPIRED`, etc.)
3. Logs must be redacted and structured.

### J. Testing Requirements

#### J1. Unit tests (required)

1. Server config validation for `upstream_auth`.
2. Credential binding logic:
   - service vs user vs agent vs user+agent
3. Resolver precedence rules (gateway-managed only; client-supplied upstream secrets are rejected/stripped).
4. Header formatting and stripping rules:
   - inbound `X-MCP-*` stripped
   - gateway-injected values stable
5. Token refresh logic:
   - refresh threshold behavior
   - handling revoked/invalid refresh tokens

#### J2. Integration tests (required)

Provide a set of fake upstream MCP servers used only for tests:

1. **No-auth upstream**: asserts receipt of `X-MCP-Principal`.
2. **API-key upstream**: requires `X-API-Key` (or configured header); gateway injects from store.
3. **OAuth2 upstream**: requires `Authorization: Bearer <token>`; gateway injects and refreshes if expired.
4. **JWT upstream**: requires bearer JWT; test static and minted modes.
5. **Header-trust upstream**: rejects if `X-MCP-Principal` missing; ensures spoofing is blocked.
6. **mTLS upstream**: Future (only when mTLS is implemented).

Each integration test must verify:
- Correct HTTP status codes and error codes for missing/expired credentials.
- Streaming behavior preserved for SSE/streamable HTTP endpoints.

#### J3. UI tests (required)

Add/extend frontend tests to cover:
- Server card displays upstream auth badge and status.
- Credential configuration flows render correctly for each auth type.
- OAuth connect flow UI states (loading/success/error).
- MCP config modal shows correct guidance for gateway-terminated upstream auth.

### K. Documentation Requirements

1. Update `docs/auth.md` to clearly separate ingress vs upstream auth responsibilities and document gateway-terminated behavior.
2. Add a new operator guide for upstream credential management:
   - configuration
   - rotation
   - troubleshooting
3. Document header trust model (`X-MCP-*`) and any compatibility mappings to existing headers (`X-User`, `X-Scopes`, etc.).

## Acceptance Criteria (Definition of Done)

1. A server can be configured with any `upstream_auth.type` listed above.
2. The UI shows the upstream auth requirement and current principal credential status.
3. Gateway-managed upstream auth works end-to-end for:
   - API key injection
   - OAuth2/provider OAuth injection (including refresh where available)
   - JWT injection (static; minted if implemented)
   - header trust model
4. Automated tests cover:
   - Resolver and validation unit tests
   - Integration tests with mock upstream servers for each method
   - UI tests for visibility and management flows
5. No secrets are logged or returned after creation; encryption-at-rest requirements are met.

## Dev/Local Development and Testing (Required)

This section defines how developers and CI should develop and test gateway-terminated upstream auth without relying on real external providers or production secrets.

### Token/Secret Storage in Dev (Local)

1. **Recoverable upstream secrets MUST be encrypted-at-rest in dev**, not stored as plaintext:
   - upstream API keys
   - upstream OAuth access/refresh tokens
   - upstream static JWT bearer tokens
   - upstream mTLS private keys (future)
2. Dev/local MUST use a gitignored master key file (KEK) stored under `.enforceai/`:
   - Example path: `.enforceai/secrets/upstream_kek` with file mode `0600`
   - The DB stores `{kid, nonce, ciphertext, tag}` (or equivalent) plus non-secret metadata
3. Any browser/UI flow MUST show secrets only once on creation and MUST NOT store them in browser storage.

### Deterministic Test Providers (No External Network)

1. Automated tests MUST NOT depend on real OAuth providers or external network calls.
2. Provide a stub OAuth/OIDC server used only for tests that implements:
   - `/authorize` (simulated auth code issuance)
   - `/token` (code exchange and refresh)
   - `/revoke` (optional)
   - JWKS endpoint for OIDC-style validation where required
3. Tests MUST be able to run in local/dev without any internet access.

### Local Upstream Test Servers (Per Auth Type)

Provide lightweight upstream MCP test servers (or HTTP proxies that simulate MCP auth expectations) for integration tests:

- `none`: asserts receipt of identity context headers (`X-MCP-Principal`).
- `api-key`: requires a configured header (default `X-API-Key`).
- `oauth2/provider-oauth`: requires `Authorization: Bearer <token>`.
- `jwt`: requires `Authorization: Bearer <jwt>`.
- `header-trust`: rejects if `X-MCP-*` identity headers are missing and verifies inbound spoofing is blocked.
- `mtls`: Future (only when mTLS is implemented).

### Developer Workflow (Manual Smoke Test)

Dev/local must support a repeatable, one-command reset and bootstrap:

1. Reset: delete `.enforceai/` (or run the existing bootstrap with `--force` if supported).
2. Bootstrap:
   - create SQLite DB
   - create encryption KEK
   - create any required signing keys for gateway tokens (if used)
3. Start the stack, configure one server with upstream auth requirements, configure the upstream credential via management API/UI, then verify:
   - client sends only gateway credential
   - upstream receives injected credential and identity context
   - missing upstream credential produces a distinct, actionable error code

## Schemas and Contracts (Normative)

This section defines the minimum stable schemas and runtime contracts required to implement upstream auth support. These are intended to be treated as source-of-truth for implementation and tests.

### Decisions Locked for This Implementation

- **Injection architecture**: `auth_server` `/validate` computes upstream injection; Nginx applies injection to upstream requests (Nginx `auth_request` pattern).
- **mTLS**: not a priority; may be deferred or implemented later.
- **stdio/pipes**: supported via an HTTP/WS bridge process that exposes stdio servers behind the gateway; gateway auth/injection remains header-based to the bridge.
- **`X-MCP-Scopes`**: space-delimited string.
- **`X-MCP-Claims`**: allowlisted JSON; initial allowlist excludes PII and has no canonical tenant/org id yet.
- **Missing upstream credential status**: `424 Failed Dependency` with stable `error_code=UPSTREAM_CREDENTIALS_REQUIRED`.
- **Upstream OAuth client secrets**: stored in an external secrets manager (Vault/KMS) with a dev/local fallback to env/file configuration.
- **Token refresh strategy (Phase 1)**: on-demand refresh only (no background refresher).
- **SSRF/egress policy**: strict allowlist; allowlist is DB-managed with optional TTL, admin-gated, and audited.
- **Tenancy (Phase 1)**: single-tenant (no canonical `tenant_id/org_id` added to `IdentityContext`).

### Server Config: `upstream_auth`

#### Canonical shape

```json
{
  "upstream_auth": {
    "mode": "gateway-managed",
    "type": "oauth2",
    "provider": "github",
    "credential_binding": "user",
    "injection": {
      "kind": "header",
      "authorization_header": "Authorization",
      "api_key_header": "X-API-Key",
      "api_key_query_param": null,
      "use_id_token_as_bearer": false,
      "additional_headers": {
        "X-Upstream-Tenant": "{metadata.tenant_id}"
      }
    },
    "authorization_model": {
      "expected_scopes": ["repo", "read:user"],
      "scope_source": "stored"
    }
  }
}
```

#### Valid enums

- `mode`: `gateway-managed` | `none`
- `type`: `none` | `api-key` | `oauth2` | `oidc` | `provider-oauth` | `jwt` | `mtls` | `header-trust`
- `credential_binding`: `service` | `user` | `agent` | `user+agent`
- `injection.kind`: `header` | `query` | `cookie` | `handshake` | `env`

#### Backward compatibility mapping

Existing fields MUST map into `upstream_auth` as follows:

- If `auth_type` is missing or equals `none`:
  - `upstream_auth.mode=none`, `upstream_auth.type=none`
- If `auth_type` equals `api_key` (legacy server configs):
  - `upstream_auth.mode=gateway-managed`, `upstream_auth.type=api-key`
  - `upstream_auth.injection.kind=header`, `api_key_header` defaults to `X-API-Key`
- If `auth_type` equals `oauth` (legacy server configs):
  - `upstream_auth.mode=gateway-managed`, `upstream_auth.type=oauth2`
  - `upstream_auth.provider` set from `auth_provider` when present
  - `upstream_auth.injection.kind=header`, `authorization_header` defaults to `Authorization`

### Upstream Credential Records

#### Storage (high level)

- Metadata is stored in plaintext (for listing and status).
- Secrets are stored encrypted-at-rest (recoverable) for upstream injection.
- Secrets are returned only once on creation, never thereafter.

#### Credential types and secret payloads

1. `api-key`:
   - secret payload: `{ "api_key": "<string>" }`
2. `oauth2` / `provider-oauth`:
   - secret payload: `{ "access_token": "<string>", "refresh_token": "<string|null>" }`
   - metadata: `{ "expires_at": "<rfc3339|null>", "scopes": ["..."], "provider": "<string|null>" }`
3. `oidc`:
   - secret payload: `{ "access_token": "<string>", "refresh_token": "<string|null>", "id_token": "<string|null>" }`
4. `jwt` (static):
   - secret payload: `{ "jwt": "<string>" }`
5. `mtls`:
   - status: Future (not implemented in the initial plan)
   - secret payload: `{ "client_cert_pem": "<string>", "client_key_pem": "<string>", "ca_bundle_pem": "<string|null>" }`
6. `header-trust`:
   - no secret payload (identity headers only)

### Management API Contracts (Normative)

These endpoints are for managing upstream credentials (gateway -> upstream). They are distinct from ingress credentials (client -> gateway).

#### Endpoint set

- `GET /enforceai/upstream/servers`
  - Returns servers that have `upstream_auth.mode=gateway-managed` and require a credential, including status for the caller’s effective principal.
- `GET /enforceai/upstream/servers/{server_id}/credentials`
  - Returns credential metadata only (never secret fields).
- `POST /enforceai/upstream/servers/{server_id}/credentials`
  - Creates or replaces the active credential for the caller’s principal binding.
  - Returns the secret once if applicable.
- `POST /enforceai/upstream/credentials/{credential_id}/revoke`
  - Revokes a credential; must be idempotent.
- OAuth flows:
  - `POST /enforceai/upstream/servers/{server_id}/oauth/start`
  - `GET  /enforceai/upstream/servers/{server_id}/oauth/callback`
  - `POST /enforceai/upstream/servers/{server_id}/oauth/disconnect`

#### Error semantics (minimum)

All error responses MUST be JSON with a stable machine-readable `error_code`:

- `UPSTREAM_CREDENTIALS_REQUIRED`: request attempted but no upstream credential exists for required binding
- `UPSTREAM_CREDENTIALS_EXPIRED`: credential exists but is expired and refresh failed or is not possible
- `UPSTREAM_CREDENTIALS_REVOKED`: credential exists but revoked
- `UPSTREAM_AUTH_MISCONFIGURED`: server config invalid/unsupported combination
- `UPSTREAM_OAUTH_STATE_INVALID`: OAuth callback state binding invalid or expired
- `UPSTREAM_OAUTH_TOKEN_EXCHANGE_FAILED`: provider token endpoint rejected exchange
- `UPSTREAM_MTLS_CONFIG_REQUIRED`: Future (reserved for mTLS support)

### Proxy-Time Injection Contract (Headers)

To make upstream auth gateway-terminated, the proxy layer needs two independent concepts:

1. Identity context to upstream (`X-MCP-*`)
2. Upstream credential injection (bearer token, API key, etc.)

#### Canonical upstream identity headers (sent to upstream)

- `X-MCP-Principal`: `user:<user_id>` or `agent:<agent_id>` (canonicalized)
- `X-MCP-Auth-Type`: ingress auth mechanism used to authenticate to the gateway (not the upstream type)
- `X-MCP-Scopes`: effective scopes (format must be consistent; recommended: space-delimited string)
- `X-MCP-Provider`: optional, only when upstream credential is provider-specific
- `X-MCP-Claims`: optional JSON, allowlisted and size-limited

##### `X-MCP-Claims` allowlist (initial)

Best-practice initial allowlist (no PII by default):

- `user_id` (canonical gateway identifier; equals EnforceAI `IdentityContext.user_id`)
- `agent_id` (canonical agent identifier; equals EnforceAI `IdentityContext.agent_id`)
- `provider` (ingress provider; equals EnforceAI `IdentityContext.provider`)

Notes:
- `tenant_id` / `org_id` are intentionally excluded until the system defines a canonical tenancy attribute in `IdentityContext` (or a strict mapping into `IdentityContext.metadata`).
- `email` is intentionally excluded by default (PII). Add only with a documented justification and explicit allowlist update.

#### Internal-only upstream injection headers (never accepted from clients)

When using Nginx `auth_request`, the auth service may return internal response headers that Nginx copies into the upstream request. These headers MUST be stripped from any inbound client requests.

Recommended internal response headers from `/validate`:

- `X-EnforceAI-Upstream-Authorization`: value for upstream `Authorization` header (e.g., `Bearer <token>`)
- `X-EnforceAI-Upstream-Api-Key`: raw API key secret (only in proxy path, never returned by management)
- `X-EnforceAI-Upstream-Api-Key-Header`: name of the header to place the API key into (default `X-API-Key`)
- `X-EnforceAI-Upstream-Mode`: `none|api-key|bearer|header-trust` (for debugging; optional)

Proxy behavior requirements:

1. The edge MUST ignore/strip inbound `X-MCP-*` and `X-EnforceAI-Upstream-*` headers from clients.
2. Only gateway-generated values may populate these headers to upstream servers.

## Risks, Concerns, and Required Clarifications

This section lists issues that should be explicitly decided or clarified before implementation to avoid ambiguity and unsafe defaults.

1. **Where upstream injection happens**: DECIDED (Nginx `auth_request` with `/validate` computing injection).
2. **Header format for `X-MCP-Scopes` and `X-MCP-Claims`**: DECIDED (space-delimited scopes; allowlisted JSON claims).
3. **Status code for “missing upstream credential”**: DECIDED (`424` + stable `error_code`).
4. **OAuth provider configuration source**: DECIDED (external secrets manager + dev env/file fallback).
5. **mTLS operational model**: Future (explicitly de-prioritized; decide when mTLS is prioritized).
6. **SSRF and egress control**: DECIDED (strict allowlist, DB-managed with optional TTL, admin-gated).
7. **Multi-tenant boundary**: DECIDED (Phase 1 is single-tenant).
8. **Token refresh strategy**: DECIDED (Phase 1 is on-demand refresh only).
9. **Nginx logging and secret leakage**:
   - Using injection headers means raw upstream secrets may transit Nginx memory.
   - Production configs MUST NOT enable Nginx debug logging for request headers, and log redaction MUST be validated in tests.
