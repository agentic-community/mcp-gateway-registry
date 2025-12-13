# Stage 4 — IdentityResolver + IdentityContext Assembly
*Created: 2025-12-13*

## Goal
Unify OIDC, gateway tokens, and API keys into a single, consistent `IdentityContext` constructed once per request, with strict credential disambiguation and correct `401/403/503` semantics.

## Non-Goals (Stage 4)
- No wiring into `auth_server/server.py` request path (Stage 5)
- No FGAC enforcement decisions or `tools/list` filtering (Stage 5)
- No management endpoints or CLI for minting/creating keys (Stage 6)
- No external network in unit tests (mock JWKS fetches)

## Inputs (Locked Decisions)
- Credential transport:
  - Canonical: `Authorization: Bearer <token>` for OIDC and gateway tokens
  - Fallback: `X-Gateway-Token: <token>` (gateway tokens only)
  - API keys: `X-API-Key: eak_<key_id>.<secret>`
  - Reject multi-credential requests (`401`)
- Agent binding:
  - `gateway-token`: `agent_id` from token claim
  - `api-key`: `agent_id` from key record
  - `oidc`: requires `X-Agent-Id` (UUIDv4), validated against registry for that `user_id` (`403` if missing/invalid/not-owned)
- Revocation (fail closed):
  - deny if `agent.revoked_at` is set
  - deny if `jti` is revoked
  - deny if `token.iat < agent.tokens_valid_after`
- Effective scopes:
  - `gateway-token`: `token.scopes ∩ agent.scopes`
  - `api-key`: `api_key.scopes ∩ agent.scopes` (or `agent.scopes` if key scopes unset)
  - `oidc`: authorization is agent-scoped; IdP-derived scopes/roles are metadata only
- Error semantics:
  - `401` missing/invalid credentials
  - `403` authenticated-but-denied (binding failures, revoked agent/credential)
  - `503` internal dependency failures (cannot safely decide; deny but signal retry)
- Auth mode selection (config):
  - `AUTH_PROVIDER = oidc | api-key | gateway-token | mixed`
  - In `mixed`, bearer tokens are routed by `iss` (gateway issuer vs configured OIDC issuers)

## Proposed Code Layout (within `auth_server/enforceai/`)
- `auth/`
  - `credentials.py` (extract exactly one credential source + agent header)
  - `resolver.py` (IdentityResolver orchestration)
- `providers/`
  - `api_key.py` (API key verification + effective scopes)
  - `gateway_token.py` (gateway token verify + revocation + binding)
  - `oidc.py` (OIDC verify + agent binding)
- `secrets/`
  - `pepper.py` (load pepper from file path; secret-safe)

This stage should reuse existing primitives:
- `auth_server/enforceai/identity.py` (`IdentityContext`)
- `auth_server/enforceai/oidc/*` (Stage 3 verifier + JWKS cache)
- `auth_server/enforceai/tokens/*` (Stage 2 gateway token verify)
- SQLite stores from Stage 1 (`SqliteAgentStore`, `SqliteApiKeyStore`, `SqliteRevocationStore`)

## Test Strategy (must pass for every phase)
Per-phase gate (required):
- `.venv/bin/python -m py_compile <changed_files...>`
- `.venv/bin/python -m pytest` (full suite + coverage gate)

Unit tests must not require network access:
- JWKS fetches must be mocked (inject fetcher into `JWKSCache`).

---

## Phase 4.1 — Credential Extraction + Ambiguity Rejection
**Goal**: deterministically select exactly one credential source and extract `X-Agent-Id` when present.

### Scope (single run)
- Implement credential extraction:
  - Accept `Authorization: Bearer <token>` and `X-Authorization: Bearer <token>` (if still needed by upstream)
  - Accept `X-Gateway-Token: <token>` (gateway-only fallback)
  - Accept `X-API-Key: eak_<key_id>.<secret>`
  - Reject ambiguous multi-credential requests (`401`)
- Normalize into a small internal model (e.g., `CredentialInput`):
  - `kind: bearer | gateway-token | api-key`
  - `value: str`
  - optional `agent_id_header: Optional[str]` (raw `X-Agent-Id`)

### Tests to add
- `tests/unit/enforceai/test_identity_credentials.py`
  - bearer parsing accepts valid value, rejects non-bearer auth scheme
  - multi-credential combinations rejected (`Authorization` + `X-Gateway-Token`, `X-API-Key` + token)
  - missing credentials produces `UnauthorizedError`
  - `X-Agent-Id` passthrough captured but not validated yet

### Exit criteria
- Full suite passes: `.venv/bin/python -m pytest`

---

## Phase 4.2 — API Key Provider (Pepper + Verify + Effective Scopes)
**Goal**: verify `X-API-Key` against stored hashed verifier, enforce expiry/revocation, and compute effective scopes.

### Scope (single run)
- Extend settings to include API key pepper:
  - `ENFORCEAI_API_KEY_PEPPER_PATH` (or `API_KEY_PEPPER_PATH` alias)
  - enforce file exists/readable when API-key mode is enabled
  - (optional) add `AUTH_PROVIDER` validation to ensure pepper is present when `api-key` or `mixed` is enabled
- Implement API key parsing:
  - format: `eak_<key_id>.<secret>` (reject malformed -> `401`)
- Implement hashing:
  - `secret_hash = HMAC-SHA256(pepper, secret)` (constant-time comparison)
  - never log the secret, pepper, or full key
- Provider behavior:
  - lookup key by `key_id` in store
  - deny if revoked or expired (`403`)
  - verify secret hash matches (`401`)
  - fetch bound agent record and deny if agent revoked (`403`)
  - compute `effective_scopes` per decision
  - return `IdentityContext(provider="api-key", user_id, agent_id, scopes, metadata=...)`

### Tests to add
- `tests/unit/enforceai/test_api_key_provider.py`
  - malformed key rejected (`401`)
  - unknown key_id rejected (`401`)
  - bad secret rejected (`401`)
  - revoked/expired key rejected (`403`)
  - agent revoked rejected (`403`)
  - effective scopes intersection behavior (key scopes unset vs set)
  - secret-safety: errors/logs do not include raw key/pepper

### Exit criteria
- Full suite passes: `.venv/bin/python -m pytest`

---

## Phase 4.3 — Gateway Token Provider (Verify + Revocation + Binding)
**Goal**: validate gateway tokens and enforce registry + revocation checks.

### Scope (single run)
- Extend settings to include a configured gateway issuer:
  - `ENFORCEAI_GATEWAY_ISSUER` (or `GATEWAY_ISSUER` alias)
  - used to (a) validate `iss` during token verification and (b) route bearer tokens in `mixed` mode
- Provider behavior:
  - accept token from `Authorization: Bearer` (when issuer matches gateway issuer) OR `X-Gateway-Token`
  - verify RS256 token using Stage 2 primitives (keyring cached; misconfig -> `503`)
  - fetch agent record and enforce:
    - agent exists and `agent.user_id == token.sub` (`403` on mismatch/missing)
    - agent not revoked (`403`)
    - token `jti` not revoked (`403`)
    - bulk revoke: deny if `token.iat < agent.tokens_valid_after` (`403`)
  - compute `effective_scopes = token.scopes ∩ agent.scopes`
  - return `IdentityContext(provider="gateway-token", ...)`

### Tests to add
- `tests/unit/enforceai/test_gateway_token_provider.py`
  - happy path with real minted token + temp SQLite stores
  - agent revoked deny (`403`)
  - `jti` revoked deny (`403`)
  - `tokens_valid_after` deny (`403`)
  - agent ownership mismatch deny (`403`)
  - keyring/store dependency failures map to `503`
  - multi-credential rejection handled upstream (Phase 4.1)

### Exit criteria
- Full suite passes: `.venv/bin/python -m pytest`

---

## Phase 4.4 — OIDC Provider (Verify + Agent Binding via `X-Agent-Id`)
**Goal**: validate OIDC JWT and bind it to a gateway-managed agent.

### Scope (single run)
- Provider behavior:
  - verify OIDC JWT via Stage 3 `OIDCVerifier` (JWKS cache injected; failures -> `401`/`503` as appropriate)
  - require `X-Agent-Id` and validate UUIDv4 (`403` if missing/invalid)
  - fetch agent record and enforce:
    - agent exists and `agent.user_id == derived_user_id` (`403`)
    - agent not revoked (`403`)
  - set `effective_scopes = agent.scopes` (IdP scopes/roles are metadata only)
  - return `IdentityContext(provider="oidc", ...)` with metadata including issuer/roles/scopes for audit

### Tests to add
- `tests/unit/enforceai/test_oidc_provider.py`
  - missing/invalid `X-Agent-Id` -> `403`
  - agent not found / wrong owner -> `403`
  - revoked agent -> `403`
  - unknown issuer / wrong audience -> `401` (via mocked JWKS)
  - metadata contains issuer and IdP roles/scopes but does not grant permissions

### Exit criteria
- Full suite passes: `.venv/bin/python -m pytest`

---

## Phase 4.5 — IdentityResolver Orchestration + End-to-End Identity Tests
**Goal**: implement `IdentityResolver` that selects the correct provider and returns a consistent `IdentityContext`.

### Scope (single run)
- Add resolver that:
  - uses Phase 4.1 credential extraction
  - enforces “exactly one credential source” (`401`)
  - selects provider based on `AUTH_PROVIDER`:
    - `oidc`: allow only OIDC bearer tokens
    - `gateway-token`: allow only gateway tokens (bearer with gateway issuer or `X-Gateway-Token`)
    - `api-key`: allow only `X-API-Key`
    - `mixed`: allow all three
  - in `mixed`, route bearer tokens by unverified `iss`:
    - `iss == ENFORCEAI_GATEWAY_ISSUER` => gateway token provider
    - `iss in OIDC_ISSUERS` => OIDC provider
    - otherwise => `401`
    - `X-API-Key` => API key provider
    - `X-Gateway-Token` => gateway token provider
    - `Authorization: Bearer` => bearer routing rules above
  - maps errors to `401/403/503` per `auth_server/enforceai/errors.py`
- Add a single integration-lite test that exercises the resolver end-to-end (no FastAPI wiring):
  - OIDC success with mocked JWKS
  - gateway token success with real minted token
  - API key success with pepper hashing

### Tests to add
- `tests/unit/enforceai/test_identity_resolver.py`
  - happy paths for each mode
  - all error semantics and multi-credential rejection
- `tests/integration/test_enforceai_identity_resolver_roundtrip.py`
  - one test that builds stores + verifier dependencies and resolves identity across all three modes

### Exit criteria (Stage 4 completion gate)
- Full suite passes: `.venv/bin/python -m pytest`
- `enforceai/session_state/latest.md` updated with Stage 4 completion and pointer to Stage 5
