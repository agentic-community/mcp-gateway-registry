# Session State — Latest

## Last Completed Work
- Repo initialized
- Base architecture files created
- Persistence backend for phase 1 selected
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

## Current Task
- Stage 4 identity resolver: `enforceai/plans/stage-4-identity-resolver-phased-plan.md` (Phase 4.1 next)

## Next Steps
1. Stage 4: wire IdentityResolver + agent binding (`X-Agent-Id`) rules
2. Stage 5: FGAC enforcement + tool visibility filtering
3. Stage 6: management APIs + CLI (self-service)

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

## Outstanding Questions
