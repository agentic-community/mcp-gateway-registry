# Session State — Latest

## Last Completed Work
- Repo initialized
- Base architecture files created
- Persistence backend for phase 1 selected
- Stage 0.1: created `auth_server/enforceai/` package skeleton with core contracts + unit tests
- Stage 0.2: added env-driven EnforceAI settings parsing + validation with unit test coverage
- Stage 0.3: added reusable pytest fixtures (RSA keys, temp SQLite, env helpers) with unit test coverage

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
- Execute Stage 0 foundation plan: `enforceai/plans/stage-0-foundation-phased-plan.md`
- Stage 1: begin runtime integration (IdentityResolver scaffold per plan)

## Next Steps
1. Stage 1.1: IdentityResolver scaffold (no enforcement yet)
2. Stage 1.2: agent registry interface + models
3. Stage 1.3: gateway token verification + effective scopes (token ∩ agent)

## Tests Executed
- `uv run python -m py_compile auth_server/enforceai/*.py tests/unit/enforceai/*.py` (pass)
- `uv run pytest -q -o addopts='' tests/unit/enforceai` (8 passed)
- `uv run python -m py_compile auth_server/enforceai/config.py tests/unit/enforceai/test_config_*.py` (pass)
- `uv run pytest -q -o addopts='' tests/unit/enforceai` (15 passed)
- `uv run python -m py_compile tests/fixtures/enforceai_fixtures.py tests/unit/enforceai/test_test_fixtures.py tests/conftest.py` (pass)
- `uv run pytest -q -o addopts='' tests/unit/enforceai` (18 passed)

## Outstanding Questions
