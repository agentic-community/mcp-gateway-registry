# Stage 3 — Generic OIDC Provider (Multi-Issuer, Local JWKS Cache)
*Created: 2025-12-13*

## Goal
Implement a generic OIDC JWT validator that supports multiple issuers, validates signatures via JWKS with local caching, and normalizes claims into an EnforceAI-friendly shape (user_id + scopes + roles-for-audit).

## Non-Goals (Stage 3)
- No request-path wiring into `auth_server/server.py` (Stage 4/5)
- No agent binding (`X-Agent-Id`) checks (Stage 4)
- No FGAC enforcement or tool filtering (Stage 5)
- No management endpoints or CLI (Stage 6)
- No disk-backed JWKS cache (memory-only)

## Inputs (Locked Decisions)
- OIDC is generic and multi-issuer, configured via `OIDC_ISSUERS` issuer map.
- `user_id` is derived as `"<iss>|<sub>"`.
- Scopes claim precedence defaults: `scp` → `scope` → `permissions` (per-issuer overrides allowed).
- Roles/groups claim precedence defaults: `roles` → `groups` → `permissions` (per-issuer overrides allowed); roles/groups are audit-only.
- Error semantics:
  - `401` for missing/invalid token (unknown issuer, bad signature, bad aud, expired, malformed)
  - `503` for internal dependency failures (e.g., cannot fetch JWKS when required to decide)

## Proposed Code Layout (within `auth_server/enforceai/`)
- `oidc/`
  - `models.py` (Pydantic models: issuer config, validated token output)
  - `claims.py` (claim extraction helpers: aud/scopes/roles normalization)
  - `jwks.py` (JWKS fetch + in-memory cache + refresh logic)
  - `verify.py` (OIDC JWT verification pipeline; multi-issuer selection)

## Test Strategy (must pass for every phase)
Per-phase gate (fast and deterministic):
- `uv run python -m py_compile <changed_files...>`
- `uv run pytest -q -o addopts='' tests/unit/enforceai -k oidc`
- `.venv/bin/python -m pytest` (full suite + coverage gate)

End-of-stage gate (Stage 3 completion):
- `make test-unit`
- `.venv/bin/python -m pytest` (full suite + coverage gate)

Unit tests must not require network access:
- Mock all JWKS fetches (patch internal fetcher; do not hit real URLs).

---

## Phase 3.1 — OIDC Issuer Config Model + Settings Alignment
**Goal**: align `EnforceAISettings` and issuer config to Stage 3 requirements without implementing verification yet.

### Scope (single run)
- Update `OIDCIssuerConfig` to support:
  - `jwks_uri` (or accept legacy `jwks_url` alias temporarily)
  - `audiences: list[str]` (accept string alias for backwards compatibility if needed)
  - `scope_claims: list[str]` and `role_claims: list[str]` with sensible defaults
  - `algorithms: list[str]` default `["RS256"]`
  - optional tuning knobs (keep minimal): `jwks_cache_ttl_seconds`, `clock_skew_seconds`
- Strengthen validation:
  - issuer keys non-empty and trimmed (already present)
  - `jwks_uri` non-empty; validate scheme (`https://` preferred; allow `http://` only if explicitly documented for local dev)
  - audience list non-empty strings

### Tests to add
- `tests/unit/enforceai/test_oidc_config_models.py`
  - parses single-issuer and multi-issuer maps
  - rejects invalid JWKS URI and empty audiences
  - validates default claim precedence fields exist and are stable

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai -k oidc`)

---

## Phase 3.2 — Claim Normalization Helpers (aud, scopes, roles)
**Goal**: normalize common OIDC claim shapes deterministically (string vs list; space-delimited scopes).

### Scope (single run)
- Implement helpers that:
  - normalize `aud` to list form and validate against configured `audiences`
  - extract scopes using claim precedence (configurable per issuer):
    - support `scp` list, `scope` space-delimited string, `permissions` list/string
  - extract roles for audit using claim precedence (configurable per issuer):
    - support list and string forms
  - never grant scopes from roles (roles are returned separately for audit/metadata only)

### Tests to add
- `tests/unit/enforceai/test_oidc_claims_normalization.py`
  - audience matching for `aud` as string vs list
  - scopes parsing across `scp`, `scope`, `permissions` with string/list variants
  - roles parsing across `roles`, `groups`, `permissions` with string/list variants
  - empty/whitespace-only entries are dropped

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai -k oidc`)

---

## Phase 3.3 — JWKS Fetch + In-Memory Cache (per issuer)
**Goal**: implement a JWKS cache that refreshes on TTL and supports a single forced refresh when required.

### Scope (single run)
- Implement a per-issuer JWKS cache with:
  - `get_jwks(issuer)`: returns cached set if fresh; otherwise fetches and caches
  - `refresh_jwks(issuer)`: forces a fetch (used for rotation/missing-kid scenarios)
  - TTL-based invalidation (`jwks_cache_ttl_seconds`)
  - safe failure behavior:
    - if cache is empty and fetch fails, surface as EnforceAI `503` dependency error
    - if cache is present and fetch fails, continue using stale cache only if doing so preserves “fail closed” semantics for the request (default: treat as `503` when the key needed to verify cannot be obtained)
- Use `httpx` for the runtime fetcher with strict timeouts.
- Design for testability: inject a fetch function/client to avoid real network.

### Tests to add
- `tests/unit/enforceai/test_oidc_jwks_cache.py`
  - first request fetches and caches
  - repeated requests within TTL do not fetch again
  - TTL expiry triggers refresh
  - fetch failure without cache maps to dependency failure
  - fetch call counts asserted (no per-request fetching)

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai -k oidc`)

---

## Phase 3.4 — OIDC JWT Verification (Multi-Issuer + JWKS Selection)
**Goal**: validate OIDC JWTs generically: issuer selection, signature verification, audience checks, time checks, and normalized outputs.

### Scope (single run)
- Implement OIDC verification pipeline:
  - read unverified `iss` and `kid` (for issuer selection and key lookup)
  - reject unknown issuer (`401`)
  - obtain JWKS via cache
  - select key by `kid`; if missing, force-refresh JWKS once and retry (rotation scenario)
  - verify signature and accepted algorithms
  - validate `exp` (required), `iat` sanity (with clock skew), `aud` match
  - derive:
    - `user_id = f\"{iss}|{sub}\"`
    - normalized scopes + roles (via Phase 3.2 helpers)
- Return a typed “validated OIDC token” output model (issuer, user_id, subject, scopes, roles, raw claim subset for audit).

### Tests to add
- `tests/unit/enforceai/test_oidc_verify_multi_issuer.py`
  - unknown issuer → `401`
  - wrong audience → `401`
  - expired token → `401`
  - signature invalid/tampered → `401`
  - algorithm mismatch → `401`
  - multi-issuer routing works (two issuers, different JWKS and audiences)
- `tests/unit/enforceai/test_oidc_jwks_rotation.py`
  - missing `kid` triggers a single refresh and then succeeds when JWKS includes new key

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai -k oidc`)

---

## Phase 3.5 — Hardening + Integration-Lite Roundtrip Test
**Goal**: add regression tests for edge cases and prove the Stage 3 components work together end-to-end without request-path wiring.

### Scope (single run)
- Add hardening tests:
  - malformed token formats map to `401` (no stack traces to caller)
  - `iss`/`sub` missing or empty → `401`
  - `aud` missing when required → `401`
  - ensure logs do not contain raw tokens or JWKS payloads (caplog-based checks)
- Add an integration-lite test (still offline) that:
  - uses a mocked JWKS fetcher
  - validates a token for issuer A and issuer B
  - asserts cache prevents refetch in repeated validation

### Tests to add
- `tests/unit/enforceai/test_oidc_hardening.py`
- `tests/integration/test_enforceai_oidc_roundtrip.py` (optional, but preferred for end-to-end wiring of cache + verify)

### Exit criteria (Stage 3 completion gate)
- `make test-unit` passes
- `.venv/bin/python -m pytest` passes
- `enforceai/session_state/latest.md` updated with Stage 3 completion and pointer to Stage 4
