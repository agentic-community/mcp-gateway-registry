# Stage 2 — Cryptography + Gateway Token Primitives (RS256)
*Created: 2025-12-13*

## Goal
Implement gateway token key loading, minting, and verification primitives (RS256) with strong validation, deterministic unit tests, and zero request-path wiring changes.

## Non-Goals (Stage 2)
- No wiring into `auth_server/server.py` or Nginx `auth_request` (Stage 4/5)
- No OIDC/JWKS fetching or validation (Stage 3)
- No API-key hashing/verification logic (later stage)
- No management endpoints or CLI (later stage)
- No background jobs/schedulers

## Inputs (Locked Decisions)
- Token format: see `enforceai/architecture/gateway_tokens.md`
- Algorithm: RS256
- Key management: mounted PEM files; `kid`-based rotation; local-only verification (no network)
- Revocation is mandatory but request-path enforcement is Stage 4/5; Stage 2 provides primitives only
- Tokens must include `exp` (no perpetual tokens)
- Max lifetime target: up to 365 days (PAT-style)

## Proposed Code Layout (within `auth_server/enforceai/`)
- `crypto/`
  - `keyring.py` (load/caches PEM keys; select verify key by `kid`)
- `tokens/`
  - `claims.py` (Pydantic claims model; time conversions; validation helpers)
  - `mint.py` (create signed JWT; set header `kid`; default `jti`)
  - `verify.py` (verify signature + claims; return typed claims)

## Test Strategy (must pass for every phase)
Per-phase gate:
- `uv run python -m py_compile <changed_files...>` (or `.venv/bin/python -m py_compile ...` if `uv run` is unreliable)
- `uv run pytest -q -o addopts='' tests/unit/enforceai` (or `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai`)

End-of-stage gate:
- `.venv/bin/python -m pytest` (full suite + coverage gate)

Use Stage 0.3 fixtures:
- `enforceai_gateway_key_files` (private PEM + public keys dir + active `kid`)
- `enforceai_sqlite_db_path` for tests that need persistence (only if you add revocation primitives here)

---

## Phase 2.1 — Claims Model + Time/Validation Helpers
**Goal**: define a single canonical representation for gateway token claims and validation rules.

### Scope (single run)
- Add `GatewayTokenClaims` (Pydantic) with required fields:
  - `iss`, `sub` (stores canonical `user_id` string), `agent_id`, `scopes`, `iat`, `exp`, `jti`
- Add helpers:
  - convert between `datetime` and JWT numeric timestamps
  - validate:
    - `exp` present and in the future (with small clock-skew leeway)
    - `iat` not too far in the future (clock-skew)
    - `exp - iat` does not exceed max lifetime (365 days default)
    - `agent_id` UUIDv4
    - `scopes` list contains non-empty strings

### Tests to add
- `tests/unit/enforceai/test_gateway_token_claims.py`
  - happy-path model parse/serialize
  - rejects missing/invalid required claims
  - lifetime/clock-skew boundary cases

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

---

## Phase 2.2 — Keyring Loader (PEM + `kid` Selection + Caching)
**Goal**: load signing and verification keys locally from mounted PEM files and select the correct verification key by `kid`.

### Scope (single run)
- Add `GatewayKeyring` that:
  - loads the active private key PEM
  - loads all public key PEMs in `ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR` (e.g., `<kid>.pem`)
  - exposes:
    - `active_kid`
    - `signing_private_key`
    - `get_public_key(kid: str) -> public_key`
- Add caching strategy:
  - cache parsed key material in memory (e.g., `@lru_cache`)
  - explicit `reload()` helper (optional) for future management tooling
- Make error messages actionable:
  - missing files/dir
  - invalid PEM
  - active `kid` missing in public keys dir (verification would be impossible)

### Tests to add
- `tests/unit/enforceai/test_gateway_keyring.py`
  - loads keys from `enforceai_gateway_key_files`
  - missing `kid` file errors clearly
  - invalid PEM errors clearly
  - caching does not re-read files on repeated calls (patch `Path.read_bytes`)

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

---

## Phase 2.3 — Token Minting (RS256, `kid`, `jti`, Required Claims)
**Goal**: mint valid gateway tokens deterministically using RS256 and the active signing key.

### Scope (single run)
- Add `mint_gateway_token(...) -> str` that:
  - sets JWT header `kid=<active_kid>`
  - sets required claims (iss/sub/agent_id/scopes/iat/exp/jti)
  - supports optional overrides for:
    - `issued_at` (for tests)
    - `expires_at` or `ttl_seconds` (for tests)
    - `jti` (for deterministic tests)
- Ensure minting never logs secrets or full tokens (no logs by default).

### Tests to add
- `tests/unit/enforceai/test_gateway_token_mint.py`
  - minted token decodes with the corresponding public key and contains required claims
  - header contains expected `kid`
  - missing/invalid inputs fail fast (agent_id format, empty scopes)

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

---

## Phase 2.4 — Token Verification (Signature + Claims + Error Semantics)
**Goal**: verify gateway tokens locally and return validated, typed claims.

### Scope (single run)
- Add `verify_gateway_token(...) -> GatewayTokenClaims` that:
  - extracts `kid` from the JWT header
  - selects the correct public key from the keyring
  - verifies signature and algorithm (`RS256` only)
  - validates claims using `GatewayTokenClaims` rules:
    - required claims present
    - `exp`/`iat` sanity and max lifetime
  - returns typed claims on success
- Map failures to EnforceAI error semantics:
  - invalid/missing token → `UnauthorizedError`
  - keyring unavailable/misconfigured → `DependencyUnavailableError` (deny, retryable)
- Do not log the raw token or key material; if logging, log only `kid`, `jti` prefix, and high-level reason.

### Tests to add
- `tests/unit/enforceai/test_gateway_token_verify.py`
  - valid token verifies
  - wrong `kid` fails
  - wrong key/tampered token fails
  - expired token fails
  - algorithm mismatch fails
  - missing required claim fails
  - error mapping uses correct EnforceAI error type

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)

---

## Phase 2.5 — Hardening: Secret-Safe Logging + Regression Pack
**Goal**: ensure key/token material never leaks and verification remains stable under edge cases.

### Scope (single run)
- Add “no secret leaks” tests using `caplog`/`capsys`:
  - private key PEM not present in logs/stdout
  - token string not present in logs/stdout (unless explicitly opted-in by a debug flag later)
- Add a small regression pack for tricky time cases:
  - clock skew boundaries (± leeway)
  - `iat > exp` invalid
  - max lifetime enforcement

### Tests to add
- Extend `tests/unit/enforceai/test_gateway_token_verify.py` (or new `test_gateway_token_no_leaks.py`)

### Exit criteria
- Phase gate passes (`py_compile` + `tests/unit/enforceai`)
- Full suite passes: `.venv/bin/python -m pytest`

---

## Stage 2 Completion Criteria (strict)
Stage 2 is complete only when:
- Keyring loader, token minting, and token verification primitives are implemented and unit-tested.
- Tokens are RS256-signed with `kid` headers and include required claims including `exp`.
- Verification is local-only (no network) and caches key material.
- No secrets or raw tokens are logged by default (tests enforce).
- Full suite passes: `.venv/bin/python -m pytest`.

