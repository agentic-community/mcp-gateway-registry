# EnforceAI Implementation Gaps vs Accepted Decisions
*Status: Informational*  
*Date: 2025-12-18*

This document records gaps between the current repository implementation and the accepted EnforceAI decisions in `enforceai/decisions/`.

## Scope
- Focus: Phase 1 EnforceAI identity + enforcement + management behavior.
- Sources of truth: `enforceai/decisions/*.md`, `enforceai/instructions/*`, and runtime code under `auth_server/enforceai/` + `/validate` integration in `auth_server/server.py`.

## Summary
The implementation largely aligns with the EnforceAI decisions for:
- Authentication modes (OIDC, gateway tokens, API keys), credential parsing, and error semantics
- Gateway token RS256 + `kid` + local key loading and max lifetime enforcement
- OIDC multi-issuer + JWKS caching + conservative claim defaults
- Revocation model (agent kill switch, jti revocation, tokens_valid_after)
- Tool visibility filtering via `X-Allowed-Tools` and Nginx/Lua response filtering
- Dual-sink audit and out-of-band retention cleanup tooling

The gaps below require follow-up to fully conform.

## Gaps

### Gap 1 — ORM requirement not implemented (Decision 0001)
**Decision:** `0001-persistence-backend.md` requires “ORM + migrations from day one, even on SQLite”.

**Current behavior:**
- Migrations exist and are applied (`auth_server/enforceai/db/migrations.py`, `auth_server/enforceai/db/data_layer.py`).
- Persistence uses direct `sqlite3` access in store implementations (example: `auth_server/enforceai/stores/sqlite/agent_store.py`).
- No ORM layer is present (no `sqlalchemy`/`sqlmodel`/`alembic` in use).

**Impact:** This is a direct mismatch with the “ORM from day one” decision and may increase migration portability risk to Postgres later.

**Recommended next step:**
- Either (A) implement the ORM layer + migrations toolchain, or (B) amend `0001` to explicitly allow “migrations + explicit SQL stores” for Phase 1, with a later ORM migration plan.

---

### Gap 2 — FGAC enforcement only covers tools (Decision 0007 / FGAC model)
**Decisions:** `0007-authorization-overlay-semantics.md` + `enforceai/architecture/fgac_model.md` treat the scope catalog as the authoritative policy model for runtime authorization.

**Current behavior:**
- `/validate` enforces FGAC only for `tools/call`.
- `tools/list` is handled by returning `X-Allowed-Tools` and filtering the upstream response (Nginx/Lua), but `/validate` does not deny other MCP methods (e.g., `resources/list`, `resources/templates/list`, `initialize`) based on catalog rules.
- Evidence: `auth_server/server.py` checks only `method in {"tools/list", "tools/call"}` for enforcement decisions.

**Impact:**
- A caller with no method-level permission for a server could still invoke non-tool methods if the upstream server accepts them, which is not aligned with “enterprise policy is authoritative” semantics.

**Recommended next step:**
- Extend `/validate` enforcement to evaluate method permissions for all MCP methods (not just tools), using the same catalog rules.
- Keep the existing “tool name required” rule only for `tools/call`.

---

### Gap 3 — Gateway token scope mismatch is not logged (Decision 0008)
**Decision:** `0008-effective-scope-source-for-gateway-tokens.md` says token scopes not present on agent scopes should be treated as a reduction and the mismatch should be logged.

**Current behavior:**
- Effective scopes are computed as `token.scopes ∩ agent.scopes` (`auth_server/enforceai/providers/gateway_token.py`), but no mismatch logging exists.

**Impact:** Operational visibility into scope-tightening or mis-issued tokens is reduced.

**Recommended next step:**
- Add a structured log entry when `set(token.scopes) - set(agent.scopes)` is non-empty.

---

### Gap 4 — Cross-user admin APIs exist (Decision 0016)
**Decision:** `0016-management-surface-cli-first-self-service.md` says cross-user administrative operations are deferred to a later explicit admin feature.

**Current behavior:**
- Cross-user admin endpoints exist under `/enforceai/admin/*` (e.g., user search, list user agents, create agent for another user) in `auth_server/enforceai/api/management_routes.py`.
- Access is guarded by `enforceai-admin` group checks, but the feature exists in Phase 1 code.

**Impact:** This contradicts the “defer cross-user admin” decision unless treated as an explicit admin feature (with an ADR and Phase 1 scope update).

**Recommended next step (choose one):**
- Option A (strict conformance): remove/disable `/enforceai/admin/*` for Phase 1.
- Option B (formalize): add a new decision clarifying that “admin endpoints are included in Phase 1 behind `enforceai-admin`”, including explicit threat model, audit requirements, and rollout rules.

## Notes
- This is not a new architectural decision; it is a conformance/gap tracking artifact placed in `enforceai/decisions/` by request.
