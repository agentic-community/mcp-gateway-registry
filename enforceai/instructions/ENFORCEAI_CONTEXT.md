# EnforceAI Gateway — Agent Context and Navigation
*Purpose*: This document is the single entry point for any agent run. It explains what this repo is being used for, where the authoritative documents live, and how to work safely across multi-session development.

## What We Are Building
This repository is based on the upstream **MCP Gateway & Registry** project. We are extending it into the **EnforceAI Security Gateway** by adding an EnforceAI identity layer and enforcement model.

High-level product direction:
- The gateway enforces **agent-scoped authorization** using a gateway-managed agent registry.
- Supported authentication modes are:
  - Generic OIDC (multi-issuer)
  - Gateway-issued tokens (RS256)
  - API keys (agent-bound)
- The enforcement point is **`auth_server`** behind Nginx `auth_request`.
- The existing scope catalog in `auth_server/scopes.yml` is reused as the Phase 1 enterprise policy catalog.

## Repo Map (Where Things Live)

### EnforceAI (authoritative for EnforceAI work)
- `enforceai/instructions/ENFORCEAI_AGENT_SETUP.md`: How an agent should initialize and close a session.
- `enforceai/instructions/ENFORCEAI_CONTEXT.md`: This context/navigation document (load early).
- `enforceai/session_state/latest.md`: Multi-session state tracking (what was done, current task, next steps).
- `enforceai/architecture/architecture_lock.md`: Non-negotiable rules (do not change unless explicitly asked).
- `enforceai/architecture/identity_model.md`: IdentityContext contract and binding rules.
- `enforceai/architecture/fgac_model.md`: Enforcement model, audit, visibility filtering, retention rules.
- `enforceai/architecture/gateway_tokens.md`: Gateway token format, signing/rotation, revocation, effective scopes.
- `enforceai/architecture/agent_registry.md`: Agent model and lifecycle requirements.
- `enforceai/mcp_gateway_identity_requirements.md`: Consolidated implementation requirements.
- `enforceai/enforceai_identity_and_gap_analysis.md`: Gap analysis vs current repo.
- `enforceai/decisions/`: Accepted decisions (ADRs). Add a new ADR whenever a new decision is made.
- `enforceai/plans/`: Phased implementation plans (follow these when coding).
- `enforceai/roadmap/`: Product roadmap (may be empty early).

### Upstream Docs (background and current system behavior)
- `docs/`: Upstream documentation (auth architecture, testing, setup, etc.).
- `docs/scopes.md`: Explains the existing scope-based FGAC model and `auth_server/scopes.yml` schema.
- `auth_server/scopes.yml`: Phase 1 enterprise policy catalog (scope definitions).
- `docs/testing.md`, `tests/README.md`: How upstream tests are structured and run.

### Runtime Code (where changes will eventually land)
- `auth_server/`: Enforcement point (FastAPI) used by Nginx `auth_request`.
- `registry/`: Registry UI + APIs (optional for enforcement path; used later for UI).
- `frontend/`: React UI (not required for Phase 1 management; CLI-first).
- `tests/`: Pytest unit/integration tests and shell-based system tests.

## How to Work in This Repo (Agent Workflow)

### Session Initialization (required)
Follow `enforceai/instructions/ENFORCEAI_AGENT_SETUP.md`. At minimum, load in this order:
1. `enforceai/architecture/architecture_lock.md`
2. `enforceai/instructions/ENFORCEAI_CONTEXT.md`
3. `enforceai/session_state/latest.md`
4. EnforceAI architecture docs in `enforceai/architecture/`
5. Relevant ADRs in `enforceai/decisions/`
6. The active plan in `enforceai/plans/` (start with Stage 0)

### Planning and Execution
- Implement only the current plan phase scope (single-run phases are defined in `enforceai/plans/`).
- Do not redesign the identity model or enforcement semantics (they are locked by `architecture_lock.md` and ADRs).
- When new information forces a decision, stop and request a decision; once decided, write an ADR in `enforceai/decisions/` and update the relevant EnforceAI docs.

### Progress Tracking (multi-session)
- Use `enforceai/session_state/latest.md` as the authoritative “handoff” between sessions.
- Update it at the end of every work session:
  - what changed
  - what tests were run
  - next phase to execute
  - open questions (only if unresolved)

## Testing (Required Before Declaring Any Phase Complete)

### Primary commands
- Unit tests: `make test-unit`
- Integration tests: `make test-integration`
- Fast local suite: `make test-fast`
- Full suite: `make test`

### Repository test notes
- Tests live in `tests/unit/` and `tests/integration/`.
- Shell-based end-to-end scripts live under `tests/` (e.g., `tests/run_all_tests.sh`), but these require a running docker-compose stack.

### Phase completion rule
Do not mark a phase complete unless:
- The phase’s required unit/integration tests pass.
- Any required compile/syntax checks pass.
- The session state is updated (`enforceai/session_state/latest.md`).

## Where to Put New Work
- Implementation plans: `enforceai/plans/`
- Decisions: `enforceai/decisions/` (ADR format, short and explicit)
- Architecture rules/contracts: `enforceai/architecture/` (only change with explicit user request for locked files)
- Temporary notes: `.scratchpad/` (gitignored; do not rely on for long-term project context)

