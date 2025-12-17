# UI Scope Catalog Management (Create/Edit/Delete) — Phased Plan
*Created: 2025-12-17*

## Goal
Allow administrators to create new scope definitions (and later edit/delete them) from the web UI, with changes persisted to the Phase 1 enterprise policy catalog (`auth_server/scopes.yml`) and applied to enforcement safely.

## Current State (What Exists)
- Scope catalog source of truth: `auth_server/scopes.yml` (Decision 0013).
- UI has a read-only viewer: `frontend/src/features/scopes/ScopesPage.tsx` (Phase 10 in `enforceai/plans/plan-enforce-gw-ui-frontend-phased.md`).
- Backend exposes a read-only endpoint: `GET /enforceai/scopes/catalog` in `auth_server/enforceai/api/management_routes.py`.
- Enforcement uses `auth_server/enforceai/fgac/catalog.py` with an LRU cache (`load_scope_catalog()`); updates require cache clearing to take effect.

## Constraints / Non-Negotiables
- No scope-catalog schema redesign in Phase 1: continue using the existing `scopes.yml` schema as-is.
- Policy is enterprise-controlled: scope catalog mutation must be admin-gated and auditable.
- Fail closed: invalid policy updates must not leave the system in a broken “cannot load catalog” state.
- No permission elevation beyond the catalog (Architecture Lock #7 still applies; this feature edits the catalog, so admin gating is critical).

## Phase 0 — Make Catalog Read Path Correct + Add Concurrency Token
**Goal**: Ensure the UI is viewing the *same* catalog that enforcement uses (configured by `ENFORCEAI_SCOPES_CATALOG_PATH`) and add an `etag` for safe concurrent edits.

### Deliverables
- Update `GET /enforceai/scopes/catalog` to load via `get_scope_catalog` (which respects `ENFORCEAI_SCOPES_CATALOG_PATH`) instead of `default_scopes_catalog_path()`.
- Add `etag` (e.g., SHA-256 of the raw catalog file bytes) to the response.
- Add `last_modified` (optional but recommended) for UI visibility.

### Tests to Add/Update
- Integration: when `ENFORCEAI_SCOPES_CATALOG_PATH` points to a temp file, `GET /enforceai/scopes/catalog` returns scopes from that file.
- Integration: changing the file contents changes `etag`.

### Required Test Gate
- `uv run python -m py_compile` for touched Python files.
- `make test-integration` (or `make test-fast` if that’s the local convention).

---

## Phase 1 — Backend “Policy Writer” + Admin Scope CRUD API
**Goal**: Add a safe, admin-only API to create (and minimally update/delete) scopes in `scopes.yml`, with validation, rollback, cache clearing, and audit events.

### Deliverables
1. **Policy writer utility (auth_server)**:
   - Reads current YAML as a mapping.
   - Applies mutations to only the requested scope (top-level key).
   - Writes back safely:
     - Create a backup file first (same directory).
     - Write updated content.
     - Re-parse and validate the updated file via `load_scope_catalog()` (cache-cleared) before returning success.
     - On validation failure: restore backup and return a 4xx with actionable error.
   - Clears catalog cache on success (`clear_scope_catalog_cache()`).

2. **Admin endpoints (auth_server)** (names can be adjusted, but keep under `/enforceai/admin/*`):
   - `POST /enforceai/admin/scopes` (create)
   - `PUT /enforceai/admin/scopes/{scope_name}` (replace/update)
   - `DELETE /enforceai/admin/scopes/{scope_name}` (delete)

3. **Concurrency control**:
   - Require `If-Match: <etag>` for `PUT`/`DELETE` (recommended for `POST` too if “create” can race).
   - Return `412 Precondition Failed` on mismatch.

4. **Guardrails & validation**:
   - Reject reserved names (`UI-Scopes`, `group_mappings`).
   - Reject empty/whitespace names.
   - Enforce schema-level rules needed for enforcement correctness:
     - If `tools/call` is included for a server permission, require `tools` to be present (either `'*'`/`all` or an explicit list), because enforcement treats missing `tools` as “no tool calls allowed”.
   - Delete behavior:
     - If the scope is referenced by `group_mappings`, either:
       - default: reject with `409 Conflict` explaining where it’s referenced, or
       - optional: allow `?force=true` to remove references from `group_mappings` as part of the delete.

5. **Audit**:
   - Emit audit events for create/update/delete with `actor_user_id`, `actor_agent_id`, `action`, `outcome`, and `X-Request-Id`.
   - Do not log the full catalog contents; include only scope name and a bounded summary (counts of servers/methods/tools).

### Tests to Add/Update
- Unit: policy writer rejects invalid input, restores backup on failure, and clears cache on success.
- Integration:
  - Admin cookie session can `POST` a new scope and it appears in `GET /enforceai/scopes/catalog`.
  - Non-admin receives `403`.
  - Missing/invalid CSRF on state-changing requests receives `403`.
  - `If-Match` mismatch yields `412`.
  - Delete conflict behavior (`409` or `force=true`) is enforced.

### Required Test Gate
- `uv run python -m py_compile` for touched Python files.
- `make test` (must pass).

---

## Phase 2 — Frontend: Create Scope (Admin Only, Minimal UX)
**Goal**: Add a “Create scope” flow in the UI that can create a valid, deny-by-default scope and optionally a simple server/tool allowlist.

### Deliverables
- UI changes in `frontend/src/features/scopes/ScopesPage.tsx`:
  - Show “Create scope” button only for `user.is_admin`.
  - Open a modal to create a scope.
- Modal form (new component recommended):
  - Required: scope name.
  - Choose one creation mode:
    - **Blank scope** (empty permission list; deny-by-default).
    - **Simple server tool scope**: select server name, include `tools/list` + `tools/call`, choose tools = `all` or list.
  - Client-side validation mirrors backend guardrails (reserved names, empty name, tools/call implies tools policy).
- API hooks:
  - `useCreateScope()` mutation hitting `POST /enforceai/admin/scopes`.
  - On success: invalidate/refetch `scopesKeys.catalog()` and auto-expand the new scope card.
- Error handling:
  - Surface 409/412/503 errors with actionable UI text (e.g., “Catalog changed; refresh and retry”).

### Tests to Add/Update
- Unit (React Testing Library + MSW):
  - “Create scope” button is hidden for non-admin users.
  - Form validation blocks invalid names.
  - Successful create triggers catalog refetch and displays the new scope.

### Required Test Gate
- `npm run typecheck` passes.
- `npm run test` passes.
- `npm run build` succeeds.

---

## Phase 3 — Frontend: Edit Scope + Safe Confirmation
**Goal**: Let admins edit an existing scope definition with a clear diff/summary before applying.

### Deliverables
- Add “Edit” action on each scope card (admin only).
- Reuse the modal in “edit mode”:
  - Pre-fill current definition.
  - Show a “Changes summary” section before save (servers/methods/tools deltas).
- Call `PUT /enforceai/admin/scopes/{scope_name}` with `If-Match` using the last loaded `etag`.
- On success: refetch catalog and keep focus on edited scope.

### Tests to Add/Update
- Unit:
  - Edit flow sends `PUT` with `If-Match`.
  - ETag mismatch (`412`) prompts a refresh path (no silent overwrite).

### Required Test Gate
- `npm run typecheck`, `npm run test`, `npm run build`.

---

## Phase 4 — Frontend: Delete Scope + Reference Safety
**Goal**: Allow admins to delete scopes without accidentally breaking `group_mappings` or active usage.

### Deliverables
- Add “Delete” action (admin only) with typed confirmation (type scope name).
- Handle backend outcomes:
  - `409 Conflict`: show “Scope is referenced by group mappings” (and optionally provide “force delete” if supported).
  - `412 Precondition Failed`: prompt refresh.
- On success: refetch catalog and remove scope from the list.

### Tests to Add/Update
- Unit:
  - Delete confirmation required.
  - `409` renders the correct guidance.

### Required Test Gate
- `npm run typecheck`, `npm run test`, `npm run build`.

---

## Phase 5 (Optional, Follow-Up) — Full Policy Management Surface
**Goal**: Manage more of `scopes.yml` from the UI without changing the schema.

### Candidates (Pick only if needed)
- UI editing for `group_mappings` (admin only).
- UI editing for `UI-Scopes` (registry visibility rules) (admin only).
- Import/export of a single scope (JSON) for review workflows.
- “Draft + apply” workflow if multiple admins will edit concurrently.

### Testing Expectations
- Integration tests for every state-changing endpoint.
- UI tests for each editor flow.

---

## Open Questions (Decisions Needed Before Phase 1)
1. Should delete default to `409` when referenced by `group_mappings`, or support `force=true` removal of references?
2. Should scope creation start with a minimal “blank scope” only, or do we require at least one server permission?
3. Do you want scope names to follow a convention (e.g., `namespace.action` like `sqlite.manage`) enforced by regex?
4. Should non-admin users see the Scopes page at all, or read-only (current behavior)?

