# EnforceAI Audit UI - Phased Implementation Plan

**Created:** 2025-12-28
**Status:** Draft
**Reference:** `docs/enforceai-audit-ui-design.md`

## Overview

This plan breaks the Audit UI implementation into 9 phases, each scoped for an agent to complete in a single session. Every phase is **fully tested** before moving to the next.

---

## Phase 1: Backend Audit Query API (Self-Service)

**Goal:** Expose a read-only audit events API for self-service users.

### Scope

1. **Extend `SqliteAuditStore`** (`auth_server/enforceai/stores/sqlite/audit_store.py`)
   - Add `query_events()` method with filters:
     - `user_id` (required for self-service)
     - `agent_id` (optional)
     - `action` (optional, list)
     - `outcome` (optional, list)
     - `request_id` (optional, exact match)
     - `server` (optional, from `details.server`)
     - `tool` (optional, from `details.tool`)
     - `since` / `until` (datetime range)
     - `limit` / `cursor` (pagination)
   - Return `AuditEventRecord` list + `next_cursor`

2. **Create response models** (`auth_server/enforceai/models/audit.py`)
   - `AuditEventResponse` (event_id, occurred_at, user_id, agent_id, action, outcome, request_id, details)
   - `AuditEventsListResponse` (items, next_cursor, server_time)

3. **Add API route** (`auth_server/enforceai/api/management_routes.py`)
   - `GET /enforceai/audit/events`
   - Query params: `since`, `until`, `limit`, `cursor`, `agent_id`, `action`, `outcome`, `request_id`, `server`, `tool`
   - Auto-filter by authenticated `user_id` (self-service only)

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Unit | `tests/unit/enforceai/stores/test_audit_store_query.py` | `query_events()` with all filter combinations, pagination, edge cases |
| Unit | `tests/unit/enforceai/models/test_audit_models.py` | Response model validation |
| Integration | `tests/integration/enforceai/api/test_audit_api.py` | `GET /enforceai/audit/events` with auth, filters, pagination |

### Acceptance Criteria

- [ ] `query_events()` returns filtered, paginated results
- [ ] API requires authentication and filters by `user_id`
- [ ] Default time window is last 60 minutes
- [ ] No maximum lookback enforced
- [ ] All unit tests pass
- [ ] All integration tests pass

---

## Phase 2: Frontend API Layer + Types

**Goal:** Add TypeScript types and API functions for audit events.

### Scope

1. **Add types** (`frontend/src/api/types.ts`)
   ```typescript
   export interface AuditEvent {
     event_id: number;
     occurred_at: string;  // ISO 8601
     user_id: string;
     agent_id: string;
     action: string;
     outcome: 'allow' | 'deny' | string;
     request_id: string | null;
     details: Record<string, unknown> | null;
   }

   export interface AuditEventsListResponse {
     items: AuditEvent[];
     next_cursor: string | null;
     server_time: string;
   }

   export interface AuditEventsQuery {
     since?: string;
     until?: string;
     limit?: number;
     cursor?: string;
     agent_id?: string;
     action?: string[];
     outcome?: string[];
     request_id?: string;
     server?: string;
     tool?: string;
   }
   ```

2. **Add API functions** (`frontend/src/api/enforceai.ts`)
   - `getAuditEvents(query: AuditEventsQuery): Promise<AuditEventsListResponse>`

3. **Add MSW handlers** (`frontend/src/test/mocks/handlers.ts`)
   - Mock handler for `GET /enforceai/audit/events`
   - Factory for mock audit events

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Unit | `frontend/src/api/__tests__/enforceai.test.ts` | `getAuditEvents()` API function |

### Acceptance Criteria

- [ ] Types match backend response shape
- [ ] API function constructs query params correctly
- [ ] MSW handlers return realistic mock data
- [ ] Unit tests pass

---

## Phase 3: Audit Explorer - Basic Table + Time Filtering

**Goal:** Create the Audit Explorer with event table and time-based filtering.

### Scope

1. **Create hooks** (`frontend/src/features/audit/hooks.ts`)
   - `useAuditEvents(filters)` - React Query hook for fetching events
   - `auditQueryKeys` - query key factory

2. **Create components**
   - `AuditExplorer.tsx` - main container
   - `AuditEventTable.tsx` - event list/table
   - `AuditTimeFilter.tsx` - time window selector (presets + custom)
   - `AuditOutcomeFilter.tsx` - allow/deny multi-select

3. **Update `AuditPage.tsx`**
   - Add tabbed interface: "Guidance" (existing) | "Explorer" (new)
   - Keep existing guidance content
   - Add explorer tab with new components

4. **Table columns** (minimum):
   - Time (UTC, local tooltip)
   - Outcome (badge: success/error)
   - Action
   - Agent (shortened UUID + copy)
   - Summary (derived from details)
   - Request ID (shortened + copy)

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Unit | `frontend/src/features/audit/__tests__/hooks.test.tsx` | `useAuditEvents` hook |
| Component | `frontend/src/features/audit/__tests__/AuditExplorer.test.tsx` | Explorer renders, loading states |
| Component | `frontend/src/features/audit/__tests__/AuditEventTable.test.tsx` | Table renders events, columns correct |
| Component | `frontend/src/features/audit/__tests__/AuditTimeFilter.test.tsx` | Presets work, custom range validation |
| Component | `frontend/src/features/audit/__tests__/AuditOutcomeFilter.test.tsx` | Multi-select behavior |
| Integration | `frontend/src/features/audit/__tests__/AuditPage.test.tsx` | Update existing tests, add tab navigation |

### Acceptance Criteria

- [ ] Explorer shows loading spinner while fetching
- [ ] Events display in descending time order
- [ ] Time presets (15m, 1h, 24h, 7d) work correctly
- [ ] Custom time range validates max 7d
- [ ] Outcome filter updates query
- [ ] Empty state shown when no events
- [ ] All tests pass

---

## Phase 4: Audit Explorer - Advanced Filters

**Goal:** Add remaining filters (agent, action, request ID, server, tool).

### Scope

1. **Create filter components**
   - `AuditAgentFilter.tsx` - searchable agent picker with "any" option
   - `AuditActionFilter.tsx` - multi-select with chips (uses `AUDIT_ACTIONS` from existing code)
   - `AuditRequestIdFilter.tsx` - exact match input
   - `AuditServerFilter.tsx` - text input (from `details.server`)
   - `AuditToolFilter.tsx` - text input (from `details.tool`)

2. **Update `AuditExplorer.tsx`**
   - Add filter bar/rail with all filters
   - Wire filters to `useAuditEvents` query params
   - Add "Clear filters" button

3. **Filter state management**
   - Use URL search params for shareable filter state
   - Sync filters with URL on change

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Component | `frontend/src/features/audit/__tests__/AuditAgentFilter.test.tsx` | Agent picker, search, selection |
| Component | `frontend/src/features/audit/__tests__/AuditActionFilter.test.tsx` | Multi-select, chip display |
| Component | `frontend/src/features/audit/__tests__/AuditRequestIdFilter.test.tsx` | Input, exact match |
| Component | `frontend/src/features/audit/__tests__/AuditServerFilter.test.tsx` | Input behavior |
| Component | `frontend/src/features/audit/__tests__/AuditToolFilter.test.tsx` | Input behavior |
| Integration | `frontend/src/features/audit/__tests__/AuditExplorer.test.tsx` | Filter combinations, clear filters, URL sync |

### Acceptance Criteria

- [ ] All filters update query correctly
- [ ] Multiple filters combine with AND logic
- [ ] Clear filters resets to defaults
- [ ] Filters persist in URL (shareable)
- [ ] Agent picker shows current user's agents
- [ ] Action filter shows categorized options
- [ ] All tests pass

---

## Phase 5: Event Details Drawer

**Goal:** Show detailed event information in a drawer/modal.

### Scope

1. **Create components**
   - `AuditEventDetailsDrawer.tsx` - drawer/modal container
   - `AuditEventDetails.tsx` - event detail display
   - `AuditDetailsJson.tsx` - pretty JSON renderer with copy

2. **Event details content**
   - Canonical identifiers: user_id, agent_id, request_id (with copy)
   - Action + outcome (badge)
   - Occurred time (UTC + local)
   - Derived human summary:
     - For `tools/call`: server, tool, reason, matched_scope
     - For `tools/list`: server, allowed_tools summary
     - For management actions: target identifiers
   - Full `details` as collapsible pretty JSON

3. **Wire to table**
   - Row click opens drawer
   - Drawer has close button
   - "Find related events" button (sets request_id filter)

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Component | `frontend/src/features/audit/__tests__/AuditEventDetailsDrawer.test.tsx` | Open/close, content display |
| Component | `frontend/src/features/audit/__tests__/AuditEventDetails.test.tsx` | All fields render, copy works |
| Component | `frontend/src/features/audit/__tests__/AuditDetailsJson.test.tsx` | JSON formatting, copy functionality |
| Integration | `frontend/src/features/audit/__tests__/AuditExplorer.test.tsx` | Row click opens drawer, find related works |

### Acceptance Criteria

- [ ] Drawer opens on row click
- [ ] All identifiers copyable
- [ ] Derived summary shows for known actions
- [ ] JSON details collapsible and copyable
- [ ] "Find related events" sets request_id filter and closes drawer
- [ ] All tests pass

---

## Phase 6: Correlation View (Request ID)

**Goal:** Show timeline of events grouped by request ID.

### Scope

1. **Create components**
   - `AuditCorrelationView.tsx` - timeline display for single request_id
   - `AuditCorrelationTimeline.tsx` - vertical timeline of events

2. **Timeline features**
   - Vertical timeline with time markers
   - Highlight allow vs deny transitions
   - Show all events for request_id (across agents if visible)
   - "Copy as JSON" for all events

3. **Integration**
   - When `request_id` filter is set, show correlation view instead of table
   - "Back to table" button clears request_id filter
   - Deep link support (`?request_id=...`)

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Component | `frontend/src/features/audit/__tests__/AuditCorrelationView.test.tsx` | Timeline renders, transitions highlighted |
| Component | `frontend/src/features/audit/__tests__/AuditCorrelationTimeline.test.tsx` | Event ordering, time markers |
| Integration | `frontend/src/features/audit/__tests__/AuditExplorer.test.tsx` | Request ID filter shows correlation view |

### Acceptance Criteria

- [ ] Correlation view shows when request_id is filtered
- [ ] Events display in chronological order
- [ ] Allow/deny transitions visually distinct
- [ ] "Copy as JSON" exports all events
- [ ] Back button returns to table view
- [ ] All tests pass

---

## Phase 7: Admin Audit API (Backend)

**Goal:** Add admin endpoints for cross-user audit viewing.

### Scope

1. **Extend `SqliteAuditStore`**
   - Update `query_events()` to support optional `user_id` (for admin)
   - Add `target_user_id` filter for management action filtering

2. **Add admin API route** (`auth_server/enforceai/api/management_routes.py`)
   - `GET /enforceai/admin/audit/events`
   - Same filters as self-service + `user_id` filter
   - Requires admin scope (e.g., `registry-admins`)

3. **Audit admin access**
   - Log admin audit queries as audit events themselves
   - Include target scope in audit details

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Unit | `tests/unit/enforceai/stores/test_audit_store_admin.py` | Admin query with user_id filter |
| Integration | `tests/integration/enforceai/api/test_audit_admin_api.py` | Admin endpoint auth, filters, audit logging |

### Acceptance Criteria

- [ ] Admin endpoint requires admin scope
- [ ] Admin can query any user's events
- [ ] Admin queries are themselves audited
- [ ] All tests pass

---

## Phase 8: Admin Mode UI

**Goal:** Add admin scope selection and cross-user audit viewing in UI.

### Scope

1. **Create components**
   - `AuditAdminScopeSelector.tsx` - target scope picker:
     - "Self" (default)
     - "Selected user" (user_id input/search)
     - "All users" (requires confirmation)
   - `AuditAdminBanner.tsx` - shows current target scope, warning

2. **Update API layer**
   - `getAdminAuditEvents(query)` - admin endpoint wrapper
   - Add `user_id` to `AuditEventsQuery` type

3. **Update `AuditExplorer.tsx`**
   - Show admin controls when user has admin scope
   - Pass target scope to API calls
   - Show banner with current scope

4. **Security UX**
   - "All users" requires explicit confirmation modal
   - Banner warns that admin access is audited

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Component | `frontend/src/features/audit/__tests__/AuditAdminScopeSelector.test.tsx` | Scope selection, confirmation modal |
| Component | `frontend/src/features/audit/__tests__/AuditAdminBanner.test.tsx` | Banner display, warning text |
| Integration | `frontend/src/features/audit/__tests__/AuditExplorer.test.tsx` | Admin mode toggle, cross-user viewing |

### Acceptance Criteria

- [ ] Admin controls only visible to admins
- [ ] Default scope is "Self"
- [ ] "All users" requires confirmation
- [ ] Banner clearly shows target scope
- [ ] Warning about auditing displayed
- [ ] All tests pass

---

## Phase 9: CSV Export (Admin-Only)

**Goal:** Add admin-only CSV export functionality.

### Scope

1. **Backend export endpoint**
   - `GET /enforceai/admin/audit/events/export`
   - Same filters as JSON endpoint
   - Limit: 10,000 rows (return error if exceeded)
   - Columns:
     - Core: `event_id`, `occurred_at`, `user_id`, `agent_id`, `action`, `outcome`, `request_id`
     - Common details: `server`, `tool`, `reason`, `matched_scope`, `provider`
     - `details_json` (full JSON string)

2. **Create components**
   - `AuditExportModal.tsx` - export configuration modal
     - Show current filters
     - Warn about sensitivity
     - Require explicit confirmation
     - Show row count estimate

3. **Wire to UI**
   - "Export CSV" button (admin only)
   - Progress indicator during download
   - Error handling for >10k rows

### Tests

| Test Type | Location | Coverage |
|-----------|----------|----------|
| Unit | `tests/unit/enforceai/api/test_audit_csv_export.py` | CSV generation, column mapping, row limits |
| Integration | `tests/integration/enforceai/api/test_audit_csv_api.py` | Export endpoint, auth, filters |
| Component | `frontend/src/features/audit/__tests__/AuditExportModal.test.tsx` | Modal UX, confirmation, error states |
| Integration | `frontend/src/features/audit/__tests__/AuditExplorer.test.tsx` | Export button visibility, download flow |

### Acceptance Criteria

- [ ] Export requires admin scope
- [ ] CSV has correct columns and data
- [ ] >10k rows returns error with guidance
- [ ] Confirmation required before export
- [ ] Sensitivity warning displayed
- [ ] Download works in browser
- [ ] All tests pass

---

## Testing Strategy Summary

### Test Categories by Phase

| Phase | Unit Tests | Component Tests | Integration Tests |
|-------|------------|-----------------|-------------------|
| 1 | 3 | - | 1 |
| 2 | 1 | - | - |
| 3 | 1 | 4 | 1 |
| 4 | 5 | - | 1 |
| 5 | - | 3 | 1 |
| 6 | - | 2 | 1 |
| 7 | 1 | - | 1 |
| 8 | - | 2 | 1 |
| 9 | 1 | 1 | 2 |

### Test Execution Commands

**Backend tests:**
```bash
# Unit tests for a phase
pytest tests/unit/enforceai/stores/test_audit_store_query.py -v

# Integration tests for a phase
pytest tests/integration/enforceai/api/test_audit_api.py -v

# All audit-related tests
pytest tests/ -k "audit" -v
```

**Frontend tests:**
```bash
# Component tests for audit feature
npm test -- --run src/features/audit

# All frontend tests
npm test -- --run
```

### Coverage Requirements

- Backend: 80% coverage on new code
- Frontend: 80% coverage on new components and hooks
- All API contracts tested with integration tests
- All filter combinations tested with unit tests

---

## Dependencies and Prerequisites

### Phase Dependencies

```
Phase 1 (Backend API)
    └── Phase 2 (Frontend Types)
        └── Phase 3 (Basic Explorer)
            ├── Phase 4 (Advanced Filters)
            ├── Phase 5 (Event Details)
            │   └── Phase 6 (Correlation View)
            └── Phase 7 (Admin API)
                └── Phase 8 (Admin UI)
                    └── Phase 9 (CSV Export)
```

### External Dependencies

- React Query (existing)
- Heroicons (existing)
- Tailwind CSS (existing)
- No new npm packages required

### Environment Requirements

- Backend running with `ENFORCEAI_DB_PATH` configured
- Audit events present in database (use bootstrap or manual insertion)
- Admin user for phases 7-9

---

## Agent Instructions

Each phase should be executed as follows:

1. **Read this plan** and the referenced design doc
2. **Read existing code** in referenced files to understand patterns
3. **Implement** the scope items in order
4. **Write tests** as specified in the Tests section
5. **Run all tests** and ensure they pass
6. **Update `enforceai/session_state/latest.md`** with:
   - Phase completed
   - Files created/modified
   - Tests executed and results
   - Any deviations from plan
   - Next phase ready indicator

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-28 | Agent | Initial plan creation |
