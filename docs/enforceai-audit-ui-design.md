# EnforceAI Audit UI — Requirements & Design (Draft)
*Created: 2025-12-28*  
*Status: Draft (design-only; no implementation)*

## 1. Background
EnforceAI emits audit events for:
- **MCP enforcement** decisions (`tools/list`, `tools/call`) from `auth_server/server.py`
- **Management/admin actions** (agents, API keys, tokens, admin mutations) from `auth_server/enforceai/api/management_routes.py`

Phase 1 audit sinks (current):
- **Stdout**: JSON lines with `event_type="enforceai_audit"`
- **SQLite**: persisted best-effort in `audit_events` table at `ENFORCEAI_DB_PATH`

Existing UI state:
- `frontend/src/features/audit/AuditPage.tsx` is guidance + “viewer coming soon”.

## 2. Goals
- Provide a clear **Audit** experience for operators/admins and self-service users.
- Make it fast to answer: **who did what, when, from where, and was it allowed/denied**.
- Make it easy to correlate activity across systems using **`X-Request-Id`**.
- Keep secrets out of the UI by default (no tokens / API key secrets).

## 3. Non-goals (Phase 1 UI)
- Implement retention cleanup inside the UI (cleanup remains out-of-band).
- Replace external log shipping / SIEM workflows.
- Guarantee immutability or non-repudiation beyond the existing sinks.

## 4. Personas and Primary Jobs
- **Developer / Self-service**: verify why a call was denied; find the request id; check which tool was blocked.
- **Security Admin (self-service)**: review their own agent management actions and enforcement outcomes.
- **Operator/Admin (cross-user)**: investigate an incident across users/agents; confirm admin changes were audited.
- **Auditor (read-only)**: export/search events for a given incident window (when APIs exist).

## 5. Audit Event Shape (Current)
Emitted JSON line and stored DB record share the same logical fields:
- `occurred_at` (DB; UTC ISO)
- `user_id` (canonical)
- `agent_id` (uuid4)
- `action` (string; examples: `tools/list`, `tools/call`, `management/agents/create`)
- `outcome` (`allow` | `deny` | other stable strings)
- `request_id` (from `X-Request-Id` or JSON-RPC `id`)
- `details` (object; varies by action)

### 5.1 Known `details` fields (examples)
- Enforcement:
  - `provider`, `server`, `tool`, `reason`, `matched_scope`, `allowed_tools`
- Management:
  - action-specific identifiers (`target_user_id`, `key_id`, `jti`, etc.)

## 6. Information Architecture (Audit “Windows”)
“Audit windows” includes both:
- **UI windows** (screens + details drawers/modals)
- **Time windows** (time-range selectors and defaults)

The Audit experience is a single route (`/audit`) with sub-views:

1. **Audit Overview (Guidance)** (existing)
   - Where audit lives (stdout + DB)
   - How to filter by request id
   - Glossary of common action names

2. **Audit Explorer (Viewer)** (new)
   - Filter/search + event table
   - Event details drawer/modal
   - Optional “correlation view” grouped by request id

3. **Retention & Export** (within Audit, not a separate system page)
   - Display retention configuration guidance and links
   - Admin-only CSV export (requires APIs)

## 7. Audit Explorer — UX Requirements

### 7.1 Default experience
- Default time window: `Last 60 minutes` (configurable).
- Default “until” anchor: **Now** (time-bounded; no live refresh required).
- Maximum lookback supported by the UI (default): **7 days**.
- Default filter scope:
  - Non-admin: current `user_id` only (implicit).
  - Admin: requires explicit “target” selection to expand scope (see Admin Mode).

### 7.2 Filters (left rail or top bar)
Minimum filters:
- **Time window**:
  - Presets (`15m`, `1h`, `24h`, `7d`)
  - Custom start/end
  - End time defaults to **Now** when unset
  - Validate range does not exceed **7d** (or show warning + clamp)
- **Outcome**: allow/deny (multi-select)
- **Action**: multi-select or search-with-chips
- **Agent**: agent_id picker (searchable) + “any”
- **Request id**: exact match
- **Server**: exact match (from `details.server`) when present
- **Tool**: exact match (from `details.tool`) when present

Nice-to-have filters:
- “Only admin actions” / “Only enforcement actions” toggles

### 7.3 Event table
Columns (minimum):
- Time (UTC, with local-time tooltip)
- Outcome (badge)
- Action
- User (admin only; otherwise hidden)
- Agent (shortened uuid + copy)
- Summary (best-effort, derived from `details`)
- Request id (shortened + copy)

Row click opens **Event Details**.

### 7.4 Event Details (drawer or modal)
Must show:
- Canonical identifiers: `user_id`, `agent_id`, `request_id`
- Action + outcome
- Occurred time
- Rendered `details` (pretty JSON)

Should show:
- “Derived” human summary:
  - For `tools/call`: server, tool, reason, matched_scope
  - For `tools/list`: server, allowed_tools policy summary
- One-click “Find related events” by request id (opens correlation view).

### 7.5 Correlation view (by request id)
When a `request_id` is provided:
- Show a timeline of all matching events (descending time).
- Highlight allow vs deny transitions.
- Provide “copy as JSON” (read-only; export is admin-only).

## 8. Admin Mode Considerations
Admin audit viewing must avoid accidental cross-user browsing:
- Require explicit admin “mode” and explicit **target scope** selection:
  - `Self` (default)
  - `Selected user_id`
  - `All users` (allowed, gated by extra confirmation)
- Visibly banner the current target scope.
- Include a warning that all admin audit access is itself auditable.

## 9. Data Source Strategy (Decision)
Decision: the UI must be able to view audit events conveniently. This requires a supported read path for events.

### Option A — “Guidance only” (current)
- No API; users view stdout/SQLite via operator tooling.
- Audit page remains documentation-only.

### Option B — Read-only Audit API (selected)
- Backend exposes a read-only endpoint that queries the SQLite `audit_events` table with server-side filtering/pagination.
- UI becomes a true viewer without direct DB access.

### Option C — Integrate with log backend (future)
- UI queries an external log store (Loki/Elastic/SIEM) via dedicated integration.

## 10. Proposed API Contract (if Option B)
This section is a UI dependency proposal (not an implementation request).

### 10.1 List events (self-service)
- `GET /enforceai/audit/events`
  - Query: `since`, `until`, `limit`, `cursor`
  - Filters: `agent_id`, `action`, `outcome`, `request_id`, `server`, `tool`

### 10.2 List events (admin)
- `GET /enforceai/admin/audit/events`
  - Same filters plus `user_id` / `target_user_id`

### 10.3 Response shape
- `items`: list of audit event records
- `next_cursor`: optional
- `server_time`: UTC ISO (for UI clock skew hints)

### 10.4 CSV export (admin-only)
- `GET /enforceai/admin/audit/events.csv`
  - Same filters as `GET /enforceai/admin/audit/events`
  - Explicit limits and a warning banner in the UI (export can be sensitive)
  - Allow exporting either a selected `user_id` or `All users` (the latter requires extra confirmation)
  - Limit exports to **10,000 rows**; if exceeded, require narrowing filters
  - CSV columns (flattened):
    - Core: `occurred_at`, `user_id`, `agent_id`, `action`, `outcome`, `request_id`
    - Common details (best-effort): `server`, `tool`, `reason`, `matched_scope`, `provider`
    - Remainder: `details_json` (full original `details` as JSON string, redacted as needed)

## 11. Security and Privacy Requirements
- Never display secrets in plain text:
  - redact obvious token-like fields in `details` (defense-in-depth)
  - do not display request headers/body contents by default
- Copy-to-clipboard must be explicit per field.
- Export must be admin-only, warn about sensitivity, and require confirmation.

## 12. Decisions Captured (from conversation)
- Audit windows include both UI windows and time windows.
- Audit Explorer supports both self-service and admin usage (admin scope selection required).
- Viewer is time-bounded, anchored on “Now”; no live refresh required.
- Maximum lookback supported by the UI (default) is `7d`.
- No free-text search requirement (structured filters only).
- Viewer is read-only (no “respond from audit” actions).
- CSV export is admin-only.
- Admin audit search may be scoped to `All users`.
- CSV export may be scoped to `All users` (extra confirmation required).
- CSV export flattens common fields and is capped at `10k` rows.

## 13. Remaining Open Questions
None.
