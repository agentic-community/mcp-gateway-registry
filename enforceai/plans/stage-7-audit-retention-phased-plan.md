# Stage 7 — Audit + Retention + Hardening (Phased Plan)
*Created: 2025-12-14*

## Goal
Make EnforceAI operable and safe in enterprise-like conditions by:
- Guaranteeing best-effort audit emission (stdout always; persistence failures never flip allow/deny)
- Providing out-of-band retention cleanup (time + size caps)
- Ensuring request-path enforcement has no network dependency and remains performant (cached key/JWKS/catalog)

## Non-Goals (Stage 7)
- No background scheduler/cron inside the auth server process
- No multi-node coordination (single instance assumed)
- No UI work

## Locked Decisions (carry forward)
- Fail closed for enforcement decisions (`401/403/503` semantics stay unchanged)
- Audit persistence failure must not deny a request; it may log error
- Cleanup is out-of-band (not on the request path)
- No external network calls in unit/integration tests (JWKS is mocked)

## Current State (as of Stage 6)
- Audit events are emitted to stdout + appended to SQLite (`audit_events`) best-effort.
- Config already includes:
  - `ENFORCEAI_AUDIT_RETENTION_DAYS`
  - `ENFORCEAI_AUDIT_MAX_DB_BYTES`

## Phase 7.1 — Audit Store Retention Primitives (DB-layer)
**Goal**: add deterministic, testable primitives to enforce retention in SQLite.

### Scope (single run)
- Extend `auth_server/enforceai/stores/interfaces.py` `AuditStore` with:
  - `delete_events_older_than(*, cutoff: datetime) -> int`
  - `delete_oldest_events(*, limit: int) -> int`
- Implement in `auth_server/enforceai/stores/sqlite/audit_store.py`:
  - Delete by time cutoff (`occurred_at < cutoff`)
  - Delete oldest N events by `occurred_at ASC` (stable ordering)
- Add small helpers:
  - Normalize datetimes to UTC
  - Guard `limit > 0`

### Tests to add/extend
- `tests/unit/enforceai/test_audit_store_sqlite.py`
  - Insert N events, delete-by-cutoff deletes expected subset
  - Insert N events, delete-oldest deletes expected count and oldest rows
  - Edge cases: empty table, cutoff with tz-naive datetime, invalid limit

### Exit criteria
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_audit_store_sqlite.py` (pass)

---

## Phase 7.2 — Retention Engine (Policy + Size Cap Computation)
**Goal**: implement a pure, testable retention algorithm that decides what to delete.

### Scope (single run)
- Add `auth_server/enforceai/audit/retention.py` with pure functions:
  - `compute_cutoff(*, now: datetime, retention_days: int) -> Optional[datetime]`
  - `enforce_time_retention(*, audit_store: AuditStore, cutoff: Optional[datetime]) -> int`
  - `enforce_size_retention(*, db_path: Path, audit_store: AuditStore, max_db_bytes: int, batch_size: int = 500) -> int`
    - Uses `db_path.stat().st_size` (no DB PRAGMA; portable)
    - Deletes oldest events in batches until under size cap (or no progress)
    - Never raises on delete failures; maps unexpected store failures to `DependencyUnavailableError` for callers
- Ensure the retention engine:
  - Treats `retention_days=0` as “no time-based deletion”
  - Treats `max_db_bytes=0` as “no size-based deletion”

### Tests to add
- `tests/unit/enforceai/test_audit_retention.py`
  - `compute_cutoff` behavior for `0`, positive days, tz-naive input
  - `enforce_time_retention` uses cutoff and returns deleted count (mock store)
  - `enforce_size_retention`:
    - When file is under cap, no deletes
    - When file is over cap, attempts deletes in batches until under cap (use a temp SQLite file + real store)
    - Stops if deletes return 0 (avoid infinite loop)

### Exit criteria
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/enforceai/test_audit_retention.py` (pass)

---

## Phase 7.3 — Operator Cleanup Command (Out-of-Band)
**Goal**: provide an operator-grade cleanup command that can be run manually or by cron/systemd timer.

### Scope (single run)
- Add `cli/enforceai_audit_cleanup.py`:
  - Argparse CLI (non-interactive)
  - Inputs via args + env:
    - `--db-path` (or `ENFORCEAI_DB_PATH`) (required)
    - `--retention-days` (or `ENFORCEAI_AUDIT_RETENTION_DAYS`) (default: 30)
    - `--max-db-bytes` (or `ENFORCEAI_AUDIT_MAX_DB_BYTES`) (default: 500_000_000)
    - `--batch-size` (default: 500)
    - `--dry-run` (log what would be deleted; no DB writes)
    - `--debug` (enable debug logging)
  - Output stable JSON summary to stdout:
    - deleted_by_time
    - deleted_by_size
    - final_db_bytes
    - started_at / finished_at / elapsed_seconds
- The command must never require network and must not depend on FastAPI app startup.

### Tests to add
- `tests/unit/cli/test_enforceai_audit_cleanup_args.py`
  - Parsing, env fallback, dry-run behavior does not delete
- `tests/integration/test_enforceai_audit_cleanup.py`
  - Create temp DB, insert many audit events, run cleanup via importing the module’s `main()` with argv
  - Assert deletion occurred and summary JSON is valid/stable

### Exit criteria
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `.venv/bin/python -m pytest -q -o addopts='' tests/unit/cli/test_enforceai_audit_cleanup_args.py tests/integration/test_enforceai_audit_cleanup.py` (pass)

---

## Phase 7.4 — Hardening + Regression Coverage + Docs
**Goal**: lock in failure policy and operational docs, and ensure no request-path regressions.

### Scope (single run)
- Add/extend regression tests:
  - Audit persistence failure does not deny `GET /validate` (enforcement path) and does not deny a management endpoint
  - JWKS cache / keyring cache / catalog cache remain in-memory and avoid per-request reloading
- Add docs under `enforceai/instructions/`:
  - How to run the cleanup command with cron/systemd
  - Recommended retention defaults for local dev vs prod
  - Where audit logs appear (stdout JSON events and SQLite table)
- Update `enforceai/session_state/latest.md` to mark Stage 7 completion and point to next stage.

### Tests to add/extend
- `tests/integration/test_enforceai_stage5_roundtrip.py` (or a new integration test)
  - Force `audit_store.append_event` to raise and assert response still `200` for allowed `tools/list` and correct deny semantics stay intact.

### Exit criteria (Stage 7 completion gate)
- `.venv/bin/python -m py_compile <changed_files...>` (pass)
- `.venv/bin/python -m pytest` (full suite + coverage gate pass)
- `enforceai/session_state/latest.md` updated

