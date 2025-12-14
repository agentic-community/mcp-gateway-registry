# EnforceAI Audit Retention (Stage 7)

This document describes where EnforceAI audit events are emitted, how retention is configured, and how to run out-of-band cleanup.

## Audit Emission
EnforceAI emits audit events in two places:

1. **Stdout**: JSON lines with `event_type="enforceai_audit"` for operational log shipping.
2. **SQLite**: persisted best-effort in the `audit_events` table in the DB at `ENFORCEAI_DB_PATH`.

Persistence failures must not deny requests. Audit persistence is best-effort and logs errors when it fails.

## Retention Configuration
Retention is configurable via environment variables:

- `ENFORCEAI_AUDIT_RETENTION_DAYS` (default: `30`): delete events older than this many days. Use `0` to disable time-based deletion.
- `ENFORCEAI_AUDIT_MAX_DB_BYTES` (default: `500000000`): size cap for the SQLite DB file. Use `0` to disable size-based deletion.

Cleanup is out-of-band and must not run on the request path.

## Cleanup Command
The cleanup command is an operator tool intended for manual runs or cron/systemd scheduling:

```bash
uv run python -m cli.enforceai_audit_cleanup --db-path /path/to/enforceai.db
```

### Inputs (Args + Env)
- `--db-path` (or `ENFORCEAI_DB_PATH`) (required)
- `--retention-days` (or `ENFORCEAI_AUDIT_RETENTION_DAYS`) (default: `30`)
- `--max-db-bytes` (or `ENFORCEAI_AUDIT_MAX_DB_BYTES`) (default: `500000000`)
- `--batch-size` (default: `500`)
- `--dry-run` (no DB writes; prints JSON summary)
- `--debug` (enable debug logging)

### Output
The command prints a stable JSON object to stdout:
- `deleted_by_time`
- `deleted_by_size`
- `final_db_bytes`
- `started_at`
- `finished_at`
- `elapsed_seconds`

## Scheduling Examples

### Cron (daily at 02:15)
```cron
15 2 * * * cd /path/to/repo && ENFORCEAI_DB_PATH=/var/lib/enforceai/enforceai.db uv run python -m cli.enforceai_audit_cleanup >> /var/log/enforceai-audit-cleanup.log 2>&1
```

### systemd (timer + service)
`/etc/systemd/system/enforceai-audit-cleanup.service`:
```ini
[Unit]
Description=EnforceAI audit cleanup

[Service]
Type=oneshot
WorkingDirectory=/path/to/repo
Environment=ENFORCEAI_DB_PATH=/var/lib/enforceai/enforceai.db
ExecStart=/usr/bin/env uv run python -m cli.enforceai_audit_cleanup
```

`/etc/systemd/system/enforceai-audit-cleanup.timer`:
```ini
[Unit]
Description=Run EnforceAI audit cleanup daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

