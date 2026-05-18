# Operations Runbooks

This folder contains step-by-step operational runbooks for the MCP
Gateway Registry. Each runbook is self-contained, copy-pasteable,
and intended to be useful at 2am with no prior context. Each
includes a procedure block, verification commands per step, and
links into the underlying code or configuration.

## Available runbooks

- [incident-response.md](incident-response.md) — Suspected credential
  leak: drop the `oauth_sessions_<ns>` collection to invalidate every
  active session in one step. Includes the trade-off vs. rotating
  `SECRET_KEY`.
- [mongodb-export-import.md](mongodb-export-import.md) — Export and
  import MongoDB / DocumentDB collections. Covers `mongoexport` /
  `mongoimport` (JSONL, for compliance pulls), `mongodump` /
  `mongorestore` (BSON, full-fidelity backups), namespace-scoped
  per-tenant operations, and the caveats around encrypted fields and
  index restoration.
- [audit-log-export.md](audit-log-export.md) — Query and export the
  `audit_events_*` collection for compliance review and security
  investigations. Two paths: REST API (recommended, admin-required)
  and direct MongoDB (operator escape hatch). Some sections are
  marked DRAFT — the direct-MongoDB paths are fully validated; the
  REST API admin-bootstrap and the disable-shipping path are not.
- [rotate-secrets.md](rotate-secrets.md) — Rotation procedures for
  `SECRET_KEY`, federation static tokens, IdP client secrets, and
  M2M client secrets. Documents what each rotation invalidates and
  the rollout sequence across replicas. DRAFT — destructive steps
  were not exercised in the validation pass. Dry-run in non-prod
  first.

## Planned runbooks

The following are tracked under [#1056](https://github.com/agentic-community/mcp-gateway-registry/issues/1056)
for future PRs:

- `telemetry-otlp-forwarding.md` — wire registry traces and metrics
  to an external OTLP collector.
- `backup-and-restore.md` — full-database backup procedures and
  point-in-time recovery for AWS DocumentDB and MongoDB CE.
- `scale-replicas.md` — pre-scale checklist and rolling-restart
  pattern.
- `federation-peer-onboard-offboard.md` — live federation peer
  lifecycle.

## Conventions

- File naming: `<topic>-<action>.md`, lowercase, hyphen-separated.
- Each runbook starts with a short summary of when to use it.
- Procedures are numbered, every step has a copy-pasteable command.
- Verification commands appear immediately after the step they verify.
- Caveats and "when to use vs. alternative X" are called out
  explicitly in their own subsection, not buried in prose.
- Code references at the end of the runbook link to the file and
  line on `main` so a reader can jump straight into the implementation.
