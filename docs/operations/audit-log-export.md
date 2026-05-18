# Operations: Audit Log Export

This runbook covers querying and exporting the registry's audit
events for compliance review, security investigations, and ad-hoc
reporting.

> **Environment-portability note.** The commands below were validated
> against the local `docker compose` stack with MongoDB CE running
> in the `mcp-mongodb` container. **On EKS and ECS deployments these
> instructions are directional only** — adapt the connection
> patterns (kubectl exec, ECS exec, DocumentDB endpoint with TLS)
> to your environment. Treat the procedures as the right shape; the
> exact invocation will differ.

---

## What's in `audit_events_*`

The registry writes one document per request to the
`audit_events_<documentdb_namespace>` collection (default install:
`audit_events_default`). Two log streams share the collection,
distinguished by the `log_type` field:

| `log_type` value | What it captures |
|---|---|
| `registry_api_access` | Calls into the registry's REST API (server registration, group management, audit queries themselves, etc.). |
| `mcp_server_access` | MCP tool invocations through the gateway (which user invoked which tool on which server). |

The relevant fields differ slightly between the two streams. See
[`registry/audit/routes.py:215-275`](../../registry/audit/routes.py#L215-L275)
for the canonical query construction the registry uses.

A registry-API record looks like this:

```json
{
  "_id": {"$oid": "..."},
  "timestamp": {"$date": "2026-05-11T16:55:56.824Z"},
  "log_type": "registry_api_access",
  "request_id": "...",
  "identity": {
    "username": "alice@example.com",
    "auth_method": "oauth2",
    "provider": "entra",
    "groups": [...],
    "is_admin": false
  },
  "request": {
    "method": "GET",
    "path": "/api/servers",
    "client_ip": "1.2.3.4"
  },
  "response": {
    "status_code": 200,
    "duration_ms": 12.4
  },
  "action": {
    "operation": "list_servers",
    "resource_type": "servers"
  },
  "authorization": {
    "decision": "ALLOW"
  }
}
```

---

## Two paths: REST API vs. direct MongoDB

| Path | Use when | Requires |
|---|---|---|
| **REST API** (`/api/audit/events`, `/api/audit/export`) | Normal operation. CSV exports for compliance. Consistent across deployments because the registry handles the query, format, and access control. | Registry running and reachable. Admin-tier session (`is_admin=true`) or M2M client with admin scopes. |
| **Direct MongoDB** (`mongosh`, `mongoexport`, `mongodump`) | Registry is down. Bulk forensic export. Ad-hoc aggregations the API doesn't expose. | Database access (privileged user). Bypasses application-level auth — anyone with this access sees everything. |

**Rule of thumb:** start with the REST API. Drop to direct MongoDB
only when the API path can't satisfy the use case.

---

## Path A: Export via REST API

> **DRAFT — admin-auth bootstrap not exercised in this PR's
> validation.** The endpoint behavior, query parameters, and CSV/JSONL
> formats below come from
> [`registry/audit/routes.py:567-918`](../../registry/audit/routes.py#L567-L918).
> The exact `Authorization` header form depends on your deployment's
> auth mode (session cookie vs. M2M token), and full validation of
> the bootstrap path is left for a follow-up. **Treat the curl
> examples as templates and confirm the auth pattern against your
> deployment before relying on them.**

### Procedure

1. Obtain an admin-tier credential. The endpoints under
   `/api/audit/*` use `require_admin` ([`routes.py:44`](../../registry/audit/routes.py#L44)),
   so the credential must produce a `user_context` with
   `is_admin == true`.

2. Query a small page first to confirm the credential is working
   and to see the schema:

   ```bash
   # Replace <ADMIN_TOKEN> with your admin bearer token / session cookie
   curl -sS \
       -H "Authorization: Bearer <ADMIN_TOKEN>" \
       "http://localhost/api/audit/events?limit=5"
   ```

   On success: a JSON object with `total`, `limit`, `offset`,
   `events` (array). On 403 Forbidden, the token is not admin.

3. Export filtered events as JSONL or CSV. All filters are optional
   and can be combined (see
   [`routes.py:821-885`](../../registry/audit/routes.py#L821-L885)
   for the full parameter list):

   ```bash
   # JSONL export, all events for one user, last 30 days
   curl -sS \
       -H "Authorization: Bearer <ADMIN_TOKEN>" \
       "http://localhost/api/audit/export?format=jsonl&username=alice@example.com&from=2026-04-18T00:00:00Z&to=2026-05-18T00:00:00Z&limit=100000" \
       -o audit-alice.jsonl

   # CSV export, all DENY decisions
   curl -sS \
       -H "Authorization: Bearer <ADMIN_TOKEN>" \
       "http://localhost/api/audit/export?format=csv&auth_decision=DENY&limit=100000" \
       -o audit-denies.csv
   ```

   Available filters: `stream` (`registry_api` or `mcp_access`),
   `from` / `to` (ISO 8601), `username` (case-insensitive partial),
   `operation`, `resource_type`, `resource_id`, `status_min` /
   `status_max` (HTTP status range), `auth_decision` (`ALLOW`,
   `DENY`, or `NOT_REQUIRED`), `limit` (1 to 100000).

4. Verify the export file:

   ```bash
   wc -l audit-alice.jsonl
   head -1 audit-alice.jsonl | python3 -m json.tool
   ```

### Caveats

- **The `username` filter does case-insensitive substring matching**
  ([`routes.py:223-227`](../../registry/audit/routes.py#L223-L227)),
  not exact equality. `username=alice` will also match
  `alice@example.com` and `MaliceSamuels@example.com`.
- **The export endpoint caps at 100000 events** per call. Larger
  windows must be paginated by `from`/`to` time ranges.
- **The export streams the response.** Large exports may take time
  but won't load the full dataset into memory at once.
- The endpoints are tied to the registry process. **If the registry
  is down, fall through to Path B.**

---

## Path B: Export via direct MongoDB (operator escape hatch)

Use this when the registry isn't running, when you need an
aggregation the REST API doesn't expose, or when you need
full-fidelity BSON output. Every command below was validated
end-to-end against the local stack.

### Procedure: Inspect

1. Connect and confirm the collection name for your namespace:

   ```bash
   docker exec mcp-mongodb mongosh --quiet mcp_registry --eval '
     db.getCollectionNames().filter(n => n.startsWith("audit_events_")).join("\n")'
   ```

2. Count events by user (top 10):

   ```bash
   docker exec mcp-mongodb mongosh --quiet mcp_registry --eval '
     db.audit_events_default.aggregate([
       {$group: {_id: "$identity.username", count: {$sum: 1}}},
       {$sort: {count: -1}},
       {$limit: 10}
     ]).toArray()'
   ```

3. Find recent denies for a single user:

   ```bash
   docker exec mcp-mongodb mongosh --quiet mcp_registry --eval '
     db.audit_events_default.find(
       {"identity.username": "alice@example.com",
        "authorization.decision": "DENY"},
       {_id: 0, timestamp: 1, "request.path": 1,
        "action.operation": 1, "authorization.reason": 1}
     ).sort({timestamp: -1}).limit(20).toArray()'
   ```

### Procedure: Export

1. Bulk export all audit events to JSONL:

   ```bash
   docker exec mcp-mongodb mongoexport \
       --db=mcp_registry \
       --collection=audit_events_default \
       --out=/tmp/audit_events_default.json
   ```

2. Filtered export — events for a user in a time range. Note that
   `timestamp` is a BSON `Date`, so the query needs the
   extended-JSON `{"$date": "..."}` form, and `username` is nested
   under `identity`:

   ```bash
   docker exec mcp-mongodb mongoexport \
       --db=mcp_registry \
       --collection=audit_events_default \
       --query='{"identity.username":"alice@example.com","timestamp":{"$gte":{"$date":"2026-04-18T00:00:00Z"}}}' \
       --jsonArray \
       --out=/tmp/audit_alice.json
   ```

3. Copy the export out of the container:

   ```bash
   docker cp mcp-mongodb:/tmp/audit_events_default.json .
   ```

For full-fidelity BSON output (preserves indexes and BSON types,
necessary for cross-environment restores), see
[mongodb-export-import.md](mongodb-export-import.md) for the
`mongodump` / `mongorestore` pair.

---

## TTL and retention

> **DRAFT — TTL behavior should be confirmed against your install
> before relying on it.** Some deployments leave the audit
> collection without a TTL index (events accumulate indefinitely);
> others apply one via configuration. Check your specific
> environment.

Inspect the indexes on the audit collection to see whether a TTL
is in place:

```bash
docker exec mcp-mongodb mongosh --quiet mcp_registry --eval '
  db.audit_events_default.getIndexes().forEach(i => print(JSON.stringify(i)))'
```

A TTL index appears with `expireAfterSeconds` set; documents older
than that age are removed by MongoDB's TTL monitor (best-effort,
not real-time).

---

## Disabling audit shipping

> **DRAFT — environment-specific.** The registry writes audit events
> via the in-process audit logger initialized at
> [`registry/main.py:394-397`](../../registry/main.py#L394-L397).
> The path to disable it varies by deployment surface (env var,
> Helm values, Terraform variable). Confirm the exact knob for your
> environment before disabling — and remember that disabling audit
> usually has compliance implications.

For most installs, audit logging is on by default and is gated by
the audit middleware registration in `registry/main.py`. Disabling
it for an emergency (e.g. a runaway log producer filling the
collection) is a deployment-config change, not a runtime flag.

The non-disruptive alternative for a runaway producer: leave audit
on, identify the producer via Path B aggregation queries above,
then address the producer.

---

## Related runbooks

- [mongodb-export-import.md](mongodb-export-import.md) — full
  export/import tooling, including the `mongodump` / `mongorestore`
  pair for cross-environment restores.
- [incident-response.md](incident-response.md) — credential-leak
  response. The audit log is your first stop for scoping the
  affected user/time-range window.

## Code references

- [`registry/audit/routes.py:30`](../../registry/audit/routes.py#L30) — audit router prefix (`/audit`, mounted at `/api`).
- [`registry/audit/routes.py:44`](../../registry/audit/routes.py#L44) — `require_admin` dependency.
- [`registry/audit/routes.py:567`](../../registry/audit/routes.py#L567) — `GET /api/audit/events` (paginated query).
- [`registry/audit/routes.py:821`](../../registry/audit/routes.py#L821) — `GET /api/audit/export` (JSONL/CSV streaming).
- [`registry/audit/routes.py:215-275`](../../registry/audit/routes.py#L215-L275) — `_build_query()`, the canonical filter-to-Mongo translation.
- [`registry/repositories/audit_repository.py:151`](../../registry/repositories/audit_repository.py#L151) — `audit_events` base collection name.
