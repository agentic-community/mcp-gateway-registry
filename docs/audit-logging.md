# Audit Logging

MCP Gateway Registry provides comprehensive audit logging for compliance, security monitoring, and operational visibility. All API requests and MCP server access events are logged to MongoDB/DocumentDB, and are expired by a TTL index that an initialization script creates — see [Data Retention](#data-retention).

![Audit Log Viewer](img/audit-log.png)

## Overview

Audit logging captures three types of events, written to one collection and
distinguished by `log_type`:

1. **Registry API Access** (`registry_api_access`) - All REST API requests to the Registry (`/api/*`, `/v0.1/*`)
2. **MCP Server Access** (`mcp_server_access`) - All MCP protocol requests proxied through the Gateway
3. **Token Mint** (`token_mint`) - Tokens the auth server signs, recorded at the signing point on both success and failure. This stream stores its fields flat (no nested `identity` / `action` blocks)

Sensitive data such as authentication tokens, session cookies, and passwords are never logged. Credential values are never recorded at all — a credential hint records only that a credential was present, never any part of its value.

## Durability and Attribution

### Durable-by-default (fail closed)

An audit trail is only dependable if it lands in a durable store (MongoDB/DocumentDB). Best-effort JSON log lines are not a durable audit trail: they can be lost on container restart, are not queryable for forensics, and are rotated away.

When audit logging is enabled (`AUDIT_LOG_ENABLED=true`) but no durable sink is available (MongoDB disabled or unreachable), the registry **refuses to start** rather than silently degrading to non-durable log lines. This is controlled by `AUDIT_LOG_REQUIRE_DURABLE` (default `true`).

- `AUDIT_LOG_REQUIRE_DURABLE=true` (default): fail closed at startup if no durable sink is available.
- `AUDIT_LOG_REQUIRE_DURABLE=false`: allow startup with a non-durable (best-effort log-line) trail, emitting a loud startup warning. Use only in local/dev.

### Per-instance attribution

`registry_api_access` records carry an `instance_id` identifying the registry replica that produced them; `MCPServerAccessRecord` and `TokenMintAuditRecord` declare no such field, so `mcp_server_access` and `token_mint` records are not attributed to a replica. Internal service tokens embed the same per-instance identifier in their subject (`<service>@<instance_id>`) so an internal action is attributable to a specific caller/replica rather than a shared service identity. The identifier is resolved from `AUDIT_INSTANCE_ID`, then `HOSTNAME` (set per-container by Docker and per-pod by Kubernetes), then the host name.

### Runtime write failures

Startup fails closed on a missing durable sink (see above), but a transient durable-write failure at request time (e.g. a momentary MongoDB unavailability) does not fail the API request — failing every call on an audit blip would be a self-inflicted denial of service. Instead, a dropped record is surfaced as a distinct `CRITICAL` `AUDIT RECORD DROPPED` log event (carrying identifying context, never the full record) so the loss is loud and alertable. A retry/dead-letter buffer that guarantees no record is lost on a transient failure is planned follow-up hardening.

### Tamper-evidence (roadmap)

Audit records are currently protected by the durable store's access controls but do not carry a per-record integrity seal (e.g. an HMAC hash chain) or an application-enforced append-only guarantee. A stronger tamper-evident guarantee is best provided by shipping the trail to an append-only external store (for example, AWS CloudWatch Logs with log-group immutability / log integrity validation) rather than by in-process hashing alone. This is planned infrastructure hardening and is tracked separately.

## Security and Privacy

### Data That Is NOT Logged

The following sensitive data is explicitly excluded from audit logs:

- **Authentication tokens** (Bearer tokens, JWT tokens)
- **Session cookies** (Cookie header values)
- **Passwords** (form fields, query parameters)
- **API keys** (full values)
- **Refresh tokens**
- **Authorization header values**

### Data Masking

A credential hint records only the *presence* of a credential, never any part of its value:

- Any credential value — bearer token, session cookie, API key — becomes the fixed marker `***`, regardless of its length.
- No suffix is emitted: the trailing characters of a short token are a large fraction of its key space, and audit records land in a store that may be read more widely than the request path. The credential *type* (`session_cookie` vs `bearer_token`) is captured separately on the identity, so this loses no diagnostic value.

Query parameters with sensitive names (token, password, key, secret, api_key, etc.) are automatically masked, by exact name and by substring, so a new variant fails closed.

## Event Schemas

### Registry API Access Event

Logged for every REST API request to the Registry.

```json
{
  "timestamp": "2026-02-06T10:30:00.000Z",
  "log_type": "registry_api_access",
  "version": "1.0",
  "request_id": "abc123-def456-...",
  "correlation_id": null,
  "identity": {
    "username": "john.doe@example.com",
    "auth_method": "oauth2",
    "provider": "keycloak",
    "groups": ["mcp-registry-admin", "developers"],
    "scopes": ["registry-admins"],
    "is_admin": true,
    "credential_type": "session_cookie",
    "credential_hint": "***"
  },
  "request": {
    "method": "POST",
    "path": "/api/servers",
    "query_params": {},
    "client_ip": "192.168.1.100",
    "forwarded_for": "10.0.0.1",
    "user_agent": "Mozilla/5.0...",
    "content_length": 1024
  },
  "response": {
    "status_code": 201,
    "duration_ms": 45.32,
    "content_length": 512
  },
  "action": {
    "operation": "create",
    "resource_type": "server",
    "resource_id": "my-mcp-server",
    "description": "Create new MCP server"
  },
  "authorization": {
    "decision": "ALLOW",
    "required_permission": "servers:write",
    "evaluated_scopes": ["registry-admins"]
  }
}
```

The durable IdP claim fields (`subject`, `canonical_id`, `principal_name`,
`object_id`, `tenant_id`, `app_id`) are **present but always `null`** on
`registry_api_access` records: the record model declares them, so they serialize
as explicit nulls, but the registry receives only a thin signed identity
assertion from the auth server rather than the raw IdP claims and so has nothing
to populate them with — see [Stream coverage](#notes-and-limitations).

Query them with a type or value test, not `$exists`. On this stream
`{"identity.subject": {"$exists": true}}` matches every record;
`{"identity.subject": {"$type": "string"}}` matches only records that carry a
real claim.

### MCP Server Access Event

Logged for every MCP protocol request proxied through the Gateway.

```json
{
  "timestamp": "2026-02-06T10:30:00.000Z",
  "log_type": "mcp_server_access",
  "version": "1.0",
  "request_id": "xyz789-...",
  "correlation_id": null,
  "identity": {
    "username": "ai-agent@example.com",
    "auth_method": "jwt_bearer",
    "provider": "entra_id",
    "groups": [],
    "scopes": ["mcp-server-cloudflare-docs"],
    "is_admin": false,
    "credential_type": "bearer_token",
    "credential_hint": "***",
    "subject": "H2mQ1-9vXk8yTn3rLp0aZ4cFdE7bGjSuVwYx1KtM2No",
    "canonical_id": "8f4a2c1e-5b3d-4e6f-9a7b-0c1d2e3f4a5b@1c2d3e4f-5a6b-7c8d-9e0f-a1b2c3d4e5f6",
    "principal_name": "ai-agent@example.com",
    "object_id": "8f4a2c1e-5b3d-4e6f-9a7b-0c1d2e3f4a5b",
    "tenant_id": "1c2d3e4f-5a6b-7c8d-9e0f-a1b2c3d4e5f6",
    "app_id": "d7e8f9a0-1b2c-3d4e-5f60-7a8b9c0d1e2f"
  },
  "mcp_server": {
    "name": "cloudflare-docs",
    "path": "/cloudflare-docs",
    "version": "1.0.0",
    "proxy_target": "http://internal-mcp-server:8080/mcp"
  },
  "mcp_request": {
    "method": "tools/call",
    "tool_name": "search_docs",
    "resource_uri": null,
    "mcp_session_id": "session-123",
    "transport": "streamable-http",
    "jsonrpc_id": "1"
  },
  "mcp_response": {
    "status": "success",
    "duration_ms": 123.45,
    "error_code": null,
    "error_message": null
  }
}
```

### Token Mint Event

Logged at the auth server's token-signing point, on success and on failure. This
stream has no nested blocks: identity, resource, and outcome fields all sit at
the top level.

```json
{
  "timestamp": "2026-02-06T10:30:00.000Z",
  "log_type": "token_mint",
  "version": "1.0",
  "request_id": "mint-4f1c8a90-...",
  "correlation_id": "xyz789-...",
  "username": "ai-agent@example.com",
  "username_hash": "user_1a2b3c4d",
  "auth_method": "oauth2",
  "provider": "entra_id",
  "internal_caller": "mcp-proxy",
  "subject": "H2mQ1-9vXk8yTn3rLp0aZ4cFdE7bGjSuVwYx1KtM2No",
  "canonical_id": "8f4a2c1e-5b3d-4e6f-9a7b-0c1d2e3f4a5b@1c2d3e4f-5a6b-7c8d-9e0f-a1b2c3d4e5f6",
  "principal_name": "ai-agent@example.com",
  "object_id": "8f4a2c1e-5b3d-4e6f-9a7b-0c1d2e3f4a5b",
  "tenant_id": "1c2d3e4f-5a6b-7c8d-9e0f-a1b2c3d4e5f6",
  "app_id": "d7e8f9a0-1b2c-3d4e-5f60-7a8b9c0d1e2f",
  "token_kind": "resource",
  "resource_type": "server",
  "resource_id": "cloudflare-docs",
  "token_path": "self_signed",
  "requested_scopes": ["mcp-server-cloudflare-docs"],
  "expires_in_seconds": 3600,
  "outcome": "success",
  "failure_reason": null
}
```

`token_mint` records have no `request` block, so they carry no client IP,
forwarded-for header, or user agent.

## Data Fields Reference

### Identity Fields

| Field | Description |
|-------|-------------|
| `username` | Human-readable display identity of the requester (email on most OIDC paths) |
| `auth_method` | Authentication method: `oauth2`, `jwt_bearer`, `anonymous` |
| `provider` | Identity provider: `cognito`, `entra_id`, `keycloak`, `okta`, `auth0`, `pingfederate` |
| `groups` | Groups the user belongs to |
| `scopes` | OAuth scopes granted to the user |
| `is_admin` | Whether the user has admin privileges |
| `credential_type` | Type of credential: `session_cookie`, `bearer_token`, `none` |
| `credential_hint` | Fixed `***` marker recording only that a credential was present. No part of the credential value is emitted — not even a suffix |
| `subject` | OIDC `sub`: opaque, stable per (user, app); protocol-level correlation |
| `canonical_id` | Durable identity: Entra `oid@tid` when both claims are present, else `subject` |
| `principal_name` | Readable principal handle, from the `upn`, `preferred_username`, or `email` claim (in that order). Entra v1.0 tokens carry `upn`; v2.0 tokens carry only `email` |
| `object_id` | Entra user Object ID (`oid`): immutable per user within a tenant |
| `tenant_id` | Entra tenant ID (`tid`) of the IdP that authenticated the caller. An IdP-side identifier — this system is single-tenant, so it does not denote a registry tenant, and it is not the gateway's own app registration |
| `app_id` | Calling application id from the `appid` / `azp` claim |

The last six fields are **nullable, not absent**. The record model declares them,
so every record carries all six keys and a claim the token did not supply
serializes as explicit `null` — including for an IdP that issues none of them.
That distinction matters when you query: `{"identity.subject": {"$exists": true}}`
matches *every* record on those streams, so filter on
`{"identity.subject": {"$type": "string"}}` to select only the records that
actually carry a value. None of the six is an input to any authorization, vault,
or OBO decision.

**Where they live.** On `registry_api_access` and `mcp_server_access` records
they sit inside the `identity` block (`identity.principal_name`, etc.). On
`token_mint` records they are **top-level** fields: that stream has no
`identity` block, and its readable identity is the flat `username` field.

### Action Fields (Registry API only)

| Field | Description |
|-------|-------------|
| `operation` | Operation type: `create`, `read`, `update`, `delete`, `list`, `toggle`, `rate`, `login`, `logout`, `search` |
| `resource_type` | Resource type: `server`, `agent`, `auth`, `federation`, `health`, `search` |
| `resource_id` | Identifier of the resource being acted upon |
| `description` | Human-readable description of the action |

### MCP Request Fields (MCP Access only)

| Field | Description |
|-------|-------------|
| `method` | JSON-RPC method name: `tools/call`, `tools/list`, `resources/read`, `resources/list`, etc. |
| `tool_name` | Name of the tool being called (for `tools/call` method) |
| `resource_uri` | URI of the resource being accessed (for `resources/read` method) |
| `mcp_session_id` | MCP session identifier |
| `transport` | Transport protocol: `streamable-http`, `sse`, `stdio` |
| `jsonrpc_id` | JSON-RPC request ID |

### Token Mint Fields (Token Mint only)

`token_mint` records share `timestamp`, `log_type`, `version`, `request_id`, and
`correlation_id` with the other streams, and carry the six IdP claim fields
(`subject`, `canonical_id`, `principal_name`, `object_id`, `tenant_id`, `app_id`)
at the **top level** rather than under `identity`. The stream-specific fields
are:

| Field | Description |
|-------|-------------|
| `username` | Raw human-readable identity of the requesting user (email → `preferred_username` → `sub`); the flat equivalent of `identity.username` |
| `username_hash` | **Deprecated**, kept for back-compat with dashboards and alerts that key on it: `user_<8 hex>`, the first 8 hex characters (32 bits) of a SHA-256 of `username`. Low entropy by design, so distinct users collide once the population reaches the tens of thousands — a grouping key, never a unique id. Use `username` instead |
| `auth_method` | Authentication method of the requesting user (`oauth2`, `network-trusted`, etc.) |
| `provider` | Identity provider, as on `identity.provider` |
| `internal_caller` | Identity of the internal service that called `/internal/tokens` (for example `mcp-proxy`) |
| `token_kind` | `user` (unrestricted within scopes) or `resource` (bound to one resource); `unknown` when the mint failed before the kind was resolved |
| `resource_type` | For resource-bound tokens: `server`, `agent`, `peer-registry`, etc. Top-level here, not under `action` |
| `resource_id` | For resource-bound tokens: the resource id, e.g. `fininfo` |
| `token_path` | Which signing path produced the token: `self_signed`, `m2m`, or `unknown` on a failure before the path was chosen |
| `requested_scopes` | Scopes requested for the token |
| `expires_in_seconds` | Token lifetime in seconds; `null` when the mint failed before a lifetime was computed |
| `outcome` | `success` or `failure` |
| `failure_reason` | Short reason when `outcome` is `failure` (for example `rate_limited`, `provider_error`) |

### Notes and Limitations

**Forward-only identity.** Recording a readable identity applies to new records
only. Historical records that stored an opaque `sub` as the username are not
backfilled — no durable map from an old `sub` to a user exists, so any backfill
would be a guess. Two consequences while the retention window rolls over:

- The distinct-identity metrics count distinct display identities, so one human
  can be counted twice: once under their old opaque `sub` and once under their
  new readable identity. This applies to every metric built on the same
  distinct-count helper over `identity.username` — DAU/WAU/MAU, the agent
  equivalents (DAA/WAA/MAA), and the executive summary's momentum counters
  `active_identities_current` / `active_identities_prior` and
  `active_agents_current` / `active_agents_prior`.
- A saved filter or alert pinned to an old `sub` value keeps matching that
  user's historical records, because the readable substring match still finds
  the `sub` that was stored as the username. Once the claim fields are populated
  it also matches their new records, through the equality match on
  `identity.subject`. So such a filter may match *more* than intended, not less.
  What it genuinely stops matching is the case where the new records carry no
  `subject` claim at all, because the token did not supply one.

**Stream coverage.** The durable claim fields are populated on the
`mcp_server_access` and `token_mint` streams, both produced by the auth server,
which holds the verified IdP claims. They are **not** present on
`registry_api_access` records. This is an intentional trust boundary, not a gap:
the auth server hands the registry a thin signed assertion (subject, session id,
groups, auth method, client id) rather than raw IdP claims, so the registry never
sees `upn`, `oid`, or `tid` and cannot record them. To correlate across streams,
match a `registry_api_access` record with an `mcp_server_access` record on
`identity.username`; match either against a `token_mint` record on that stream's
top-level `username` field, since `token_mint` has no `identity` block.

## Data Retention

Audit logs are automatically expired using MongoDB/DocumentDB TTL (Time-To-Live) indexes.

### Default Retention

- **Default retention period**: 7 days
- **TTL index field**: `timestamp`

### Configuring Retention

Set the `AUDIT_LOG_MONGODB_TTL_DAYS` environment variable to customize retention:

```bash
# Keep logs for 30 days
export AUDIT_LOG_MONGODB_TTL_DAYS=30

# Keep logs for 90 days (compliance requirement)
export AUDIT_LOG_MONGODB_TTL_DAYS=90
```

The TTL index is created by whichever initialization script matches your storage
backend — the application never creates it:

```bash
# DocumentDB (wraps scripts/init-documentdb-indexes.py)
./scripts/init-documentdb.sh

# MongoDB CE
python scripts/init-mongodb-ce.py
```

On Kubernetes the `setup-mongodb` Job runs the MongoDB CE script for you, and
takes its retention from Helm values:

```yaml
mongodb-configure:
  mongodb:
    auditTtlDays: 90
```

Set it there rather than on the collection. The Job re-runs on every
`helm upgrade`, so a TTL applied out-of-band is reconciled back to whatever the
chart says.

### Important Notes

- TTL indexes run approximately once per minute in MongoDB/DocumentDB
- Documents may persist slightly longer than the TTL value
- Changing the TTL means recreating the index, which both scripts do for you.
  **Shortening** retention is refused by default: the TTL monitor would delete
  every record older than the new window, normally within a minute and
  irreversibly, so the script logs the refusal, leaves the longer retention in
  place, and continues. Set `AUDIT_LOG_MONGODB_TTL_ALLOW_SHRINK=true`
  (`mongodb-configure.mongodb.auditTtlAllowShrink`) when you intend to discard
  those records. Lengthening applies immediately.
- With no init script run, nothing expires. Confirm with
  `db.audit_events_default.getIndexes()` that an index reports
  `expireAfterSeconds`
- For compliance requirements, consider also streaming logs to a long-term archive

## Storage

### MongoDB Collection

Audit events are stored in the `audit_events_{namespace}` collection. Both
initialization scripts create the same index set, under the same names:

| Index | Purpose |
|-------|---------|
| `request_id` + `log_type` (unique) | Fast lookup by request ID within a stream. One request writes a record on more than one stream and they share a `request_id`, so uniqueness is on the pair — a single-field unique index rejected the second write and the record was dropped |
| `identity.username` + `timestamp` | Query by user over time range |
| `action.operation` + `timestamp` | Query by operation type over time range |
| `action.resource_type` + `timestamp` | Query by resource type over time range |
| `mcp_server.name` | Distinct / filter queries by MCP server name |
| `log_type` + `resource_type` + `resource_id` + `timestamp: -1` | Query the `token_mint` stream, whose resource fields are top-level |
| `log_type` + one identity claim + `timestamp: -1`, ×9 | Correlate an IdP-side identifier back to gateway activity. Named `audit_claim_{mcp,mint}_{claim}_idx` |
| `timestamp` (TTL) | Automatic expiration after the configured number of days |

Each claim index is pinned to a single stream with a `partialFilterExpression`, so
it only stores the records it can serve. Measured on a representative 30k-record
mix, 2.09 MB across 13 pinned indexes versus 3.46 MB across 9 unpinned ones, and
an insert updates a subset rather than all of them. `sparse` would not work here:
on a compound index it keeps a document when *any* indexed field exists, and
`log_type` always exists.

There are 9 rather than 13 because `registry_api_access` gets none. Its records
nest the claim fields the same way `mcp_server_access` does, but the registry
receives a thin signed assertion rather than raw IdP claims, so those fields are
permanent nulls on that stream and the audit API does not search them there. It is
also the largest stream, so indexing four always-null fields on it was the most
expensive way to serve no query.

These indexes bound a query to one stream. They do not make the identity filter
seek: it is a single `$or` mixing claim equality with case-insensitive regex on the
readable fields, and an `$or` uses an index union only when every branch is
indexable, which a case-insensitive regex never is. Seeking straight to a claim
value needs the equality branches split out of that `$or`, which is a change to the
query rather than to the schema.

### Schema changes on upgrade

The application never creates or alters an audit index. Only an initialization
script does, so a release that adds an index changes nothing until that script
runs against your database.

On Kubernetes the `setup-mongodb` Job runs it as a `post-install,post-upgrade`
Helm hook, at hook weight `-10` so it precedes the other bootstrap jobs. It
executes the registry image's own `scripts/init-mongodb-ce.py` — there is no copy
of the script in the chart. Consequences worth knowing before you upgrade:

- **The script version is the IMAGE version, not the chart version.** The Job runs
  whatever `scripts/init-mongodb-ce.py` is baked into
  `registry:<global.image.tag>`. Bumping the chart alone therefore runs the OLD
  script; the schema change lands when the registry image is rebuilt and the tag
  is bumped. This is the deliberate trade for removing the chart's copy of the
  script: the copy shipped with the chart but drifted from source, whereas the
  image is built from source every time. Chart and image are released under the
  same version, so upgrading both — as a normal release does — is correct.
- **Repackage the subcharts first.** `charts/*/charts/*.tgz` is gitignored and
  only rebuilt on demand, so a plain `helm upgrade` after a `git pull` deploys the
  *previous* subchart and silently skips the migration:

  ```bash
  cd charts/mcp-gateway-registry-stack
  helm dependency build && helm dependency update
  ```

- **The hook gates the release.** `helm upgrade` waits for the Job. If MongoDB is
  unreachable the Job fails and the upgrade fails, rather than succeeding against
  a database whose schema was never updated. That is deliberate — the previous
  behaviour was to leave the schema stale and say nothing.
- **Index builds are proportional to collection size.** A large `audit_events`
  collection can take minutes. Raise
  `mongodb-configure.job.activeDeadlineSeconds` (default 1800) and your
  `helm upgrade --timeout` together; a deadline shorter than the build marks the
  release failed while the build continues in the background.
- **Retention is never silently shortened.** See
  [Configuring Retention](#configuring-retention).
- **If the unique index fails to build**, the collection already holds a true
  duplicate `(request_id, log_type)` pair — which the pre-migration single-field
  index made possible. The Job logs the exact aggregation to list the offending
  pairs and leaves the old index in place, so uniqueness stays enforced while you
  delete the surplus copies. Re-run the upgrade afterwards.

The migration is forward-only. Rolling the chart back leaves the newer indexes in
place; they are additive and the older application ignores them.

### Storage Sizing

Typical event sizes:
- Registry API event: ~1-2 KB
- MCP Server Access event: ~1-2 KB

Estimated storage (without compression):
- 1,000 requests/day for 7 days: ~14 MB
- 10,000 requests/day for 30 days: ~600 MB
- 100,000 requests/day for 90 days: ~18 GB

## Viewing Audit Logs

### Admin UI

Administrators can view audit logs in the Registry UI:

1. Navigate to **Settings** > **Audit** > **Audit Logs**
2. Select log stream: **Registry API** or **MCP Access**
3. Apply filters (time range, username, operation, status)
4. Click any row to view full event details
5. Export filtered results as JSONL or CSV

### API Access

Query audit events programmatically:

```bash
# Get recent Registry API events
curl -H "Authorization: Bearer $TOKEN" \
  "https://registry.example.com/api/audit/events?stream=registry_api&limit=50"

# Get MCP access events for a specific user
curl -H "Authorization: Bearer $TOKEN" \
  "https://registry.example.com/api/audit/events?stream=mcp_access&username=john.doe"

# Export events as JSONL
curl -H "Authorization: Bearer $TOKEN" \
  "https://registry.example.com/api/audit/export?stream=registry_api&format=jsonl"
```

### MongoDB/DocumentDB Direct Query

```javascript
// Find all events for a user in the last 24 hours
db.audit_events_default.find({
  "identity.username": "john.doe@example.com",
  "timestamp": { $gte: new Date(Date.now() - 24*60*60*1000) }
}).sort({ timestamp: -1 })

// Count events by operation type
db.audit_events_default.aggregate([
  { $match: { log_type: "registry_api_access" } },
  { $group: { _id: "$action.operation", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
])

// Find failed MCP requests
db.audit_events_default.find({
  "log_type": "mcp_server_access",
  "mcp_response.status": "error"
})
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIT_LOG_ENABLED` | `true` | Enable/disable audit logging |
| `AUDIT_LOG_MONGODB_TTL_DAYS` | `7` | Log retention period in days |

### Non-Blocking Design

Audit logging is designed to never impact request processing:

- Logging happens asynchronously after the response is sent
- A failure to record an event never fails the request; the loss is surfaced as a
  `CRITICAL` `AUDIT RECORD DROPPED` log event instead (see
  [Runtime write failures](#runtime-write-failures))
- Each record is written with its own insert (`insert_one` in
  `registry/repositories/audit_repository.py`); there is no write batching

## Compliance Considerations

This section describes what the system records and what it does not. It is not
legal advice and asserts no compliance guarantee: the operator is the data
controller and determines their own obligations. Audit logs are commonly used as
evidence for frameworks such as SOC 2 and ISO/IEC 27001; what those frameworks
require of a given deployment is outside the scope of this document.

### Personal data

Audit records identify people, by design — attribution is the point of an audit
trail. Rather than restate the schema, see [Identity Fields](#identity-fields)
and the per-stream field tables above for exactly what is stored. Beyond the
identity block, note that records also carry group membership and scopes, the
client IP, forwarded-for and user-agent values, and — on `registry_api_access`
records — the request's query parameters, whose values are masked only when the
parameter *name* looks credential-like (see [Data Masking](#data-masking)).

Not recorded: credential values (reduced to a fixed `***` marker), request and
response payloads, tool arguments, and the IdP `name` claim.

### Retention

Records expire through a TTL index driven by `AUDIT_LOG_MONGODB_TTL_DAYS`
(default `7`). That index is created only by an initialization script, never by
the application, so a deployment that has not run one retains audit records
indefinitely — see [Data Retention](#data-retention).

### Access and deletion

Every route under `/api/audit/*` is a `GET` guarded by `require_admin`, so the
API provides no way to alter or delete a record — including an administrator's
own activity. Records leave the store by TTL expiry. `GET /api/audit/export`
produces a filtered extract as JSONL or CSV.

### Operator responsibilities

These are deployment decisions the product does not make for you:

- the lawful basis for this processing, and the notices and records that go with it
- the retention period, and confirming the TTL index actually exists
- who holds admin, since the audit UI exposes other users' personal data
- how you respond to data-subject requests, given there is no deletion endpoint
- whether to stream the trail to a SIEM for longer retention or alerting

## Troubleshooting

### Logs Not Appearing

1. Verify audit logging is enabled: `AUDIT_LOG_ENABLED=true`
2. Check MongoDB connection: ensure the Registry can write to the database
3. Look for warnings in Registry logs: `grep "audit" registry.log`

### TTL Not Working

1. Verify the TTL index exists: `db.audit_events_default.getIndexes()`
2. Note that MongoDB TTL runs approximately every 60 seconds
3. Documents may persist up to 60 seconds beyond their expiration time

### Missing Events

1. Check if the request completed (cancelled requests may not be logged)
2. Verify the log stream filter matches the event type
3. For MCP access, ensure the path is not an API path (starts with `/api/`)
