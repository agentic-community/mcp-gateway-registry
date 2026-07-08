# Operations: MongoDB Export and Import

This runbook covers exporting and importing MongoDB / DocumentDB
collections used by the MCP Gateway Registry. Each section is a
self-contained procedure intended to be useful at 2am.

> **Environment-portability note.** The commands below were validated
> against the local `docker compose` stack with MongoDB CE running
> in the `mcp-mongodb` container. **On EKS and ECS deployments these
> instructions are directional only** — adapt the connection
> patterns (kubectl exec, ECS exec, DocumentDB endpoint with TLS)
> to your environment. Treat the procedures as the right shape; the
> exact invocation will differ.

The registry stores all per-namespace state under collections suffixed
with the deployment's `documentdb_namespace` (default: `default`). The
suffix is appended at write time by
[`get_collection_name()`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/repositories/documentdb/client.py#L54-L58),
so on a stock install the collections are
`oauth_sessions_default`, `audit_events_default`,
`mcp_servers_default`, etc. Use `show collections` in `mongosh` to
list the exact names for your install before running any of these
procedures.

---

## When to use which tool

The MongoDB toolset includes two distinct export/import pairs. Pick
based on what you actually need.

| Pair | Format | Use for | Preserves |
|------|--------|---------|-----------|
| `mongoexport` / `mongoimport` | JSONL (one JSON document per line) | Compliance exports, ad-hoc inspection, cross-tool data extraction, single-collection migrations | Document content. **Does not** preserve indexes, options, or BSON types beyond what JSON can express. |
| `mongodump` / `mongorestore` | BSON + metadata | Full-fidelity backups, disaster recovery, restoring an entire collection or database | Document content + indexes + collection options + BSON types (e.g. `Timestamp`, `Long`, `Decimal128`). |

**Rule of thumb:** use `mongodump` for backups you intend to restore,
use `mongoexport` for data you intend to read or hand to another tool.

---

## Connecting to the database

The commands below assume you are running them either on the database
host or against a reachable endpoint. Your connection string and
authentication depend on the deployment.

```bash
# Local MongoDB (compose stack — runs in the mcp-mongodb container)
docker exec mcp-mongodb mongosh "mongodb://localhost:27017/<DOCUMENTDB_DB_NAME>"

# AWS DocumentDB (with the standard CA bundle)
mongosh "mongodb://<DOCUMENTDB_HOST>:27017/<DOCUMENTDB_DB_NAME>" \
    --tls --tlsCAFile /path/to/global-bundle.pem \
    --username <DOCUMENTDB_USERNAME>
```

Replace `<DOCUMENTDB_DB_NAME>` with the value from your environment
(default: `mcp_registry`). For all subsequent commands, run inside
the container with `docker exec mcp-mongodb <cmd>` for compose
deployments, or run from a host that can reach the DocumentDB
endpoint with the appropriate `--tls --tlsCAFile` flags.

---

## Export a single collection (mongoexport)

Use this for compliance pulls, ad-hoc inspection, or any case where
the consumer wants JSON.

### Procedure

1. List the collections in your namespace to confirm the exact name:

   ```bash
   docker exec mcp-mongodb mongosh --quiet mcp_registry \
       --eval 'db.getCollectionNames().sort().join("\n")'
   ```

   Expected: a sorted list including (on a default-namespace install)
   `audit_events_default`, `mcp_servers_default`,
   `oauth_sessions_default`, etc.

2. Export the target collection to JSONL. Each line is one document:

   ```bash
   docker exec mcp-mongodb mongoexport \
       --db=mcp_registry \
       --collection=audit_events_default \
       --out=/tmp/audit_events_default.json
   ```

   Output ends with `exported N record(s)`. Copy the file out of the
   container with `docker cp mcp-mongodb:/tmp/audit_events_default.json .`.

3. (Optional) Restrict by query, fields, or sort. Audit events store
   the user under nested `identity.username`, and `timestamp` is a
   BSON `Date` (use the `{"$date": "..."}` extended-JSON form for
   range comparisons):

   ```bash
   # Only events from a specific user, last 30 days, output as one
   # JSON array (use --jsonArray for downstream tools that expect it)
   docker exec mcp-mongodb mongoexport \
       --db=mcp_registry \
       --collection=audit_events_default \
       --query='{"identity.username":"alice@example.com","timestamp":{"$gte":{"$date":"2026-04-18T00:00:00Z"}}}' \
       --jsonArray \
       --out=/tmp/audit_alice.json
   ```

4. Verify the export looks correct before sharing:

   ```bash
   docker exec mcp-mongodb wc -l /tmp/audit_events_default.json
   docker exec mcp-mongodb head -c 200 /tmp/audit_events_default.json
   ```

   The line count should match the source collection's
   `db.audit_events_default.countDocuments()` for an unfiltered export.

---

## Import a collection (mongoimport)

Use this to load a JSONL export back into the database, typically
into a different collection or a different cluster (DR drill,
test-environment seed, etc.).

### Procedure

1. Confirm the destination collection name. **Do not import on top
   of a live collection without explicit reason** — the `--drop` flag
   below removes the existing collection before inserting:

   ```bash
   docker exec mcp-mongodb mongosh --quiet mcp_registry \
       --eval 'db.audit_events_default_restored.countDocuments()'
   ```

2. Import the file. The `--drop` flag drops the destination first so
   the import is atomic on the collection:

   ```bash
   docker exec mcp-mongodb mongoimport \
       --db=mcp_registry \
       --collection=audit_events_default_restored \
       --drop \
       --file=/tmp/audit_events_default.json
   ```

   Output ends with `N document(s) imported successfully. 0 document(s) failed to import.`

3. Verify counts match:

   ```bash
   docker exec mcp-mongodb mongosh --quiet mcp_registry \
       --eval 'db.audit_events_default_restored.countDocuments()'
   ```

### Caveats

- **Indexes are not restored.** `mongoimport` writes documents only;
  any indexes on the source collection (e.g. the TTL index on
  `oauth_sessions_*.expires_at`, the unique index on
  `oauth_sessions_*.session_id`) must be recreated. The registry
  recreates its own indexes at startup if you import into the live
  collection name, but restoring into a renamed collection (e.g.
  `*_restored`) leaves it unindexed. Use `mongodump` / `mongorestore`
  instead if you need full-fidelity restoration.
- **Encrypted fields** (e.g. `oauth_sessions_*.encrypted_id_token`)
  remain encrypted under the original `SECRET_KEY`. Importing into
  a deployment with a different `SECRET_KEY` will produce records the
  registry cannot decrypt.

---

## Full-fidelity backup and restore (mongodump / mongorestore)

Use this for backups you intend to restore — it preserves indexes,
collection options, and BSON types.

### Procedure: Backup

1. Dump a single collection (writes BSON + metadata to a directory):

   ```bash
   docker exec mcp-mongodb mongodump \
       --db=mcp_registry \
       --collection=oauth_sessions_default \
       --out=/tmp/dump
   ```

   Output: `/tmp/dump/mcp_registry/oauth_sessions_default.bson` plus a
   `*.metadata.json` describing indexes and options.

2. Or dump the entire database (every collection in `mcp_registry`):

   ```bash
   docker exec mcp-mongodb mongodump \
       --db=mcp_registry \
       --out=/tmp/dump
   ```

3. Copy the dump out of the container:

   ```bash
   docker cp mcp-mongodb:/tmp/dump ./mcp-registry-dump-$(date +%Y%m%d-%H%M%S)
   ```

### Procedure: Restore

1. Copy the dump back into the target container:

   ```bash
   docker cp ./mcp-registry-dump-20260518-143000 mcp-mongodb:/tmp/restore
   ```

2. Restore. **The `--drop` flag drops each collection before
   restoring it from the dump** — be deliberate:

   ```bash
   docker exec mcp-mongodb mongorestore \
       --db=mcp_registry \
       --drop \
       /tmp/restore/mcp_registry
   ```

3. Verify document counts and at least one index per critical
   collection:

   ```bash
   docker exec mcp-mongodb mongosh --quiet mcp_registry --eval '
     ["oauth_sessions_default","audit_events_default","mcp_servers_default"]
       .forEach(c => print(c, db.getCollection(c).countDocuments(),
                           db.getCollection(c).getIndexes().length))'
   ```

### Caveats

- **`mongorestore` does not invalidate active sessions.** If you are
  restoring to recover from a session-related incident, follow up
  with the [credential-leak runbook](incident-response.md) to drop
  the `oauth_sessions_*` collection after the restore.
- **AES-GCM-encrypted fields** require the `SECRET_KEY` from the
  source deployment. Cross-environment restores must replicate the
  source `SECRET_KEY` or accept that encrypted fields will be
  unreadable.
- **DocumentDB has version-specific compatibility.** When restoring
  a `mongodump` taken from MongoDB CE into DocumentDB (or vice versa),
  use `mongorestore --noIndexRestore` and recreate indexes after
  the data load to avoid index-feature incompatibilities. See the
  [AWS DocumentDB compatibility matrix](https://docs.aws.amazon.com/documentdb/latest/developerguide/mongo-apis.html)
  for the exact set of supported index options.

---

## Scoping to a single tenant (namespace)

Multi-tenant deployments use `documentdb_namespace` to isolate state
between tenants. Each tenant has its own collection set
(`oauth_sessions_<ns>`, `audit_events_<ns>`, etc.). To export or
back up a single tenant, list collections matching the suffix and
operate on each:

```bash
# List all collections for tenant 'acme'
docker exec mcp-mongodb mongosh --quiet mcp_registry --eval '
  db.getCollectionNames()
    .filter(n => n.endsWith("_acme"))
    .sort()
    .join("\n")'

# Dump every collection for tenant 'acme'. mongodump does not have a
# regex include flag, so loop over matching collection names. Each
# collection becomes its own .bson + .metadata.json under the same
# output directory, suitable for a single mongorestore pass.
docker exec mcp-mongodb sh -c '
  set -e
  rm -rf /tmp/dump-acme
  for coll in $(mongosh --quiet mcp_registry --eval "
        db.getCollectionNames().filter(n => n.endsWith(\"_acme\")).sort().join(\"\n\")
      "); do
    mongodump --db=mcp_registry --collection="$coll" --out=/tmp/dump-acme
  done'
```

Restore the tenant by pointing `mongorestore` at the dump directory:

```bash
docker exec mcp-mongodb mongorestore --db=mcp_registry --drop /tmp/dump-acme/mcp_registry
```

`mongorestore` does support a regex include filter via `--nsInclude`
if you want to scope a restore from a larger dump:

```bash
# Restore only tenant 'acme' collections from a full-database dump
docker exec mcp-mongodb mongorestore \
    --nsInclude='mcp_registry.*_acme' \
    /tmp/full-dump
```

### Caveats

- **`idp_m2m_clients`, `okta_m2m_clients`, and `_telemetry_state`
  are not namespace-suffixed.** They are global per-database. A
  per-tenant dump that uses the `_<ns>$` filter above will not
  capture them; include them explicitly if your DR plan requires it.

---

## Related runbooks

- [incident-response.md](incident-response.md) — when to drop
  `oauth_sessions_*` to invalidate active sessions (after a session
  leak or restore).

## Code references

- [`registry/repositories/documentdb/client.py:54-58`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/repositories/documentdb/client.py#L54-L58) — `get_collection_name()` namespace suffixing.
- [`registry/auth/session_store.py:23`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/auth/session_store.py#L23) — `oauth_sessions` base collection name.
- [`registry/repositories/audit_repository.py:151`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/repositories/audit_repository.py#L151) — `audit_events` base collection name.
- [`registry/core/config.py:751`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/core/config.py#L751) — `documentdb_namespace` setting (default: `default`).
