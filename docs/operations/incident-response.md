# Operations: Incident Response

This page documents incident-response procedures for the MCP Gateway Registry.
Each section is a self-contained runbook intended to be useful at 2am.

---

## Suspected credential leak: invalidate all active sessions

If you suspect a session-cookie or session-record leak (for example: a
database backup was misplaced, an operator's laptop was lost, or telemetry
shows anomalous session activity), you can invalidate every active session
in one step.

The OAuth session payload — username, groups, AES-GCM-encrypted
`id_token` — lives in MongoDB / DocumentDB in the
`oauth_sessions_<namespace>` collection. Dropping the collection removes
every session record at once. The collection is recreated automatically
on the next session write; no service restart is required.

### Procedure

1. Connect to the MongoDB / DocumentDB cluster as a privileged user:

   ```bash
   # Local MongoDB
   mongosh "mongodb://localhost:27017/<DOCUMENTDB_DB_NAME>"

   # AWS DocumentDB (with the standard CA bundle)
   mongosh "mongodb://<DOCUMENTDB_HOST>:27017/<DOCUMENTDB_DB_NAME>" \
       --tls --tlsCAFile /path/to/global-bundle.pem \
       --username <DOCUMENTDB_USERNAME>
   ```

2. Drop the session collection. Replace `<namespace>` with your deployment
   namespace (run `show collections` first to confirm the exact name —
   the suffix matches the value of `DOCUMENTDB_DB_NAME` /
   `MCP_NAMESPACE` for your install):

   ```javascript
   use <DOCUMENTDB_DB_NAME>;
   show collections;  // confirm the oauth_sessions_<namespace> name
   db.oauth_sessions_<namespace>.drop();
   ```

3. All active users are immediately logged out and must re-authenticate.
   The next OAuth callback will recreate the collection with its
   indexes (TTL on `expires_at`, unique on `session_id`).

4. Verify by tailing the auth_server / registry logs for the next
   browser session — you should see a fresh `Created session for user
   <name>` entry in auth_server, and `registry_session_store_resolve_total`
   counters should resume.

### When to use this vs. rotating SECRET_KEY

| Action | Effect | Use when |
|--------|--------|----------|
| Drop `oauth_sessions_*` collection | Invalidates all active sessions. Stored encrypted `id_token`s become unrecoverable (already true even without rotation, since the records are gone). | Cookie or session-record leak suspected. Targeted, fast. |
| Rotate `SECRET_KEY` | Invalidates all active sessions AND all stored encrypted `id_token`s (encrypted under the old key). Requires restarting all auth_server and registry replicas. | `SECRET_KEY` itself is compromised. Heavier; affects encryption key for all future sessions too. |

Use the targeted collection drop unless `SECRET_KEY` is the suspected
compromise vector — then do both.

### Follow-up

- File an incident ticket noting the time of the drop and the suspected
  cause.
- Notify users that they have been logged out (status page / email).
- If `SECRET_KEY` is the suspected compromise vector, rotate it next and
  restart all replicas.
- Review audit logs for the suspected leak window to scope user impact.
