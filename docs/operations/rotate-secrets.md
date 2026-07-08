# Operations: Rotate Secrets

This runbook covers rotation procedures for the registry's
process-level secrets. Each section identifies what the secret
encrypts or signs, what rotation invalidates, the rollout sequence
across replicas, and the verification step.

> **Environment-portability note.** The commands below illustrate the
> compose-stack workflow. **On EKS and ECS deployments these
> instructions are directional only** — use your platform's
> equivalents (kubectl rollout, ECS service update, Helm upgrade)
> for restarts, and your secret-manager pattern (Kubernetes
> Secrets, AWS Secrets Manager, SSM Parameter Store, etc.) for the
> secret-update step. The *order* and *blast radius* documented below
> apply across deployments; the *invocation* differs.

> **DRAFT — destructive validation not exercised.** The procedures
> below were authored from the implementation in
> [`registry/auth/session_crypto.py`](../../registry/auth/session_crypto.py)
> and related modules. The destructive steps (rotating `SECRET_KEY`,
> invalidating sessions) were **not** executed against the running
> validation stack; doing so would log out all users mid-session.
> Before running this in production, do a dry-run in a non-prod
> environment first.

---

## Quick reference: what each secret protects

| Secret | What it does | Rotation invalidates | Restart required |
|---|---|---|---|
| `SECRET_KEY` | (1) HMAC-signs internal-auth JWTs between auth-server, registry, and mcpgw. (2) Derives an AES-GCM key (via HKDF) that encrypts `id_token` blobs in the OAuth session store. | All in-process JWT trust + every encrypted `id_token` in `oauth_sessions_*`. Active sessions cannot decrypt their stored `id_token` after rotation. | **Yes — all auth_server and registry replicas.** The AES-GCM cipher is cached in a process-wide singleton ([`session_crypto.py:37`](../../registry/auth/session_crypto.py#L37)) and won't re-derive on a config reload. |
| Federation static token | Authenticates one peer registry to another for federation API access. | Inbound federation calls from peers using the old token (they get 401 until they update their config). | No restart for the *issuing* registry; peer registries must re-fetch and reload. |
| IdP client secret (Keycloak, Entra, Cognito) | OAuth2 client_secret used by the auth-server to exchange authorization codes for tokens. | Pending OAuth flows fail (5-minute grace window). New logins use the new secret. | **Yes — auth-server replicas** (env-var-based config, not hot-reloaded). |
| M2M client secret | Authenticates a non-interactive M2M client to the registry / IdP. | API calls from the M2M client using the old secret (they get 401 until config update). | No registry restart. The M2M client itself must rotate. |

**Rule of thumb:** rotate the smallest scope that addresses the
threat. Targeted rotations (federation token, M2M client secret) are
nearly free. `SECRET_KEY` is the heaviest — it logs out every user
and requires a coordinated rolling restart.

---

## Rotate `SECRET_KEY`

> **DRAFT — destructive.** Do not run this against a prod
> environment without a dry-run pass and a maintenance window. All
> active users will be logged out and will need to re-authenticate.

### When to use

- `SECRET_KEY` itself is suspected compromised (config leak, env-var
  exposure in logs, repository misconfiguration).
- Periodic rotation per your security policy.

For a session-leak incident where `SECRET_KEY` is **not** the
suspected compromise vector, prefer the targeted collection drop
in [incident-response.md](incident-response.md) — same effect on
active sessions, no restart required.

### Procedure

1. Generate a new key. The implementation requires at least 32
   bytes ([`config.py:766-771`](../../registry/core/config.py#L766-L771)):

   ```bash
   # 64-character hex string (32 bytes of entropy)
   openssl rand -hex 32
   ```

2. Update the secret in your deployment surface. The process
   below assumes the same `SECRET_KEY` is shared by all
   `auth_server` and `registry` replicas (this is required —
   different replicas with different keys cannot decrypt each
   other's session writes):

   - **docker compose**: edit the value in `.env` (or your
     extra-env file) and `docker compose up -d` to reload.
   - **Kubernetes / Helm**: update the `global.secretKey` value
     (chart `values.yaml`) or the underlying `Secret` resource;
     `helm upgrade` rolls the deployments.
   - **ECS / Terraform**: update the SSM parameter / Secrets
     Manager secret referenced by the task definition; redeploy
     the service.

3. Restart the affected services so the new key is loaded into
   the AES-GCM cipher singleton. **The order matters:** restart
   `auth_server` first (it issues internal JWTs to the registry),
   then the `registry` replicas, then any process that mints
   internal tokens (mcpgw):

   ```bash
   # docker compose
   docker compose restart auth-server
   docker compose restart registry
   docker compose restart mcpgw
   ```

   For multi-replica deployments, do a rolling restart per service
   so request handling stays available. Helm and ECS handle this
   natively; for compose, restart one replica at a time if you've
   scaled past 1.

4. Drop the `oauth_sessions_*` collection. The encrypted
   `id_token`s in that collection are no longer decryptable under
   the new key, so the records are dead weight. The collection
   recreates itself on the next session write
   (see [incident-response.md](incident-response.md) for the exact
   command).

5. Verify a fresh login works:

   - Open a new browser session and complete an OAuth login.
   - Tail the auth_server logs for `Created session for user
     <name>`.
   - Confirm the session record decrypts on subsequent requests
     (no `Failed to decrypt id_token` warnings in registry logs).

### Caveats

- **Cookie-only deployments before this rewrite carried the
  encrypted token in the cookie.** This deployment carries it in
  MongoDB. The blast radius is the same: rotation invalidates
  stored tokens. ([`session_crypto.py:13-16`](../../registry/auth/session_crypto.py#L13-L16))
- **Half-rolled deploys break SSO.** A registry replica running
  with the new key cannot decrypt sessions written by an
  auth-server still running on the old key. Keep the rollout
  window short.

---

## Rotate federation static token

### When to use

- A peer registry's static token is leaked or rotated by the peer.
- Periodic rotation of inbound federation credentials.

The federation static token is configured at
[`config.py:251-252`](../../registry/core/config.py#L251-L252)
(`federation_static_token_auth_enabled`, `federation_static_token`).
It authenticates inbound federation API calls; the issuing peer
registry includes it in its outbound requests to ours.

### Procedure

1. Generate a new token (use a value at least 32 bytes for
   parity with `SECRET_KEY`):

   ```bash
   openssl rand -hex 32
   ```

2. Coordinate with the peer registry operator — they must update
   the token in their outbound federation config at the same time
   you update it in your inbound config. There is **no overlap
   window** with a single static token. (For a window, use OAuth
   federation auth, which supports rotation via IdP token issuance.)

3. Update the secret in your deployment surface (same channels as
   `SECRET_KEY` above). Federation auth re-reads the token via the
   settings module on each request, so **no restart is required**
   — but a restart is harmless if you prefer to be explicit.

4. Verify the peer's reconciliation worker resumes successfully:

   - Use [`api/registry_management.py peer-status <peer_id>`](../../api/registry_management.py)
     against your registry — the peer should report `enabled` and
     have a recent `last_synced_at`.
   - Tail the registry logs for `Federation auth: ALLOW` entries
     from the peer's IP after rotation; before rotation you may
     see `Federation auth: DENY (invalid token)`.

### Caveats

- **No rotation grace period.** The cutover is atomic — both sides
  need the new value or one side will start denying.
- **Two-sided coordination.** This is the fragile part: agree with
  the peer operator on a rotation timestamp before doing it.

---

## Rotate IdP client secret

> **DRAFT — IdP-specific steps not exercised in this PR's
> validation.** The procedure below describes the sequence at the
> level of "what you do with the auth-server"; the IdP-side steps
> (Keycloak admin console, Entra app registrations, Cognito user
> pool client) vary by provider and are not documented here.

### When to use

- IdP client_secret is suspected compromised.
- Periodic rotation per IdP policy (Entra and many Cognito setups
  expire client secrets on a schedule).

### Procedure

1. **In the IdP**: rotate the client secret. Most IdPs let you
   *add* a new secret while keeping the old one active for a
   grace period — do that if available, so step 2 has overlap.

2. **In the deployment**: update the env var consumed by the
   auth-server (typically `OAUTH_CLIENT_SECRET_<PROVIDER>` or
   provider-specific). Update via the same secret-management
   channel as `SECRET_KEY`.

3. **Restart auth-server replicas** so the new secret is picked
   up. The auth-server reads the secret at process start.

4. **In the IdP**: revoke the old secret once you've verified
   logins work with the new one (next step).

5. Verify a fresh login completes end-to-end:

   - Open a new browser session, complete the OAuth flow.
   - Tail the auth-server logs for the token-exchange success
     entry.
   - Look for `400 invalid_client` errors during the cutover —
     they indicate either the IdP and the auth-server have
     desynced, or you missed a replica restart.

### Caveats

- **Pending OAuth flows during the rotation will fail.** Anyone
  mid-login will see an error and need to re-initiate. Do this in
  a low-traffic window or accept the visible failures.
- **Active sessions are unaffected.** The client_secret is used at
  token-exchange time, not for ongoing requests; sessions
  established before rotation continue working until they expire
  normally.

---

## Rotate M2M client secret

### When to use

- A specific machine-to-machine client's credential is suspected
  compromised.
- Periodic rotation for high-privilege clients.

The registry stores M2M client config in `idp_m2m_clients` /
`okta_m2m_clients` (not namespace-suffixed — see
[mongodb-export-import.md](mongodb-export-import.md)). The actual
secret lives in the IdP, not in the registry's MongoDB.

### Procedure

1. **In the IdP**: rotate the M2M client's secret using the
   IdP's API or admin console. Use the IdP's rotation feature if
   available so the old secret is valid during the cutover window.

2. **In the M2M client deployment**: update the secret in the
   client's config (env var, secret-manager entry). Restart the
   client process so the new secret is picked up.

3. **In the IdP**: once the M2M client is verified working,
   revoke the old secret.

4. Verify via the registry's audit log:

   ```bash
   docker exec mcp-mongodb mongosh --quiet mcp_registry --eval '
     db.audit_events_default.find(
       {"identity.username": "<m2m-client-id>",
        "authorization.decision": "ALLOW"}
     ).sort({timestamp: -1}).limit(3).toArray()'
   ```

   You should see ALLOW decisions immediately after the M2M client
   restart. DENY decisions during this window indicate the
   client picked up the old secret or the IdP rotation didn't
   take effect.

### Caveats

- **No registry restart.** The registry doesn't cache M2M client
  secrets — it validates each request against the IdP.
- **Audit-log scoping.** If you have many M2M clients, narrow the
  query above by `identity.username` (the client_id) so you don't
  read 18000 events.

---

## Coordinated rotation order

If you rotate multiple secrets in one maintenance window (e.g.
post-incident, full-credential refresh), do them in this order:

1. **IdP client secret** first. Affects only token-exchange (one
   moment per login), narrowest blast radius among heavy rotations.
2. **Federation static token** second. Coordinated with peers
   beforehand; instantaneous cutover.
3. **M2M client secrets** third. Per-client, can be staged.
4. **`SECRET_KEY` last.** This logs everyone out; do it after the
   above so users don't have to re-login twice (once for the IdP
   rotation, once for `SECRET_KEY`).

After all rotations, drop `oauth_sessions_*` once (per
[incident-response.md](incident-response.md)) to clean up records
that can no longer decrypt under the new `SECRET_KEY`.

---

## Related runbooks

- [incident-response.md](incident-response.md) — drop
  `oauth_sessions_*` after a `SECRET_KEY` rotation. Also the
  preferred response when sessions are leaked but `SECRET_KEY`
  itself is not the compromise vector.
- [audit-log-export.md](audit-log-export.md) — query the audit log
  during rotation to confirm M2M clients and federation peers
  reconnect successfully.

## Code references

- [`registry/auth/session_crypto.py:13-23`](../../registry/auth/session_crypto.py#L13-L23) — rotation behavior documented inline; restart-required is by design.
- [`registry/auth/session_crypto.py:37`](../../registry/auth/session_crypto.py#L37) — process-wide AES-GCM cipher singleton.
- [`registry/auth/internal.py:49-63`](../../registry/auth/internal.py#L49-L63) — internal-JWT signing with `SECRET_KEY`.
- [`registry/core/config.py:71`](../../registry/core/config.py#L71) — `secret_key` setting.
- [`registry/core/config.py:251-252`](../../registry/core/config.py#L251-L252) — federation static token settings.
- [`registry/core/config.py:766-771`](../../registry/core/config.py#L766-L771) — `SECRET_KEY` startup validation (required, 32+ bytes).
