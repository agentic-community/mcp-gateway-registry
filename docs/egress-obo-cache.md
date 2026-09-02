# Egress OBO Exchanged-Token Cache

*Status: Feature documentation. Off by default (`EGRESS_OBO_CACHE_ENABLED=false`).*

The OBO (on-behalf-of) exchanged-token cache is an **opt-in** optimization for
the same-IdP token-exchange hop the auth-server performs when an MCP server calls
a first-party API on the user's behalf (`obo_exchange`). By default the gateway
is **stateless**: every request that needs an OBO token re-runs the exchange
against the IdP. With the cache enabled, the auth-server stores each exchanged
token in the per-user SecretStore so that repeated calls by the **same principal
to the same audience** reuse a still-valid token instead of re-exchanging.

This is a latency/throughput and IdP-load optimization, not a correctness
requirement. Leaving it off preserves the stateless per-request exchange, which
is the safest default for high-sensitivity deployments (see
[Security trade-off](#security-trade-off)).

---

## Table of Contents

- [Settings](#settings)
- [Security trade-off](#security-trade-off)
- [Auth-server prerequisite](#auth-server-prerequisite)
- [Metrics](#metrics)
- [Backend storage behavior](#backend-storage-behavior)
- [Operational notes](#operational-notes)

---

## Settings

Three environment variables control the cache. They are read by **both** the
registry and the auth-server processes: the cache itself runs in the auth-server,
but `config.Settings` is shared and both build it, so set all three on both
services / subcharts wherever the feature is enabled.

| Env var                                | Default | Meaning                                                                                                                                                                                                                                   |
|----------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EGRESS_OBO_CACHE_ENABLED`             | `false` | Opt-in: cache exchanged OBO tokens in the per-user SecretStore so repeated calls by the same principal to the same audience reuse a still-valid token instead of re-exchanging. Default off preserves the stateless per-request exchange. |
| `EGRESS_OBO_CACHE_MAX_TTL_SECONDS`     | `300`   | Hard cap (seconds) on the OBO cache reuse window, applied below the token's real expiry. Bounds revocation/conditional-access re-evaluation latency on cache hits.                                                                        |
| `EGRESS_OBO_CACHE_EXPIRY_SKEW_SECONDS` | `30`    | Never reuse a cached OBO token within this many seconds of its expiry. Must be `< EGRESS_OBO_CACHE_MAX_TTL_SECONDS`.                                                                                                                      |

The effective reuse window for any cached token is
`min(token_expiry - EGRESS_OBO_CACHE_EXPIRY_SKEW_SECONDS, now + EGRESS_OBO_CACHE_MAX_TTL_SECONDS)`.

Helm value keys live under `egressAuth` in both the `registry` and `auth-server`
charts: `oboCacheEnabled`, `oboCacheMaxTtlSeconds`, `oboCacheExpirySkewSeconds`.

---

## Security trade-off

Every re-exchange is a fresh authorization decision. When the auth-server runs
the OBO exchange against the IdP, the IdP re-evaluates the request:
per-request conditional-access policy, session validity, and token revocation are
all re-checked. On a **cache hit the gateway skips that round-trip**, so those
checks are not re-evaluated for the duration of the reuse window.

Concretely, revocation and conditional-access re-evaluation latency increases by
up to `min(remaining-skew, MAX_TTL)` on cache hits — a token revoked or a policy
tightened at the IdP is not observed by the gateway until the cached entry falls
outside its reuse window.

For high-sensitivity deployments, keep `EGRESS_OBO_CACHE_MAX_TTL_SECONDS` small
(so the worst-case staleness window is short) or leave the cache off entirely.
The stateless default trades throughput for immediate re-authorization on every
call.

---

## Auth-server prerequisite

Today only the registry instantiates the SecretStore
(`get_secret_store()`); the auth-server has never been a SecretStore client.
**Enabling this cache makes the auth-server a SecretStore client for the first
time**, because the cache reads and writes exchanged tokens through the per-user
SecretStore.

Therefore, before enabling the cache, the auth-server must receive the same
secret-store backend configuration the registry service already has:

- `SECRET_STORE_BACKEND`, plus the backend-specific config:
    - **OpenBao** (EKS/Helm): `OPENBAO_ADDR`, `OPENBAO_KV_MOUNT`,
      `OPENBAO_AUTH_METHOD`, `OPENBAO_ROLE` (and `OPENBAO_NAMESPACE` /
      `OPENBAO_TOKEN` where the registry uses them).
    - **AWS Secrets Manager** (ECS): `SECRETS_MANAGER_PATH_PREFIX`,
      `AWS_SECRETS_REGION`, `SECRETS_MANAGER_KMS_KEY_ID`.
- The corresponding backend authorization for the **auth-server identity**,
  **scoped to the reserved `obo-cache` namespace** (never the whole vault) and
  **provisioned automatically, gated on `EGRESS_OBO_CACHE_ENABLED`**:
    - **OpenBao** (EKS/Helm): the stack's `openbao-init` job writes a scoped
      `mcp-obo-cache` policy (limited to `<kv>/data/<prefix>/b2JvLWNhY2hl/*` — the
      base64url-encoded `obo-cache` segment — create/read/update, no delete) and
      binds it to a dedicated `auth-server-obo-cache` Kubernetes-auth role on the
      auth-server ServiceAccount. Set `auth-server.egressAuth.openbao.authMethod: kubernetes`.
    - **AWS Secrets Manager** (ECS): the Terraform module attaches a scoped IAM
      policy to the auth-server task role — Secrets Manager CRUD (no `DeleteSecret`)
      on `<prefix>/b2JvLWNhY2hl/*` only, plus the CMK if configured.
      The auth-server can read/write only its own `obo-cache` entries, enforced at
      **two layers**: (1) the in-process SecretStore handle is namespace-clamped — it
      refuses any `auth_method` other than `obo-cache`, so a bug or a future caller
      cannot even attempt a PAT/3LO read; and (2) the backend IAM/OpenBao policy is
      scoped to the same namespace. **Caveat:** under OpenBao **token** auth the
      scoped `auth-server-obo-cache` role does not apply and the token's own policy
      governs — use `kubernetes` auth (the chart default) so the scoped role is the
      enforcing control, or scope the token's policy yourself.

If the cache is left off, the auth-server does not touch the SecretStore and none
of the above is required.

---

## Metrics

The cache path emits the following counters, labeled by `idp` (store errors are
labeled by `op`):

| Metric                               | Meaning                                                                                                                                                                                                             |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mcpgw_obo_cache_hit_total`          | A cached exchanged token was reused.                                                                                                                                                                                |
| `mcpgw_obo_cache_miss_total`         | No usable cached token; an exchange was required.                                                                                                                                                                   |
| `mcpgw_obo_exchange_performed_total` | An OBO token exchange was run against the IdP.                                                                                                                                                                      |
| `mcpgw_obo_exchange_failure_total`   | An OBO token exchange against the IdP failed.                                                                                                                                                                       |
| `mcpgw_obo_cache_store_error_total`  | Writing an exchanged token to the SecretStore failed (the request still succeeds; the token is simply not cached).                                                                                                  |
| `mcpgw_obo_cache_unstorable_total`   | An exchange succeeded but the IdP returned no usable `expires_in`, so nothing could be cached. Nonzero ⇒ the cache can never populate for that IdP (opaque/short-lived tokens) — a config signal, not a tuning one. |

Hit ratio (`hit / (hit + miss)`) is the headline signal for whether the cache is
earning its keep. Alert on `exchange_failure` (IdP trouble) and
`cache_store_error` (degraded store); a rising `unstorable` count means the IdP
never returns a cacheable lifetime.

---

## Backend storage behavior

The cache stores one entry per `(user, idp, audience, scope-set)` tuple. How those entries behave depends on the
SecretStore backend:

- **Store operations per miss:** each miss performs one read and one write against the backend. While waiting on another
  replica's in-flight exchange, a contended cold miss also re-reads on a backoff. The hit ratio therefore determines how
  much backend traffic the cache adds alongside the IdP exchange.
- **Entry lifecycle:** the cache does not delete entries. A near-expiry entry is treated as a miss and overwritten in
  place on the next exchange for the same tuple. Neither backend applies a server-side TTL to these entries (OpenBao KV
  as configured by the chart, or AWS Secrets Manager), so an entry for a tuple that does not recur stays stored, holding
  an expired token, until it is overwritten or removed. On AWS Secrets Manager each entry is a separate secret,
  encrypted with the configured KMS key, and standard Secrets Manager pricing applies to stored secrets and API calls.
- **Sizing guidance:** the number of stored entries grows with principal/audience/scope cardinality, so factor that into
  capacity planning on either backend. Automatic cleanup of expired `obo-cache` entries is not yet implemented.

---

## Operational notes

- **Clock skew:** `expires_at` is written by the exchanging replica's clock and
  read against the reading replica's clock. `EGRESS_OBO_CACHE_EXPIRY_SKEW_SECONDS`
  (minimum 1s) must cover both the token-expiry margin **and** inter-replica NTP
  drift; the 30s default is comfortable for well-synced hosts.
- **Cross-replica single-flight** relies on the Mongo-backed lease. If that lease
  backend is unavailable the auth-server logs a startup WARNING and degrades to
  **per-replica** single-flight (a cold-miss herd may issue one exchange per
  replica rather than one globally) — correct, just less efficient.
- **Single-flight timing invariant:** `put_timeout < exchange_timeout < lease_ttl
  < wait_total_s`, so a waiter always outlasts a dead lease-holder and is promoted
  instead of stampeding the IdP when the exchange is slow.

---

## Related documentation

- [Per-User Egress Credential Vault (Third-Party OBO)](egress-credential-vault.md) — the credential vault and OBO
  exchange this cache sits in front of.
- [Unified Parameter Reference](unified-parameter-reference.md) — all deployment-surface parameters.
