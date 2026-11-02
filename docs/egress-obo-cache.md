# Egress OBO Exchanged-Token Cache

*Status: Feature documentation. Off by default (`EGRESS_OBO_CACHE_ENABLED=false`).*

The OBO (on-behalf-of) exchanged-token cache is an **opt-in** optimization for the token exchange the auth-server runs
when an MCP server needs a token for the user (`obo_exchange`). By default the gateway is **stateless**: every request
that needs an OBO token runs the exchange against the IdP again. With the cache on, the auth-server stores each
exchanged token in the per-user SecretStore, and later calls by the **same user to the same audience** reuse that token
while it is still valid.

The cache saves latency and IdP load. Nothing depends on it for correctness. Leaving it off keeps the stateless
exchange, which is the safest default for high-sensitivity deployments (see [Security trade-off](#security-trade-off)).

---

## Table of Contents

- [Settings](#settings)
- [When the cache is used](#when-the-cache-is-used)
- [Security trade-off](#security-trade-off)
- [Auth-server prerequisite](#auth-server-prerequisite)
- [Metrics and audit](#metrics-and-audit)
- [Backend storage behavior](#backend-storage-behavior)
- [Operational notes](#operational-notes)

---

## Settings

Three environment variables control the cache. Both the registry and the auth-server read them: the cache runs in the auth-server, but both processes build the shared `config.Settings`, so set all three on both services or subcharts wherever you enable the feature.

| Env var                                | Default | Meaning                                                                                                                                                                                                                                   |
|----------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EGRESS_OBO_CACHE_ENABLED`             | `false` | Opt-in: cache exchanged OBO tokens in the per-user SecretStore so repeated calls by the same principal to the same audience reuse a still-valid token instead of re-exchanging. Default off preserves the stateless per-request exchange. |
| `EGRESS_OBO_CACHE_MAX_TTL_SECONDS`     | `300`   | Hard cap (seconds) on the OBO cache reuse window, applied below the token's real expiry. Bounds revocation/conditional-access re-evaluation latency on cache hits.                                                                        |
| `EGRESS_OBO_CACHE_EXPIRY_SKEW_SECONDS` | `30`    | Never reuse a cached OBO token within this many seconds of its expiry. Must be `< EGRESS_OBO_CACHE_MAX_TTL_SECONDS`.                                                                                                                      |

A cached token is reused until `min(token_expiry - EGRESS_OBO_CACHE_EXPIRY_SKEW_SECONDS, exchange_time + EGRESS_OBO_CACHE_MAX_TTL_SECONDS)`.

The Helm values live under `egressAuth` in both the `registry` and `auth-server` charts: `oboCacheEnabled`, `oboCacheMaxTtlSeconds`, `oboCacheExpirySkewSeconds`.

---

## When the cache is used

The auth-server reads from or writes to the cache only when `/validate` verified the request's bearer JWT with Entra or Keycloak, the two IdPs that have an OBO exchange. `/validate` then signs a SHA-256 digest of that token into the internal hop token, and the OBO hop uses the cache only when its subject token matches the digest. Every other request runs the stateless exchange, so the IdP still checks the assertion:

- A request that authenticated with a session cookie. `/validate` checks the cookie first and never verifies a bearer header sent alongside it, so a cached token would otherwise be one forged header away from anyone holding the cookie.
- A request whose bearer is a gateway-minted (self-signed) token, or a token from an IdP with no OBO path.
- A request with no stable user id (`egress_user`) on the hop token.

Cache entries are keyed on the immutable user id, the IdP kind, the target audience and a SHA-256 digest of the requested scopes, so a token cached for one user, audience or scope set is never served for another.

---

## Security trade-off

Every exchange is a fresh authorization decision: the IdP re-checks conditional-access policy, session validity and
revocation. A cache hit skips that round trip, so the IdP does not re-check anything for the length of the reuse window.

A token revoked, or a policy tightened, at the IdP therefore takes up to `min(remaining - skew, MAX_TTL)` longer to take
effect on cache hits. The gateway notices only once the cached entry leaves its reuse window.

For high-sensitivity deployments, keep `EGRESS_OBO_CACHE_MAX_TTL_SECONDS` small so that window stays short, or leave the
cache off. The stateless default gives up throughput in exchange for a fresh IdP decision on every call.

---

## Auth-server prerequisite

Without the cache the auth-server never opens the SecretStore (`get_secret_store()`). **Turning the cache on makes the
auth-server a SecretStore client**, because it reads and writes exchanged tokens there. It then needs the same
secret-store configuration the registry already has:

- `SECRET_STORE_BACKEND`, plus the backend settings:
    - **OpenBao** (EKS/Helm): `OPENBAO_ADDR`, `OPENBAO_KV_MOUNT`, `OPENBAO_AUTH_METHOD`, `OPENBAO_ROLE`, and `OPENBAO_NAMESPACE` where the registry uses it.
    - **AWS Secrets Manager** (ECS): `SECRETS_MANAGER_PATH_PREFIX`, `AWS_SECRETS_REGION`, `SECRETS_MANAGER_KMS_KEY_ID`.
- `EGRESS_CREDENTIAL_ENCRYPTION_KEY`, if the registry encrypts stored credentials. The auth-server then encrypts cached OBO tokens with the same key. Terraform injects the registry's key secret into the auth task when the cache is on. On Helm, set `auth-server.egressAuth.credentialEncryptionKey` to the registry's value; the chart renders it only when `oboCacheEnabled` is `"true"`. On Compose, both services read the same `.env` value.

The auth-server's access to the store is limited to the reserved `obo-cache` namespace in two ways. In process, its SecretStore handle refuses any `auth_method` other than `obo-cache`, so a bug cannot even attempt to read a user's PAT or 3LO entry. At the backend, a scoped policy grants only the namespace's path, `<prefix>/b2JvLWNhY2hl/*` (`b2JvLWNhY2hl` is the base64url encoding of `obo-cache`). Helm and Terraform create that policy automatically, and only when the cache is enabled:

- **OpenBao (Helm stack):** the stack's `openbao-init` job writes an `mcp-obo-cache` policy (create, read and update on `<kv>/data/<prefix>/b2JvLWNhY2hl/*`, no delete) and binds it to an `auth-server-obo-cache` Kubernetes-auth role on the auth-server ServiceAccount. The auth-server chart defaults to `openbao.authMethod: kubernetes` and `openbao.role: auth-server-obo-cache`. With token auth the scoped role does not apply and the token's own policy governs, so keep `kubernetes` auth or scope the token yourself. A standalone auth-server chart install gets no `openbao-init` job, so create the policy and role in OpenBao yourself before turning the cache on, or the auth-server's OpenBao login fails.
- **AWS Secrets Manager (Terraform):** the module attaches a policy to the auth-server task role that allows `CreateSecret`, `GetSecretValue`, `PutSecretValue` and `DeleteSecret` on `<prefix>/b2JvLWNhY2hl/*` in the deployment's own region and account, plus the customer-managed KMS key if one is configured. The cache never deletes entries, but the store deletes its own overflow shards when a user's entries outgrow one secret.
- **Docker Compose:** Compose has no namespace scoping. The auth-server gets a vault token only from `AUTH_SERVER_OPENBAO_TOKEN`, which is empty by default. To use the cache on Compose with the bundled dev OpenBao, set it to the same value as `OPENBAO_TOKEN`. That root token can read every stored egress credential, not only cached OBO tokens.

With the cache off, the auth-server never touches the SecretStore and none of this is required.

---

## Metrics and audit

The cache emits these counters. Most carry an `idp` label; the store-error counter carries `op` and `kind` instead.

| Metric                               | Meaning                                                                                                                                                                                                                       |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mcpgw_obo_cache_hit_total`          | A cached exchanged token was reused.                                                                                                                                                                                          |
| `mcpgw_obo_cache_miss_total`         | No usable cached token; an exchange was required.                                                                                                                                                                             |
| `mcpgw_obo_exchange_performed_total` | An OBO token exchange was run against the IdP.                                                                                                                                                                                |
| `mcpgw_obo_exchange_failure_total`   | An OBO token exchange against the IdP failed or timed out.                                                                                                                                                                    |
| `mcpgw_obo_cache_store_error_total`  | A store read (`op=get`), store write (`op=put`) or lease call (`op=lease`) failed, and the request fell back to a fresh exchange. `kind=backend` is an outage or timeout; `kind=unexpected` is a bug, also logged at ERROR. |
| `mcpgw_obo_cache_unstorable_total`   | An exchange succeeded but the IdP returned no usable `expires_in`, so nothing could be cached. A nonzero count means the cache can never fill for that IdP, which is a configuration signal rather than a tuning one.        |

The hit ratio, `hit / (hit + miss)`, shows whether the cache is doing any work. Alert on `exchange_failure` for IdP trouble and on `cache_store_error` for a degraded store. A `store_error` count with `kind=unexpected` above zero needs a code fix, because the cache is then missing on every request.

Every OBO token the hop forwards still writes a `token_mint` audit record. A cache hit records `token_path="obo_cache"`; a live exchange records `token_path="obo_exchange"`. The IdP's own sign-in logs therefore line up with the `obo_exchange` records only.

---

## Backend storage behavior

The cache stores one entry per `(user, idp, audience, scope set)`. The SecretStore backend determines how those entries behave:

- **Store operations per miss:** each miss does one read and one write. A cold miss that waits on another replica's exchange also re-reads with backoff. The hit ratio therefore decides how much backend traffic the cache adds next to the IdP exchange.
- **Entry lifecycle:** the cache never deletes entries. An entry near expiry counts as a miss and is overwritten on the next exchange for the same key. Neither backend expires these entries on its own (OpenBao KV as the chart configures it, or AWS Secrets Manager), so an entry whose key never recurs stays stored, holding an expired token, until something overwrites or removes it. On OpenBao each entry is its own KV path. On AWS Secrets Manager each user's cached entries share one secret, which spills into overflow shard secrets when it grows past the size target. Entries are encrypted with the configured KMS key, and standard Secrets Manager pricing applies to stored secrets and API calls.
- **Sizing:** the number of stored entries grows with the number of distinct users, audiences and scope sets, so plan capacity for that on either backend. Automatic cleanup of expired `obo-cache` entries is not built yet.

---

## Operational notes

- **Clock skew:** the replica that exchanges writes `expires_at` by its own clock, and the replica that reads compares against its own. `EGRESS_OBO_CACHE_EXPIRY_SKEW_SECONDS` (minimum 1s) has to cover both the expiry margin and clock drift between replicas. The 30s default is plenty for hosts kept in sync with NTP.
- **Cross-replica single-flight** uses the Mongo-backed lease, so a burst of cold misses for one key runs one exchange across all replicas. If the lease backend is unavailable at startup, the auth-server logs a WARNING and falls back to single-flight within each replica, so a burst may run one exchange per replica. A lease error at request time skips single-flight for that request and does not cache the result.
- **Latency budget:** nginx gives each MCP request `MCP_PROXY_TIMEOUT` plus 30 seconds, and the upstream call can use all of `MCP_PROXY_TIMEOUT`, so getting an OBO token has to finish within those 30 seconds. Each store read is capped at 1 second, each write at 2 seconds and the exchange at 10 seconds. A request that finds another exchange in progress waits up to 11 seconds, including any time spent waiting for the lease, before running its own exchange. The slowest path stays under 30 seconds.
- **Crashed holder:** the replica running an exchange holds a 15-second lease. If it dies mid-exchange, requests waiting on it give up after 11 seconds and each run their own exchange without caching. That is the same IdP load as running with the cache off, and the next request after the lease expires fills the cache again.
- **IdP app or tenant change:** the cache key does not include the IdP tenant or the gateway's client id. After either changes, entries cached under the old configuration stay in use until their reuse window ends, at most `EGRESS_OBO_CACHE_MAX_TTL_SECONDS`. To drop them sooner, delete the `obo-cache` namespace from the store.

---

## Related documentation

- [Per-User Egress Credential Vault (Third-Party OBO)](egress-credential-vault.md): the credential vault and OBO exchange this cache sits in front of.
- [Unified Parameter Reference](unified-parameter-reference.md): all deployment-surface parameters.
