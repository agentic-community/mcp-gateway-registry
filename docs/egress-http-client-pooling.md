# Egress HTTP connection pooling — operations guide

The gateway pools its outbound (egress) HTTP clients so repeated egress calls
reuse warm TCP+TLS connections instead of handshaking on every request. This
covers the OBO token exchange, the 3LO exchange/refresh, the egress-token vend,
the MCP-proxy egress stream, the registry health-check loop, and the browser
login OAuth callback (authorization-code token exchange + userinfo).

Pooling is on by default and requires no configuration. This page documents the
tuning knobs, the observability signal, and the correct rollback lever.

## Configuration

All four settings are non-secret and wired across Docker (`.env`), ECS Terraform
(`.tfvars`), and EKS Helm (`values.yaml`). See
[`docs/unified-parameter-reference.md`](./unified-parameter-reference.md) for the
per-surface names.

| Env var | Default | Meaning |
|---|---|---|
| `EGRESS_HTTP_POOL_MAX_CONNECTIONS` | `100` | Max total connections per pooled client. Bounds FD/ephemeral-port use under burst. Size above expected concurrent MCP streams + short calls. |
| `EGRESS_HTTP_POOL_MAX_KEEPALIVE` | `20` | Max idle keep-alive connections per pooled client. Clamped to `MAX_CONNECTIONS` at startup. |
| `EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS` | `30` | Idle keep-alive expiry. **Set this below the shortest upstream/LB idle timeout** (see below). |
| `EGRESS_HTTP_POOL_CONNECT_RETRIES` | `1` | httpx transport connect-establishment retries. |

These are consumed by **both** the registry and the auth-server processes (the
guard/pool code is shared), so set them identically on both.

## Tuning `KEEPALIVE_EXPIRY` (the keep-alive race)

A pooled keep-alive connection can be closed server- or load-balancer-side after
an idle period. httpx does **not** auto-retry a non-idempotent POST that lands on
such a half-open connection, so the first request after an idle gap can fail.

- Set `EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS` **below** the shortest idle
  timeout of anything the gateway calls out to (IdP token endpoints, the ALB in
  front of the registry vend, upstream MCP servers). Many ALBs/NLBs and IdP fronts
  close idle keep-alives at 10–60s; the 30s default is a starting point, not a
  guarantee — verify against your fronting infrastructure.
- The OBO exchange and the egress-token vend additionally wrap their POST in a
  single transparent reconnect retry (they are idempotent), so a reset there is
  self-healed once. The 3LO exchange/refresh and the streaming paths do **not**
  retry (see below).

## Observability: `mcpgw_registry_egress_conn_reset_total{site}`

Counts pooled-client keep-alive resets that triggered the transparent reconnect
retry, labeled by `site` (`obo` | `vend`). A rising count means
`KEEPALIVE_EXPIRY_SECONDS` is set **above** an upstream idle timeout — lower it.

**Scope caveat (important):** this counter only covers the two POST hops that use
the reconnect helper (`obo`, `vend`). It does **not** cover:

- the **3LO** exchange/refresh — its POST is deliberately *not* retried, because
  authorization-code and refresh-token grants are single-use/rotating and a blind
  re-POST could double-spend the grant; a reset there surfaces as a transient
  "token endpoint unreachable" that the refresh worker retries on its next pass;
- the **streaming** hops (MCP-proxy egress stream, health initialize/probe) — a
  streaming POST cannot be wrapped by the reconnect helper; a reset surfaces as a
  502 (the MCP client) or a failed health check (retried next cycle).
- the **login callback** (authorization-code token exchange + userinfo) — its
  calls are pooled but not wrapped in the reconnect helper; a reset surfaces as a
  failed login that the user retries.

So do not treat this counter as capturing *all* keep-alive resets — it captures
the two idempotent POST hops only. Watch egress latency/error dashboards for the
streaming and 3LO paths.

## Rollback

To back out pooling behavior at runtime **without redeploying code**:

- Set `EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS=0` — disables keep-alive reuse
  (each request opens a fresh connection) while keeping the pooled client objects.

Do **not** use `EGRESS_HTTP_POOL_MAX_CONNECTIONS=1` as a rollback: a single
connection is still kept alive and reused, and it serializes all egress. It is
not equivalent to the old per-call behavior.

A full code rollback is reverting the egress call sites to per-call clients.

## Security notes

- The SSRF guard is unchanged: every request is validated and pinned to a public
  IP by the guarded transport *before* pool checkout, and the connection pool is
  keyed by the pinned IP, so a rebound hostname re-resolves to a new pool entry
  (rebind-safe). `verify` is part of the pooled-client key.
- Shared clients hold no shared identity state: no default auth headers (every
  credential rides a per-request header) and cookie persistence is disabled via a
  no-store cookie jar, so a `Set-Cookie` can never be replayed onto another
  request/user sharing the client.
- Because the pool is keyed by the pinned IP, two hostnames that resolve to the
  same public IP can share one TLS connection (connection coalescing). This is
  safe — each request is independently pinned and carries the correct `Host` — but
  do **not** enable HTTP/2 on these clients, which would coalesce far more
  aggressively across hostnames on a shared cert.
- The browser-login OAuth callback (authorization-code exchange + userinfo) pools
  its calls on the **plain** (un-SSRF-guarded) shared client, not the guarded
  credentialed-OAuth client: its target is the operator-configured login IdP from
  `oauth2_providers.yml` (for Keycloak/PingFederate the in-cluster `KEYCLOAK_URL` /
  base URL, which defaults to `http://`), and the HTTPS-only credentialed-OAuth
  guard would reject an `http://` in-cluster token endpoint and break login. The
  target is static operator config — never request- or registrant-derived — so the
  plain client is appropriate; credentials still ride per-request.
