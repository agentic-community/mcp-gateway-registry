# Egress HTTP connection pooling — operations guide

The gateway pools its outbound (egress) HTTP clients so repeated egress calls
reuse warm TCP+TLS connections instead of handshaking on every request. This
covers the OBO token exchange, the 3LO exchange/refresh, the egress-token vend,
the MCP-proxy egress stream, the registry health-check loop, and the browser
login OAuth callback (authorization-code token exchange + userinfo).

The `mcpgw` MCP server pools the same way: its Registry API
calls (`list_services`, `list_agents`, `list_skills`, `get_skill_content`,
`search_registry`, `intelligent_tool_finder`, `healthcheck`) and its Keycloak M2M
`client_credentials` token POST now all ride one process-lifetime pooled client
instead of opening a fresh `httpx.AsyncClient` per call.

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

These are consumed by **three** processes — the registry, the auth-server, and
the `mcpgw` MCP server — so set them identically on all three.

The registry and the auth-server share the guard/pool code
(`registry/utils/url_guard.py` + `registry/core/config.py`). `mcpgw` cannot: its
image (`docker/Dockerfile.mcp-server`) copies only `servers/mcpgw/`, so it has no
`registry` package and no pydantic `Settings`. It therefore parses the same four
env vars itself with bare `os.getenv` in `servers/mcpgw/http_pool.py`, using the
same defaults and bounds (including the keep-alive clamp to `MAX_CONNECTIONS`).
The pool code on the mcpgw side is duplicated **by necessity, not shared** — a
change to one side has to be mirrored to the other by hand.

Implementation note (both sides): the limits are constructed **on the httpx
transport**, not on the `AsyncClient`. `httpx.AsyncClient` ignores its own
`limits=` whenever an explicit `transport=` is supplied
(`AsyncClient._init_transport` returns the given transport untouched), so passing
the limits to the client leaves the pool on httpx's defaults (100 / 20 / 5s) and
makes all four settings inert. `tests/unit/utils/test_shared_http_clients.py` and
`tests/unit/servers/mcpgw/test_http_pool.py` assert the configured values reach
the live connection pool, not just the `httpx.Limits` object.

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
- On the `mcpgw` side *every* pooled hop is wrapped in that same single
  transparent reconnect retry, because every one of them is idempotent (GET
  reads, the semantic-search POSTs, and the `client_credentials` grant).

## Observability: `mcpgw_registry_egress_conn_reset_total{site}`

Counts pooled-client keep-alive resets that triggered the transparent reconnect
retry, labeled by `site`:

| `site` | Emitted by | Hop |
|---|---|---|
| `obo` | auth-server (meter `mcp-auth-server`) | OBO token exchange POST |
| `vend` | auth-server (meter `mcp-auth-server`) | egress-token vend POST |
| `mcpgw_registry` | mcpgw (meter `mcp-gateway-mcpgw`) | mcpgw → Registry API calls |
| `mcpgw_m2m_token` | mcpgw (meter `mcp-gateway-mcpgw`) | mcpgw → Keycloak M2M `client_credentials` token POST |

A rising count means `KEEPALIVE_EXPIRY_SECONDS` is set **above** an upstream idle
timeout — lower it.

The two `mcpgw_*` label values are emitted by the **mcpgw process** and are
scraped from mcpgw's own Prometheus listener (`mcpgw-server:9464`, scrape job
`mcp-mcpgw`), not from the auth-server that emits `obo`/`vend`
(`auth-server:9464`, job `mcp-auth-server`). The same metric name therefore spans
services: aggregate by `job`/`instance` (and `site`) rather than summing the bare
name, or one service's resets get silently folded into another's.

**Scope caveat (important):** on the registry/auth-server side this counter only
covers the two POST hops that use the reconnect helper (`obo`, `vend`). It does
**not** cover:

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

On `mcpgw`, `mcpgw_registry` + `mcpgw_m2m_token` cover every hop mcpgw makes
*itself* (all are idempotent, all are wrapped), which is more than the
registry/auth-server side gets. They do **not** cover fastmcp's own OAuth hops when
`OIDC_ENABLED=true`: `OAuthProxy` builds clients inline for `/token` (login, client
refresh, proactive refresh) and `/revoke` with no injection point in fastmcp 3.4.7,
so those four call sites are unpooled, unguarded and uncounted. They fire per
login/refresh rather than per tool call. The `/certs` JWKS fetch *is* covered —
`JWTVerifier` accepts a public `http_client=` and is wired to the pooled client.

## mcpgw scope

- mcpgw gets **one** process-lifetime, connection-pooled client covering every
  egress hop it makes: the 7 Registry API calls (`/api/servers`, `/api/agents`,
  `/api/skills`, skill content, `/api/search/semantic` ×2,
  `/api/servers/health`) and the Keycloak M2M `client_credentials` token POST.
- That client is **SSRF-guarded**, like the registry/auth-server pooled clients:
  `servers/mcpgw/http_pool.py::_GuardedAsyncTransport` validates and pins the
  destination on **every** request, before pool checkout. mcpgw's targets are
  operator config (`REGISTRY_BASE_URL`, `KEYCLOAK_INTERNAL_URL`), never request- or
  registrant-derived, so there is no policy decision to make — but configuration
  trust says nothing about what those hostnames *resolve to*, and both hops carry a
  privileged credential (registry API token / M2M access token) and return the
  response body to the MCP caller. So the guard hard-denies the classes that would
  turn a DNS answer into credential exfiltration: cloud metadata /
  workload-identity endpoints (EC2 IMDS, ECS task creds, EKS Pod Identity,
  Alibaba), link-local, unspecified, multicast and reserved — including the IPv6
  wrappers that embed an IPv4 address (IPv4-mapped, NAT64, 6to4, Teredo), which are
  unwrapped before classification.
- The guard also restricts requests to the exact `(host, port)` pairs the process is
  configured for — derived from `REGISTRY_BASE_URL` and `KEYCLOAK_INTERNAL_URL` — so
  a future tool cannot quietly borrow this client, and its credentials, for a third
  destination. Both fall back to the same defaults `servers/mcpgw/server.py` uses
  (they are imported from one place, so the URL the tools build and the destination
  the guard admits cannot disagree), and Keycloak is admitted even when M2M and OIDC
  are both off: a spurious entry is harmless, a missing one would be an outage. The
  set is only ever empty if a configured URL is unparseable — a safety valve that
  normal config cannot reach, since httpx would reject such a URL anyway. The IP
  classification always applies either way.
- Unlike the registry's default profile, mcpgw **allows** private-unicast, CGNAT
  and loopback addresses — matching the registry's own `EGRESS_UPSTREAM_PROFILE`
  semantics — because it legitimately talks to in-cluster service names (`registry`,
  `keycloak`) and to `localhost` in dev/stdio mode.
- Multi-address fallback is preserved, **including the timeout budget**: every
  resolver answer is validated, the transport tries each validated address in
  resolver order, and the caller's connect budget is *divided* across those
  attempts. Unguarded httpx hands the hostname to the connect layer, which walks the
  addresses inside one connect budget; pinning replaces that, so without the split an
  N-address host would multiply the caller's timeout by N (a 30s call against a dead
  dual-stack upstream would hang ~120s with the default one connect retry). With the
  split the ceiling matches unguarded httpx, where `EGRESS_HTTP_POOL_CONNECT_RETRIES`
  still multiplies attempts per address. A dual-stack host with an unreachable AAAA
  (`localhost` → `::1`, IPv4-only listener) connects; a single denied answer fails the
  whole request closed — answer ordering can never salvage it.
- The address that last connected is tried **first** on the next request, provided it
  is still among that request's freshly validated answers. Because the pool is keyed
  by the pinned IP, plain resolver order would make every request open a fresh
  connect to an unreachable first answer before reaching the warm keep-alive on the
  second, and those failed attempts evict the idle connections — measured at one new
  TCP connection per request for `localhost` → `::1, 127.0.0.1`, versus 9 connections
  for 2000 requests with the preference. It is a reordering hint, not a cache: an
  address DNS no longer returns is never used (rebind-safe). One residual
  divergence: only the *connect* budget is divided, so a saturated pool can spend one
  pool-acquisition budget per address (irrelevant at the default
  `MAX_CONNECTIONS=100` against 8 hops).
- URL userinfo is rejected outright: credentials belong in a per-request header, and
  pinning would otherwise carry `user:pass@` onto the rewritten request.
- What mcpgw's guard deliberately omits from the registry's: operator
  allowlists/profiles, the structural `validate_url` checks, and
  `coerce_ip_literal`'s obfuscated IPv4-literal spellings (decimal/octal/hex). Those
  exist for attacker-*supplied* URLs. An obfuscated literal is still handled, by a
  different mechanism: it does not parse as an IP, so it falls through to the
  resolver branch where `getaddrinfo` canonicalizes it and every answer is
  classified.
- The denial table is hand-duplicated from `url_guard`. A test asserts equivalence
  with `url_guard._ip_denial_reason(ip, allow_private=True)`, so drift on either side
  fails a test instead of silently narrowing the guard.
- **Per-request timeouts are preserved** (30s on the Registry API calls, 15s on the
  M2M token POST); the client's default timeout is only a fallback for a call that
  sets none.
- A **no-store cookie jar** is installed, so a `Set-Cookie` is never persisted
  onto a later or concurrent request sharing the client.
- Credentials ride **per-request headers only**; the shared client carries no
  default auth headers.
- Teardown is wired through `FastMCP(lifespan=...)`, which runs on stdio exit and
  on SIGINT. Under **SIGTERM** (`docker stop`, ECS task stop, Kubernetes pod
  termination) fastmcp 3.4.7 skips it — uvicorn re-raises the signal with the
  default handlers restored once `server.serve()` returns, so the lifespan
  shutdown never runs. That is benign for an httpx pool (no write buffering to
  flush, and the kernel reclaims the sockets), so do **not** rely on a guaranteed
  pool close on SIGTERM.

## Rollback

To back out pooling behavior at runtime **without redeploying code**:

- `EGRESS_HTTP_POOL_MAX_KEEPALIVE=0` — the most precise lever: the pool retains zero
  idle connections, so nothing is ever reused. Valid on all three processes.
- `EGRESS_HTTP_POOL_KEEPALIVE_EXPIRY_SECONDS=0` — equivalent in effect (an idle
  connection is retired immediately rather than never retained) and also valid on all
  three processes.

Either way `mcpgw_registry_egress_conn_reset_total` should fall to zero once reuse is
off — that is the confirmation signal that the lever took effect.

Do **not** use `EGRESS_HTTP_POOL_MAX_CONNECTIONS=1` as a rollback: a single
connection is still kept alive and reused, and it serializes all egress. It is
not equivalent to the old per-call behavior.

Note the limits of a runtime rollback on mcpgw: disabling reuse does **not** disable
the two other behaviors this change introduced — the multi-address fallback loop and
the single reconnect retry. Neither has an env lever; backing those out is a code
change.

A full code rollback is reverting the egress call sites to per-call clients — for
mcpgw, the eight call sites in `servers/mcpgw/server.py` (and dropping the
`JWTVerifier(http_client=...)` injection).

## Security notes

- **Registry / auth-server:** the SSRF guard is unchanged — every request is
  validated and pinned to a public IP by the guarded transport *before* pool
  checkout, and the pool is keyed by the pinned IP, so a rebound hostname re-resolves
  to a new pool entry (rebind-safe). `verify` is part of the pooled-client key (those
  clients are keyed per `(profile, verify)`; mcpgw has a single client and never
  varies `verify`).
- **mcpgw:** same per-request validate-and-pin, but private/CGNAT/loopback are
  allowed by design and the destination set is restricted to the two configured
  `(host, port)` pairs. See "mcpgw scope" above.
- Shared clients hold no shared identity state: no default auth headers (every
  credential rides a per-request header) and cookie persistence is disabled via a
  no-store cookie jar, so a `Set-Cookie` can never be replayed onto another
  request/user sharing the client.
- Because the pool is keyed by the pinned IP, two hostnames that resolve to the same
  IP can share one connection (coalescing). This is safe — each request is
  independently pinned and carries the correct `Host` — but do **not** enable HTTP/2
  on these clients, which would coalesce far more aggressively across hostnames on a
  shared cert. It is not reachable on mcpgw in practice: it has two destinations,
  they are distinct container IPs / Service Connect aliases / ClusterIPs in every
  supported deployment, both default to plain HTTP (no cert identity involved), and
  the destination allowlist rejects anything else.
- The browser-login OAuth callback (authorization-code exchange + userinfo) pools
  its calls on the **plain** (un-SSRF-guarded) shared client, not the guarded
  credentialed-OAuth client: its target is the operator-configured login IdP from
  `oauth2_providers.yml` (for Keycloak/PingFederate the in-cluster `KEYCLOAK_URL` /
  base URL, which defaults to `http://`), and the HTTPS-only credentialed-OAuth
  guard would reject an `http://` in-cluster token endpoint and break login. The
  target is static operator config — never request- or registrant-derived — so the
  plain client is appropriate; credentials still ride per-request.
