# Gateway Generic Proxy — Design

Status: Foundation landed dark (PR #1628, stack for #1565). Feature is off by
default and a no-op in production until a later slice flips the flags.

Related design docs: [egress-auth-design.md](egress-auth-design.md),
[internal-hop-authentication.md](internal-hop-authentication.md),
[virtual-mcp-server.md](virtual-mcp-server.md),
[storage-architecture-mongodb-documentdb.md](storage-architecture-mongodb-documentdb.md).

## Problem

Today the gateway can reverse-proxy only two entity types: MCP servers (at
`/mcp-proxy/...`) and A2A agents (at `/agent/...`). Every other registry
entity — skills, custom entities, and agents that want a uniform hop — has no
gateway-fronted route, so a client must reach it directly. That defeats the point
of a gateway: one authenticated ingress, one audit point, one egress policy.

This feature lets **any** entity opt into being served through the gateway by
setting `is_proxied=true` and (where it has no native backend) a
`proxy_target_url`. The registry then renders an nginx location block that routes
authenticated traffic through a single uniform auth-server hop to the entity's
backend, with SSRF defense in depth at every layer.

## Scope and non-goals

- **In scope for the generic hop:** `a2a_agent`, `skill`, `custom`.
- **Unchanged:** MCP servers keep their legacy `/mcp-proxy/...` route; existing
  agent routes keep `/agent/...`. Nothing existing is relocated.
- **Alias-only:** virtual servers never emit a generic block.
- **Ships dark:** `gateway_generic_proxy_enabled`,
  `gateway_canonical_namespace_enabled`, `gateway_proxy_allow_private_targets`
  all default `false`. With the feature off, no location block renders, `/validate`
  mints no generic token, and the render path issues **zero** additional per-tick
  DB queries.

## Path model — what changes and what does not

`/proxy/{entity_type}/{path}/` is the **internal** auth-server endpoint nginx
forwards to; a client never calls it. The client-facing location is
`/{entity_type}/{path}`.

| Entity | Route today | After this feature (when enabled) |
|---|---|---|
| MCP server | `/mcp-proxy/...` | unchanged |
| A2A agent (existing) | `/agent/...` | unchanged |
| A2A agent (opted-in) | — | additional `/a2a_agent/...` generic route |
| Skill / custom (opted-in) | not proxyable | new `/skill/...`, `/{custom-type}/...` |
| Virtual server | alias-only | alias-only (no generic block) |

`gateway_canonical_namespace_enabled` is defined but read nowhere in this
foundation PR — a dormant placeholder. Its intent (per its config docstring) is a
later slice that emits canonical `/entity_type/path` blocks **alongside** the
legacy flat aliases (additive, including for MCP servers), gated until the matching
`/validate` entity-derivation and canonical-alias minting land.

---

## DocumentDB / MongoDB changes

This is the only storage-facing part of the feature, and it is deliberately small.
There is **no schema migration and no backfill**.

### 1. New persisted fields (the `ProxyableMixin`)

`registry/schemas/proxy_mixin.py` defines `ProxyableMixin`, mixed into the five
storage models (server, agent, skill, virtual server, custom entity) and their
request/patch models. These fields therefore become part of each entity document:

| Field | Type | Written by | Meaning |
|---|---|---|---|
| `is_proxied` | `bool` (default `false`) | admin opt-in (API) | when true, the entity gets a gateway route |
| `proxy_target_url` | `str \| None` | admin (API) | backend the gateway forwards to; required for skills/custom (they have no native backend URL) |
| `proxy_resolved_ips` | `list[str]` | resolve-and-validate refresh | IPs the hostname last resolved to (egress re-validation bookkeeping) |
| `proxy_target_host` | `str \| None` | refresh | original hostname preserved for Host/SNI |
| `proxy_disabled_reason` | `str \| None` | refresh | set when the refresh auto-disables a route (e.g. target now resolves to a denied IP); when non-null the entity is treated as NOT proxied |

Because every field has a default, this is a **purely additive, backward-compatible
schema change**. Existing documents simply lack the keys; `is_proxied` reads as its
default (`false`), so a legacy row is "not proxied" with no migration step. A
document is never rejected on read for a missing or even a bad proxy value — see
the read-safety note below.

### 2. New index — `idx_is_proxied`

`registry/repositories/documentdb/_identity_url_sidecar.py` adds
`ensure_is_proxied_index(collection, name)`, called from each DocumentDB
repository's `ensure_indexes()` (server, agent, skill, custom, virtual):

- **Preferred:** a *partial* index
  `create_index("is_proxied", partialFilterExpression={"is_proxied": True})` so
  only the (few) proxied documents are indexed.
- **Fallback:** AWS DocumentDB may reject `partialFilterExpression`, so on failure
  it creates a plain single-field index `idx_is_proxied_plain`. The
  `{"is_proxied": True}` query uses either.
- **Never raises.** If both fail, `list_proxied` still works via an unindexed scan —
  the index is an optimization, not a correctness requirement. Crucially it runs
  *after* `_indexes_created` is set, so an index failure can't leave that guard
  `False` and re-run every index on every op.

### 3. New query — `list_proxied()`

Each DocumentDB repository gets `list_proxied()` (the base interface returns `[]`,
so non-DocumentDB backends are a safe no-op). It is a **projected** query returning
raw dicts, not reconstructed model objects:

```python
projection = {
    "is_proxied": 1, "is_enabled": 1, "proxy_target_url": 1,
    "proxy_resolved_ips": 1, "proxy_target_host": 1,
    "proxy_disabled_reason": 1, "sync_metadata": 1,
}
cursor = collection.find({"is_proxied": True}, projection)  # then _id -> "path"
```

Two deliberate properties:
- **Projection, not full doc:** only the columns the nginx render +
  `resolve_proxy_target` need, keeping the render hot path cheap.
- **Raw dicts, not model reconstruction:** a bypass-written invalid row (federation
  raw write, migration, manual edit) cannot crash the nginx reload by throwing
  during Pydantic reconstruction. On any error the method logs and returns `[]`
  (fail-closed to "nothing to render").

### 4. Zero queries while disabled

`_fetch_generic_proxied_resources()` (the render-time collector) early-returns `[]`
when `gateway_generic_proxy_enabled` is false, **before** touching any repository.
An upgraded-but-not-enabled deployment pays no new per-tick fan-out. There is a test
asserting zero new DB queries in the disabled state.

### Storage read-safety (why no raising validator on stored docs)

The mixin has **no raising field validator** on `proxy_target_url`. Storage models
are reconstructed from the DB on every read; a raising egress check would make a
denied-but-stored target throw on load, and because listings skip rows that fail to
build, the bad record would silently vanish from every listing — hiding exactly what
an admin needs to fix. So the raising check lives only on the request/patch models
(the API edge, built from client payloads, never from stored data). For stored data
the enforcement is that the render/fetch path simply does not produce a route for a
denied target.

---

## End-to-end request sequence (example: a proxied skill)

Assume a skill registered at path `skills/proxy-demo`, flagged `is_proxied=true`
with `proxy_target_url=https://backend.internal.example/api`, and the feature
enabled. A client calls `GET {ROOT_PATH}/skill/skills/proxy-demo/tools`.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant N as nginx (gateway)
    participant V as auth-server /validate
    participant P as auth-server /proxy hop
    participant G as url_guard (guarded_async_client)
    participant B as Skill backend

    C->>N: GET /skill/skills/proxy-demo/tools (Bearer or session cookie)
    Note over N: Matches the generic location /skill/...<br/>sets generic_backend_url, generic_proxy_kind=skill,<br/>entity_path=skills/proxy-demo
    N->>V: auth_request /validate<br/>X-Generic-Proxy-Kind, X-Entity-Path,<br/>X-Original-Method (server-set, not client)
    Note over V: 1. is_generic_request is true (kind non-empty)<br/>2. CSRF gate. state-changing verb + cookie auth gives 403.<br/>GET here is safe and passes<br/>3. verb maps to scope method. MCP wildcard does NOT authorize a verb<br/>4. scope check against the entity path
    alt authorized
        V-->>N: 200 + X-Internal-Token-Generic (generic audience),<br/>X-User, X-Scopes
    else denied
        V-->>N: 401/403 goes to auth_error or forbidden_error
        N-->>C: 401/403
    end
    N->>P: proxy_pass /proxy/skill/skills/proxy-demo/tools/<br/>X-Internal-Token-Generic, X-Upstream-Url is ignored
    Note over P: verify_generic_proxy_token.<br/>decode against GENERIC_PROXY_AUDIENCE, bind entity_type + full path.<br/>upstream_url comes from the TOKEN, not the header
    Note over P: feature latch active, else 503.<br/>build confined outbound URL (sub-path on pinned base).<br/>assert_outbound_host_pinned
    P->>G: guarded_async_client (PROXY_PROFILE, no redirect follow)
    Note over G: resolve + validate + PIN connection IP.<br/>metadata, private, link-local denied.<br/>re-validated on every redirect hop
    G->>B: GET https://backend.internal.example/api/tools
    B-->>G: 200 body (bounded read)
    G-->>P: response
    Note over P: response-header allowlist drops Set-Cookie, HSTS, CSP, framing.<br/>gateway then sets its own security headers
    P-->>N: buffered response
    N-->>C: response
```

### Reading the shapes

1. **nginx generic location block.** Rendered per proxied entity. It sets
   `$generic_backend_url` (a *separate* variable from the MCP `$backend_url`, so the
   MCP token mint never fires here), `$generic_proxy_kind`, and `$entity_path`, then
   issues the `auth_request /validate` subrequest and forwards the server-set marker
   headers.
2. **`/validate` (the authz brain).** Discriminates the generic hop by the non-empty
   `X-Generic-Proxy-Kind`. Runs the CSRF gate first (a state-changing verb under
   ambient cookie auth is refused with 403 before any token mint; Bearer callers
   exempt; gated by `gateway_generic_require_bearer_for_writes`). Maps the HTTP verb
   to a scope method — the legacy MCP `methods:["*"]` wildcard does **not** authorize
   an HTTP verb (that needs explicit `["GET","HEAD"]` or the distinct `http:*`
   token). On success it mints the generic-audience internal token into
   `X-Internal-Token-Generic`.
   - **Marker-spoof invariant:** the marker headers are trusted only because the
     shared `/validate` block redefines them via `proxy_set_header` from nginx
     variables the location set — never from client headers.
3. **`/proxy/{entity_type}/{entity_path:path}` (the uniform hop).**
   `verify_generic_proxy_token` decodes the token against `GENERIC_PROXY_AUDIENCE`
   and binds both `entity_type` and the full registered path on a segment boundary.
   The destination is taken from the **token claim** `upstream_url`, not the inbound
   `X-Upstream-Url` header (which is ignored). It builds a confined outbound URL
   (sub-path appended to the pinned base) and asserts the host stayed pinned.
4. **Fetch via `guarded_async_client` (PROXY_PROFILE).** Resolves, validates every
   IP, and pins the connection IP inside the transport for the request and each
   redirect. `follow_redirects=False`: a 30x is returned to the client verbatim so
   the next hop re-enters the gateway and is re-authorized. Same egress policy as
   registration time, so decisions can't drift.
5. **Response shaping.** A response-header allowlist drops `Set-Cookie`/HSTS/CSP/
   framing; the gateway sets its own security headers. Body is read bounded
   (`gateway_generic_client_max_body_size` / `gateway_generic_max_concurrency`).

### Failure shapes

- Feature latch off (flag disabled or startup egress self-check failed) → `503`.
- Missing/invalid/wrong-audience token → `401` at the hop.
- Egress blocked at connect time (pinned target resolved to a denied IP) → `502`
  ("Upstream not permitted"), logged by reason only (never the raw target).
- Upstream timeout → `504`; other upstream error → `502`.

---

## How a route comes to exist (registration → render)

```mermaid
sequenceDiagram
    autonumber
    participant A as Admin (API)
    participant S as Entity service (register/update)
    participant M as DocumentDB
    participant R as nginx render (config regen)

    A->>S: register/update skill (is_proxied true, proxy_target_url set)
    Note over S: request model validator. STATIC literal-IP egress check, 422 on deny
    Note over S: validate_and_pin_proxy_target. DNS resolve + validate EVERY IP.<br/>4xx on deny. writes proxy_resolved_ips, proxy_target_host
    S->>M: persist doc (proxy fields + pin bookkeeping)
    Note over R: on next config regen with feature enabled
    R->>M: list_proxied per repo (indexed is_proxied true, projected)
    Note over R: resolve_proxy_target drops federated, disabled,<br/>auto-disabled, and targetless rows
    Note over R: safe_generic_block. SKIP-not-fail render guard checking<br/>scheme, egress, entity-type grammar, dot-dot segments, breakout chars.<br/>plus cross-placeholder collision dedup, generic is lowest precedence
    R->>R: emit the entity_type/path location block, run nginx -t, reload
```

The egress policy is enforced at three points on the one canonical `url_guard`:
the API edge (static, 422), the service register/update layer (DNS-aware, 4xx), and
fetch time (pinned transport). A federation sync stripping proxy fields on ingest
and export means peer data can never become a live local route.

---

## Configuration (all fail-closed)

| Setting | Default | Purpose |
|---|---|---|
| `gateway_generic_proxy_enabled` | `false` | master switch; gates render fetch + hop |
| `gateway_canonical_namespace_enabled` | `false` | dormant placeholder (future canonical aliases) |
| `gateway_proxy_allow_private_targets` | `false` | relax loopback/private/CGNAT only (metadata never overridable) |
| `gateway_generic_require_bearer_for_writes` | `true` | refuse state-changing verbs under ambient cookie auth |
| `gateway_generic_client_max_body_size` | `1m` | inbound body cap (strict nginx size-token validator) |
| `gateway_generic_max_concurrency` | (set) | in-flight cap on the hop |
| `gateway_generic_tls_verify` | (set) | upstream TLS verification mode |
| `gateway_egress_selfcheck_enabled` | (set) | probe metadata IPs at startup; disable feature if egress not enforced |

## Observability

- `mcpgw_registry_gateway_generic_blocks_dropped_total{reason=invalid|collision}` —
  counts render-time drops.
- `gateway_egress_policy_unverified` — gauge set to 1 when the startup egress
  self-check found metadata reachable (feature latched off for the process).
