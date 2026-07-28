# Rate Limiting: Design and Implementation Guide

This document describes the application-level rate limiting in the MCP Gateway Registry (issue #295): what it does, how it is designed at a high level, and how it is implemented down to the data model, the enforcement path, and the failure modes. It is written to be read by an engineer who needs to operate, extend, or debug the feature.

## Table of Contents

1. [Operational Guidance (start here)](#operational-guidance-start-here)
2. [What This Is (and Is Not)](#what-this-is-and-is-not)
3. [Two Layers of Rate Limiting](#two-layers-of-rate-limiting)
4. [High-Level Design](#high-level-design)
5. [The Two Axes and Windows](#the-two-axes-and-windows)
6. [Data Model](#data-model)
7. [The Enforcement Path](#the-enforcement-path)
8. [Correctness Across Replicas](#correctness-across-replicas)
9. [Failure Modes](#failure-modes)
10. [Latency](#latency)
11. [Observability](#observability)
12. [Configuration](#configuration)
13. [Managing Limits (Admin API and CLI)](#managing-limits-admin-api-and-cli)
14. [Where the Code Lives](#where-the-code-lives)
15. [Extending It](#extending-it)

---

## Operational Guidance (start here)

**By default, rate limiting is completely off — every authenticated caller has unlimited access, exactly as before.** Nothing throttles until an operator opts in. Two things are *always* unlimited regardless of config: **admins** (they bypass all caller limits so an operator can never lock themselves out) and **control-plane `/api/*`** calls (the dashboard/login are never throttled). Rate limiting applies only to **data-plane** MCP/A2A calls by non-admin callers.

```
                    ┌─────────────────────────────────────────────┐
                    │  DEFAULT: RATE_LIMITING_ENABLED=false         │
                    │  → unlimited access for everyone (no change)  │
                    └───────────────────────┬───────────────────────┘
                                            │  operator opts in
                                            ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ 1. ENABLE enforcement across your deployment surface            │
        │    Set the mandatory param on registry + auth-server:           │
        │        RATE_LIMITING_ENABLED=true                               │
        │    (per-surface names + optional tuning params → unified ref)   │
        └───────────────────────────────┬───────────────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ 2. CREATE a rate-limit group (a caller limit definition)        │
        │        rate-limit-set --axis caller --entity-type group ...     │
        └───────────────────────────────┬───────────────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ 3. ADD callers to the group (membership, keyed on identity)     │
        │    a user   → --subject-type user   --subject <username>        │
        │    an agent → --subject-type client --subject <client_id>       │
        └───────────────────────────────┬───────────────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ 4. TEST it: burst calls as that caller and watch 429s           │
        └───────────────────────────────────────────────────────────────┘
```

### Step 1 — Enable enforcement (mandatory param)

Rate limiting is enabled by a **single mandatory parameter**, set on **both** the `registry` and `auth-server` services:

```
RATE_LIMITING_ENABLED=true
```

Everything else has a safe default (backend, fail-open, cache TTL, timeout, floors). For the per-surface names (Docker `.env`, Terraform `.tfvars`, Helm `values.yaml`) and the optional tuning parameters, see the [unified parameter reference → Group 32, Rate Limiting](../unified-parameter-reference.md#group-32--rate-limiting). For Cognito **agent/M2M** callers there is one extra provider step (`COGNITO_M2M_CLIENT_IDS`) — see [docs/idp/cognito.md](../idp/cognito.md#machine-to-machine-m2m--agent-clients).

### Step 2 — Create a rate-limit group

A group carries a per-window limit. It has two numbers — one for human users, one for agents — and you set whichever apply (at least one). On windows `<= 60s` a floor applies (user `>= 20/min`, agent `>= 10/min` by default), so pick values at or above it.

```bash
# A caller group "power-users": humans 60/min, agents 30/min
uv run python api/registry_management.py --token-file .token --registry-url "$REG" \
  rate-limit-set --axis caller --entity-type group --name power-users \
  --user-max-requests 60 --agent-max-requests 30 --window-seconds 60
```

### Step 3 — Add an existing user and/or agent to the group

Membership is keyed on the caller's identity from the validated token — a **username** for a human, a **client_id** for an agent — and is the ONLY source of a caller's rate-limit groups (decoupled from IdP/authz groups).

```bash
# Add an existing human user by username
uv run python api/registry_management.py --token-file .token --registry-url "$REG" \
  rate-limit-member-set --subject-type user --subject <username> --groups power-users

# Add an existing agent by its client_id (M2M)
uv run python api/registry_management.py --token-file .token --registry-url "$REG" \
  rate-limit-member-set --subject-type client --subject <client_id> --groups power-users
```

(You can also manage all of the above from the UI: **Settings → IAM → Rate Limits**, and edit membership on **Users** / **M2M Accounts**.)

### Step 4 — Test it

Drive calls **as that caller** (their own token, not admin) against an MCP server and watch the limit trip:

```bash
uv run python tests/scripts/call_mcp_tool.py \
  --server-url "$REG/<server>/mcp" --tool <tool> --tool-args '{}' \
  --token-file <caller-token> --registry-url "$REG" --count <limit+5>
# → the first N succeed (200), then HTTP 429 with X-RateLimit-* and Retry-After
```

For full end-to-end walkthroughs (including the response headers, metrics, floors, per-agent M2M, target limits, and failure modes), see the [Testing](#testing) guides below.

### Per-caller-per-target limits (the `caller_target` axis)

To cap each caller *independently per target* (a shared multi-tenant backend where one noisy caller against server X must not consume another caller's headroom against X, and X and Y are counted separately), create a `caller_target` group and add members exactly as for the caller axis — only the `--axis` differs:

```bash
# Each member may call ANY single target at most 60/min (humans) / 30/min (agents),
# with an independent counter per target.
cd api && uv run python registry_management.py \
  rate-limit-set --axis caller_target --entity-type group --name per-server-cap \
  --user-max-requests 60 --agent-max-requests 30 --window-seconds 60 \
  --token-file ../.token --registry-url "$REG"

# Add a caller (same membership mechanism as the caller axis).
uv run python registry_management.py rate-limit-member-set \
  --subject-type user --subject alice --groups per-server-cap \
  --token-file ../.token --registry-url "$REG"
```

Verify: burst `alice` against server X past 60 → 429s for X; her quota against server Y is untouched; and a second caller `bob` against X is unaffected by alice.

### Quarantine (kill switch): block a caller or a target immediately

Quarantine drops **all** data-plane traffic from a caller or to a target — an absolute block, not a rate. The two reserved groups (`quarantine-callers`, `quarantine-targets`) are **auto-seeded empty at startup**, so there is nothing to create: you just move a subject in.

```bash
cd api
# Quarantine a compromised agent (client_id) — all its calls are dropped instantly.
uv run python registry_management.py rate-limit-quarantine-add \
  --subject-type client --subject <client_id> --token-file ../.token --registry-url "$REG"

# Quarantine a user, or take an MCP server / A2A agent out of rotation:
#   --subject-type user   --subject alice
#   --subject-type server --subject mcpgw
#   --subject-type agent  --subject /booking-agent

# List everything quarantined; remove when done.
uv run python registry_management.py rate-limit-quarantine-list --token-file ../.token --registry-url "$REG"
uv run python registry_management.py rate-limit-quarantine-remove \
  --subject-type client --subject <client_id> --token-file ../.token --registry-url "$REG"
```

Operational notes:
- A quarantined subject is denied as a **plain 403** (no `X-RateLimit-*` headers, no 429 rewrite) — it is an access decision, not a throttle. Callers see `403 Access forbidden`.
- **Admins cannot be quarantined at all.** Adding a caller (user or client) that belongs to an admin group to quarantine is refused with a `403` at the API — on both the `rate-limit-quarantine-add` path and the general membership-PUT path (so the reserved group can't be added by the back door). The check resolves the live admin population and **fails closed** (a `503` if it can't be verified), so an operator can never lock the administrators out of the gateway. (Enforcement also still skips caller gates for admins as a second layer.) A quarantined **target** is blocked for everyone, admins included — targets are never admins.
- Changes take effect within one cache TTL (~30s).
- **Global off-switch:** disable a reserved group's definition (`rate-limit-disable --id quarantine:group:quarantine-callers:1`) to turn the whole caller/target kill switch off without emptying the group. The groups cannot be deleted.
- Quarantine only acts while `RATE_LIMITING_ENABLED=true` and is **fail-open by default** (a memberships-store outage allows traffic); set `RATE_LIMIT_QUARANTINE_FAIL_CLOSED=true` to fail closed. It is best-effort containment — pair it with IdP credential revocation.
- UI (admin only): quarantine a **caller** from **Settings → IAM → Users** or **M2M Accounts** via the block icon on each row (admin-group users have no such control and are refused server-side); quarantine a **target** (server/agent) from the **Quarantined targets** box in **Settings → IAM → Rate Limits → Quarantine**, which also shows both groups' live member counts, a per-member remove, and the global on/off toggle. See also the [FAQ: How do I quarantine a user, agent, or MCP server?](../faq/quarantine-a-caller-or-target.md).

---

## What This Is (and Is Not)

MCP tool calls and A2A agent calls pass through the gateway with authentication and authorization, but historically there was no cap on **how often** an authenticated caller could invoke them. A single user or agent could call a tool thousands of times a minute and overwhelm a backend MCP server, exhaust a downstream API quota, or run up cost.

This feature adds **application-level, identity/group/target-aware** rate limiting: limits that are expressed in terms of *who* is calling (via their group membership) and *what* they are calling (a specific MCP server or A2A agent), enforced at the one hop every call already crosses. A specific user or agent is limited by placing it in a rate-limited group.

It is **not** a volumetric DoS control. That job belongs to the coarse per-IP limiting at the nginx edge (see the next section). The two are complementary layers.

## Two Layers of Rate Limiting

The gateway has two independent rate-limiting layers that solve different problems:

| | nginx edge limiting | Application-level limiting (this doc) |
|---|---|---|
| **Where** | nginx, at the inbound edge | auth-server `/validate` hop |
| **Keyed on** | Source IP | Identity (user/client) and target entity (server/agent), via groups |
| **Sees** | Raw connection only | The authenticated user, their groups, the target server/agent |
| **Protects against** | Volumetric floods (DoS) | Per-caller / per-target quota abuse, backend overload, cost |
| **Algorithm** | `limit_req` (leaky bucket) + `limit_conn` | Fixed-window counters |
| **State** | nginx shared memory (per node) | DocumentDB counters (shared across replicas) |
| **Failure mode** | Fails closed (429 when zone full) | Fails open (availability guardrail) |

nginx structurally cannot do the application-level job: at the edge it has not yet authenticated the caller, so it cannot see the user, their groups, or the target server. Conversely, the application layer should not try to absorb a raw volumetric flood, because by the time a request reaches `/validate` it has already consumed nginx worker and auth resources. Each layer does what only it can.

This document covers **only the application-level layer**. The nginx edge limiting is configured in the nginx templates and the `trusted_real_ip_cidrs` Terraform variable.

## High-Level Design

```
                         (nginx auth_request subrequest)
 MCP/A2A client ─ call ─►  nginx  ──► /validate (auth-server, async)
                                        │
                         1. resolve identity + groups + target (existing)
                         2. authorize (existing; fails closed)
                         3. RATE LIMIT (new): every applicable gate must pass
                            ├─ caller gates:  group (per-caller), per window
                            └─ target gates:  (entity_type:name, per window)  [if defined]
                               │
                               ▼
                         RateLimiter ──► RateLimiterBackend (interface)
                               │              ├─ DocumentDBBackend  (v1)
                               │              └─ RedisBackend       (future)
                               ▼
                         allow → 200 (existing)   |   deny → 403 + X-RateLimit-* headers
                                                          │  (nginx rewrites to 429)
                                                          ▼
                                          client sees 429 + Retry-After
```

### Why the throttle leaves `/validate` as a 403, not a 429

`/validate` is only ever reached as nginx's `auth_request` subrequest, never directly by a client. nginx's `auth_request` module forwards **only** 401 and 403 from the subrequest to the parent location; any other status (including 429) is turned into a **500** at the parent ("auth request unexpected status: 429"). So a throttle is signalled as a **403** carrying the `X-RateLimit-*` headers plus an `X-RateLimit-Throttled: 1` marker. The data-plane location blocks capture those headers with `auth_request_set $rl_*`, and the shared `@forbidden_error` named location rewrites the response into a real **429 + Retry-After** when the marker is set (a genuine authorization 403, with the marker absent, falls through to the plain forbidden response). The `$rl_*` variables are declared at http scope via a `map` with an empty default, so control-plane and registry-only-mode locations that never capture them still resolve the references. See `docker/nginx_rev_proxy_*.conf` and `registry/core/nginx_service.py`.

Enforcement lives in the auth-server `/validate` endpoint, **after** authorization and immediately before the success response is built. It is gated by `RATE_LIMITING_ENABLED` (default `false`), so an existing deployment sees no behavior change until an operator opts in.

Three design commitments shape everything else:

- **No new required infrastructure.** Counters live in the DocumentDB/MongoDB the gateway already runs. A `RateLimiterBackend` interface leaves room for a Redis backend later without touching the enforcement logic.
- **Correct when horizontally scaled.** The auth-server and registry can run as multiple replicas; the counters are shared state with atomic increments, so N replicas enforce a single limit, not N times the limit.
- **Availability first.** Rate limiting sits on the critical path of every call. If the counter store is unreachable, the default is to allow (fail open) and log loudly, rather than convert a limiter blip into a full gateway outage. Authorization continues to fail closed independently.

## The Two Axes and Windows

A limit applies to one of two **axes**:

- **Caller axis (Limit A):** a caller (a user or agent), across *all* targets, may not exceed N requests per window. Limits target a **group** (name = group). A group carries **two separate numbers** — `user_max_requests` (applied to human callers) and `agent_max_requests` (applied to agent/M2M callers); at least one is required. The caller type is derived from the token: a genuine machine (`client_credentials`) token ⇒ agent, otherwise user. The distinction is made from the token's own `client_id` **claim** (present only on `client_credentials` tokens), NOT the `azp`-derived `client_id` that every OIDC token carries — otherwise a human browser/password-grant token (whose `azp` is the web OAuth client) would be misclassified as an agent. The matching number is used; a group that does not set that type's number does not gate that caller. Enforced **per caller** (each caller gets their own quota). If a caller is in several groups with limits at the same window, the **most restrictive** wins.

  **Group resolution is decoupled from the token's authz groups.** No IdP emits rate-limit groups, and reusing the token's `groups` claim would be wrong (adding a rate-limit group could change the caller's scopes, since authz groups map to scopes). Instead, a caller's rate-limit groups come **solely** from a dedicated `rate_limit_memberships` collection, keyed by the caller's **username** and/or **client_id** from the validated token. A **specific user or agent** is rate-limited by adding a membership mapping it to a rate-limited group. The token's `groups` claim is never consulted by the limiter.

  Two lockout safeguards apply to the caller axis:
  - **Scope: data-plane only.** Caller limits are enforced only on MCP/A2A calls (requests that classify to a target). Control-plane `/api/*` traffic (the dashboard, login post-steps, config reads) is **exempt**, so a caller limit can never break the UI or lock an operator out of the registry.
  - **Admin bypass.** A caller with the admin role skips caller gates entirely (target gates still apply). An operator cannot rate-limit themselves out of the tool that manages the limits.
  - **Config-time floors.** `RATE_LIMIT_USER_FLOOR_PER_MIN` (default 20) and `RATE_LIMIT_AGENT_FLOOR_PER_MIN` (default 10) are pure config (no API). On short windows (`<= 60s`), a group definition whose user/agent number is below its floor is **rejected at config time**. Longer (hourly/daily volume) windows are exempt so legitimate low volume caps still work.
- **Target axis (Limit B):** a target *entity*, across *all* callers combined, may not exceed `max_requests` per window. Only enforced for targets that define a limit. This protects a weak backend from combined load, and is **not** bypassed by admin.

The target axis is **entity-type-generic**. v1 enforces two target kinds:

- `mcp_server` — an MCP server (name = the server path, e.g. `mcpgw`)
- `a2a_agent` — an A2A agent (name = the agent path, e.g. `/booking-agent`)

The model also reserves `mcp_tool` and `a2a_skill` (name = `<parent>:<leaf>`) for a later phase; those require reading the JSON-RPC payload and are not enforced in v1 (the admin API rejects them with a clear message so an operator never gets a silently inert limit).

- **Target-group (`server_group`) — per-member uniform, a convenience label.** A single `target`-axis definition can name a **set of servers** instead of one, via an on-definition `members: list[str]` (server paths). The semantics are **per-member uniform**: each server in the group gets its **own independent `max_requests`/window bucket** — this is *not* a shared or pooled limit across the group. It exists so an admin writes one definition (and maintains one membership list) instead of one definition per server, and can add or remove servers later with no new definitions. Because the counter subject at enforcement time is always the *live request's* server, a `server_group` limit for server `foo` keys as `tgt:server_group:foo:<window>` — a distinct bucket per member, and distinct from any individual `target:mcp_server:foo` limit. A server that is both in a group **and** has its own `mcp_server` target limit therefore gets **two independent buckets, both enforced** (all-must-pass). Overlapping groups stack the same way (one bucket per group). `server_group` is servers-only in this version (agents are a later phase); a member need not be a currently-registered server (an unregistered path simply never produces a gate). Create it with `--axis target --entity-type server_group --members "srvA,srvB"`. It is not admin-bypassable (like every target limit).

- **Caller-target axis (Limit C, issue #1504):** a caller gets its own **independent quota per target**. Neither the caller axis (which pools a caller's traffic across all targets) nor the target axis (which pools all callers against one target) can express "caller A may call server X only N times/min, with a *separate* N/min against server Y, and caller B's usage of X does not consume A's." A `caller_target` definition is a **group** carrying per-caller-type limits exactly like the caller axis; the difference is only the counter subject, which is the composite `<identity>|<target_entity_type>:<target_name>`. It is data-plane only, admin-bypassed (like the caller axis), and subject to the same config-time floor. Create it with `--axis caller_target`.

- **Quarantine (kill switch, issue #1504):** an absolute, immediate block of all data-plane traffic from a caller or to a target — not a rate. Modeled as **two reserved, auto-seeded rate-limit groups**, `quarantine-callers` and `quarantine-targets`, created empty at first startup so an operator never has to create anything: they just **move a subject into the group** (`rate-limit-quarantine-add --subject-type <user|client|server|agent> --subject <s>`). A quarantined subject is denied as an **early return before any counter work**, and — critically — as a **plain 403 without the `X-RateLimit-Throttled` marker**, so nginx returns a plain 403 rather than rewriting to a 429 (quarantine is an access decision, not a throttle). A quarantined **caller** is admin-bypassable (no self-lockout); a quarantined **target** is not (out of rotation for everyone). Disabling a reserved group's sentinel is the **global off-switch**. The reserved groups cannot be deleted or shadowed by a rate definition. Quarantine follows the limiter's fail-open policy by default (a memberships-store error allows), with an opt-in `RATE_LIMIT_QUARANTINE_FAIL_CLOSED` to deny instead; it is best-effort, **not breach containment** — pair it with IdP credential revocation.

**Windows and volume limits.** A window is any length from one second up to a full day (`window_seconds` ≤ 86400). A per-day volume cap ("no more than 5000 calls/day") is therefore the *same mechanism* as a per-second burst cap, just a longer window. A single subject may hold **several limits at different windows at once** — e.g. `100/min` *and* `5000/day`. Each window is enforced as its own independent gate, and all applicable gates must pass. Most-restrictive resolution applies only *within* the same window.

## Supported Use-Cases

These are the traffic-management use-cases the rate limiter supports today. Each maps to one primitive (axis + entity type); pick the row that matches the intent, then configure as shown.

| Use case (intent) | Primitive | How to configure | Bucket semantics |
|-------------------|-----------|------------------|------------------|
| Cap one caller's total rate across **all** servers (contain a runaway/looping agent) | `caller` group | Define a `caller` group limit, add the user/client as a membership | One bucket per caller (all targets pooled) |
| Each member of a group gets its own quota **per server** (fairness / don't punish healthy fan-out) | `caller_target` group | Define a `caller_target` group limit, add members | One bucket per (caller, server) pair |
| Protect **one** server from combined load (capacity protection) | `target` / `mcp_server` | Define a `target` limit naming the server | One shared bucket for that server, across all callers |
| Protect **one** A2A agent from combined load | `target` / `a2a_agent` | Define a `target` limit naming the agent path | One shared bucket for that agent, across all callers |
| Apply the **same** per-server cap to a **set** of servers, each individually (server tiers, e.g. "fragile backends") — **NEW** | `target` / `server_group` | Define a `target` `server_group` with `--members`; each member limited to `max_requests` individually | One independent bucket **per member server** (not pooled) |
| Instantly block all traffic from a caller or to a target (incident response kill switch) | `quarantine` | `rate-limit-quarantine-add` the subject into the reserved group | No rate; hard 403 drop until removed |
| Layer a burst cap and a volume cap on the same subject | any axis, two windows | Define two definitions at different `--window-seconds` (e.g. 60 and 86400) | Independent gate per window; all must pass |

**Deliberately NOT supported** (by design, to avoid config explosion and confusing semantics):

- **Differentiated limits per named server for the same caller** ("caller A gets N1 to server X but N2 to server Y"). Use per-server `target` limits, or a `caller_target` group (uniform N per server). Tiering by server class is expressed with `server_group`, not per-pair rules.
- **A single shared/pooled budget across a group of servers or callers** ("the whole team gets 100/min combined to the finance servers"). Every group primitive here is per-member; there is no cross-member shared counter. A pooled bucket would be a separate future axis.

## Data Model

Two collections, both namespaced (`_<documentdb_namespace>`).

### Definitions: `mcp_rate_limits`

Each document is one `RateLimitDefinition`:

```python
class RateLimitDefinition(BaseModel):
    axis: str            # "caller" | "target" | "caller_target" | "quarantine"
    entity_type: str     # group axes: "group"; target: "mcp_server" | "a2a_agent" | "mcp_tool" | "a2a_skill" | "server_group"
    name: str            # group name, server path, or agent path (reserved for quarantine sentinels)
    max_requests: int    # target axis, >= 1 (server_group: applied to EACH member individually)
    user_max_requests: int   # group axes (caller / caller_target)
    agent_max_requests: int  # group axes (caller / caller_target)
    window_seconds: int  # 1 .. 86400
    members: list[str]   # server_group target entity ONLY: the member server paths
    scope: str           # quarantine axis only: "caller" | "target"
    fail_closed: bool    # deny on backend error (security-critical only); default false
    enabled: bool        # toggle without deleting; default true
```

`members` is present only on a `server_group` target definition (absent/`None` on every other definition, which is what keeps pre-existing documents behaving identically). It is **not** part of the `_id`, so editing the member list replaces the same document — adding or removing a server never changes the definition's identity.

The `caller_target` axis reuses the group shape (`user_max_requests` / `agent_max_requests`); only its counter subject differs (the `<identity>|<entity_type>:<name>` composite). The `quarantine` axis is a **sentinel**: it carries no rate, only a `scope`, and its power is its reserved `name` (`quarantine-callers` / `quarantine-targets`), recognized by the limiter as a kill switch. Quarantine membership reuses the existing `rate_limit_memberships` `groups[]` (a caller adds `quarantine-callers`; a `server:`/`agent:` subject joins `quarantine-targets`).

The document `_id` is `"<axis>:<entity_type>:<name>:<window_seconds>"` — for example `caller:group:developers:60` or `target:a2a_agent:booking-agent:60`. Putting the window in the `_id` is what lets a subject carry a burst limit and a daily limit as two separate documents.

`axis` and `entity_type` are validated against per-axis allowlists; an unknown combination is rejected (fail closed).

### Counters: `rate_limit_counters`

Ephemeral, TTL-expiring documents. The `_id` encodes axis abbreviation, entity type, subject, window length, and the integer window index:

```
_id = "clr:group:alice@example.com:60:474320"     (a caller, 60s window)
      "tgt:a2a_agent:booking-agent:86400:329"      (a target, daily window)
      "tgt:server_group:mcpgw:60:474320"           (a server_group member; one bucket PER member server)
count        = <int>          # atomically incremented
window_start = <datetime>     # start of the fixed window
expire_at    = <datetime>     # window_start + 2*window; TTL index target
```

Because the window index is part of the `_id`, correctness never depends on precise TTL expiry — an old window's document is simply never touched again, and the periodic TTL sweep reaps it. A TTL index on `expire_at` (`expireAfterSeconds=0`) does the cleanup.

## The Enforcement Path

On each `/validate` call, when `RATE_LIMITING_ENABLED`:

1. **Identity** is taken from the validated token only — the `client_id` **claim** (present only on a genuine M2M/`client_credentials` token) ⇒ agent keyed on that client; otherwise `username` ⇒ user. The `azp`-derived `client_id` that every OIDC token carries is deliberately NOT used for this (it would misclassify humans as agents and bucket all web users under one client). It is **never** read from a client-supplied header. This is a hard security property: keying on a client-controlled value would let a caller split load across fabricated identities.
2. **Target classification** maps the request to `(entity_type, name)`: an `/agent/...` path → `("a2a_agent", agent_path)`; otherwise a server path → `("mcp_server", server_name)`; neither → skip the target axis.
3. **Gates are built.** Caller definitions (all windows for the caller's groups) are fetched in one cached query and grouped by window; within each window the most-restrictive wins. Target definitions (all windows for the classified entity) are fetched in one cached query — for an `mcp_server` target this also matches any enabled `server_group` definition whose `members` list contains this server, so a server picks up its group limits automatically (each keyed per this individual server). Each `(axis, window)` becomes a gate.
4. **Gates are enforced sequentially, tightest-window-first**, stopping at the first denial.
5. On denial, the auth-server raises `HTTPException(403)` with `X-RateLimit-Throttled: 1`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `Retry-After`, and `Connection: close`. The 403 (not 429) is deliberate: nginx `auth_request` forwards only 401/403 from the subrequest and would turn a 429 into a 500. The `@forbidden_error` nginx location detects the `X-RateLimit-Throttled` marker and rewrites the response into a real **429 + Retry-After** for the client (see "Why the throttle leaves `/validate` as a 403" above).

### Why sequential, tightest-window-first

This ordering is the crux of a subtle correctness property: **a request rejected by one gate must not consume any other gate's quota.** Consider a caller with `5/min` and `20/day`. Under a naive concurrent design (increment all counters, then check), a burst of rejected traffic would still advance the daily counter and could exhaust the daily budget in minutes — a self-inflicted lockout driven entirely by *rejected* requests.

Two mechanisms together prevent this:

- **The backend increments only when under the limit** (`incr_if_allowed`, described below). A denied gate performs no increment.
- **Gates run tightest-window-first and short-circuit.** The burst gate is checked before the daily gate; when it denies, the daily gate is never touched.

The cost is that the *allowed* path makes one DB round trip per configured gate (typically two to four) instead of one. That is a deliberate trade for correct cross-window behavior, and it is bounded (see [Latency](#latency)).

## Correctness Across Replicas

The counter increment is a single atomic conditional update:

```python
find_one_and_update(
    {"_id": doc_id, "count": {"$lt": max_requests}},   # only if under the limit
    {"$inc": {"count": 1}, "$setOnInsert": {...}},
    upsert=True,
    return_document=AFTER,
)
```

For a counter that is below its limit, this atomically increments and returns the new value. For a counter already at its limit, the `{"count": {"$lt": max_requests}}` predicate does not match, so with `upsert=True` MongoDB tries to *insert* a new document with the same `_id` — which the unique `_id` index rejects with `DuplicateKeyError`. The backend catches that and reports "at limit, not incremented."

Two replicas racing on the boundary both go through the same atomic compare-and-increment, so the aggregate can never exceed `max_requests`, regardless of replica count. This is the entire reason for a shared counter store, and it also means the stored count can never run above the limit.

## Failure Modes

- **Backend error or timeout → fail open by default.** Each counter op runs under a hard timeout (`RATE_LIMIT_BACKEND_TIMEOUT_MS`, default 250 ms). On error or timeout, the limiter logs a warning, emits an error metric, and **allows** the request — unless the specific limit is marked `fail_closed`, or the global `RATE_LIMIT_FAIL_OPEN` is `false`. A slow store therefore fails *fast* into "not limiting," never into "every authenticated request hangs."
- **A malformed definition is skipped**, logged, and never breaks the auth path.
- **An unexpected error in the enforcement wrapper is swallowed** — rate limiting must never turn into a 500 on `/validate`.
- **Feature disabled → complete no-op.** No DB access, no import cost.

The fail-open default is a deliberate, argued exception to the project's "admission fails closed" principle: rate limiting is an *availability guardrail*, not an authorization admission gate. Authorization continues to fail closed on its own.

## Latency

`/validate` runs on almost every authenticated request, so added latency matters.

- **Definitions cost zero DB reads in steady state** — both caller and target reads are served from an in-process cache (`RATE_LIMIT_DEFINITIONS_CACHE_TTL_SECONDS`, default 30 s), and `/validate` already holds a warm DocumentDB connection pool.
- **The only per-call DB work is the counter increments** — one indexed primary-key op per configured gate (typically two to four). On a co-located DocumentDB / same-region MongoDB with write concern `w:1`, each is low single-digit milliseconds.
- **A slow store cannot hang the gateway** — the 250 ms per-op timeout bounds the worst case and fails open.
- **Escape hatch:** if counter latency is ever unacceptable, the `RateLimiterBackend` interface allows a Redis/ElastiCache backend (sub-millisecond) with no change to the enforcement logic.

The `mcpgw_rate_limit_backend_duration_ms` histogram measures the real per-op latency; watch its p99 against the timeout to decide when to tune windows or move to Redis.

## Observability

All labels are bounded (no per-user/per-server *name* labels, which would be unbounded — those live in the WARNING log and the status endpoint).

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `mcpgw_rate_limit_throttled_total` | Counter | `axis` (`clr` / `tgt` / `ctg`), `entity_type`, `window_seconds` | Times an axis entity was throttled (a gate denied). Split by `axis` to see caller vs target vs **caller_target** throttles. "How many times did a caller / a target / a caller-per-target quota get rate-limited?" |
| `mcpgw_rate_limit_quarantine_denied_total` | Counter | `scope` (`caller` / `target`), `entity_type` | Requests dropped because a caller or target is **quarantined** (kill switch). "How many calls were blocked by quarantine, split by whether the caller or the target was quarantined?" |
| `mcpgw_rate_limit_quarantine_members` | ObservableGauge | `group` (`quarantine-callers` / `quarantine-targets`) | Current member count of each kill-switch group, sampled each export cycle (correct across replicas; not on the request path). "How many callers / targets are currently quarantined?" |
| `mcpgw_rate_limit_checks_total` | Counter | `axis`, `entity_type`, `outcome` | Total gate checks (allow/deny) — the denominator for a throttle rate. `ctg` appears in `axis`. |
| `mcpgw_rate_limit_errors_total` | Counter | `axis` (incl. `qtn` for a quarantine-read error) | Backend errors (fail-open/closed events). |
| `mcpgw_rate_limit_backend_duration_ms` | Histogram | `backend`, `op` | Per-op latency of the counter-store round trip. |

The new caller_target axis needs **no new throttle metric** — it reuses `mcpgw_rate_limit_throttled_total` with `axis="ctg"`. Quarantine adds two dedicated series (`_quarantine_denied_total`, `_quarantine_members`). All labels stay bounded (no subject-name label); per-subject attribution stays in the WARNING app log and the `GET /api/rate-limit-quarantine` list.

Recommended alerts: sustained `mcpgw_rate_limit_errors_total` (limiter effectively off), `mcpgw_rate_limit_backend_duration_ms` p99 approaching the timeout, and any nonzero `mcpgw_rate_limit_quarantine_denied_total` if quarantine is meant to be a rare, deliberate action.

## Configuration

All parameters are off-by-default-safe and wired across Docker Compose, Terraform, and Helm. See the [unified parameter reference](../unified-parameter-reference.md#group-32--rate-limiting) for the per-surface names.

| Env var | Default | Read by | Purpose |
|---------|---------|---------|---------|
| `RATE_LIMITING_ENABLED` | `false` | both | Master switch. |
| `RATE_LIMIT_BACKEND` | `documentdb` | both | Counter backend (only `documentdb` in v1). |
| `RATE_LIMIT_FAIL_OPEN` | `true` | both | Global fail-open on backend error. |
| `RATE_LIMIT_QUARANTINE_FAIL_CLOSED` | `false` | both | Deny on a quarantine-membership read error (default fail-open). Best-effort containment, not breach containment. |
| `RATE_LIMIT_DEFINITIONS_CACHE_TTL_SECONDS` | `30` | both | In-process definitions cache TTL. |
| `RATE_LIMIT_BACKEND_TIMEOUT_MS` | `250` | both | Hard per-op counter timeout. |
| `RATE_LIMIT_USER_FLOOR_PER_MIN` | `20` | registry | Config-time floor: a caller group's `user_max_requests` on a window `<= 60s` must be `>=` this, else the definition is rejected. Registry-only (validates definitions). |
| `RATE_LIMIT_AGENT_FLOOR_PER_MIN` | `10` | registry | Same, for a group's `agent_max_requests`. Registry-only. |

The first five are read by **both** the `registry` and `auth-server` services (keep them in agreement); the two floors are read by the **registry only** (it validates group definitions at config time). The auth-server enforces limits; the registry validates and stores them.

**Provider note (Cognito agents):** agent/M2M rate limiting requires the caller's machine token to be accepted at `/validate`. On **Amazon Cognito**, a `client_credentials` app-client id must be allowlisted via `COGNITO_M2M_CLIENT_IDS` (or `*`), and the client mapped to registry groups via the M2M-clients store — see [docs/idp/cognito.md → Machine-to-machine (M2M / agent) clients](../idp/cognito.md#machine-to-machine-m2m--agent-clients). Keycloak/Okta/Auth0/PingFederate use their own `*_m2m_client_id` settings.

## Testing

Hands-on, copy-pasteable end-to-end guides (create principals, groups, and memberships, then watch limits trip):

- [End-to-End Test Guide](../e2e_testing.md) — the happy-path walkthrough: create the test principals, group (caller) limits, response headers, metrics + logs, the floor safeguards, per-agent (M2M) limits, and target limits.
- [Advanced / Edge-Case Test Guide](../e2e_testing_advanced.md) — failure modes and correctness invariants: concurrency/atomicity, window reset + `Retry-After`, deny-does-not-consume, caller-type classification, membership-vs-authz, admin bypass, data-plane-only scope, enable/disable, and fail-open/fail-closed.

## Managing Limits (Admin API and CLI)

Limit **definitions** are managed at runtime, not through env vars. All endpoints are admin-only.

- `GET /api/rate-limits` — list all definitions.
- `GET /api/rate-limits/{id}` — read a single definition (404 if absent).
- `PUT /api/rate-limits/{id}` — create/update. The `_id` is derived from the body; the URL id must match exactly (a colon-bearing tool/skill name is never parsed out of the URL).
- `POST /api/rate-limits-enabled/{id}?enabled=true|false` — enable/disable in place without re-specifying the definition (a distinct prefix so the greedy `{id:path}` doesn't swallow the action).
- `DELETE /api/rate-limits/{id}` — delete.
- `GET /api/rate-limits-status?identity=&entity_type=&name=` — introspect matching definitions.

Memberships (which caller belongs to which rate-limit group; id = `<subject_type>:<subject>`):

- `GET /api/rate-limit-memberships` — list all memberships.
- `GET /api/rate-limit-memberships/{id}` — read one (404 if absent).
- `PUT /api/rate-limit-memberships/{id}` — create/update (`_id` derived from body; URL must match).
- `DELETE /api/rate-limit-memberships/{id}` — delete.

A **read-only** view of the current definitions also appears on the **Settings → System Config** page (group "Rate Limit Definitions (read-only)"), fed live from `/api/config/full`. The UI is view-only by design; all mutations go through the API/CLI above.

CLI equivalents:

```bash
# 100 requests/minute for the "developers" group, plus a 5000/day volume cap.
# Caller/group axes take per-caller-type limits (--user-max-requests / --agent-max-requests,
# at least one), NOT --max-requests (that is the target axis).
uv run python registry_management.py rate-limit-set \
  --axis caller --entity-type group --name developers \
  --user-max-requests 100 --agent-max-requests 100 --window-seconds 60
uv run python registry_management.py rate-limit-set \
  --axis caller --entity-type group --name developers \
  --user-max-requests 5000 --agent-max-requests 5000 --window-seconds 86400

# 500 requests/minute aggregate to the "mcpgw" MCP server, across all callers
uv run python registry_management.py rate-limit-set \
  --axis target --entity-type mcp_server --name mcpgw --max-requests 500 --window-seconds 60

# 100 requests/minute to EACH server in the "fragile-backends" group, individually
# (per-member uniform: airegistry-tools gets its own 100/min bucket, aws-kb gets its
# own 100/min bucket -- not a shared pool). Add/remove servers by re-running with an
# updated --members list; no new definitions needed.
uv run python registry_management.py rate-limit-set \
  --axis target --entity-type server_group --name fragile-backends \
  --max-requests 100 --window-seconds 60 --members "airegistry-tools,aws-kb"

uv run python registry_management.py rate-limit-list
uv run python registry_management.py rate-limit-get --id caller:group:developers:60
uv run python registry_management.py rate-limit-disable --id caller:group:developers:60
uv run python registry_management.py rate-limit-enable --id caller:group:developers:60
uv run python registry_management.py rate-limit-delete --id caller:group:developers:60
```

To rate-limit a **specific user or agent**, define a group limit and add a
**rate-limit membership** mapping that user/agent to the group. Memberships live
in the `rate_limit_memberships` collection (separate from authz groups) and are
managed with the `rate-limit-member-*` commands / `/api/rate-limit-memberships`:

```bash
# 1. Define the group's limit (caller/group axis uses per-caller-type limits)
uv run python registry_management.py rate-limit-set \
  --axis caller --entity-type group --name rate-limited-testers \
  --user-max-requests 3 --agent-max-requests 3 --window-seconds 60

# 2a. Map a USER (by username) to that rate-limit group
uv run python registry_management.py rate-limit-member-set \
  --subject-type user --subject alice --groups rate-limited-testers

# 2b. ...or map an AGENT (by client_id) to that group
uv run python registry_management.py rate-limit-member-set \
  --subject-type client --subject my-agent-client-id --groups rate-limited-testers

# Inspect / remove memberships
uv run python registry_management.py rate-limit-member-list
uv run python registry_management.py rate-limit-member-delete --id user:alice
```

The membership is keyed on the username / client_id the auth-server reads from the
validated token, so it takes effect on the caller's next request. No IdP change or
re-authentication is required (the token's `groups` claim is not involved).

## Where the Code Lives

| Path | Responsibility |
|------|----------------|
| `registry/rate_limiting/models.py` | `RateLimitDefinition`, `RateLimitDecision`, allowlists |
| `registry/rate_limiting/backend.py` | `RateLimiterBackend` ABC + `IncrResult` (the `incr_if_allowed` contract) |
| `registry/rate_limiting/documentdb_backend.py` | Fixed-window conditional-`$inc` counters, TTL index, latency histogram |
| `registry/rate_limiting/definitions_repository.py` | Cached CRUD over `mcp_rate_limits` |
| `registry/rate_limiting/memberships_repository.py` | Cached CRUD over `rate_limit_memberships` + `get_groups_for(username, client_id)` |
| `registry/rate_limiting/limiter.py` | `RateLimiter.check()` — gate building, sequential enforcement |
| `registry/api/rate_limit_routes.py` | Admin CRUD + status API |
| `auth_server/rate_limiting_config.py` | Env-var constants + the `RateLimiter` singleton |
| `auth_server/server.py` | `_classify_rate_limit_target` + `_enforce_rate_limit`, called in `/validate` |
| `registry/observability/meters.py` | The four metrics |

## Extending It

- **A new target kind** (e.g. a database, a queue): add it to `TARGET_ENTITY_TYPES`, teach `_classify_rate_limit_target` to recognize its request shape, done. No schema change.
- **Per-tool / per-skill limits:** already modeled (`mcp_tool`, `a2a_skill`); wiring is a later phase that adds a JSON-RPC-payload classifier. Before enabling, close the composite-`name` delimiter concern (validate that a raw name component contains no `:`, or store the counter key as separate indexed fields instead of a concatenated `_id`).
- **A Redis backend:** implement `RateLimiterBackend` against Redis (an atomic Lua script for the conditional increment), select it via `RATE_LIMIT_BACKEND`. The enforcement logic does not change.
- **Sliding window / token bucket:** a different algorithm is a new backend or a new limiter strategy; the fixed-window boundary-burst limitation is documented and acceptable for v1.
