# Theory of the System

This document is the *theory of the program*: the mental model of **why the system is shaped the
way it is**, held by the people who built it. It is not a feature list and not a changelog. It is
the causal story — the beliefs the system acts on, the invariants it holds, and the consequences
if those invariants are violated — so a new contributor (or a staff engineer doing diligence) can
rebuild the builders' model without archaeology through the commit history.

Every invariant below was checked against the code and the design docs before it was written
here. Where the code contradicts a comfortable summary, the contradiction is stated rather than
smoothed over. This document *indexes and narrates* the design docs under
[`docs/design/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docs/design/);
it does not replace them.

---

## 1. What problem this system believes it is solving

The belief: **AI assets proliferate faster than any team can govern them.** MCP servers, agents,
skills, and one-off tools multiply across an organization; each arrives with its own credentials,
its own endpoint, its own discovery story, and no shared inventory or audit trail. Left alone
this produces credential sprawl in dotfiles, no way to answer "what can this user reach," and no
offboarding story.

The system's response is a single, governed **control plane** for every AI asset type, paired
with an optional **gateway** data plane that enforces access at one authenticated ingress. The
consequence the design keeps returning to: *discovery, authorization, and audit are centralized;
data movement is not centralized unless it must be.* Most of the invariants below are corollaries
of that one sentence.

The project grew from an MCP-server gateway into a general-purpose AI-asset registry. That
evolution matters to the theory: agents, skills, and admin-defined custom entities were added as
*new entity types on the same registration/search/access-control/audit spine*, not as separate
subsystems. The single-spine choice (invariant 1) is why that growth stayed cheap.

---

## 2. Core invariants

Each invariant states the rule, why it exists, and what breaks if it is violated. Verdicts
reflect what the code on `main` actually does as of this writing.

### 2.1 One control plane for every AI asset type

Servers, agents, skills, and custom entities share **one** schema-driven registration path, one
search index, one access-control model, and one audit taxonomy. Adding an entity type means
adding a schema and a handful of route handlers, not a new subsystem.

- **Why:** the product's growth thesis. If each asset type had its own registration, search, and
  auth, every new type would multiply the governance surface and the places a check could be
  forgotten.
- **In code:** the unified `semantic_search` endpoint returns servers, agents, skills, and
  virtual servers from one path ([`registry/api/search_routes.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/api/search_routes.py)),
  filtered by a shared `entity_type` field in the search repository. The registration-gate check
  is invoked uniformly from `server_routes.py`, `agent_routes.py`, and `skill_routes.py`. Custom
  entity types are schema-driven at runtime
  ([`registry/api/custom_type_routes.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/api/custom_type_routes.py)) and mirror
  the skill save-then-index pattern. The audit model enumerates one resource-type taxonomy
  ([`registry/audit/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/registry/audit/)).
- **If violated:** a new asset type that bypasses the shared spine (its own search, its own
  registration) reintroduces exactly the sprawl the system exists to prevent, and creates a place
  where an access check can be silently missing.

### 2.2 The gateway is a generic reverse proxy; the registry is the control plane

The two halves of the system are a clean plane split. The **gateway is a generic HTTP reverse
proxy** (built on nginx): it does routing, TLS termination, and access enforcement at one
authenticated ingress, and it is fundamentally protocol-agnostic. The **registry is the control
plane**: it holds the inventory, the access model, and the audit trail, and it decides *what the
gateway is allowed to route to*. The gateway moves bytes; the registry decides.

- **Why it is generic:** the proxy started MCP-only — that was the only protocol in scope when the
  project began, and A2A did not yet exist — but nothing about the reverse-proxy design is
  MCP-specific. It routes HTTP. That is why extending it to be an **A2A gateway comes naturally**
  (an A2A endpoint is just another HTTP backend behind the same auth-gated location block), and why
  the same mechanism can front **skills served over HTTP** and, in the future, **plain REST APIs**.
  The reverse-proxy-vs-application-gateway decision was made precisely to preserve this protocol
  independence ([`docs/design/architectural-decision-reverse-proxy-vs-application-layer-gateway.md`](architectural-decision-reverse-proxy-vs-application-layer-gateway.md)):
  a nginx reverse proxy routes any HTTP endpoint, whereas an application-layer gateway would have
  hardcoded MCP semantics.
- **The plane split as it stands today:** the gateway (nginx) and the registry (FastAPI) are
  distinct concerns that happen to be deployed together. The registry generates the nginx config
  from its inventory; nginx enforces it on the data path. This is why the same auth-gated
  location-block pattern applies uniformly to whatever protocol sits behind it.
- **Direction of travel (not yet on `main`):** the intended evolution is to **separate the two into
  independently-scaling containers** — the gateway as its own container that scales with data-plane
  traffic, and the registry as its own FastAPI container that scales with control-plane load
  (registration, search, UI). They scale on different axes: a burst of proxied tool/agent traffic
  should not require scaling the registry, and a large registry/search workload should not require
  scaling the proxy. Today they are co-deployed; treat the clean plane split above as the design
  intent that makes that separation a packaging change rather than a redesign.
- **If violated:** baking protocol-specific logic into the proxy (making it "the MCP gateway"
  rather than "a reverse proxy that currently fronts MCP") forecloses the A2A/skills/REST
  generalization and re-couples the data plane to one protocol. Coupling the registry and gateway
  so they cannot scale independently forces you to over-provision one to serve the other's load.

### 2.3 The registry is a control plane, never the data path, for agent-to-agent traffic

For A2A, the registry does discovery and auth validation only. Once two agents find each other,
they communicate **peer-to-peer**; their traffic never flows through the registry or gateway.

- **Why:** a central agent-traffic bottleneck would cap throughput and add latency to every
  agent interaction, for no governance gain that discovery-time authorization does not already
  provide.
- **In code:** [`docs/design/a2a-protocol-integration.md`](a2a-protocol-integration.md) is explicit
  ("The registry itself is NOT involved in agent-to-agent communication … they communicate
  peer-to-peer with no registry intermediation"), and there is no A2A traffic-proxying code on
  `main`.
- **Nuance (verified):** an **opt-in** A2A reverse-proxy capability is in flight (adds an
  `invoke_agent` scope, nginx/Lua agent-card rewriting, and a feature flag). It is **not merged to
  `main`** — it lives on a feature branch. So the peer-to-peer default holds today; the proxy, if
  merged, is explicitly opt-in and does not change the default. Treat "A2A never flows through the
  gateway" as the current-main invariant and the proxy as a bounded, flagged exception.
- **If violated:** making the registry a mandatory A2A hop turns a discovery service into a
  latency-critical data plane with a single point of failure.

### 2.4 Gateway vs registry-only is an explicit, configuration-level distinction

Whether the system runs as a full gateway (nginx integration, reverse-proxy enforcement) or as a
registry only (discovery/governance, no data-plane) is a deliberate config choice, not an
emergent behavior.

- **Precision (verified — corrects a common mislabel):** the master switch is **`DEPLOYMENT_MODE`**
  (`WITH_GATEWAY` / `REGISTRY_ONLY`) in [`registry/core/config.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/core/config.py),
  which gates nginx updates and even the UI title. This is *distinct* from **`REGISTRY_MODE`**
  (`FULL` / `SKILLS_ONLY` / `MCP_SERVERS_ONLY` / `AGENTS_ONLY`), which controls *which entity-type
  features/tabs* are enabled (enforced by [`registry/middleware/mode_filter.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/middleware/mode_filter.py)).
  The two are orthogonal and interact (a `WITH_GATEWAY + SKILLS_ONLY` combination auto-corrects to
  `REGISTRY_ONLY`). If you are reasoning about "is there a data plane," look at `DEPLOYMENT_MODE`,
  not `REGISTRY_MODE`.
- **Why:** the registry is useful on its own (discovery for coding assistants and agents) without
  taking on the operational weight of a proxy. Making the distinction explicit means neither mode
  is a degraded version of the other.
- **If violated:** coupling the two axes, or inferring the mode implicitly, produces deployments
  that silently enable or disable a data plane the operator did not choose.

### 2.5 Configuration parity across the three deployment surfaces is an invariant, not a courtesy

Every configuration parameter is expressed on all three surfaces — Docker Compose (`.env`), ECS
Terraform (`.tfvars`), and EKS Helm (`values.yaml`) — and a new parameter is not "done" until it
is wired through all three.

- **Why:** operators pick a surface and must not discover that a feature is unconfigurable on
  theirs. Reviewers need one place to confirm a parameter was propagated.
- **In code:** [`docs/unified-parameter-reference.md`](../unified-parameter-reference.md) is the
  living cross-surface index ("configured identically across three deployment surfaces"), and the
  parity is mechanically defended by the `reserved-env-names.txt` files under each chart plus the
  matched reserved-list / `extraEnv` test invariants described in `CLAUDE.md`.
- **If violated:** a parameter wired on one surface only becomes a silent "works on my deployment"
  bug that surfaces at production cutover.

### 2.6 Admission fails closed; notification fails open

The registration **gate** (external admission control) blocks registration when it cannot get an
allow decision. The registration **webhook** (external notification) is fire-and-forget: a failed
delivery is logged and never blocks the caller.

- **Why:** admission is a security control — if you cannot prove an asset is allowed, you must not
  persist it. Notification is an integration convenience — its failure must never take down the
  registration path it observes.
- **In code:** [`registry/services/registration_gate_service.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/services/registration_gate_service.py)
  ("Blocking registration (fail-closed)"; "Registrations will be blocked until the gate is
  available") vs [`registry/services/webhook_service.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/services/webhook_service.py)
  ("fire-and-forget: failures are logged at WARNING but never propagated"). The gate also strips
  credential fields from the payload it sends outward.
- **If violated:** a fail-open gate is not a gate; a fail-closed webhook couples your registry's
  availability to an external listener's uptime.

### 2.7 Auth is agnostic across a supported set of IdPs — but that set is a closed, hand-written factory

Any of six identity providers — Keycloak, Amazon Cognito, Microsoft Entra ID, Okta, Auth0,
PingFederate — can be selected via `AUTH_PROVIDER`, and the system does not couple to any one of
them. Group-to-scope mapping lives in the database, not in provider-specific code.

- **Precision (verified — corrects an overstatement):** this is **not** generic "point at any OIDC
  issuer" discovery. Each provider is a hand-written class in
  [`auth_server/providers/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/auth_server/providers/) selected by a factory; **adding a new
  provider requires writing a new provider class**
  ([`docs/design/idp-provider-support.md`](idp-provider-support.md): "Create Provider Class / Create
  IAM Manager / Update Factory"). Some providers resolve endpoints via
  `.well-known/openid-configuration` (Keycloak, PingFederate, Okta); others hardcode (Cognito's
  JWKS URL). So: agnostic across the supported six = true; "zero-code OIDC discovery for any
  provider" = false. State it as the former.
- **Why:** enterprises bring their own IdP; coupling to one would make the system a non-starter for
  everyone else. A closed factory (rather than open discovery) is the deliberate trade — more code
  per provider, in exchange for handling each provider's real-world quirks.
- **If violated:** provider-specific assumptions leaking into shared auth code turns "supports your
  IdP" into "supports the one we tested."

### 2.8 MCP spec compliance is a first-class, tracked commitment

The gateway implements the MCP authorization discovery contract: Protected Resource Metadata at
`/.well-known/oauth-protected-resource`, and a `WWW-Authenticate` header on MCP-facing 401s
pointing at that metadata.

- **Why:** MCP clients (coding assistants) rely on this discovery handshake to connect without
  hand-configured client IDs. Tracking the spec is what makes one-command assistant integration
  possible.
- **In code:** [`registry/auth/oauth_metadata.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/auth/oauth_metadata.py) defines
  the PRM path; a dedicated [`registry/middleware/mcp_www_authenticate.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/middleware/mcp_www_authenticate.py)
  attaches the header ("RFC 9728 §5.1"), with matching RFC citations in the nginx config generator.
- **If violated:** dropping the discovery header silently breaks the zero-config assistant
  onboarding that several features depend on.

---

## 3. Boundaries — what this system deliberately does NOT do

Negative space is part of the theory. Each of these is a stated non-goal, not a missing feature.

- **It is not part of A2A data traffic** (§2.3) — discovery and auth only.
- **It does not manage PKI or agent identities.** ANS integration is read-only "bring your own ANS
  ID"; the registry stores, displays, and re-verifies trust metadata but never issues certificates
  ([`docs/design/ans-integration.md`](ans-integration.md)).
- **It is single-tenant, not multi-tenant SaaS.** The cookie/session design explicitly assumes a
  single tenant and documents subdomain risks as acceptable only under that assumption
  ([`docs/design/cookie-security-design.md`](cookie-security-design.md)).
- **The session cookie carries no identity payload** — only an opaque signed `session_id`; groups,
  scopes, and the id_token live server-side ([`docs/design/session-flow-cookie-based.md`](session-flow-cookie-based.md)).
- **Third-party egress tokens never touch the user's machine.** The gateway injects upstream
  credentials from a vault; client auth headers are ingress-only and stripped on egress
  ([`docs/design/egress-auth-design.md`](egress-auth-design.md)).
- **Internal HS256 hop tokens prove key-possession, not origin.** Asymmetric per-hop signing is a
  documented deferral, not an accident ([`docs/design/internal-hop-authentication.md`](internal-hop-authentication.md)).
- **Virtual MCP servers do not stream (no SSE) and do no per-backend load balancing**
  ([`docs/design/virtual-mcp-server.md`](virtual-mcp-server.md)).
- **New IdP support is a code change, not zero-config discovery** (§2.7).
- **It does not replace GitHub Releases.** Detailed per-version notes live in
  [`docs/release-notes/`](../release-notes/1.26.0.md); this repo does not try to be the canonical
  release channel.

---

## 4. How to read the codebase

A half-page map of which directory owns which invariant.

| Directory | Owns | Notable contents |
|---|---|---|
| [`registry/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/registry/) | The control plane: registration, search, access control, audit, health, config, all entity types. | `api/` (route handlers per entity type + `search_routes.py`, `custom_type_routes.py`), `services/` (`registration_gate_service.py`, `webhook_service.py`, `federation/`), `repositories/` (storage-agnostic data access), `auth/` (`oauth_metadata.py`, `internal.py`), `egress_auth/`, `search/` + `embeddings/`, `middleware/` (`mode_filter.py`, `mcp_www_authenticate.py`), `audit/`. |
| [`auth_server/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/auth_server/) | The single `/validate` chokepoint; per-provider IdP classes; session store; group→scope enrichment. | `providers/{keycloak,cognito,entra,okta,auth0,pingfederate}.py`, `factory.py`, `session_store.py` |
| [`frontend/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/frontend/) | React/TypeScript admin UI. | `src/` |
| [`cli/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/cli/) | Command-line client + service management. | |
| [`docker/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docker/) | The gateway data plane: nginx templates + Lua (`virtual_router.lua`, `agent_card_rewrite.lua`). | `lua/`, `nginx_rev_proxy_*.conf` |
| [`charts/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/charts/) | EKS/Helm deployment + reserved-env-name enforcement. | subcharts + `mcp-gateway-registry-stack`, `reserved-env-names.txt` |
| [`terraform/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/terraform/) | ECS Fargate IaC (the ECS surface). | `aws-ecs/modules/mcp-gateway/` |
| [`credentials-provider/`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/credentials-provider/) | Egress credential/token handling + refresh. | |

The **three-way boundary that explains the most code:** `registry/` decides *what is allowed*
(control plane), `auth_server/` *proves who you are and enforces the check* (the `/validate`
chokepoint), and `docker/` nginx + Lua *moves the bytes* (data plane). When a change feels like
it is in the wrong place, it is usually because it crossed one of these boundaries.

---

## 5. Index of design docs

One line each; read the doc for the full rationale.

- [a2a-protocol-integration.md](a2a-protocol-integration.md) — control-plane-only A2A: discover + auth, then peer-to-peer.
- [agent-skills-architecture.md](agent-skills-architecture.md) — skills as URL-referenced `SKILL.md` guidance, distinct from executable servers.
- [agentcore-scanner-design.md](agentcore-scanner-design.md) — decoupled AgentCore scanner/registrar + separate cron token-refresher.
- [ans-integration.md](ans-integration.md) — read-only "bring your own ANS ID"; registry never manages PKI.
- [anthropic-api-implementation.md](anthropic-api-implementation.md) — Anthropic MCP Registry REST API v0.1 compatibility shim.
- [architectural-decision-reverse-proxy-vs-application-layer-gateway.md](architectural-decision-reverse-proxy-vs-application-layer-gateway.md) — why nginx reverse-proxy over an app-layer gateway.
- [authentication-design.md](authentication-design.md) — three identity paths (human OAuth2, API tokens, M2M) resolving to DB-stored scopes.
- [aws-agent-registry-federation.md](aws-agent-registry-federation.md) — read-only pull federation of Bedrock AgentCore registries with per-registry STS AssumeRole.
- [cookie-security-design.md](cookie-security-design.md) — single-tenant, explicitly-configured domain cookies; `secure` gated on `X-Forwarded-Proto`.
- [database-abstraction-layer.md](database-abstraction-layer.md) — repository pattern + factory selecting backend via `STORAGE_BACKEND`.
- [egress-auth-design.md](egress-auth-design.md) — ingress-only client headers; gateway injects vaulted upstream creds keyed by a 4-tuple.
- [federation-architecture.md](federation-architecture.md) — symmetric peer-to-peer pull sync, Fernet-encrypted static tokens, `peer_id`-namespaced read-only items.
- [hybrid-search-architecture.md](hybrid-search-architecture.md) — vector + keyword fused with Reciprocal Rank Fusion (k=60), replacing saturating additive scoring.
- [idp-provider-support.md](idp-provider-support.md) — multi-IdP via a closed factory keyed on `AUTH_PROVIDER`; new provider = new code.
- [internal-hop-authentication.md](internal-hop-authentication.md) — short-lived HS256 per-hop tokens, audience-scoped, fail closed, ignore plaintext `X-User`.
- [server-versioning.md](server-versioning.md) — versions as separate documents; nginx `map` routing; only active version indexed/health-checked.
- [session-flow-cookie-based.md](session-flow-cookie-based.md) — opaque signed `session_id`; payload server-side in Mongo; id_token AES-GCM encrypted.
- [session-flow-jwt-bearer.md](session-flow-jwt-bearer.md) — stateless programmatic access validated only at `/validate`; four token kinds converge on one user-context derivation.
- [storage-architecture-mongodb-documentdb.md](storage-architecture-mongodb-documentdb.md) — MongoDB CE (dev) and DocumentDB (prod) share one repository; sole divergence is vector search.
- [virtual-mcp-server.md](virtual-mcp-server.md) / [virtual-mcp-server-explained.md](virtual-mcp-server-explained.md) — one aggregating endpoint fronting many backends, routed in nginx Lua; no SSE, no per-backend LB.

**Known doc drift to be aware of when citing** (flagged, not fixed here): the scoring description
in `storage-architecture-mongodb-documentdb.md` predates the RRF change in
`hybrid-search-architecture.md`; `idp-provider-support.md` names fewer providers than the code now
implements. Prefer the code and the newer doc when they disagree.

---

## 6. How to change this system without breaking its theory

Before merging a change that touches auth, routing, registration, or configuration, check it
against the invariants above. The [`pr-review`](https://github.com/agentic-community/mcp-gateway-registry/tree/main/.claude/skills/pr-review/) skill encodes the
security-specific version of this list; the theory-level checks are:

1. **New asset type or entity behavior?** It must ride the shared registration/search/access-control/
   audit spine (§2.1), not a parallel one.
2. **New protocol or backend type behind the gateway?** Keep the proxy generic — front it as an
   auth-gated HTTP backend; do not bake protocol-specific logic into the reverse proxy, and do not
   couple the change to co-deployment of gateway and registry (§2.2).
3. **New outbound or agent-traffic path?** Confirm you are not turning the registry into a data
   plane it is designed not to be (§2.3).
4. **New mode-dependent behavior?** Gate it on the correct axis — `DEPLOYMENT_MODE` for
   gateway-vs-registry, `REGISTRY_MODE` for entity-type enablement (§2.4).
5. **New configuration parameter?** Wire it through all three surfaces and update
   [`docs/unified-parameter-reference.md`](../unified-parameter-reference.md) (§2.5).
6. **New external integration?** Decide explicitly whether it is admission (fail closed) or
   notification (fail open) — and implement the matching failure mode (§2.6).
7. **Auth change?** Keep it provider-agnostic across the supported six; do not leak one provider's
   assumptions into shared code (§2.7).
8. **MCP-facing endpoint?** Preserve the PRM / `WWW-Authenticate` discovery contract (§2.8).

A change that violates an invariant is not automatically wrong — but it must be a *deliberate*
change to the theory, argued in the PR description, not an accidental erosion of it.
