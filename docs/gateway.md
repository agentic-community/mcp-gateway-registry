# The Gateway: Enterprise Data Plane for MCP and A2A

This solution has **two parts**:

- **The Registry** — the control plane. It is the system of record for MCP servers, A2A agents, skills, groups, and access policy. It handles registration, discovery, search, and the admin surfaces where operators define who can reach what.
- **The Gateway** — the data plane. It is the single ingress every runtime call flows through: an authenticating, authorizing, rate-limiting reverse proxy that sits in front of your MCP servers and A2A agents.

This document focuses on **the Gateway**. The registry is covered elsewhere (see [Registry docs](README.md) and the [design directory](design/)). The point of the gateway is simple to state: **every MCP and A2A call your organization makes goes through one governed, observable, policy-enforcing hop** — the things an enterprise needs before it can let agents loose on real tools and real data.

## Why a Data Plane

Without a gateway, each MCP client connects directly to each MCP server, and each agent talks directly to each other agent. Authentication is ad hoc, authorization is per-server (if it exists at all), there is no central rate limiting, no unified audit trail, and no consistent way to let a user reach a third-party tool "as themselves." That does not scale to an enterprise with hundreds of tools and many teams.

The gateway collapses all of that into one data plane. A client authenticates once, to the gateway. The gateway decides what that caller may reach, meters how often they may reach it, proxies the call to the real backend, and records it. The backends can stay simple; the policy lives in one place.

## What Flows Through the Gateway

### MCP server traffic

MCP clients (Claude Code, Cursor, custom agents, CI jobs) connect to the gateway rather than directly to individual MCP servers. The gateway terminates the connection, authenticates the caller, authorizes the specific server (and, where configured, the specific tool), and reverse-proxies the request to the registered backend. To the client it looks like one endpoint that exposes many servers; to each backend it looks like a single trusted caller. The gateway serves MCP 2026-07-28 clients and earlier clients on the same URLs; see [MCP 2026-07-28 support](mcp-2026-07-28-migration.md).

### A2A agent traffic

The gateway does the same for Agent-to-Agent traffic. In **reverse-proxy mode**, each enabled agent gets gateway routes that proxy its A2A traffic — both the agent card (discovery) and the JSON-RPC invocations — through the same authenticating, authorizing hop. Agent-to-agent calls become first-class, governed traffic instead of a private mesh the enterprise cannot see. (A registry-only discovery mode is also available when you want cataloging without putting the gateway in the data path. See [A2A protocol integration](design/a2a-protocol-integration.md).)

In both cases the gateway is the chokepoint, and that chokepoint is where the enterprise features below are enforced.

## The Enterprise Features the Gateway Provides

### Fine-Grained Access Control (FGAC)

Access is decided per caller and per resource, not all-or-nothing. A caller's identity and group membership come from your IdP (Keycloak, Entra, Okta, Auth0, Cognito, PingFederate). Policy — expressed in terms of groups and the servers, agents, tools, and skills they may reach — lives in the registry control plane. At call time the gateway resolves the caller's identity and groups, then authorizes the exact target: this user, in these groups, may reach this MCP server (and this tool), or invoke this agent (and this skill). Authorization **fails closed** — if policy is missing or ambiguous, the call is denied. This is what lets an enterprise give different teams different, least-privilege slices of the tool catalog through one endpoint.

### Rate Limiting

The gateway meters how often callers may invoke tools and agents, in two complementary layers:

- **Edge (per-IP) limiting** at nginx protects against volumetric floods before a request is even authenticated.
- **Application-level, identity/target-aware limiting** at the authorization hop caps how many requests a given caller (via their rate-limit group membership) may make, and how many a given MCP server or A2A agent may absorb across all callers — at windows from per-second bursts to per-day volume quotas. A specific user or agent is limited by mapping it (by username or client_id) to a rate-limited group in a dedicated memberships collection, kept separate from IdP/authz groups.

Together they protect weak backends from overload, keep one tenant from starving others, and bound the cost and quota consumption of downstream APIs. Limits are identity- and target-aware because the gateway, unlike a raw network appliance, knows exactly who is calling and what they are calling. See [Rate limiting design](design/rate-limiting.md).

### Dynamic Client Registration (DCR)

Enterprises cannot hand-provision an OAuth client for every MCP client an employee might run. The gateway supports **Dynamic Client Registration** (RFC 7591): a well-formed MCP client can register itself with the configured IdP and be issued a fresh `client_id`, then complete a standard OAuth authorization-code + PKCE flow to obtain a token the gateway will accept. This is how tools like Claude Code, Cursor, and Kiro onboard against the gateway without a manual IT ticket per user per tool. See [Keycloak MCP client guide (DCR + OAuth)](keycloak-mcp-clients.md).

### Egress Authentication

Inbound auth answers "who is this caller and what may they reach." **Egress auth** answers the other half: how does the gateway let that caller reach an *authentication-protected third-party* MCP server — GitHub, Slack, Atlassian — **as themselves**, without the client ever handling the third-party credential? The gateway holds per-user credentials in a secret store and attaches them on the outbound hop, so the user's own identity and permissions apply at the third party, while the sensitive token never touches the coding assistant. This is what makes it safe to let an agent act on a user's behalf against real SaaS tools. See [Egress authentication design](design/egress-auth-design.md) and the [Egress credential vault](egress-credential-vault.md).

## How They Fit Together

```
                    ┌──────────────────────────────────────────────┐
                    │                  GATEWAY                      │
                    │              (the data plane)                 │
   MCP / A2A        │                                               │      registered
   clients  ──────► │  authenticate → FGAC authorize → rate limit   │ ───► MCP servers
   & agents         │       → proxy → audit    (+ egress auth        │      & A2A agents
                    │                            on the way out)     │
                    └───────────────────────┬──────────────────────┘
                                             │ reads identity/policy from
                                             ▼
                    ┌──────────────────────────────────────────────┐
                    │                 REGISTRY                      │
                    │             (the control plane)               │
                    │  servers · agents · skills · groups · policy  │
                    │       registration · discovery · admin        │
                    └──────────────────────────────────────────────┘
```

The registry decides *what the policy is*; the gateway *enforces it on every call*. An operator manages servers, agents, groups, and rate-limit definitions through the registry's admin surfaces; the gateway reads that state and applies it in the data path. The two are deployed together but separable — you can run the registry for discovery only, or put the gateway inline to get the full enforcing data plane.

## Where to Go Next

- [Fine-grained access control and auth](design/authentication-design.md)
- [Rate limiting design and implementation](design/rate-limiting.md)
- [A2A protocol integration](design/a2a-protocol-integration.md)
- [Egress authentication design](design/egress-auth-design.md)
- [Keycloak MCP client guide (DCR + OAuth)](keycloak-mcp-clients.md)
- [Unified configuration reference](unified-parameter-reference.md)
- [MCP 2026-07-28 support](mcp-2026-07-28-migration.md)
