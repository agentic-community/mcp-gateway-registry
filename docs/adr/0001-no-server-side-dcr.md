# ADR 0001: The registry will not act as a DCR authorization server

- **Status:** Accepted (2026-08-02)
- **Deciders:** Registry maintainers
- **Related:** Issue #995, closed PR #511 (`Feat/DCR`), umbrella #988

## Decision

The MCP Gateway Registry will **not** implement RFC 7591 Dynamic Client Registration (DCR) as a server-side feature. The registry will not expose its own client registration endpoint and will not act as an authorization server that mints OAuth clients on demand.

This decision is narrow and specific. It bans exactly one role: **the registry as a DCR authorization server**. It does not ban DCR everywhere. Two DCR-adjacent behaviors are explicitly still supported (see "What this does not forbid" below).

## Context

DCR (RFC 7591) lets a client register itself with an authorization server at connect time and receive a freshly minted `client_id`. A registry that implemented this server-side would run its own `/register` endpoint against which any IDE could create clients.

The MCP 2025-06-18 authorization spec lists DCR as a `SHOULD` with an explicit fallback clause, and the November 2025 spec direction (Aaron Parecki, RFC 8693-based ID-JAG plus Client ID Metadata Documents) moves away from DCR as the recommended path.

A prior attempt, PR #511 (`Feat/DCR`), tried to add server-side DCR and was closed. This ADR records the decision explicitly so the discussion is not reopened ad hoc.

### Why we decline server-side DCR

The reasons are primarily operational. They hold regardless of which way the spec trends:

- **Unbounded database growth.** Every client registration is a new persistent record, forever. An authorization server that accepts DCR has to store every client any IDE ever creates.
- **No standard revocation.** Once a client is registered via DCR, there is no standard mechanism to un-register it. Cleanup becomes a manual, out-of-band chore.
- **Rate-limiting and abuse surface.** A public `/register` endpoint has to be defended against registration-based abuse.
- **Enterprise IT loses control.** Employees could register arbitrary clients without an administrator in the loop.

We already control the upstream IdP configuration (Entra, Keycloak, Okta, Cognito, Auth0) and can register clients there when genuinely needed. Delegating to the upstream IdP gives us the "self-describing client" use case without owning any of the four problems above.

### A note on the original rationale

Issue #995 originally leaned heavily on "the spec is retiring DCR." That framing has aged poorly: as of 2026, real-world resources are moving **toward** RFC 7591, not off it. Atlassian's Rovo MCP `authv2` resource, for example, only mints MCP-audience tokens to clients registered via DCR (see PR #1519). The durable justification for this ADR is therefore the **operational** cost of running a DCR authorization server, not a claim that the ecosystem is abandoning DCR.

## What this does not forbid

This ADR bans the registry acting as a DCR **authorization server**. It does not affect either of the following, both of which are supported today:

1. **Delegated DCR against the upstream IdP.** The registry advertises a spec-compliant discovery chain (RFC 9728 Protected Resource Metadata, then RFC 8414 Authorization Server metadata). An IDE that supports DCR uses that chain to register a client at the **upstream IdP's** `registration_endpoint` (for example Keycloak, Auth0, or Okta). The registry is not the authorization server in this flow, it points the IDE at the real one. See [Dynamic Client Registration](../connection-methods/dynamic-client-registration.md).

2. **The registry as a DCR client on egress.** For some backends the gateway must register **itself** as an OAuth client at the backend's authorization server before it can obtain a working token (for example Atlassian Rovo `authv2`, which requires a public PKCE client registered via RFC 7591). Here the registry is the OAuth **client**, not the authorization server. This is a different role and is unaffected by this decision. See PR #1519.

## Alternatives considered

- **DCR server-side (the registry runs its own `/register`).** Rejected for the four operational reasons above. This is the option this ADR exists to decline.
- **DCR client-side only, delegated to the upstream IdP.** Accepted and already implemented. This is the supported path for zero-touch client provisioning.
- **Client ID Metadata Documents (CIMD).** Complementary, tracked separately under umbrella #988. CIMD covers the self-describing-client use case without any of DCR's server-side drawbacks.
- **Pre-registered clients only.** Supported for operators who want an administrator in the loop; the operator registers the client in the upstream IdP once.

## Consequences

- **For operators:** to provision clients, configure them in the upstream IdP, or rely on the delegated-DCR discovery chain the registry already advertises. The registry itself will never hand out a `client_id`.
- **For contributors:** pull requests that add a server-side DCR / client registration endpoint to the registry will be declined with a pointer to this ADR. Discovery-chain work (PRM, AS metadata) and egress DCR-client work (the gateway registering itself at a backend) are unaffected and welcome.
- **For future maintainers:** the reasoning lives here so the #511 discussion does not have to be re-run from scratch.

## Revisit trigger

Reopen this decision if either of the following happens:

- A future MCP spec revision makes server-side DCR a `MUST` for the resource-server / gateway role, or
- A major coding assistant regresses to DCR-server-only and cannot use the delegated-IdP discovery chain.

If revisited, supersede this ADR with a new one rather than editing the decision in place, so the history stays traceable.

## References

- [RFC 7591 Dynamic Client Registration](https://datatracker.ietf.org/doc/html/rfc7591)
- [RFC 9728 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728)
- [RFC 8414 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414)
- [Aaron Parecki, MCP Authorization Spec Update (2025-11-25)](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update)
- [Dynamic Client Registration connection method](../connection-methods/dynamic-client-registration.md)
- Issue #995 (this decision), umbrella #988, closed PR #511 (`Feat/DCR`), PR #1519 (egress DCR client)
