# Decision 0007 — Authorization Overlay Semantics
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI requires agent-scoped authorization and forbids deriving agent permissions from IdP roles/groups.
We also need enterprise-controlled policy definitions and a streamlined request-path enforcement flow.

## Decision
- Runtime authorization is computed from agent scopes (and optional allowed-tools) only.
- Enterprise policy is the authoritative scope catalog defining what each scope allows.
- Unknown/removed scopes grant no permissions (fail closed).
- Any per-user baseline constraints are enforced at agent provisioning time (agent scopes cannot exceed the gateway-defined baseline).

## Consequences
- Enforcement stays simple and deterministic on the request path.
- Policy changes to scope definitions take effect immediately for all agents using those scopes.
- If per-user baselines are needed, they must be implemented in provisioning flows (or added later as an additional runtime cap).
