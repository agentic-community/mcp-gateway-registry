# FGAC Model — EnforceAI Gateway

## Enforcement Logic
1. Extract IdentityContext
2. Determine tool + action
3. Validate:
   allow = action in scopes AND tool in allowed_tools
4. Default-deny for unknown actions
5. Log all decisions

## Overlay Semantics (Decision)
- Runtime authorization is agent-scoped:
  - Allow/deny is computed from the agent's scopes (and optional `allowed_tools`) only.
  - IdP roles/groups must not grant additional permissions.
- Enterprise policy is the authoritative scope catalog (what each scope allows).
  - Unknown/removed scopes must grant no permissions (fail closed).
  - Scope validity and any per-user baseline constraints are enforced at agent provisioning time.

## Token Interaction (Decision)
- Gateway tokens may carry scopes, but they must not elevate beyond current agent registry scopes.
- For `gateway-token` requests, compute:
  - `effective_scopes = token.scopes ∩ agent.scopes`

## Audit Fields
- user_id
- agent_id
- action
- tool
- decision
- reason
- timestamp
