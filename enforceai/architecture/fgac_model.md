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
  - Phase 1 policy catalog source is `auth_server/scopes.yml` (reuse existing scope schema and evaluator).

## Token Interaction (Decision)
- Gateway tokens may carry scopes, but they must not elevate beyond current agent registry scopes.
- For `gateway-token` requests, compute:
  - `effective_scopes = token.scopes ∩ agent.scopes`

## Tool Visibility (Decision)
- `tools/list` responses must be filtered to avoid exposing tools the caller cannot use.
- Phase 1 rule: only return tools that are callable under the current effective authorization:
  - Allowed by enterprise policy catalog (`auth_server/scopes.yml`) for `tools/call`
  - And allowed by agent-level `allowed_tools` if it is set
- If the caller is not authorized to call any tools on a server, `tools/list` must return an empty tool list for that server.

## Audit Fields
- user_id
- agent_id
- action
- tool
- decision
- reason
- timestamp

## Audit Sink and Failure Policy (Decision)
- Phase 1 audit is dual-sink:
  - Emit structured audit events to stdout (JSON).
  - Persist audit events to the local SQLite audit table for investigation and retention.
- Failure policy (pragmatic):
  - If audit persistence fails, do not fail the request solely due to audit storage failure.
  - Emit a high-severity log event indicating audit persistence failure so operators can remediate.

## Audit Retention (Decision)
- SQLite audit retention is hybrid and configurable:
  - Time-based retention (e.g., `ENFORCEAI_AUDIT_RETENTION_DAYS`)
  - Size-based cap (e.g., `ENFORCEAI_AUDIT_MAX_DB_BYTES`)
- Cleanup is performed out of band (not on the request path) to avoid impacting `/validate` latency.
