# Decision 0018 — Tool Visibility Filtering (`tools/list`)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
MCP clients rely on `tools/list` for discovery. Exposing tools that the caller cannot execute leads to broken workflows (clients attempt calls that will be denied) and leaks information about capabilities.
EnforceAI requires agent-scoped authorization and supports optional `allowed_tools` restrictions.

## Decision
- Filter `tools/list` responses to return only tools that are callable under effective authorization:
  - Allowed by enterprise policy catalog (`auth_server/scopes.yml`) for `tools/call`
  - And allowed by agent-level `allowed_tools` if configured
- If the caller cannot call any tools on a server, `tools/list` returns an empty tool list for that server.

## Consequences
- Improves client reliability and reduces capability leakage.
- Requires response filtering in the gateway/proxy layer (not just request-time validation).
