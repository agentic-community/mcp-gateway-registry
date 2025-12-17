# Archived: Legacy Authentication Doc (Removed)

This file previously documented a legacy dual-token model (ingress + egress) and client-side passthrough of upstream credentials.

That model is deprecated and intentionally removed from this repository’s active documentation.

## Current Model (Canonical)

- Agents authenticate only to the gateway.
- Upstream credentials and upstream connections are managed by the gateway (gateway-terminated upstream authentication).
- Client-visible MCP configs must not include upstream API keys, OAuth tokens, or mTLS material.

See:
- `docs/auth.md`
- `enforceai/mcp_upstream_auth_requirements.md`
