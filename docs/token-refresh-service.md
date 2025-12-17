# Token Refresh Service (Deprecated)

This repository previously documented a background “token refresh service” that refreshed provider/egress tokens and regenerated MCP client configuration files.

That model is deprecated in favor of a gateway-terminated design:

- Agents authenticate only to the gateway.
- The gateway manages all upstream credentials and upstream token refresh on demand.
- Client configs must not contain upstream provider tokens.

For client-side gateway access tokens, use the gateway’s normal ingress auth (OIDC/client credentials) or the JWT token vending flow for headless/coding-assistant usage.
