# Connection Method: Dynamic Client Registration (DCR)

This is one of three ways an AI coding assistant (Cursor, Claude Code, Codex)
can obtain the OAuth `client_id` it needs to log in to a gateway-protected MCP
server. See [the connection methods overview](../ai-coding-assistants-setup.md#how-coding-assistants-connect-three-methods)
for how this compares to a pre-registered client id and Client ID Metadata
Documents.

## What it is

Dynamic Client Registration (RFC 7591) lets the IDE **register itself** with the
IdP at connect time. The user adds an MCP server, the IDE calls the IdP's
registration endpoint, receives a freshly-created `client_id`, and proceeds
through the OAuth/PKCE login. No operator pre-registers anything.

## When to use it

DCR is the natural fit for a **public or multi-tenant registry** serving many
users and servers you cannot pre-arrange clients for. It is zero-touch: any user,
any IDE, any server, no admin in the loop.

The registry's spec-compliant discovery chain drives it automatically:

```
claude mcp add https://gateway.example.com
# -> 401 with WWW-Authenticate
# -> read Protected Resource Metadata (PRM)
# -> read Authorization Server metadata
# -> DCR: register a client at the registration_endpoint
# -> authorization-code + PKCE login
# -> first tool call
```

With IdPs that support RFC 7591 DCR (Keycloak, Auth0, Okta), no manual
`client_id` paste is required.

## Trade-offs to be aware of

DCR is powerful but has two costs that matter at enterprise scale:

1. **Anonymous registration is a governance concern.** "Anyone can create a
   client in our IdP" is something many enterprises disable. When DCR is off,
   use the [pre-registered client id method](client-id.md) instead.
2. **Client sprawl.** Every IDE on every machine that ever connected leaves a
   client record behind. Over time the IdP fills with disposable clients that
   nobody prunes. (Keycloak deployments can use
   `keycloak/setup/cleanup-stale-dcr-clients.sh` to reap them.)

## Prerequisites

- The IdP must have DCR enabled (and reachable
  `registration_endpoint` in its Authorization Server metadata).
- The gateway must advertise its Protected Resource Metadata and emit the
  `WWW-Authenticate` header on 401 (see
  [OAuth discovery endpoints](../oauth-discovery-endpoints.md)).

## How the Connect dialog decides to rely on DCR (important nuance)

The Connect dialog does not detect DCR capability directly. It assumes
**DCR is available when, and only when, `auth_provider === "keycloak"`**. Two
consequences follow:

- For Claude Code and Codex on a Keycloak deployment, the dialog emits a
  DCR-style config (no embedded token) and expects the IDE to self-register. For
  Cursor, and for all IDEs on non-Keycloak providers, it does NOT rely on DCR and
  falls back to the static token (see the fallback matrix in
  [the client-id method](client-id.md#fallback-behavior-when-no-client-id-is-configured)).
- The assumption is provider-name-based, not a real capability check. A Keycloak
  with DCR **disabled** will still be treated as DCR-capable, so a Claude
  Code/Codex config that relies on DCR will not actually complete. In that case,
  use the [pre-registered public client id](client-id.md) method instead - that is
  exactly the scenario it exists for. Conversely, Okta and Auth0 do support DCR but
  are NOT treated as DCR-capable by the dialog today, so they fall back to the
  static token unless a client id is configured.

## Related

- [Connection methods overview](../ai-coding-assistants-setup.md#how-coding-assistants-connect-three-methods)
- [Pre-registered public client id](client-id.md)
- [Client ID Metadata Documents](client-id-metadata-documents.md)
- [OAuth discovery endpoints](../oauth-discovery-endpoints.md)
- [Keycloak MCP clients](../keycloak-mcp-clients.md)
