# Connection Method: Client ID Metadata Documents (CIMD)

> **Status: partial.** The registry can now **publish** its own Client ID
> Metadata Document (so it can act as an OAuth *client* to external CIMD-aware
> IdPs). On the ingress side it forwards an incoming CIMD `client_id` to the
> upstream IdP, which is the party that fetches and validates the document.
> See "Current status" below.

This is one of three ways an AI coding assistant could obtain the OAuth identity
it needs to log in to a gateway-protected MCP server. See
[the connection methods overview](../ai-coding-assistants-setup.md#how-coding-assistants-connect-three-methods)
for how it compares to a pre-registered client id and Dynamic Client
Registration.

## What it is

Client ID Metadata Documents (CIMD) take a different approach: the `client_id` is
not a pre-registered record at all. It is an `https` URL that points to a small
JSON metadata document describing the client. The authorization server fetches
that URL on demand to learn who the client is, instead of looking up a stored
registration.

## Why it matters

CIMD aims to keep the zero-touch benefit of Dynamic Client Registration while
removing its bookkeeping cost:

- No registration call.
- No stored client record in the IdP.
- No client sprawl to clean up.

It is the newest of the three approaches and currently the least universally
supported across IdPs and IDEs, which is why it is future-facing rather than
available today.

## Current status in this registry

Two sides, matching how CIMD works in practice:

### The registry as a CIMD client (publisher) -- available, opt-in

When `CIMD_PUBLISHER_ENABLED=true`, the registry serves a Client ID Metadata
Document at:

    GET /.well-known/mcp-client-metadata

That document's URL **is** the registry's `client_id` when it authenticates as
an OAuth client to an external CIMD-aware IdP. It is public, unauthenticated, and
cacheable (`Cache-Control: public, max-age=3600`). Configure it with:

| Setting | Purpose |
| --- | --- |
| `CIMD_PUBLISHER_ENABLED` | Serve the document (default off; 404 when off). |
| `CIMD_CLIENT_NAME` | Human-readable client name (default "AI Registry Tools"). |
| `CIMD_REDIRECT_URIS` | CSV of allowed callbacks (default `{REGISTRY_URL}/oauth2/egress/callback`). |
| `CIMD_SCOPE` | Space-separated requested scopes (default the advertised OIDC scopes). |
| `CIMD_LOGO_URI`, `CIMD_CONTACTS` | Optional metadata; omitted when unset. |

**Operator consequence:** because the document's URL is the `client_id`, renaming
or moving the endpoint changes the client_id and breaks any CIMD-aware IdP
mid-flight. Treat the path as stable.

### The registry consuming an incoming CIMD client_id -- delegated to the IdP

A coding assistant that uses a CIMD `client_id` sends it to the **authorization
server**, which the registry advertises as the **upstream IdP** (RFC 8414 AS
metadata; the registry is not itself the authorization server -- see
[ADR: no server-side DCR](../adr/0001-no-server-side-dcr.md)). The **IdP** fetches
and validates the CIMD document. On the token leg, the gateway's `/oauth/token`
proxy forwards the CIMD `client_id` verbatim to the IdP's token endpoint. CIMD
consumption therefore works through the same discovery chain as the other
methods, with no registry-side client fetch or validation.

This requires an upstream IdP that supports CIMD (e.g. Microsoft Entra; Keycloak
26.6+ has experimental support). IdPs without CIMD keep using the pre-registered
client id or DCR methods.

## Related

- [Connection methods overview](../ai-coding-assistants-setup.md#how-coding-assistants-connect-three-methods)
- [Pre-registered public client id](client-id.md)
- [Dynamic Client Registration](dynamic-client-registration.md)
