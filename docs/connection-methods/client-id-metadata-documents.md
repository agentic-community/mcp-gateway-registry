# Connection Method: Client ID Metadata Documents (CIMD)

> **Status: coming soon.** This method is not yet supported by the MCP Gateway
> Registry. This page describes the concept and is a placeholder for future
> documentation.

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

Not implemented. When CIMD support lands, this page will document:

- How to configure the gateway to advertise/accept CIMD client ids.
- Which IdPs and IDEs support it.
- How it interacts with the registry's group-based authorization.

## Related

- [Connection methods overview](../ai-coding-assistants-setup.md#how-coding-assistants-connect-three-methods)
- [Pre-registered public client id](client-id.md)
- [Dynamic Client Registration](dynamic-client-registration.md)
