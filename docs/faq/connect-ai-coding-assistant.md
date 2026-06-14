# How do I get my AI coding assistant to work with this registry?

**Short answer**: point your assistant (Cursor, Claude Code, Codex) at a server's
**Connect** dialog in the registry UI, copy the generated config, and paste/run
it. The assistant authenticates against your identity provider (IdP) and connects.
How the OAuth `client_id` is provided depends on your IdP and assistant - that is
what the detailed docs below cover.

## Start here

1. **Understand the three connection methods** and which fits your setup:
   [How coding assistants connect](../ai-coding-assistants-setup.md#how-coding-assistants-connect-three-methods).
   In short:
   - **Pre-registered public client id** (`IDE_OAUTH_CLIENT_ID`) - the operator
     registers one public client; best for enterprise/IdP-with-DCR-disabled.
   - **Dynamic Client Registration (DCR)** - the IDE self-registers; zero-touch.
   - **Client ID Metadata Documents (CIMD)** - on the roadmap, not yet implemented.
2. **Check the compatibility matrix** (which method works for your assistant x
   IdP):
   [Compatibility matrix](../ai-coding-assistants-setup.md#compatibility-matrix-coding-assistant-x-identity-provider).

## Pick your path

### Most setups: pre-registered public client id

Read [Connection method: pre-registered public client id](../connection-methods/client-id.md).
It explains the `IDE_OAUTH_CLIENT_ID` / per-server `oauth_client_id` settings, the
`IDE_OAUTH_CALLBACK_PORT` (needed for Okta/Cognito/Entra), and has a per-IdP setup
section. Create the public client with the script for your IdP:

| IdP | Setup script |
| --- | --- |
| Keycloak | `setup/idp/keycloak/setup-ide-public-client.sh` |
| Amazon Cognito | `setup/idp/cognito/setup-ide-public-client.sh` |
| Okta | `setup/idp/okta/setup-ide-public-client.sh` |
| Microsoft Entra ID | `setup/idp/entra/setup-ide-public-client.sh` (login blocked pending #990) |

Then set the printed `IDE_OAUTH_CLIENT_ID` (and `IDE_OAUTH_CALLBACK_PORT` for
strict IdPs) on the registry and restart.

### Zero-touch on Keycloak: DCR

If your IdP supports anonymous DCR (Keycloak does), you do not need to register a
client. See [Connection method: DCR](../connection-methods/dynamic-client-registration.md).
Note the dialog only emits the DCR-style config when the provider is Keycloak.

## Key things to know (save yourself a debugging session)

- **Access comes from groups, not scopes.** The user must be in an IdP group that
  is mapped to a registry scope in the `mcp_scopes` collection, or login succeeds
  but every server returns 403. (Okta needs a `groups` claim added to the
  authorization server; Cognito/Keycloak/Entra include groups by default.)
- **Strict IdPs need a fixed callback port.** Okta, Cognito, and Entra match the
  redirect URI literally including the port. Set `IDE_OAUTH_CALLBACK_PORT` and
  register `http://localhost:<port>/callback`. Only **Claude Code** can pin the
  port (`--callback-port`); Codex/Cursor cannot and fall back to the static token
  on those IdPs.
- **After a redeploy, log out and back in.** A stale session cookie makes the
  Connect dialog fail to load (403) and silently fall back to the static token.
- **The client id is public, not a secret.** It is safe to commit and to show in
  configs; it identifies the application, not the user.

## Related

- [AI coding assistants setup guide](../ai-coding-assistants-setup.md)
- [Pre-registered public client id](../connection-methods/client-id.md)
- [Dynamic Client Registration (DCR)](../connection-methods/dynamic-client-registration.md)
- [Client ID Metadata Documents (CIMD)](../connection-methods/client-id-metadata-documents.md)
- [Unified parameter reference](../unified-parameter-reference.md) (`IDE_OAUTH_CLIENT_ID`, `IDE_OAUTH_CALLBACK_PORT`, `MCP_ADVERTISED_SCOPES`)
