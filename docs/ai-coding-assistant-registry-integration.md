# AI Coding Assistants and the MCP Gateway Registry: One-Command Integration

> **What this doc is**: a tour of the simplest, most modern way for AI coding
> assistants and AI desktop apps to connect to the MCP Gateway Registry, where
> the assistant itself handles client registration and OAuth, and the human
> just clicks "Allow" in a browser. No `.env` files, no copy-pasted tokens, no
> shared service accounts.
>
> **Status**: tested end-to-end with [Codex CLI](https://github.com/openai/codex),
> [Claude Code](https://docs.claude.com/en/docs/claude-code/overview)
> (CLI on macOS, Linux, Windows),
> [Claude.ai Custom Connectors](https://support.anthropic.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp),
> and [Kiro CLI](https://kiro.dev/) (Windows). Microsoft Entra ID support via
> the Client ID Metadata Document (CIMD, `draft-parecki-oauth-client-id-metadata-document`)
> is the next milestone; see [Roadmap](#roadmap) below.
>
> **Umbrella tracking issue**: [#988 Coding-Assistant OAuth Integration](https://github.com/agentic-community/mcp-gateway-registry/issues/988)

## TL;DR

When the gateway is configured with Keycloak as its identity provider, a
spec-compliant MCP client can connect with no manual configuration:

1. The assistant points itself at the gateway URL (one CLI command, or a URL
   pasted into a settings page).
2. The assistant performs Dynamic Client Registration ([RFC 7591](https://datatracker.ietf.org/doc/html/rfc7591))
   against Keycloak through the gateway's published `/.well-known/oauth-authorization-server`
   metadata.
3. The user is bounced to a browser, logs into Keycloak, sees a consent screen
   listing the scopes the assistant is asking for, and clicks **Allow**.
4. The assistant gets back a token scoped to the user's groups, and the
   gateway accepts it on every subsequent MCP request.

That's it. No tokens to copy, no client secrets to manage, no role drift
between humans and shared service accounts.

## Why DCR matters

For the operator, Dynamic Client Registration removes three classes of pain:

- **No per-client provisioning**. You don't pre-create a Keycloak client for
  each assistant your team uses. The assistant registers itself the first time
  it talks to the gateway.
- **Per-user authorization, not per-tool**. The token the assistant gets back
  carries the *user's* group membership, so the user gets exactly the MCP
  servers and tools their groups grant them, no more and no less.
- **Real audit trails**. Every call into the gateway is attributable to a
  specific human, not to a shared `mcp-tools-readonly` token that you can't
  revoke without breaking everyone.

For the end user, DCR removes one big pain: **there is no "set up
authentication" step before you can use the gateway**. You discover the URL,
paste it in, log in, and you're done.

## Tested integrations

### Codex CLI

[Codex CLI](https://github.com/openai/codex) is OpenAI's terminal-based AI
coding agent. It supports MCP via Streamable HTTP transport and performs the
same DCR + OAuth flow as Claude Code: the assistant registers itself with
Keycloak, opens your browser for login and consent, and the connection is live.

**Demo walkthrough** (3 minutes): https://app.vidcast.io/share/25ed9141-b890-423c-b063-96da5cd9d533

To add the `airegistry-tools` MCP server:

```bash
codex mcp add ai-registry --transport streamable-http --url https://mcpgateway.acme.corp/airegistry-tools/mcp
```

Replace `mcpgateway.acme.corp` with your own gateway hostname.

Once connected, Codex gains access to the registry's `search_registry` tool.
The AI assistant can then discover and connect to any enterprise-approved MCP
server in the catalog on behalf of the user, without the user needing to know
individual server URLs.

What happened under the hood:

1. Codex fetched `/.well-known/oauth-protected-resource` from the gateway,
   which named Keycloak as the authorization server.
2. Codex fetched `/.well-known/oauth-authorization-server` from Keycloak,
   which named the DCR endpoint.
3. Codex POSTed an RFC 7591 registration to Keycloak, getting back a freshly
   minted client ID and secret.
4. Codex initiated an authorization-code+PKCE flow, bouncing you to the
   browser.
5. After consent, Codex exchanged the code for an access token.
6. Codex attached `Authorization: Bearer <token>` on every subsequent MCP
   call.

**Enterprise value**: developers on Codex get access to the full catalog of
organization-approved MCP servers through a single registry connection,
with per-user authorization and audit trails. No shared tokens, no manual
URL distribution.

### Claude Code (CLI)

**Demo walkthrough**: https://app.vidcast.io/share/09a267f4-3a5c-43a4-91fb-f128ed0e7983

To add the `airegistry-tools` MCP server (which gives Claude Code access to
the registry's discovery and search tools):

```bash
claude mcp add --transport http airegistry-tools https://mcpgateway.acme.corp/airegistry-tools/mcp
```

Replace `mcpgateway.acme.corp` with your own gateway hostname.

Then inside Claude Code:

```
/mcp
> select airegistry-tools
> /authorize
```

Claude Code opens your browser to Keycloak, you log in, you click **Allow**
on the consent screen, the browser redirects back to a localhost URL that
Claude Code is listening on, and the connection completes. Subsequent
sessions reuse the cached token until it expires.

What happened under the hood:

1. Claude Code fetched `/.well-known/oauth-protected-resource` from the
   gateway, which named Keycloak as the authorization server.
2. Claude Code fetched `/.well-known/oauth-authorization-server` from
   Keycloak (rehomed onto the gateway's hostname so CORS works), which named
   the DCR endpoint.
3. Claude Code POSTed an RFC 7591 registration to Keycloak, getting back a
   freshly minted client ID and secret.
4. Claude Code initiated an authorization-code+PKCE flow, which is what
   bounced you to the browser.
5. After consent, Claude Code exchanged the code for an access token at
   Keycloak's token endpoint.
6. Claude Code attached `Authorization: Bearer <token>` on every MCP call.

If you're curious about the exact wire-level sequence, see the level-300
section of [Keycloak: MCP Client Guide](keycloak-mcp-clients.md).

### Claude.ai Custom Connectors

In Claude.ai (web), under **Settings -> Connectors -> Browse connectors -> +
Custom Connector**:

- **Name**: anything (e.g., `My MCP Gateway`)
- **URL**: `https://<your-gateway>/<server-path>/mcp`

Click **Add**. Claude.ai performs the same DCR + OAuth flow as Claude Code, but
in the browser tab where you're already logged in, so the consent screen is
the only interactive step.

The first time a new gateway connects from Claude.ai's infrastructure, you'll
see logs like this on the gateway:

```
160.79.106.x - "POST /realms/mcp-gateway/clients-registrations/openid-connect HTTP/1.1" 201
160.79.106.x - "GET /realms/mcp-gateway/protocol/openid-connect/auth HTTP/1.1" 200
160.79.106.x - "POST /realms/mcp-gateway/protocol/openid-connect/token HTTP/1.1" 200
Claude-User - "POST /<server-path>/mcp HTTP/1.1" 200
```

That's a freshly registered Claude.ai-owned client, an authorization round
trip with the user, a token exchange, and then MCP traffic. No manual setup
on either side.

> **Note**: Claude.ai sends DCR registration requests from its own IPs
> (currently in the 160.79.106.x range). The gateway's Keycloak Trusted Hosts
> policy already includes `claude.ai`, `localhost`, and `127.0.0.1` so these
> registrations succeed. If you tighten that policy, Claude.ai connectors
> will stop being able to register.

### Kiro CLI

Kiro CLI on Windows performs the same DCR + OAuth flow as Claude Code:
the assistant registers itself, opens your browser to Keycloak, you click
**Allow** on the consent screen, and the connection completes.

To add the `airegistry-tools` MCP server:

```powershell
kiro-cli mcp add --name airegistry-tools --scope global --url https://mcpgateway.acme.corp/airegistry-tools/mcp
```

Replace `mcpgateway.acme.corp` with your own gateway hostname. Then in Kiro:

```
/mcp
```

Kiro opens your browser, you log in, you accept scopes, and the listing
flips from `loading / 0 tools` to the actual tool list within a few seconds.

> **Caveat for remote-development setups**: Kiro's OAuth callback listens
> on `127.0.0.1:<random-port>` on the machine where Kiro itself is running.
> If you're running Kiro on a remote box (e.g., over SSH on EC2) but the
> browser is on your laptop, you need an SSH local-forward tunnel for that
> port before the consent click, otherwise the redirect lands on your
> laptop with nothing listening. The simplest fix is to run Kiro on the
> same machine as the browser. If you do need the tunnel:
>
> ```powershell
> ssh -i <key> -L <port>:localhost:<port> -N user@host
> ```
>
> Bring the tunnel up *before* triggering the OAuth flow, and re-tunnel if
> Kiro picks a new ephemeral port on retry.

## What's *not* yet a one-command setup

These integrations work, but they currently require the user to paste a
static token into the assistant's config file rather than going through DCR:

- **Roo Code** (VS Code extension): supports MCP via custom HTTP headers, but
  no OAuth flow yet. Use a service-account JWT generated by
  [`credentials-provider/keycloak/get_m2m_token.py`](../credentials-provider/keycloak/).
- **Cursor**: limited DCR support depending on version. Static tokens are the
  reliable fallback today.

These will move to one-command DCR as their MCP client implementations catch
up to the 2025-06-18 spec. Tracking: see issue [#988](https://github.com/agentic-community/mcp-gateway-registry/issues/988).

## Roadmap

The umbrella issue [#988](https://github.com/agentic-community/mcp-gateway-registry/issues/988)
breaks the work into phases. Where we are today:

| Phase | What it adds | Status | Issue |
|-------|--------------|--------|-------|
| 1 | PRM + AS metadata + WWW-Authenticate on 401 (the foundation that lets clients discover Keycloak via the gateway) | Implemented (this PR) | [#989](https://github.com/agentic-community/mcp-gateway-registry/issues/989) |
| 2 | Entra v1 `api://` scope verbatim pass-through + audience normalization | Designed | [#990](https://github.com/agentic-community/mcp-gateway-registry/issues/990) |
| 3 | RFC 8707 resource parameter enforcement + token proxy | Designed | [#991](https://github.com/agentic-community/mcp-gateway-registry/issues/991) |
| 4 | CIMD consumer: accept a CIMD URL as `client_id` on `/authorize` (this is what makes Entra one-command, since Entra doesn't speak DCR) | Designed | [#993](https://github.com/agentic-community/mcp-gateway-registry/issues/993) |
| 5 | ID-JAG (RFC 8693 token exchange) receiver on the token proxy | Designed | [#994](https://github.com/agentic-community/mcp-gateway-registry/issues/994) |

CIMD (Phase 4) is the path to bringing the same one-command experience to
Microsoft Entra ID. Entra doesn't implement RFC 7591 DCR, but it does
support CIMD (`draft-parecki-oauth-client-id-metadata-document`), which
achieves the same outcome by letting the client publish its own metadata
at a URL that the IdP fetches. Keycloak 26.6+ has experimental CIMD support
on the producer side; we're tracking the consumer side at [#993](https://github.com/agentic-community/mcp-gateway-registry/issues/993).

## Setup checklist for operators

If you're standing up a gateway that you want to be DCR-ready for Claude Code
and Claude.ai today:

1. Deploy with `AUTH_PROVIDER=keycloak`. See [Keycloak: Agent M2M & Operations Guide](keycloak-agent-m2m.md)
   for the full Keycloak setup, then come back here.
2. Run `bash keycloak/setup/upgrade-realm-for-dcr.sh` once. This is idempotent
   and applies the four DCR-enabling changes to the realm: groups mapper on
   the `basic` scope, audience mapper on the `basic` scope, widened
   "Allowed Client Scopes" policy, and a relaxed "Trusted Hosts" policy
   that includes `claude.ai`.
3. Set `MCP_ADVERTISED_SCOPES="profile email offline_access"` in `.env`
   (quoted; this is the list of scopes the gateway advertises in PRM, and
   Keycloak rejects DCR registrations that ask for scopes outside its allowlist).
4. Confirm the gateway publishes `/.well-known/oauth-protected-resource` and
   `/.well-known/oauth-authorization-server` on every server path you want
   to expose. Both should return JSON without authentication.
5. Hand the URL to a user. Tell them to run
   `claude mcp add --transport http <name> <url>`.

That's the whole loop. From the user's perspective, all they ever see is a
URL, a login screen, and a consent dialog.

## Troubleshooting

If a Claude Code or Claude.ai connection fails, check these in order:

1. **`curl -i https://<gateway>/<server>/mcp`** without a token should
   return `401` with a `WWW-Authenticate: Bearer resource_metadata="..."` header.
   If it doesn't, the middleware isn't wired in for that server.
2. **`curl https://<gateway>/.well-known/oauth-protected-resource`** should
   return JSON naming Keycloak as the authorization server. If 404, the
   wellknown route isn't mounted.
3. **`docker-compose logs keycloak | grep TRUSTED_HOST`** when DCR fails.
   Claude.ai DCR will be rejected if `claude.ai` isn't in the trusted hosts
   list. Re-run `upgrade-realm-for-dcr.sh`.
4. **`docker-compose logs keycloak | grep "Invalid scope"`** when DCR fails
   with a 400. The client asked for a scope that isn't in
   `MCP_ADVERTISED_SCOPES`. Either add it, or trim the advertisement to
   match what Keycloak actually has.
5. **DCR client bloat**: Claude Code creates a new DCR client per machine,
   per re-link. Run `bash keycloak/setup/cleanup-stale-dcr-clients.sh --dry-run`
   periodically (or in cron) to see what's accumulated, then drop `--dry-run`
   to delete.

## Related documentation

- [Keycloak: MCP Client Guide](keycloak-mcp-clients.md) - the level-100
  through level-400 walkthrough of exactly what happens on the wire
- [Keycloak: Agent M2M & Operations Guide](keycloak-agent-m2m.md) - service
  accounts and operational procedures for agent (non-coding-assistant) M2M
  flows
- [AI Coding Assistants Setup Guide](ai-coding-assistants-setup.md) - the
  static-token configuration path for clients that don't yet do DCR (Roo
  Code, Kiro, older Cursor)
- [Authentication Guide](auth.md) - identity-provider-agnostic overview
