# What changes when I upgrade to a release with MCP 2026-07-28 support?

For most deployments, nothing you need to act on. Existing MCP clients keep working on the same URLs, and clients of the new MCP 2026-07-28 revision start working too. One access-control change needs a check before you upgrade: see [Check your scopes](#check-your-scopes).

Support for MCP 2026-07-28 starts in release 1.37.0. For how the gateway handles each entry of the 2026-07-28 changelog, see [MCP 2026-07-28 support](../mcp-2026-07-28-migration.md).

## What does not change

- **Existing clients.** Clients that open with `initialize` and send an `Mcp-Session-Id` (Claude Code, Cursor, the registry CLI, scripts using the MCP SDK before 2.x) work as before, on direct and virtual servers.
- **URLs and registrations.** No route moves, and you do not need to register any server, virtual server or group again.
- **Tool calls.** The gateway authorizes `tools/list` and `tools/call` as before.
- **Admin groups.** A group whose methods list is `all`, such as `registry-admins`, keeps every method.
- **Deployment settings.** No Helm value, Terraform variable or environment variable is new or required.

## What changes for clients

- **New-protocol clients work.** A 2026-07-28 client sends no `initialize` and no session id. Direct servers pass its requests to the backend. Virtual servers answer it themselves and translate for backends that only speak the older protocol.
- **A direct route does not translate.** A client that speaks only 2026-07-28 cannot use a direct route to a server that speaks only the older protocol, because the gateway forwards requests unchanged. A client in auto mode (the MCP SDK default) asks `server/discover` first and falls back to the older handshake, so it works.
- **Headers must match the body.** A 2026-07-28 request whose `Mcp-Method`, `Mcp-Name` or `MCP-Protocol-Version` header disagrees with its body gets HTTP 400 with error `-32020`. A correct client never sends that; it points to a client bug.
- **`ping` on virtual servers.** A 2026-07-28 client gets "method not found" for `ping`, which the new revision removed. Older clients still get an answer.
- **No server push yet.** `subscriptions/listen`, the new way to receive change notifications, answers "method not found". Clients carry on and see changes the next time they ask. Progress messages from a long tool call still arrive at the end of the call.
- **Larger tool calls.** Requests up to 1 MB now go through. Tool calls above about 10 KB used to fail with HTTP 500.
- **Sign-in headers.** A 401 carries one `WWW-Authenticate` header (it carried two), and a 403 now names the protected-resource metadata in `WWW-Authenticate`.

## Check your scopes

**What changed.** Before 1.37.0, a scope entry with `tools: ["*"]` (all tools) also allowed every MCP method, whatever its methods list said. It now allows every tool for `tools/call`, and only the methods the entry lists. `server/discover` and `notifications/*` need no entry: the gateway allows them for any scope that grants access to the server.

**Who is affected.** A group entry that has `tools: ["*"]` and a methods list that leaves out a method its users call. The usual candidates are `resources/read`, `prompts/list` and `prompts/get` (for servers that offer resources or prompts), and `completion/complete` and `logging/setLevel` (for clients that use them). Entries with named tools keep working as before, and so do the default scopes the registry seeds.

**How to check and fix it:**

1. **Find wildcard tool entries.** In **Settings > IAM > Groups**, open each group and look for server entries whose tools are `*`.
2. **Compare the methods list** with the methods the server offers and its users call, using the list above.
3. **See what the gateway refuses.** After the upgrade, the auth server logs each refusal as `Access denied: server='<server>' method='<method>' tool='<tool>'`. Search the auth-server log for `Access denied`: `docker compose logs auth-server` on Docker Compose, the auth-server pod logs on Kubernetes, or the auth-server CloudWatch log group on Amazon ECS.
4. **Add the missing methods.** Edit the group in **Settings > IAM > Groups**, or send `PATCH /api/management/iam/groups/<group>` with the group's full `scope_config.server_access` list. The PATCH replaces the whole list.
5. **Groups with a `server: "*"` entry.** The API no longer saves an access entry that covers every server (`server: "*"` or `"all"`), so you cannot save a group that still has one as it is. Recreate the group with per-server entries and grant broad visibility through `ui_permissions` (`"all"`) instead.

A client that hits this sees a plain HTTP 403 that does not name the method, so the auth-server log is the place to find the cause.

## What changes for operators

- **HTTPS servers list their tools again on images built from 1.37.0.** Tool discovery moved to MCP SDK 2.x. Servers that require each user to sign in (Slack, Datadog, Atlassian, GitHub with OAuth) still answer the registry's background fetch with 401 and show 0 tools in the registry until you connect a discovery identity for them. Users who have connected their own account still get the tools through the gateway.
- **A health status label changed.** A server whose host does not resolve (a stopped container, a typo, a DNS outage) now reads `unhealthy: host could not be resolved` instead of `unhealthy: url blocked by SSRF guard`.
- **HTTP+SSE-only servers log a warning.** Registering a server whose only transport is `sse` logs that MCP 2026-07-28 deprecates HTTP+SSE and that support may end no earlier than 2027-07-28. The registry still accepts the server.
- **mcpgw and the security scanner moved to MCP SDK 2.x.** mcpgw (the AI Registry tools server) serves both protocol revisions with the same tool names and schemas.
- **Optional: issuer check for egress sign-in.** A server's egress OAuth configuration accepts an `expected_issuer`. When it is set, the gateway refuses a sign-in reply that names a different issuer. See [egress credential vault](../egress-credential-vault.md#setting-expected_issuer).
- **Optional: consent prompt for new clients.** With `EGRESS_CONSENT_USE_ELICITATION=true`, a 2026-07-28 client that supports URL prompts gets the "connect your account" step as a prompt it can show the user, then retries on its own. Without the setting, every client gets a tool result with the link, as before.

## Troubleshooting

**A user gets HTTP 403 on a call that worked before the upgrade.** Search the auth-server log for `Access denied` with that user's server and method, then add the method as described in [Check your scopes](#check-your-scopes).

**A server shows 0 tools in the registry, but users can call its tools.** The server requires each user to sign in, and the registry's background fetch has no user. In the server's edit form, turn on **Discovery Identity (OAuth 2.1)**, save, then press **Connect account for discovery**, so the registry can list tools and run security scans with that account. Until then, avoid pressing Rescan: a scan that cannot authenticate fails, and a failed scan disables the server.

**A 2026-07-28 client fails on a direct route.** The backend speaks only the older protocol. Use the client in auto mode, or reach the server through a virtual server, which translates.

**A client gets HTTP 400 with error `-32020`.** Its headers name a different method, tool or protocol version than its request body. Fix the client; the gateway refuses such requests before authorization on purpose.

**A server reads `unhealthy: host could not be resolved`.** The server's host name does not resolve from the registry. Check that the container or service is running and that the name in its URL is right.

## Related documentation

- [MCP 2026-07-28 support](../mcp-2026-07-28-migration.md): each changelog entry with how the gateway handles it
- [1.37.0 release notes](../release-notes/1.37.0.md)
- [Scopes](../scopes.md)
- [Egress credential vault](../egress-credential-vault.md)
