# MCP Apps

[MCP Apps](https://apps.extensions.modelcontextprotocol.io/) is the first official MCP extension. It lets a tool open a small interactive web page, such as a chart, a form or a diagram, that the chat client shows inside the conversation. The user clicks and types in the page, and the page can call more tools on the same server. The stable spec is dated [2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx).

The chat client (the "host") renders the page. The gateway never renders it. Its job is to pass the right MCP messages through unchanged and to apply the user's permissions to each one.

## Support at a glance

| Route or feature | Status |
|---|---|
| Direct server routes (`/<server>/mcp`) | Supported |
| Virtual servers (`/virtual/<id>/mcp`) | Not supported yet |
| Registry UI: MCP APP badge on the server card and on each tool | Supported |
| Pages that set `_meta.ui.domain` (Claude only) | Supported on direct routes; the gateway rewrites the value, see below |

## How it works

A server that ships an MCP App does three things:

1. It publishes the page as an MCP resource with a `ui://` address and the MIME type `text/html;profile=mcp-app`.
2. It points a tool at that page with `_meta.ui.resourceUri`.
3. It checks whether the client declared the `io.modelcontextprotocol/ui` extension when it connected, and lists its UI tools only if it did.

When the model calls the tool, the host fetches the page with `resources/read`, shows it in a sandboxed iframe, and passes it the tool's result. The page talks back to the host over `postMessage`, and the host sends any tool call the page asks for to the server as an ordinary `tools/call`.

On a direct route the gateway forwards all of this unchanged: the host's `initialize` with its extension list, the tool `_meta`, `resources/read` for the page, and every `tools/call` from the page. Each request goes through the same access check as any other MCP request.

## What users need

A user needs `resources/read` in the methods their scope grants for the server. Without it the tool still runs, but the host cannot fetch the page and the user sees text only. The page's own tool calls need `tools/call` for those tools, like any other call.

```json
{
  "server": "budget-allocator",
  "methods": ["initialize", "notifications/initialized", "tools/list", "tools/call", "resources/list", "resources/read"],
  "tools": "*"
}
```

## What the registry shows

The health check connects to each server declaring `io.modelcontextprotocol/ui`, so a server that follows the spec lists its UI tools to the registry as it would to a host. For each tool that points at a page, the registry stores two fields in the tool list:

| Field | Meaning |
|---|---|
| `ui_resource_uri` | The `ui://` address of the page. The registry keeps it only if it starts with `ui://` and is at most 2,048 characters. |
| `ui_visibility` | Who may call the tool: `model`, `app`, or both. The spec default is both. |

The registry keeps nothing else from the tool's `_meta`. The server-level field `num_mcp_app_tools` counts the tools that have a page. It appears in `GET /api/servers`, in `GET /api/server_details/<path>`, and in `registry_management.py server-get`. When the response includes the tool list, the count covers only the tools the caller can see. With `include_tools=false` it is the count the health check stored.

## Search and app-only tools

Semantic search returns `ui_resource_uri` on each tool result and on each server's `matching_tools`, so an agent that finds a tool through `search_registry` can tell that it opens a page. The page lives on the same server as the tool: read it with `resources/read` on that server's route.

A tool whose `ui_visibility` is `["app"]` exists for its page to call, and the spec says the model must not see it. The registry leaves these tools out of the search index and out of the virtual-server tool picker, because both feed what a model reads. The tools modal on the server card still lists them, marked APP ONLY.

In the UI, a server with at least one such tool shows an **MCP APP** badge on its card. In the card's tool list, each tool with a page shows MCP APP, and a tool that only the page may call also shows APP ONLY.

The health check rewrites the stored tool list when the set of UI tools changes, even if the tool count stays the same. A server registered before this feature gets its fields on its next health check.

## Connecting from Claude

Claude renders MCP Apps on claude.ai, in Claude Desktop and on mobile, for connectors added by URL as a custom connector. In Claude Desktop, open **File > Settings > Customize > Connectors** and add a custom connector. Claude connects to a custom connector from Anthropic's servers, so the gateway URL must be reachable from the public internet. Claude Code shows the tool's text result and does not render the page.

Use the server's gateway URL, for example `https://<your-gateway>/budget-allocator/mcp`, and sign in when Claude asks.

## Known limitations

### Virtual servers

A virtual server does not support MCP Apps yet:

- The router opens its own sessions to each backend without the UI extension, so a backend that checks for it hides its UI tools.
- The virtual server's `initialize` reply does not advertise `resources` or the UI extension.
- A page calls tools by their backend names. A virtual server that renames a tool rejects that call, and an app-only tool has to be mapped like any other tool.
- Two backends that use the same `ui://` address collide, because `resources/read` goes to whichever backend lists it first.

### `ui.domain` in Claude

A page can set `_meta.ui.domain` to get a stable sandbox origin. Claude accepts only `<hash>.claudemcpcontent.com`, where `<hash>` is the first 32 hex characters of the SHA-256 of the connector URL ([Claude's troubleshooting guide](https://claude.com/docs/connectors/building/mcp-apps/troubleshooting)). A backend computes the value from its own URL, but behind the gateway the user connects to the gateway's URL, so Claude would refuse the page with "There was a problem displaying content".

On direct routes the gateway fixes this. When a `resources/read` reply carries a `ui.domain` ending in `.claudemcpcontent.com`, the auth server's proxy replaces it with the value for the URL the client connected to: the gateway's public origin, then the route path exactly as the client sent it (a trailing slash counts), then any query string. Values in any other format are left alone, and so is every other method. If the reply cannot be parsed, it is forwarded unchanged.

The public origin comes from configuration, never from request headers: `AUTH_SERVER_EXTERNAL_URL`, then `REGISTRY_URL`, then `settings.registry_url`. If none is set, the gateway does not rewrite the value. Two cases are not covered:

- A deployment reachable under more than one host name: the value matches the first configured origin only.
- A deployment that serves MCP routes under a path prefix: the origin drops the path, so the hash does not match.

After the rewrite the page runs under a different sandbox origin than the backend chose. A page that relies on that exact origin, for example for an OAuth callback or a CORS allowlist on the backend's own API, may need its backend to accept the gateway-derived origin as well.

### Servers that store state per MCP session

Claude can send the model's tool calls and the page's tool calls on different MCP sessions. A server that keys its state on the session (for example, a game that stores the board per session) loses track of it, with or without the gateway in between. This comes from how that server is built, not from the gateway.

### `subscriptions/listen`

The gateway answers the 2026-07-28 `subscriptions/listen` method with "method not found". A page that waits for change notifications does not get them. See [MCP 2026-07-28 support](mcp-2026-07-28-migration.md).

## Servers to test with

These public servers serve MCP Apps and need no sign-in. They are run by others and may change or go away.

| Server | URL | Notes |
|---|---|---|
| Budget Allocator (official ext-apps example, hosted by a community gallery) | `https://openwork-mcp-app-gallery-fable.vercel.app/apps/budget-allocator/mcp` | Stateless. Renders through the gateway in Claude. |
| draw.io | `https://mcp.draw.io/mcp` | Sets `ui.domain`, so it exercises the rewrite above. |

The [ext-apps repository](https://github.com/modelcontextprotocol/ext-apps/tree/main/examples) has more examples you can run yourself.
