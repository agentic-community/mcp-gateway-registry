# MCP 2026-07-28 support

The MCP Gateway Registry serves clients of MCP [2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) and clients of earlier revisions (`2025-11-25` and before) on the same URLs. This page goes through every entry in the [2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) and says what the gateway does about it.

Each entry has a status, a **100 level** explanation (what it means in plain words) and a **200 level** explanation (how the gateway does it, with the files involved).

## Terms used on this page

- **Handshake-era client or server:** one that speaks `2025-11-25` or earlier. It opens with `initialize` and sends an `Mcp-Session-Id` on every call.
- **2026-07-28 client or server:** one that speaks the new revision. Every request stands alone and carries its protocol version in `params._meta`.
- **Dual-era server:** one that speaks both, on the same endpoint.
- **Direct route:** one registered server behind one URL, for example `/github/mcp`. The gateway checks access, then forwards the request unchanged, so the client and the server must share a protocol version.
- **Virtual route:** a virtual server, for example `/virtual/dev-essentials/mcp`, that combines tools from several servers. The Lua router (`docker/lua/virtual_router.lua`) reads each request and talks to each backend in the protocol that backend speaks.
- **Pre-consent:** the state before a user connects their own account for a server that uses per-user egress OAuth (for example GitHub or Slack). The gateway answers some requests itself until the user connects.

## At a glance

| Changelog entry | Status |
|---|---|
| Major 1. No protocol-level sessions | Supported |
| Major 2. No `initialize`; version and capabilities in `_meta` | Supported |
| Major 3. `server/discover` | Supported |
| Major 4. `subscriptions/listen` | Not supported yet (clean error) |
| Major 5. `ping`, `logging/setLevel` removed; log level in `_meta` | Supported |
| Major 6. Tasks move to an extension | Direct routes: passed through. Virtual routes: not offered |
| Major 7. Multi round-trip requests | Supported |
| Major 8. `resultType` on every result | Supported |
| Major 9. No SSE resumability | No change needed |
| Minor 1. `extensions` capability field | Passed through |
| Minor 2. OpenTelemetry trace context in `_meta` | Passed through |
| Minor 3. Deterministic `tools/list` order | Supported |
| Minor 4. `Mcp-Method`, `Mcp-Name`, `x-mcp-header` | Supported |
| Minor 5. `ttlMs` and `cacheScope` on cacheable results | Partly supported (see [known gaps](#known-gaps)) |
| Minor 6. Resource not found is `-32602` | Partly supported (see [known gaps](#known-gaps)) |
| Minor 7. RFC 9207 `iss` check | Supported (opt-in per server) |
| Minor 8. `application_type` in client registration | Supported |
| Minor 9. Credentials bound to their authorization server | Supported |
| Minor 10. Any JSON Schema 2020-12 keyword in tool schemas | Supported |
| Minor 11. `elicitationId` and `notifications/elicitation/complete` removed | Supported |
| Minor 12. Error code renumbering | Supported |
| Deprecated 1. Roots, Sampling, Logging | Passed through |
| Deprecated 2. HTTP+SSE transport | Still routed; warning on SSE-only registration |
| Deprecated 3. `includeContext` values | Passed through |
| Deprecated 4. Dynamic Client Registration | Both CIMD and DCR available |

## Major changes

### Major 1. Remove protocol-level sessions and `Mcp-Session-Id`

**Status:** supported.

**100 level.** A 2026-07-28 client never opens a session, so it has no session id to send. The gateway serves it without one on both route types, and handshake-era clients keep using sessions as before.

**200 level.** On a direct route the proxy holds no session state; it forwards `Mcp-Session-Id` when a handshake-era client sends one. On a virtual route the router recognizes a 2026-07-28 request by its `_meta` protocol version or a `2026-` `MCP-Protocol-Version` header and stops requiring a session. To reach handshake-era backends, which still need one, the router creates its own client session for that user and virtual server, caches it, and checks it on every use. The client never sees it. A 2026-07-28 backend gets no session at all. The router ignores any `Mcp-Session-Id` a 2026-07-28 client sends.

### Major 2. No `initialize` handshake; version and capabilities travel in `_meta`

**Status:** supported.

**100 level.** A 2026-07-28 client puts its protocol version, its name and what it can do into every request. The gateway reads them from there when it needs them.

**200 level.** The virtual router reads `io.modelcontextprotocol/protocolVersion` from `params._meta`, requires the `MCP-Protocol-Version` header to match it, and answers an unknown version with HTTP 400 and `-32022`, listing the versions it supports. When it calls a 2026-07-28 backend it copies the client's whole `_meta` (client info, capabilities, progress token, log level, trace context) and adds what is missing. The auth server's metrics middleware (`auth_server/metrics_middleware.py`) reads the client name and protocol version from `_meta` on every request, since there is no `initialize` to read them from. The router's own `server/discover` answer names the gateway in `io.modelcontextprotocol/serverInfo`; results forwarded from a backend keep whatever `serverInfo` that backend sent.

A handshake-era client opens a session, then sends the session id with each call:

```http
POST /virtual/dev-essentials/mcp
Content-Type: application/json

{"jsonrpc": "2.0", "id": 1, "method": "initialize",
 "params": {"protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "my-client", "version": "1.0"}}}
```

```http
POST /virtual/dev-essentials/mcp
Mcp-Session-Id: vs-6f1c...
MCP-Protocol-Version: 2025-11-25

{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
```

A 2026-07-28 client sends the same call on its own, with the version and client details in `_meta` and mirrored into headers:

```http
POST /virtual/dev-essentials/mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/list

{"jsonrpc": "2.0", "id": 2, "method": "tools/list",
 "params": {"_meta": {
   "io.modelcontextprotocol/protocolVersion": "2026-07-28",
   "io.modelcontextprotocol/clientInfo": {"name": "my-client", "version": "1.0"},
   "io.modelcontextprotocol/clientCapabilities": {}}}}
```

A version the router does not know gets HTTP 400:

```json
{"jsonrpc": "2.0", "id": 2,
 "error": {"code": -32022, "message": "Unsupported protocol version: 2027-01-01",
           "data": {"supported": ["2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"],
                    "requested": "2027-01-01"}}}
```

### Major 3. `server/discover`

**Status:** supported.

**100 level.** Servers must answer a new question, "which versions do you speak and what can you do". Virtual servers answer it themselves. The registry also asks it of every backend, to learn which protocol each one speaks.

**200 level.**
- On a virtual route the router answers `server/discover` itself: `2026-07-28` plus the handshake-era versions it bridges, the `tools`, `resources` and `prompts` capabilities, and its `serverInfo`.
- On a direct route the request goes to the backend, which answers for itself. A handshake-era backend answers "method not found", and a client in auto mode falls back to `initialize`.
- Before a user has connected their account for an egress server, the gateway answers `server/discover` with `-32601` on purpose. Clients in auto mode, the SDK default, then fall back to `initialize`, and the gateway's local handshake answer carries the connect instructions.
- The router sends `server/discover` to each backend before its first call and caches the backend's protocol era for 5 minutes per backend.
- The health check (`registry/health/service.py`) sends `server/discover` first. A `supportedVersions` answer or a 2026-07-28 error code marks a live 2026-07-28 server. Anything else falls back to the `initialize` and `ping` check, and the health check remembers a handshake-era answer for an hour so later checks skip the extra request.
- Tool discovery (`registry/core/mcp_client.py`) uses the MCP SDK 2.x client in auto mode, which asks `server/discover` and falls back to `initialize`.


The virtual router's answer:

```json
{"jsonrpc": "2.0", "id": 1,
 "result": {
   "resultType": "complete",
   "supportedVersions": ["2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"],
   "capabilities": {"tools": {"listChanged": false}, "resources": {}, "prompts": {}},
   "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "mcp-gateway-virtual-server", "version": "1.0.0"}}}}
```

The gateway's answer before a user has connected an egress account. A handshake-era backend answers with the same code, in its own words:

```json
{"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}}
```

### Major 4. `subscriptions/listen` replaces the GET stream and `resources/subscribe`

**Status:** not supported yet.

**100 level.** A 2026-07-28 client can keep one connection open to hear about changes, such as a tool list that changed. The gateway cannot hold such a connection open yet, so it answers "method not found". Clients carry on without pushed updates and see changes the next time they ask.

**200 level.** The direct-route proxy answers `subscriptions/listen` with `-32601` before contacting the backend, because it buffers each response and an open-ended stream would hold a pooled connection forever (#1890). The virtual router answers it as an unknown method (404 `-32601`). Supporting it needs a streaming relay on both route types, which also changes how progress notifications and long calls work; that is a separate design.


The direct-route answer:

```json
{"jsonrpc": "2.0", "id": 5,
 "error": {"code": -32601, "message": "Method not found: subscriptions/listen",
           "data": {"detail": "This method is not supported through the gateway proxy."}}}
```

### Major 5. `ping`, `logging/setLevel` and `notifications/roots/list_changed` removed; log level per request

**Status:** supported.

**100 level.** These requests no longer exist in 2026-07-28. Virtual servers answer them as unknown for 2026-07-28 clients and keep answering `ping` for older clients. A per-request log level reaches the backend unchanged.

**200 level.** The virtual router answers `ping` and any other method 2026-07-28 does not define with HTTP 404 and `-32601` when the client speaks 2026-07-28, and keeps the handshake-era behavior otherwise. `io.modelcontextprotocol/logLevel` is part of the client `_meta` the router copies to 2026-07-28 backends. On direct routes all of these pass through to the backend.


`ping` from a 2026-07-28 client on a virtual route, answered with HTTP 404:

```json
{"jsonrpc": "2.0", "id": 9, "error": {"code": -32601, "message": "Method not found: ping"}}
```

### Major 6. Tasks move to the `io.modelcontextprotocol/tasks` extension

**Status:** passed through on direct routes; not offered on virtual routes.

**100 level.** Long-running "tasks" are now an optional extension. On a direct route the gateway forwards task requests to the server. Virtual servers do not offer the extension, so clients do not use it there.

**200 level.** The direct-route proxy forwards `tasks/get` and `tasks/update` like any other method, subject to the scope's method list. The virtual router does not list the tasks extension in its `server/discover` capabilities, and it answers `tasks/*` as an unknown method.

### Major 7. Multi round-trip requests replace server-initiated requests

**Status:** supported.

**100 level.** A server that needs something more, such as a user's answer, now replies "input required" with what it needs, and the client sends the same request again with the answer. The gateway forwards these exchanges unchanged. It can also use the same form to ask a 2026-07-28 user to connect an account.

**200 level.**
- Both route types forward an `input_required` result and the retry with its `inputResponses` and `requestState` unchanged, so the backend's own checks on its `requestState` still apply.
- For an egress server before consent, with `EGRESS_CONSENT_USE_ELICITATION=true`, a `tools/call`, `prompts/get` or `resources/read` from a 2026-07-28 client that declares `elicitation.url` gets an `input_required` result holding one URL-mode `elicitation/create` with the connect link, and a gateway `requestState` that starts with `mcpgw-egress-consent.`.
- On the retry, `_strip_gateway_consent_retry()` in `auth_server/server.py` removes the gateway's `requestState` and `inputResponses` before the gateway authorizes and forwards the body, so it checks the same bytes it sends and the upstream never sees a prompt it did not send. A declined or cancelled prompt ends with an `isError` result carrying the link.
- With the setting off (the default), every client gets the `isError` result with the link, as before. See [egress credential vault](egress-credential-vault.md).


The consent prompt a 2026-07-28 client gets:

```json
{"jsonrpc": "2.0", "id": 7,
 "result": {
   "resultType": "input_required",
   "inputRequests": {
     "egress_connect": {
       "method": "elicitation/create",
       "params": {"mode": "url",
                  "message": "Connect your github account to use this server.",
                  "url": "https://gateway.example.com/oauth2/egress/connect?server=%2Fgithub&elicitationId=..."}}},
   "requestState": "mcpgw-egress-consent.Qx3..."}}
```

The client's retry, and what the gateway forwards after removing its own two fields:

```json
{"method": "tools/call",
 "params": {"name": "search_issues", "arguments": {"q": "bug"},
            "requestState": "mcpgw-egress-consent.Qx3...",
            "inputResponses": {"egress_connect": {"action": "accept"}},
            "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}}
```

```json
{"method": "tools/call",
 "params": {"name": "search_issues", "arguments": {"q": "bug"},
            "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}}
```

### Major 8. Every result carries `resultType`

**Status:** supported.

**100 level.** Every answer must now say whether it is complete or still needs input. The gateway adds that label wherever it writes or reshapes an answer for a 2026-07-28 client.

**200 level.**
- Lists the virtual router builds carry `resultType: "complete"`, and so does its `server/discover` answer.
- Results the router forwards from a handshake-era backend to a 2026-07-28 client gain `resultType: "complete"`, in a JSON body or in SSE `data:` lines. The router re-encodes them with a cjson instance that keeps `[]` as an array. An existing `resultType`, an error and a body that is not JSON pass through unchanged.
- The auth server's own pre-consent answers (lists, `ping`, the connect tool's result, the consent result, the missing-PAT result) carry `resultType: "complete"` when the request is 2026-07-28.
- On a direct route the backend's results pass through as the backend wrote them.


A handshake-era backend's `tools/call` result, and what the router sends to a 2026-07-28 client:

```json
{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "hi"}], "isError": false}}
```

```json
{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "hi"}], "isError": false, "resultType": "complete"}}
```

### Major 9. No SSE resumability or `Last-Event-ID`

**Status:** no change needed.

**100 level.** The gateway never resumed broken streams, so nothing changes.

**200 level.** The direct-route proxy and the virtual router collect each backend response before replying and never handled `Last-Event-ID`.

## Minor changes

### Minor 1. `extensions` field in client and server capabilities

**Status:** passed through.

**100 level.** Capabilities can now name optional extensions. The gateway forwards them and does not offer any on virtual servers.

**200 level.** Client capabilities in `_meta` reach 2026-07-28 backends unchanged. The router's `server/discover` answer lists no extensions.

### Minor 2. OpenTelemetry trace context in `_meta`

**Status:** passed through.

**100 level.** Clients can put tracing information into a request. It reaches the backend unchanged.

**200 level.** `traceparent`, `tracestate` and `baggage` are part of the client `_meta` the router copies to 2026-07-28 backends. Direct routes forward the body unchanged. The gateway does not start its own spans from these keys.

### Minor 3. Deterministic `tools/list` order

**Status:** supported.

**100 level.** A tool list comes back in the same order every time, which helps clients cache it.

**200 level.** The virtual router builds `tools/list` in the order of the virtual server's mapping file. On a direct route, per-user filtering keeps the backend's order.

### Minor 4. `Mcp-Method`, `Mcp-Name` and `x-mcp-header` request headers

**Status:** supported.

**100 level.** A 2026-07-28 client copies the method and the tool name into HTTP headers. The gateway checks that the headers and the body agree, because it authorizes the body while a server may act on the headers. On virtual servers it also sets these headers itself when it calls a 2026-07-28 backend.

**200 level.** `docker/lua/capture_body.lua` runs before authorization on every MCP route and compares `Mcp-Method` with the body's method, `Mcp-Name` (base64 form decoded) with `params.name` or `params.uri`, and `MCP-Protocol-Version` with the `_meta` version. A mismatch gets HTTP 400 and `-32020`. nginx skips the check for requests without these headers, which covers every handshake-era client. For 2026-07-28 backends the router sets the headers from the exact body it sends and adds `Mcp-Param-*` headers from the backend's `x-mcp-header` tool annotations, dropping any tool whose annotation breaks the spec's rules. See [scopes](scopes.md#mcp-2026-07-28-request-headers).


A request whose header names a different tool than its body, answered with HTTP 400:

```http
POST /github/mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: list_issues

{"jsonrpc": "2.0", "id": 4, "method": "tools/call",
 "params": {"name": "delete_repository", "arguments": {}, "_meta": {...}}}
```

```json
{"jsonrpc": "2.0", "id": 4, "error": {"code": -32020, "message": "Header mismatch: Mcp-Name does not match params.name"}}
```

A tool whose schema marks a parameter with `x-mcp-header`, and the header the router adds when it calls a 2026-07-28 backend:

```json
{"name": "get_weather",
 "inputSchema": {"type": "object",
                 "properties": {"region": {"type": "string", "x-mcp-header": "Region"}}}}
```

```http
Mcp-Param-Region: us-east-1
```

### Minor 5. `ttlMs` and `cacheScope` on cacheable results

**Status:** partly supported.

**100 level.** Lists and resource reads now say how long a client may reuse them and whether a shared cache may store them. When the gateway filters a list per user, it marks the list private.

**200 level.** Lists the virtual router builds for 2026-07-28 clients carry `cacheScope: "private"` and `ttlMs: 60000`. On a direct route, the auth server rewrites `cacheScope: "public"` to `"private"` on a `tools/list` it filters per user. Two cases are not covered yet; see [known gaps](#known-gaps).


A list the virtual router builds for a 2026-07-28 client:

```json
{"jsonrpc": "2.0", "id": 2,
 "result": {"tools": [...], "resultType": "complete", "cacheScope": "private", "ttlMs": 60000}}
```

### Minor 6. Resource not found is `-32602`

**Status:** partly supported.

**100 level.** The error code for a missing resource changed. Direct routes pass the backend's own error through. Virtual servers still use the old code; see [known gaps](#known-gaps).

**200 level.** The virtual router answers an unknown tool, resource or prompt with `-32601` for every client.

### Minor 7. RFC 9207 `iss` in authorization responses

**Status:** supported, opt-in per server.

**100 level.** When the gateway signs a user in to a provider such as GitHub, the provider's reply can name who issued it. An operator can record the expected issuer for a server, and the gateway then refuses a reply that names a different one.

**200 level.** `expected_issuer` is an optional field on a server's egress and discovery OAuth configuration (API, CLI and server forms). The gateway records it in the encrypted consent state when consent begins. The shared callback `/oauth2/egress/callback` reads `iss`, and `_check_response_issuer()` in `registry/egress_auth/service.py` refuses a reply whose `iss` differs, before the code exchange. The callback accepts a reply without `iss`. The user login flow gives each identity provider its own callback URL, which RFC 9207 accepts as a defense on its own. See [egress credential vault](egress-credential-vault.md#setting-expected_issuer).


The field in a `POST /api/servers/{server_path}/egress-auth` body:

```json
{"egress_auth_mode": "oauth_user",
 "egress_provider": "custom",
 "client_id": "my-client",
 "client_secret": "YOUR_CLIENT_SECRET",
 "custom_authorize_url": "https://idp.example.com/authorize",
 "custom_token_url": "https://idp.example.com/token",
 "expected_issuer": "https://idp.example.com"}
```

### Minor 8. `application_type` in client registration

**Status:** supported.

**100 level.** A client that registers itself with a sign-in server must say what kind of application it is. The registry's own client description says it is a web application.

**200 level.** The registry's Client ID Metadata Document at `/oauth/client-metadata.json` (off by default) carries `"application_type": "web"`. MCP clients that register with the gateway's identity provider send their own `application_type`, which the provider handles.


```json
{"client_id": "https://gateway.example.com/oauth/client-metadata.json",
 "client_name": "AI Registry Tools",
 "client_uri": "https://gateway.example.com",
 "redirect_uris": ["https://gateway.example.com/oauth2/egress/callback"],
 "grant_types": ["authorization_code", "refresh_token"],
 "response_types": ["code"],
 "token_endpoint_auth_method": "none",
 "scope": "...",
 "application_type": "web"}
```

### Minor 9. Credentials bound to the authorization server that issued them

**Status:** supported.

**100 level.** A client may use a saved credential only with the sign-in server that issued it. The gateway records where each egress credential came from and stops using it if the server's sign-in settings change.

**200 level.** Each stored egress credential records `bound_token_url`, the provider's token endpoint captured when consent began. The callback refuses a code if the endpoint changed during consent, and every vend and refresh refuses a credential whose server now uses a different token endpoint, which sends the user back through consent.

### Minor 10. Any JSON Schema 2020-12 keyword in `inputSchema` and `outputSchema`

**Status:** supported.

**100 level.** Tool argument descriptions may now use the full JSON Schema language. The gateway keeps them intact when it passes them on.

**200 level.** The schema normalization used for virtual servers (`_ensure_mcp_compliant_schema` in `registry/core/nginx_service.py` and `_ensure_mcp_schema` in the router) leaves an object schema alone, narrows a type list such as `["object", "null"]` to `"object"` (the root type must be `"object"`), and moves `$defs` and `definitions` to the root when it has to wrap a non-object schema, so `#/$defs/...` references still resolve. Tools found by health-loop discovery keep their schema whichever key stored it (`registry/utils/tool_schema.py`). The gateway does not resolve `$ref`; clients do.


A root type list, before and after:

```json
{"type": ["object", "null"], "properties": {"q": {"type": "string"}}}
```

```json
{"type": "object", "properties": {"q": {"type": "string"}}}
```

A non-object schema the gateway has to wrap, before and after. The definitions move to the root so `#/$defs/Item` still resolves:

```json
{"type": "array", "items": {"$ref": "#/$defs/Item"}, "$defs": {"Item": {"type": "string"}}}
```

```json
{"type": "object",
 "properties": {"value": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
 "$defs": {"Item": {"type": "string"}}}
```

### Minor 11. `elicitationId` and `notifications/elicitation/complete` removed

**Status:** supported.

**100 level.** In the new revision a client learns that an outside step, such as signing in, finished by retrying its request. The gateway's 2026-07-28 consent prompt works that way. Older clients keep the older form.

**200 level.** The `input_required` consent prompt has no `elicitationId` field and the gateway sends no completion notification; its own `requestState` marks the retry. The connect link still carries an `elicitationId` query parameter, which the gateway's connect page uses. The `-32042` URL elicitation for `2025-11-25` clients keeps its `elicitationId`, which that revision defines.

### Minor 12. Error code renumbering

**Status:** supported.

**100 level.** Some error codes moved to new numbers. The gateway uses the new numbers when it answers 2026-07-28 clients.

**200 level.** Header mismatch is `-32020` (nginx and the virtual router) and unsupported protocol version is `-32022` (virtual router). The gateway never sends `-32021` itself. The health check treats any of `-32020`, `-32021` and `-32022` as proof of a live 2026-07-28 server. The gateway sends the `-32042` URL elicitation only to handshake-era clients.

## Deprecated features

### Deprecated 1. Roots, Sampling and Logging

**Status:** passed through.

**100 level.** These features still work during the deprecation window. The gateway does not depend on them and forwards them when a client and server use them.

**200 level.** Requests and notifications for these features pass through direct routes, and the multi round-trip forms pass through virtual routes as described under Major 7.

### Deprecated 2. HTTP+SSE transport

**Status:** still routed; warning on SSE-only registration.

**100 level.** The spec now deprecates the older HTTP+SSE way of connecting. The gateway still supports it, and registering a server that offers only that transport now writes a warning to the log.

**200 level.** `register_server` logs a warning when `supported_transports` is exactly `["sse"]`, naming 2027-07-28 as the earliest date the gateway may drop support (the spec's twelve-month deprecation window). See [API reference](api-reference.md).

### Deprecated 3. `includeContext` values `"thisServer"` and `"allServers"`

**Status:** passed through.

**100 level.** This is a sampling setting agreed between a client and a server.

**200 level.** The gateway forwards it without reading or changing it.

### Deprecated 4. Dynamic Client Registration in favor of Client ID Metadata Documents

**Status:** both available.

**100 level.** Clients should identify themselves with a published description document. Registering a new client at each sign-in is deprecated. The registry publishes such a document for itself, and MCP clients can still register dynamically with the gateway's identity provider.

**200 level.** The registry serves its CIMD document at `/oauth/client-metadata.json` when the feature is on. Dynamic Client Registration for MCP clients stays available through the identity provider; see [Keycloak MCP client guide](keycloak-mcp-clients.md).

## Other changes in the release

The schema fix for `number` types and the governance and process changes (the feature lifecycle policy and the SEP workflow) need no gateway change.

## Gateway components

| Component | Protocol support |
|---|---|
| Virtual router (`docker/lua/virtual_router.lua`) | Serves both revisions; bridges a 2026-07-28 client to handshake-era backends and talks 2026-07-28 to backends that support it |
| Direct-route proxy (`auth_server/server.py`) | Forwards both revisions unchanged; header and body checks in `capture_body.lua` |
| Registry tool discovery (`registry/core/mcp_client.py`) | MCP SDK 2.x, auto mode for streamable HTTP, handshake mode for HTTP+SSE |
| Health check (`registry/health/service.py`) | `server/discover` first, then `initialize` and `ping` |
| Security scanner | `cisco-ai-mcp-scanner` 4.8.7 with a fallback patch for handshake-era servers |
| mcpgw (`servers/mcpgw`) | fastmcp 4 on MCP SDK 2.3, serves both revisions |
| CLI (`cli/mcp_utils.py`) | Handshake-era client; works against every route |

## Known gaps

- **`subscriptions/listen` and live progress.** They need a streaming relay on both route types, which has its own design.
- **`ttlMs` and `cacheScope` in two places.** The auth server's pre-consent lists and `resources/read` results the virtual router forwards from handshake-era backends lack them. A client that checks for them rejects those answers.
- **Not-found error code on virtual routes.** The router answers `-32601` where 2026-07-28 uses `-32602`.
- **One place for protocol rules.** Several files each apply the result rules and keep their own protocol constants; [#1997](https://github.com/agentic-community/mcp-gateway-registry/issues/1997) moves them into one helper and one constants module per language.

## Upgrade notes

- A scope with `tools: ["*"]` now grants every tool for `tools/call` and no other method. A custom scope that relied on the old behavior for a method it does not list, such as `resources/read` or `prompts/get`, gets 403 for that method until someone adds it. Any scope that grants access to a server allows `server/discover` and `notifications/*` on it. See [scopes](scopes.md).
- The gateway now authorizes MCP request bodies up to 1 MB. Tool calls above about 10 KB used to fail with a 500.

## Testing against a live gateway

- `tests/e2e/mcp_client_matrix.py` calls one tool per route with a handshake-era client and with the MCP SDK 2.x in legacy, auto and 2026-07-28 modes.
- `tests/e2e/mcp_2026_gateway_checks.py` checks `server/discover`, header and body mismatch rejection, `cacheScope`, and the 401 and 403 challenges.

Both exit non-zero on an unexpected result. Register the upstream servers they use from `cli/examples/` (`mcp-everything.json`, `pingmcp-modern.json`, `pingmcp-dual.json`, `virtual-server-mcp-compat.json`).

## Related documentation

- [Virtual MCP server design](design/virtual-mcp-server.md)
- [Egress credential vault](egress-credential-vault.md)
- [Scopes](scopes.md)
- [Unified configuration reference](unified-parameter-reference.md)
