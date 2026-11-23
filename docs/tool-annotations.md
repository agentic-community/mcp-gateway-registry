# MCP Tool Annotations

An MCP server can describe how each of its tools behaves. The registry stores that description when it reads the server's `tools/list`, and returns it wherever it returns the tool. Agents and MCP clients use it to decide, for example, whether to ask a person before they call a tool.

## What the registry stores

For each tool, the registry keeps three optional fields from `tools/list`, under their MCP names:

| Field | Type | What it says |
|-------|------|--------------|
| `title` | string | A display name for the tool |
| `annotations` | object | Behavior hints (below) |
| `outputSchema` | object | JSON Schema for the tool's structured result |

`annotations` can hold any of these keys:

| Key | Type | Meaning when true |
|-----|------|-------------------|
| `readOnlyHint` | boolean | The tool does not change anything |
| `destructiveHint` | boolean | The tool can delete or overwrite data |
| `idempotentHint` | boolean | Calling it twice with the same arguments has the same effect as calling it once |
| `openWorldHint` | boolean | The tool talks to systems outside the server (the web, a third-party API) |
| `title` | string | A display name, as above |

See the [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) for the full definitions.

The registry stores what the server sent and nothing more:

- An explicit `false` stays `false`.
- A hint the server left out stays out. The registry does not fill in the spec's defaults, so a missing `destructiveHint` means "not declared", not "destructive".
- A tool with no hints has no `annotations` key at all.
- The registry drops keys it does not recognize and values of the wrong type. A string `"true"` is not a boolean, so it is dropped.

## Annotations are advisory

The server writes these hints, and the registry does not check them. A server can mark a tool that deletes data as `readOnlyHint: true`. The registry never uses annotations for access control, tool blocking, or any other decision.

If you build an agent that skips confirmation for read-only tools, only trust `readOnlyHint` from servers you trust. Treat any tool you cannot show to be read-only as one that changes data.

## When the registry reads them

The registry reads annotations every time it fetches a server's tools:

- when the server is registered,
- on the first health check after the registry starts,
- when the server becomes healthy again after being unhealthy,
- when someone opens the server's tool list and the registry fetches it live (`GET /api/servers/{path}/tools`),
- on a schedule, for servers that stay healthy, when `TOOL_REFRESH_ENABLED` is on (see [Tool list refresh](tool-list-refresh.md)).

A fetch rewrites the stored tool list when any tool's name, description, input schema, `title`, `annotations` or `outputSchema` changed, even if the number of tools did not. The registry compares one hash over the whole list. Servers registered before the registry kept annotations pick them up on the first health check after an upgrade.

## Where they appear

### Server and tool APIs

`GET /api/servers?include_tools=true`, `GET /api/servers/{path}` and `GET /api/servers/{path}/tools` return each tool's stored fields:

```json
{
  "name": "delete_note",
  "description": "Delete one note by id.",
  "inputSchema": {"type": "object", "properties": {"note_id": {"type": "string"}}, "required": ["note_id"]},
  "annotations": {"readOnlyHint": false, "destructiveHint": true},
  "outputSchema": {"type": "object", "additionalProperties": {"type": "string"}}
}
```

### Virtual server tool list

`GET /api/virtual-servers/{path}/tools` returns each resolved tool with the backend tool's `annotations` (null when it has none), including for a tool renamed with an alias.

### Search

`POST /api/search/semantic` adds `annotations` to each entry in `tools[]` and to each entry in a server's `matching_tools[]`. The value is `null` when the server declared none. `outputSchema` and `title` are not part of search results; read them from the server API.

```json
{
  "server_path": "/notes",
  "tool_name": "delete_note",
  "inputSchema": {"type": "object", "properties": {"note_id": {"type": "string"}}},
  "annotations": {"readOnlyHint": false, "destructiveHint": true},
  "relevance_score": 0.91
}
```

The `search_registry` tool on the [AI registry tools server](ai-registry-tools.md) returns the same `tools` and `servers` arrays, so an agent that searches through MCP sees the same field.

An annotation in a search result is data about a tool, not an annotation the MCP client sees on a tool it can call. A client applies its own approval rules only to tools in a `tools/list` it received, so an agent that finds tools through search has to read `annotations` from the result and decide for itself.

### MCP `tools/list`

A direct connection to a server through the gateway returns the server's own `tools/list`, annotations included.

A [virtual MCP server](design/virtual-mcp-server.md) passes each backend tool's `title`, `annotations` and `outputSchema` through to its `tools/list`, including for renamed tools. When a backend cannot be reached, or the caller still has to connect an account, the virtual server answers from its stored copy, which carries the same fields.

### Registry UI

The tool list on a server card, and on a virtual server card, shows a `READ ONLY` badge when the server declares `readOnlyHint: true`, and a `DESTRUCTIVE` badge when it declares `destructiveHint: true` on a tool that is not read-only. Each badge's tooltip says the server declared it and the registry does not verify it.

### Command line

`registry_management.py server-search` prints a `Hints` line under each tool that declares any, and `--json` returns the raw `annotations` object:

```text
  cancel_booking (from /ai.autonomad-travel)
    Relevance: 88.00%
    Hints (declared by the server): not read-only, destructive, open-world
```

The CLI lists only the hints set to true, plus "not read-only" when a server says `readOnlyHint: false`.

## Example: the AI registry tools server

Every tool on the built-in [AI registry tools server](ai-registry-tools.md) (`/airegistry-tools/`) only reads from the registry. Each one declares a title and these annotations, so the server works as a quick check that a deployment stores and returns them:

```json
{"readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
```

## Troubleshooting

A tool shows no annotations:

1. Check what the server sends. Run `tools/list` against the server directly and look for an `annotations` object on the tool.
2. If the server sends them but the registry does not show them, trigger a fetch: open the server's tool list in the UI, or call `GET /api/servers/{path}/tools`.
3. Check the values' types. The registry drops a hint whose value is not a JSON boolean.
