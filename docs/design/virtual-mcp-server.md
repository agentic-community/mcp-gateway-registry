# Virtual MCP Server - Design Document

**Date**: 2026-02-10
**Status**: Implemented
**PR**: [#459](https://github.com/agentic-community/mcp-gateway-registry/pull/459)
**Issue**: [#129](https://github.com/agentic-community/mcp-gateway-registry/issues/129)

---

## 1. Overview

A Virtual MCP Server is a gateway-level construct that aggregates tools, resources, and prompts from multiple backend MCP servers into a single unified endpoint. Instead of connecting to individual MCP servers, clients connect to a virtual server that presents a curated, access-controlled view of capabilities drawn from any combination of registered backends.

### Problem Statement

Organizations deploying multiple MCP servers face several operational challenges:

- **Client complexity**: Each client must discover, connect to, and manage sessions with every individual MCP server it needs
- **Tool sprawl**: Teams cannot curate role-specific or project-specific tool bundles from existing servers
- **Naming conflicts**: Two backend servers may expose tools with the same name (e.g., both GitHub and GitLab expose a `search` tool)
- **Version drift**: No mechanism to pin a client to a specific backend server version while allowing others to upgrade
- **Access control gaps**: Authorization is all-or-nothing per server, with no per-tool granularity

### Solution

Virtual MCP Servers solve these problems by introducing a composition layer at the gateway:

```
                    +-----------------------+
                    |   Virtual MCP Server  |
                    |  /virtual/dev-tools   |
                    +-----------+-----------+
                                |
              +-----------------+-----------------+
              |                 |                 |
        +-----+-----+    +-----+-----+    +-----+-----+
        |  /github   |    |  /slack   |    |  /jira    |
        |  Backend   |    |  Backend  |    |  Backend  |
        +-----+-----+    +-----+-----+    +-----+-----+
              |                 |                 |
         search-repo       post-message      create-ticket
         create-pr         list-channels     search-issues
```

A client connecting to `/virtual/dev-tools` sees `search-repo`, `post-message`, `create-ticket`, `list-channels`, `create-pr`, and `search-issues` as a single flat tool list, regardless of which backend provides each tool.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| Tool aggregation | Merge tools from multiple backends into one endpoint |
| Tool aliasing | Rename tools to resolve conflicts or improve clarity |
| Version pinning | Lock a tool mapping to a specific backend server version |
| Scope-based access control | Server-level and per-tool scope requirements |
| Session multiplexing | One client session maps to N backend sessions transparently |
| Resource and prompt aggregation | Aggregate `resources/list` and `prompts/list` across backends |

---

## 2. Architecture

### System Context

```
+------------------------------------------------------------+
|                     MCP Gateway                             |
|                                                             |
|  +------------------+       +---------------------------+  |
|  |  FastAPI Registry |       |     Nginx Reverse Proxy   |  |
|  |  (Port 7860)      |       |     (Port 80/443)         |  |
|  |                    |       |                           |  |
|  |  - CRUD API        |<----->|  - Auth validation        |  |
|  |  - Tool catalog    |       |  - Location routing       |  |
|  |  - Session store   |       |  - Lua router execution   |  |
|  +------------------+       +-------------+-------------+  |
|                                           |                 |
|                              +------------+------------+    |
|                              | virtual_router.lua      |    |
|                              | - JSON-RPC dispatch     |    |
|                              | - Session multiplexing  |    |
|                              | - Tool aggregation      |    |
|                              | - Alias translation     |    |
|                              +------------+------------+    |
|                                           |                 |
+-------------------------------------------|----------------+
                                            |
                 +----------+----------+----------+
                 |          |          |          |
              Backend    Backend    Backend    Backend
              Server A   Server B   Server C   Server D
```

### Request Lifecycle

1. The client sends an MCP JSON-RPC request to `/virtual/{server-slug}/mcp`.
2. Nginx validates the caller and authorizes the virtual server and virtual tool name before Lua handles the request.
3. Lua loads `/etc/nginx/lua/virtual_mappings/{id}.json`, checks the virtual server's required scopes and any alias-level override, and rewrites an aliased `tools/call` to the backend's original tool name.
4. For every backing request, Lua explicitly captures `/_vs_auth_<backend>` with the rewritten JSON-RPC body. This verifies the same caller has the backing server's method/tool grant and returns a signed token bound to that registered backend and its resolved upstream. `ngx.location.capture` does not run `auth_request` on a captured backend location; the explicit check is required.
5. Lua calls `/_vs_backend_<backend>` only after that grant. A PAT or `oauth_user` backend routes through auth-server's `/mcp-proxy/<backend>/` vend/inject hop. Plain backends retain their direct proxy, with ingress credentials and internal tokens stripped. The resolved backend path, not the virtual alias, keys the egress vault.
6. Virtual `initialize` creates the client session locally. Backend `initialize` is performed when a backend is first used; both stateful session IDs and successful sessionless initialization are remembered per caller and mapped backend version. `tools/list` and other routed methods also require the backing grant.

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `virtual_server_models.py` | Pydantic data models for configuration, requests, and responses |
| `virtual_server_repository.py` | MongoDB persistence (CRUD on `virtual_servers` collection) |
| `virtual_server_service.py` | Business logic: validation, tool resolution, nginx reload coordination |
| `tool_catalog_service.py` | Aggregates available tools across all enabled backend servers |
| `virtual_server_routes.py` | REST API endpoints for management |
| `nginx_service.py` | Generates nginx location blocks, backend proxies, and Lua mapping files |
| `virtual_router.lua` | Runtime JSON-RPC routing, session management, tool aggregation |
| `backend_session_repository.py` | MongoDB persistence for backend session tracking |
| Frontend components | React management UI with multi-step wizard |

---

## 3. Data Model

### Virtual Server Configuration

The primary configuration document stored in MongoDB:

```python
class VirtualServerConfig:
    path: str                           # e.g., "/virtual/dev-tools"
    server_name: str                    # e.g., "Dev Tools"
    description: Optional[str]
    tool_mappings: List[ToolMapping]    # At least one required
    required_scopes: List[str]         # Server-level scope requirements
    tool_scope_overrides: List[ToolScopeOverride]
    tags: List[str]
    supported_transports: List[str]    # Default: ["streamable-http"]
    is_enabled: bool                   # Controls nginx routing
    num_stars: float                   # Average rating (0.0-5.0)
    rating_details: List[dict]         # Individual ratings [{user, rating}]
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime
```

### Tool Mapping

Each tool mapping connects a tool from a backend server to the virtual server:

```python
class ToolMapping:
    tool_name: str                     # Original tool name on backend
    alias: Optional[str]               # Renamed tool in virtual server
    backend_server_path: str           # e.g., "/github"
    backend_version: Optional[str]     # Pin to specific version
    description_override: Optional[str]
```

The effective tool name exposed to clients is `alias` if set, otherwise `tool_name`. This enables conflict resolution when two backends expose tools with the same name.

### Tool Scope Override

Per-tool access control layered on top of server-level scopes:

```python
class ToolScopeOverride:
    tool_alias: str                    # Matches alias or tool_name
    required_scopes: List[str]         # Additional scopes for this tool
```

### Backend Session

`backend_sessions` stores a client session ID, the internal `/_vs_backend_<path>` location (plus a pinned-version suffix when present), the authenticated owner, and a `stateless` flag. `backend_session_id` is a real ID for stateful backends and `null` only after a successful stateless `initialize`. MongoDB expires inactive records using `last_used_at`. The client-facing `Mcp-Session-Id` belongs to the virtual server; it is never forwarded as a backend session ID.

### Storage Design

| Collection | `_id` | Purpose |
|------------|-------|---------|
| `virtual_servers` | path (e.g., `/virtual/dev-tools`) | Virtual server configuration |
| `backend_sessions` | `{client_session_id}:{backend_location}[:version]` | Owner-bound stateful/stateless initialization with idle TTL |
| `client_sessions` | `{session_id}` | Client session metadata for audit |

Indexes on `virtual_servers`:
- `is_enabled` (for listing active servers)
- `tags` (for filtering)
- `server_name` (for search)
- Compound: `is_enabled` + `tags`

---

## 4. Session Management

### Two-Tier Caching

The Lua router keeps backend initialization state in the 30-second nginx shared-dictionary L1 cache and an owner-bound MongoDB L2 record with a one-hour idle TTL. The key includes the virtual client session, backing location, and pinned backend version; L1 also includes the authenticated owner. Every backing request still reauthorizes the rewritten method/tool before using either cache.

Mapping files serialize an unpinned `backend_version` as JSON `null`. Lua-cjson decodes that sentinel as userdata rather than Lua `nil`; the router normalizes it to an empty version before choosing a backend session key or setting `X-MCP-Server-Version`. Only a non-empty mapped string can select a pinned version.

A successful backend `initialize` stores either its `Mcp-Session-Id` or an explicit `stateless: true` state if no ID was returned. A JSON-RPC error, non-200 response, or missing result is not stateless success and is not cached. A stateless backend receives subsequent requests without `Mcp-Session-Id` and is not initialized again just because L1 expires; L2 records the successful sessionless state. Stateful sessions retain their session ID and can be reinitialized if the backend rejects a stale session. Auth failures do not trigger session retries.

### Session Lifecycle

1. Client `initialize` on the virtual endpoint mints its client-facing `vs-...` session ID.
2. Every routed request first checks the virtual grant and then checks the backing server grant on the rewritten method and original backend tool name.
3. On an L1/L2 miss, Lua authorizes and sends backend `initialize`; a successful stateful session ID or explicit stateless result is stored for that owner, backend and pinned version.
4. Only a stateful backend's session-related HTTP 400, 404, or 410 can invalidate its cached session and trigger one reinitialize. An authorization failure does not retry or become a stateless marker.

---

## 5. Nginx Configuration Generation

When a virtual server is created, updated, toggled, or deleted, the registry regenerates the nginx configuration. This process is serialized with an `asyncio.Lock` to prevent concurrent reloads.

### Generated Artifacts

For each enabled virtual server, the registry generates a virtual ingress location and JSON mapping file; each unique mapped backend also gets an internal authorization location and dispatch location:

**1. Location Block** (in `nginx.conf`):

```nginx
# Virtual MCP Server: Dev Tools
location /virtual/dev-tools/ {
    set $virtual_server_id "dev-tools";
    auth_request /validate;
    auth_request_set $auth_scopes $upstream_http_x_scopes;
    auth_request_set $auth_user $upstream_http_x_user;

    rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
    content_by_lua_file /etc/nginx/lua/virtual_router.lua;
}
```

**2. Internal Backend Locations** (one authorization and dispatch pair per unique backend):

```nginx
location = /_vs_auth_github {
    internal;
    set $backend_url "https://github-mcp.example.com/mcp";
    proxy_set_header X-Original-URL $scheme://$host/github/mcp;
    proxy_set_header X-Body $http_x_body;
    proxy_set_header X-Resolved-Upstream $backend_url;
    proxy_pass http://auth-server:8888/validate;
}
location = /_vs_backend_github {
    internal;
    proxy_set_header X-Internal-Token $http_x_internal_token;
    proxy_pass http://auth-server:8888/mcp-proxy/github/;
}
```

The second location above is for a credentialed backend; a plain backend's dispatch location proxies to its MCP endpoint directly and clears gateway credentials. Lua calls the authorization location explicitly before dispatch and refuses a response without a signed backend token.

**3. JSON Mapping File** (`/etc/nginx/lua/virtual_mappings/dev-tools.json`):

```json
{
  "required_scopes": ["mcp-access"],
  "tools": [
    {
      "name": "search-repo",
      "original_name": "search",
      "backend_location": "/_vs_backend_github",
      "backend_version": null,
      "egress_auth_mode": "pat",
      "description": "Search repositories",
      "required_scopes": ["github-access"],
      "inputSchema": { "type": "object", "properties": { "query": { "type": "string" } } }
    }
  ],
  "tool_backend_map": {
    "search-repo": {
      "original_name": "search",
      "backend_location": "/_vs_backend_github",
      "backend_version": null,
      "required_scopes": ["github-access"]
    }
  }
}
```

The mapping file is read by the Lua router at request time. It provides pre-computed lookup tables so the router does not need to query the registry API for tool metadata on every request.

---

## 6. Tool Aliasing and Version Pinning

### Tool Aliasing

Tool aliasing solves naming conflicts and improves tool discoverability:

```
Backend /github exposes:  search, create_pr, list_repos
Backend /gitlab exposes:  search, create_mr, list_projects
```

Without aliasing, both `search` tools would collide. With aliasing:

```json
{
  "tool_mappings": [
    { "tool_name": "search", "alias": "github-search", "backend_server_path": "/github" },
    { "tool_name": "search", "alias": "gitlab-search", "backend_server_path": "/gitlab" }
  ]
}
```

The client sees `github-search` and `gitlab-search`. When the client calls `github-search`, the Lua router translates it back to `search` before proxying to the `/github` backend.

### Version Pinning

Version pinning locks a tool mapping to a specific backend server version:

```json
{
  "tool_name": "search",
  "alias": "search-repo",
  "backend_server_path": "/github",
  "backend_version": "v1.5.0"
}
```

The mapping includes `backend_version` per tool. Lua selects that version before backend authorization and dispatch, so both the signed upstream claim and backend session key use the selected version. A client-supplied `X-MCP-Server-Version` cannot override the mapping. The active version is used for unpinned tools.

---

## 7. Access Control

### Scope Validation Flow

Both grants are required. `/validate` first authorizes the caller for the virtual server and requested alias; Lua also checks virtual `required_scopes` and alias-specific overrides. Before each backend `initialize`, `tools/list`, or `tools/call`, Lua authorizes the resolved backing server with the rewritten JSON-RPC method and **original** backend tool name. The backing grant uses the same scopes and blocked-tool check as direct calls to that server. A grant to the virtual server alone does not authorize the backing server, and a backing grant alone does not authorize the virtual server. A missing grant prevents credential vending and upstream requests.

Each `tools/list` backing fetch requires backend list access. A successful discovery for a plain backend can be cached per authenticated caller, but the backing grant is checked on every cache hit. Egress-backed discovery is not cached: consent and available tools can vary per user and change when credentials are connected or revoked.

### Example

```json
{
  "required_scopes": ["mcp-access"],
  "tool_scope_overrides": [
    { "tool_alias": "search-repo", "required_scopes": ["github-read"] },
    { "tool_alias": "create-pr", "required_scopes": ["github-write"] }
  ]
}
```

The virtual mapping above requires `mcp-access`, then `github-read` for the `search-repo` alias or `github-write` for `create-pr`. The caller must **also** have a backing-server scope that grants `/github` `initialize`, `tools/list`, and the corresponding original tool (`search` or `create_pr`) on `tools/call`. Without that backing grant, the alias cannot be called even if every virtual override is satisfied. Without the virtual grant, a direct `/github` grant does not unlock `/virtual/dev-tools`.

---

## 8. JSON-RPC Method Routing

The Lua router (`virtual_router.lua`) implements the full MCP protocol for virtual endpoints:

### Method Dispatch Table

| Method | Backend Calls | Caching | Session Required | Notes |
|--------|--------------|---------|-----------------|-------|
| `initialize` | None | No | Creates session | Returns virtual server capabilities |
| `ping` | None | No | No | Responds directly |
| `notifications/initialized` | None | No | No | Returns HTTP 202 Accepted per MCP spec |
| `notifications/cancelled` | None | No | No | Returns HTTP 202 Accepted per MCP spec |
| `tools/list` | All distinct backends | Plain: per-user 60s; egress: none | Yes | Backing grants rechecked; aggregated and scope-filtered |
| `tools/call` | Single backend | No | Yes | Alias translated, routed to owner backend |
| `resources/list` | All distinct backends | 60s TTL | Yes | Aggregated with lookup map |
| `resources/read` | Single backend | No | Yes | Routed via lookup map |
| `prompts/list` | All distinct backends | 60s TTL | Yes | Aggregated with lookup map |
| `prompts/get` | Single backend | No | Yes | Routed via lookup map |

**HTTP Method Handling:**
- `POST` - JSON-RPC requests and notifications
- `GET` - Returns HTTP 405 (server-initiated SSE streams not supported)
- `DELETE` - Returns HTTP 405 (client-initiated session termination not supported)

### Backend Request Ordering

Lua sends backend authorization and backend requests sequentially, not via `capture_multi()`. Each backing method is checked before it can reach the backend or egress vend. `tools/list` merges per-backend results only after those checks; a failed egress discovery cannot be treated as a successful cached list or substituted with mapping-only metadata.

---

## 9. API Endpoints

### Management API

All management endpoints are served by FastAPI on the registry port.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/virtual-servers` | Admin | Create a new virtual server |
| `GET` | `/api/virtual-servers` | User | List all virtual servers |
| `GET` | `/api/virtual-servers/{path}` | User | Get a specific virtual server |
| `PUT` | `/api/virtual-servers/{path}` | Admin | Update a virtual server |
| `DELETE` | `/api/virtual-servers/{path}` | Admin | Delete a virtual server |
| `POST` | `/api/virtual-servers/{path}/toggle` | Admin | Enable or disable a virtual server |
| `GET` | `/api/virtual-servers/{path}/tools` | User | Get resolved tools with full metadata |
| `GET` | `/api/tool-catalog` | User | Browse all available tools across backends |

### Internal API (Lua Router <-> FastAPI)

Nginx exposes these locations only for Lua subrequests. The backing authorization location checks the same caller against the resolved server path and exact rewritten body before any backend dispatch.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/_vs_auth_<backend>` | Authorize backing server and mint an upstream-bound internal token |
| `POST` | `/_vs_backend_<backend>` | Dispatch to auth-server's egress broker or a plain direct backend |
| `GET` | `/_internal/sessions/backend/{client_id}:{backend}` | Read owner-bound stateful ID or explicit stateless state |
| `PUT` | `/_internal/sessions/backend/{client_id}:{backend}` | Persist successful backend initialization state |
| `DELETE` | `/_internal/sessions/backend/{client_id}:{backend}` | Invalidate a stale stateful backend session |
| `POST` | `/_internal/sessions/client` | Create client-facing virtual session |

### Path Validation

Virtual server paths must match the pattern `/virtual/{slug}` where `slug` is lowercase alphanumeric with hyphens. Path traversal attacks are prevented by normalizing and validating paths before any database or filesystem operation.

### Rating Endpoints

Virtual servers support user ratings (1-5 stars):

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/virtual-servers/{path}/rate` | User | Submit or update a rating |
| `GET` | `/api/virtual-servers/{path}/rating` | User | Get rating info for a virtual server |

**Rating Request:**
```json
{
  "rating": 4
}
```

**Rating Response:**
```json
{
  "average_rating": 4.2,
  "message": "Rating submitted successfully"
}
```

**Get Rating Response:**
```json
{
  "num_stars": 4.2,
  "rating_details": [
    {"user": "alice", "rating": 5},
    {"user": "bob", "rating": 4}
  ]
}
```

---

## 10. Search and Discovery

Virtual servers are indexed for semantic search alongside regular MCP servers and A2A agents.

### Indexed Fields

The following fields are included in the vector embedding for semantic search:

- Server name
- Description
- Tags (prefixed with "Tags: ")
- Tool names (alias or original name from each tool mapping)
- Tool description overrides

### Search Collection

Virtual servers are stored in the unified `mcp_embeddings_{dimensions}` collection (e.g., `mcp_embeddings_384` for 384-dimension models) with `entity_type: "virtual_server"`. This enables cross-entity search queries that return servers, agents, and virtual servers in a single response. The dimension suffix matches the configured embedding model (384 for sentence-transformers, 1536 for OpenAI/Bedrock Titan).

### Search Document Structure

```json
{
  "_id": "/virtual/dev-tools",
  "entity_type": "virtual_server",
  "path": "/virtual/dev-tools",
  "name": "Dev Tools",
  "description": "Aggregated development tools",
  "tags": ["development", "tools"],
  "is_enabled": true,
  "tools": [
    {"name": "github_search"},
    {"name": "jira_search"}
  ],
  "embedding": [0.12, -0.34, ...],
  "metadata": {
    "server_name": "Dev Tools",
    "num_tools": 5,
    "backend_count": 2,
    "backend_paths": ["/github", "/jira"],
    "required_scopes": ["mcp-access"],
    "supported_transports": ["streamable-http"],
    "created_by": "admin"
  }
}
```

### Search Result Format

When virtual servers match a search query, they appear in the `virtual_servers` array:

```json
{
  "servers": [...],
  "agents": [...],
  "virtual_servers": [
    {
      "entity_type": "virtual_server",
      "path": "/virtual/dev-tools",
      "server_name": "Dev Tools",
      "description": "Aggregated development tools",
      "relevance_score": 0.85,
      "tags": ["development", "tools"],
      "backend_paths": ["/github", "/jira"],
      "tool_count": 5,
      "matching_tools": [
        {"tool_name": "github_search"}
      ]
    }
  ],
  "tools": [...],
  "skills": [...]
}
```

### Indexing Lifecycle

Virtual servers are indexed/re-indexed when:
- Created via `POST /api/virtual-servers`
- Updated via `PUT /api/virtual-servers/{path}`
- Toggled via `POST /api/virtual-servers/{path}/toggle`
- Deleted (removed from search index)

---

## 11. Frontend Management UI

The management UI provides a multi-step wizard for creating and editing virtual servers:

### Wizard Steps

1. **Basics**: Server name (auto-generates path slug), description, tags, transport selection
2. **Tool Selection**: Interactive picker showing all available tools grouped by backend server, with search filtering
3. **Configuration**: Per-tool alias assignment, version pinning, scope overrides, description overrides
4. **Review**: Summary of the complete configuration before submission

### Dashboard Integration

Virtual servers appear on the main dashboard alongside regular MCP servers. They are visually distinguished with a different color scheme and a "Virtual" badge. A dedicated "Virtual MCP" filter tab in the dashboard allows viewing only virtual servers.

### Key Components

| Component | Purpose |
|-----------|---------|
| `VirtualServerList` | Table view with search, toggle, edit, delete |
| `VirtualServerCard` | Dashboard card with status, tool count, backend count |
| `VirtualServerForm` | 4-step creation/edit wizard |
| `ToolSelector` | Searchable tool picker grouped by backend |
| `useVirtualServers` | React hook for CRUD with optimistic updates |

---

## 12. Validation and Error Handling

### Creation-Time Validation

When a virtual server is created or updated, the service layer performs the following validations:

1. **Path format**: Must match `/virtual/[a-z0-9-]+`
2. **Path uniqueness**: No existing virtual server with the same path
3. **Backend existence**: Each `backend_server_path` must reference a registered, enabled server
4. **Tool existence**: Each `tool_name` must exist in the referenced backend's tool list
5. **Alias uniqueness**: No two tool mappings may produce the same effective name
6. **Scope override validity**: Each `tool_alias` in scope overrides must match an existing tool mapping

### Runtime Error Handling

| Error Condition | Lua Router Behavior |
|----------------|---------------------|
| Missing `Mcp-Session-Id` header | Returns JSON-RPC error: "Missing session" |
| Invalid/expired client session | Returns JSON-RPC error: "Invalid session" |
| Backend returns HTTP 400+ | Invalidates cached session, retries with fresh `initialize` |
| Backend unreachable | Returns JSON-RPC error with backend details |
| User lacks required scope | Returns HTTP 403 with scope details |
| Unknown tool name in `tools/call` | Returns JSON-RPC error: "Tool not found" |
| Unknown JSON-RPC method | Returns JSON-RPC error: "Method not found" |

---

## 13. Performance Characteristics

### Caching Strategy

| Data | Cache Location | TTL | Invalidation |
|------|---------------|-----|--------------|
| Backend initialization state | L1 (shared dict) | 30s | State-appropriate session rejection or TTL |
| Backend initialization state | L2 (MongoDB) | 1 hour idle | Stateful session rejection or idle TTL |
| Plain enriched tool list | L1 (per-user shared dict) | 60s | Backing grant rechecked on hits; TTL |
| Credentialed enriched tool list | None | Per request | Consent and tool availability remain current |
| Resource/prompt lookup maps | L1 (shared dict) | 60s | On TTL |
| Mapping files | Disk | Until regenerated | On CRUD mutation |

### Stress Test Results

Testing with a production-representative configuration:

| Scenario | Requests | Throughput | Error Rate |
|----------|----------|------------|------------|
| Concurrent `tools/list` | 1,000 | 68.9 req/s | 0% |
| Concurrent `tools/call` | 1,000 | 57.9 req/s | 0% |
| Mixed workload | 1,000 | 5.2 req/s | 0% |
| Session storm (100 concurrent inits) | 100 | 43.7 req/s | 0% |

### Latency Overhead

Virtual routing adds a virtual grant and a fresh backend-bound authorization check for each backing request, plus a vend/inject hop for credentialed backends. Earlier latency benchmarks predate this dual-grant egress path and do not measure its overhead.

---

## 14. Deployment Considerations

### Multi-Instance Behavior

- Each nginx worker maintains its own L1 shared dict cache
- L2 (MongoDB) provides cross-instance session consistency
- Nginx config regeneration is triggered by the registry instance that receives the mutation
- In multi-registry deployments, a mechanism for cross-instance nginx reload propagation would be needed (not currently implemented)

### Configuration Reload

When a virtual server is mutated:

1. The service acquires a global `asyncio.Lock` to serialize reload operations
2. Full nginx configuration is regenerated (including all virtual and non-virtual servers)
3. Mapping JSON files are written to disk
4. `nginx -s reload` is issued
5. The lock is released

This approach is simple and correct but means all virtual server mutations are serialized. For typical management workloads (infrequent CRUD), this is not a bottleneck.

### Resource Sizing

| Resource | Sizing Guidance |
|----------|----------------|
| Shared dict memory | 2 MB covers ~10K cached entries |
| MongoDB `backend_sessions` | TTL-indexed, self-cleaning |
| Mapping files on disk | ~1-10 KB per virtual server |
| Nginx location blocks | One per virtual server + one per unique backend |

---

## 15. Limitations and Future Work

### Current Limitations

- **No resource subscriptions**: `listChanged` notifications from backends are not propagated through virtual servers
- **No per-backend load balancing**: Each backend location maps to a single upstream; horizontal scaling of a backend requires external load balancing
- **No streaming support**: The current Lua router buffers full request/response bodies; SSE streaming through virtual servers is not implemented
- **Single-instance nginx reload**: Config regeneration assumes a single nginx instance; multi-instance coordination is not built in

### Future Enhancements

- **Dynamic tool routing**: Route a single tool to different backends based on request parameters or user attributes
- **Weighted backend selection**: Load balance across multiple instances of the same backend
- **SSE pass-through**: Support streaming transports for long-running tool calls
- **Cross-instance reload coordination**: Notify peer registry instances when nginx config changes
- **Tool usage analytics**: Track per-tool invocation counts, latency, and error rates at the virtual server level
- **Template virtual servers**: Pre-defined virtual server templates for common tool bundles

---

## 16. File Reference

```
registry/
  schemas/
    virtual_server_models.py          # Pydantic data models
    backend_session_models.py         # Session tracking models
  services/
    virtual_server_service.py         # Business logic and validation
    tool_catalog_service.py           # Cross-backend tool aggregation
  repositories/
    interfaces.py                     # Repository interfaces
    documentdb/
      virtual_server_repository.py    # MongoDB persistence
      backend_session_repository.py   # Session persistence with TTL
  api/
    virtual_server_routes.py          # REST API endpoints
  core/
    nginx_service.py                  # Nginx config + mapping generation

docker/
  lua/
    virtual_router.lua                # Lua JSON-RPC router

frontend/
  src/
    types/virtualServer.ts            # TypeScript type definitions
    hooks/useVirtualServers.ts        # React data hooks
    components/
      VirtualServerList.tsx           # List/table view
      VirtualServerCard.tsx           # Dashboard card
      VirtualServerForm.tsx           # Multi-step wizard
      ToolSelector.tsx                # Interactive tool picker

tests/
  unit/
    test_virtual_server_models.py     # Model validation tests
    test_virtual_server_service.py    # Service layer tests
    test_virtual_server_nginx.py      # Nginx generation tests
    test_backend_session_repository.py # Session repository tests
  integration/
    test_virtual_server_api.py        # API endpoint tests
  e2e/
    test_virtual_mcp_protocol.py      # MCP protocol E2E tests
    test_virtual_mcp_latency.py       # Latency benchmarks
    test_virtual_mcp_stress.py        # Stress tests
```
