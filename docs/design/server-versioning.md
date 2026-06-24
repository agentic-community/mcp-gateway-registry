# MCP Server Version Routing - Design Document

**Date**: 2026-01-29
**Status**: Implemented
**Issue**: [#370](https://github.com/agentic-community/mcp-gateway-registry/issues/370)

---

## 1. Overview

MCP Server Version Routing enables **multiple versions of the same MCP server** to run simultaneously behind a single gateway endpoint. Traffic routes to the active (default) version unless a client explicitly requests a specific version via the `X-MCP-Server-Version` HTTP header.

### Use Cases

- **Canary deployments**: Register a new version as inactive, test it with the version header, then promote it to active
- **Version pinning**: Clients that depend on a specific server version can pin to it with a header
- **Instant rollback**: Switch the active version back to a previous one without redeployment
- **Deprecation lifecycle**: Mark old versions as deprecated with sunset dates before removal

### Example

```bash
# Request to active version (default behavior, no header needed)
curl -X POST https://gateway.example.com/context7 \
  -d '{"method": "tools/list"}'
# Routes to v2.0.0 (current active version)

# Request to a specific inactive version
curl -X POST https://gateway.example.com/context7 \
  -H "X-MCP-Server-Version: v1.5.0" \
  -d '{"method": "tools/list"}'
# Routes to v1.5.0 (legacy version)
```

---

## 2. Two Version Concepts

The registry tracks **two independent version values** for each server. They serve different purposes and are determined differently.

| Aspect | User-Provided Version (Routing Label) | MCP Server Version (Software Identity) |
|--------|---------------------------------------|----------------------------------------|
| **Purpose** | Traffic routing between backend deployments | Identifies the actual software running at the backend |
| **Who controls it** | Platform admin / operator | MCP server developer (set in server code) |
| **When it is set** | At registration time via API or CLI | Discovered at runtime during health checks |
| **How it is determined** | Admin provides it explicitly (e.g., `v1.0.0`, `v2.0.0`) | Read from the MCP `initialize` response `serverInfo.version` field |
| **Mutability** | Changes only via explicit admin action (register, switch default) | Changes whenever the upstream server deploys a new build |
| **Stored as** | `version` field on the server document | `mcp_server_version` field on the server document |
| **Example values** | `v1.0.0`, `v2.0.0`, `beta-3` | `2.14.4`, `1.25.0`, `0.9.1` |
| **Multiple can coexist** | Yes, each version is a separate document with its own backend URL | No, only the active version is health-checked |

### Why Two Versions Exist

These are fundamentally **different things at different conceptual levels**:

- The **user-provided version** is an operational label. It answers: "Which backend deployment should receive traffic for this path?" An admin registers `/context7` with version `v1.0.0` pointing to `https://mcp.context7.com/mcp`, and later registers `v2.0.0` pointing to `https://mcp-v2.context7.com/mcp`. The two versions can run simultaneously with independent backend URLs.

- The **MCP server version** is a software fact. It answers: "What version of the code is running at this backend URL right now?" The server at `https://mcp.context7.com/mcp` may report itself as `2.14.4` today and `2.14.5` tomorrow after a deployment. The admin's routing label (`v1.0.0`) does not change.

They are **never merged or conflated**. An MCP server version change is an informational event, not a routing change. If the upstream server silently upgrades, the registry detects it during health checks and stores the previous/current values for observability.

### MCP Server Version Change Detection

When a health check detects that `mcp_server_version` has changed:

| Field | Purpose |
|-------|---------|
| `mcp_server_version` | Current version reported by the running server |
| `mcp_server_version_previous` | The version before the most recent change |
| `mcp_server_version_updated_at` | ISO timestamp of when the change was detected |

The frontend shows a subtle green dot indicator next to the MCP server version badge when the version changed within the last 24 hours. No acknowledgement workflow is required -- this is informational only.

---

## 3. Storage Design: Separate Documents per Version

Each version of a server is stored as a **separate document** in MongoDB/DocumentDB. The active version uses the original path as its `_id` (backward compatible), and inactive versions use a compound `path:version` ID.

### Active Version Document

This document appears in all listings, search results, health checks, and the dashboard.

```json
{
  "_id": "/context7",
  "server_name": "Context7 MCP Server",
  "version": "v2.0.0",
  "proxy_pass_url": "https://mcp.context7.com/mcp",

  "is_active": true,
  "version_group": "context7",
  "other_version_ids": ["/context7:v1.5.0"],

  "description": "Up-to-date Docs for LLMs and AI code editors",
  "tags": ["documentation", "search", "libraries"],
  "supported_transports": ["streamable-http"],
  "num_tools": 12,
  "num_stars": 4.5,
  "is_enabled": true,
  "registered_at": "2026-01-10T00:00:00Z",
  "updated_at": "2026-01-14T00:00:00Z",

  "mcp_server_version": "2.14.5",
  "mcp_server_version_previous": "2.14.4",
  "mcp_server_version_updated_at": "2026-01-28T15:30:00Z"
}
```

### Inactive Version Document

This document is hidden from listings and search. It is accessible only via the version management API.

```json
{
  "_id": "/context7:v1.5.0",
  "server_name": "Context7 MCP Server",
  "version": "v1.5.0",
  "proxy_pass_url": "https://v1.mcp.context7.com/mcp",

  "is_active": false,
  "version_group": "context7",
  "active_version_id": "/context7",
  "status": "deprecated",
  "sunset_date": "2026-06-01",

  "description": "Legacy version for backward compatibility",
  "tags": ["documentation", "search", "libraries"],
  "supported_transports": ["streamable-http"],
  "num_tools": 10,
  "is_enabled": true,
  "registered_at": "2025-11-15T00:00:00Z",
  "updated_at": "2026-01-14T00:00:00Z"
}
```

### Version-Specific Fields

These fields are added to the standard server document schema to support versioning:

| Field | Type | Present On | Description |
|-------|------|-----------|-------------|
| `version` | `str` | Both | The user-provided version label (e.g., `v2.0.0`) |
| `is_active` | `bool` | Both | `true` for the active version, `false` for inactive |
| `version_group` | `str` | Both | Groups all versions of the same server (derived from path) |
| `other_version_ids` | `list[str]` | Active only | Array of `_id` values for all inactive versions |
| `active_version_id` | `str` | Inactive only | The `_id` of the currently active version document |
| `status` | `str` | Inactive | Version lifecycle status: `stable`, `beta`, `deprecated` |
| `sunset_date` | `str` | Inactive | ISO date after which this version will be removed |

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Active version keeps original path as `_id` | Backward compatibility -- existing nginx location blocks, health checks, and API references continue to work unchanged |
| Inactive versions use `path:version` compound `_id` | Guarantees uniqueness within the collection and is easy to parse |
| `is_active` field for filtering | All listing and dashboard queries add `is_active: true`, keeping inactive versions out of normal views |
| `version_group` for linking | Enables efficient queries to populate the version selector modal without scanning the full collection |
| Each version is a complete document | Versions can have different descriptions, tool counts, ratings, and backend URLs |

### Why Separate Documents Instead of Embedded Array

Two storage approaches were evaluated:

| Criteria | Embedded Array | Separate Documents (chosen) |
|----------|---------------|----------------------------|
| Search pre-filtering | Requires `$elemMatch` or application logic | Simple `is_active: true` filter |
| Each version as independent entity | Awkward -- tools, ratings, descriptions nested in array | Natural -- each doc has full metadata |
| Document size | Grows with versions | Fixed size per document |
| Version swap complexity | Array element update | Document insert/delete (more complex, but infrequent) |
| Listing queries | Need to exclude inactive array items | Simple query filter |

The separate-documents design was chosen because **search filtering is critical** (see Section 5) and each version is a complete entity with its own tools, ratings, and metadata.

---

## 4. Nginx Version Routing

### Map Directive

The nginx configuration uses a `map` directive for O(1) version lookup based on the URI path and the `X-MCP-Server-Version` request header. The map is auto-generated whenever servers are registered, updated, or versions are changed.

```nginx
map "$uri:$http_x_mcp_server_version" $versioned_backend {
    default "";

    # context7 versions
    "~^/context7(/.*)?:$"           "https://mcp.context7.com/mcp";
    "~^/context7(/.*)?:latest$"     "https://mcp.context7.com/mcp";
    "~^/context7(/.*)?:v2.0.0$"     "https://mcp.context7.com/mcp";
    "~^/context7(/.*)?:v1.5.0$"     "https://v1.mcp.context7.com/mcp";
}
```

Each entry maps a `path:version` combination to a backend URL. Three entries exist for the active version: empty header (no version specified), `latest` keyword, and the explicit version string.

### Location Block

For multi-version servers, the location block uses a variable-based `proxy_pass` instead of a hardcoded URL:

```nginx
location /context7 {
    # ... existing auth_request, headers, transport config ...

    set $backend_url "https://mcp.context7.com/mcp";  # Default fallback
    if ($versioned_backend != "") {
        set $backend_url $versioned_backend;
    }

    proxy_pass $backend_url;
    add_header X-MCP-Version-Routing "enabled" always;
}
```

Single-version servers continue to use direct `proxy_pass` with no map entries (fully backward compatible).

### Has-Versions Detection

The nginx config generator checks `server_info.get("other_version_ids", [])` to determine whether a server has multiple versions. If the array is non-empty, the location block uses the variable-based pattern. This check uses `other_version_ids` (the actual MongoDB field), not a `versions` field.

### Request Flow

```
Client Request
  POST /context7
  X-MCP-Server-Version: v1.5.0  (optional)
       |
       v
  Nginx Map Lookup
  Key: "/context7:v1.5.0"
  Result: "https://v1.mcp.context7.com/mcp"
       |
       v
  Location /context7
  $backend_url = map result (or default fallback)
  proxy_pass $backend_url
       |
       v
  Backend: https://v1.mcp.context7.com/mcp
```

### Request/Response Headers

| Header | Direction | Required | Description |
|--------|-----------|----------|-------------|
| `X-MCP-Server-Version` | Request | No | Target version (`v1.0.0`, `v2.0.0`, `latest`, or omit for default) |
| `X-MCP-Version-Routing` | Response | Auto | Indicates version routing is active for this server (`enabled`) |

---

## 5. Search and Listing Integration

### Dashboard Listings

All listing queries filter by `is_active: true`, ensuring only the active version of each server appears in the dashboard:

```python
cursor = collection.find({"is_active": True})
```

Inactive versions are invisible in normal listings and only accessible via the version management API.

### Semantic Search

The registry uses hybrid semantic search combining vector similarity with tokenized keyword matching. The search backend is either **MongoDB-CE** (client-side cosine similarity) or **AWS DocumentDB** (native `$vectorSearch` pipeline with HNSW index). Both backends use the same indexing strategy for versioning.

#### Search Index Structure

Server embeddings are stored in a separate collection (`mcp_embeddings_{dimension}_{namespace}`) with this document structure:

```json
{
  "_id": "/context7",
  "entity_type": "mcp_server",
  "path": "/context7",
  "name": "Context7 MCP Server",
  "description": "Up-to-date Docs for LLMs...",
  "tags": ["documentation", "search"],
  "is_enabled": true,
  "text_for_embedding": "Context7 MCP Server Up-to-date Docs... Tags: documentation, search...",
  "embedding": [0.042, -0.018, ...],
  "metadata": { ... },
  "indexed_at": "2026-01-28T12:00:00Z"
}
```

The embedding text is built from the server's name, description, tags, and tool names/descriptions.

#### How Inactive Versions Are Excluded

There is always **exactly one search document per server path** in the embeddings collection. That document always contains the active version's metadata. Inactive versions never get their own search documents.

This is enforced by which code paths call `index_server()`:

| Operation | Calls `index_server()`? | What happens in the embeddings collection |
|-----------|------------------------|-------------------------------------------|
| `register_server()` (first registration) | Yes | Creates search document at `_id: /context7` with this version's data |
| `register_server()` (new version of existing server) | No (calls `add_server_version()` internally) | No change -- the existing search document stays as-is with the active version |
| `update_server()` | Yes | Overwrites the search document with updated metadata |
| `set_default_version()` | Yes | Overwrites the search document with the **new** active version's data |
| `add_server_version()` | No | No change -- inactive versions are not indexed |

Every call to `index_server()` **recomputes the embedding vector from scratch**. It rebuilds the embedding text by concatenating the provided version's `server_name`, `description`, `tags`, and `tool_list` (tool names and descriptions), then generates a fresh embedding vector from that text. The resulting document is written via `replace_one({"_id": path}, doc, upsert=True)`, which overwrites whatever was previously stored at that path. There is no separate removal step -- the old active version's embedding data is simply replaced by the new active version's embedding data.

This means that if `v2.0.0` has 15 tools and a different description than `v1.0.0`'s 10 tools, switching the active version causes the search document to reflect `v2.0.0`'s content with a new embedding vector that captures its tools and description.

Inactive version documents (stored in the server collection at compound IDs like `/context7:v1.0.0`) have **no corresponding entry** in the embeddings collection. They were never added there.

**Example**: Context7 has three versions (`v1.0.0`, `v1.5.0`, `v2.0.0`) with `v2.0.0` active. The embeddings collection contains exactly one document at `_id: /context7` with `v2.0.0`'s name, description, tags, and tools as the embedding text. When `set_default_version()` switches to `v1.5.0`, `index_server()` rebuilds the embedding text from `v1.5.0`'s metadata (which may have different tools, description, and tags), generates a new embedding vector, and overwrites the search document. A search for "documentation tools" can only ever match this single document -- the two inactive versions have no search presence.

This means inactive versions never consume search result slots, which is the critical requirement for search quality.

#### MongoDB-CE vs AWS DocumentDB Search Behavior

| Aspect | MongoDB-CE | AWS DocumentDB |
|--------|-----------|----------------|
| Vector index | Regular B-tree index (no native vector support) | HNSW vector index (cosine similarity, M=16, efConstruction=128) |
| Search method | Client-side: fetches all embeddings, computes cosine similarity in Python | Native: `$vectorSearch` aggregation pipeline |
| Keyword matching | Tokenized matching in Python (stopwords removed, tokens > 2 chars) | Aggregation pipeline with `$addFields` for text boost scoring |
| Re-ranking | `relevance = normalized_vector_score + (text_boost * 0.05)` | `relevance = normalized_vector_score + (text_boost * 0.1)` |
| Pre-filtering of inactive versions | Same -- inactive versions are not in the search collection | Same -- inactive versions are not in the search collection |

Both backends produce the same result: only active versions appear in search results.

#### Keyword Boost Scoring

The hybrid search applies keyword boosts on top of vector similarity scores:

| Match Location | Boost Points |
|---------------|-------------|
| Path match | 5.0 |
| Name match | 3.0 |
| Description match | 2.0 |
| Tags match | 1.5 |
| Tool name/description match | 1.0 per tool |

The boost is multiplied by a factor (0.05 for MongoDB-CE, 0.1 for DocumentDB) and added to the normalized vector score. This ensures exact name matches rank higher than semantically similar but differently-named servers.

### Summary of Filtering Strategy

| Filter | When Applied | Mechanism |
|--------|-------------|-----------|
| Active vs. inactive version | **Index time** (pre-filter) | Only active version is written to search collection |
| Enabled vs. disabled server | **Query time** (post-filter) | `is_enabled` metadata returned with results |
| User access control | **Query time** (post-filter) | API layer checks user permissions |

This design ensures that inactive versions never waste search result slots, which is the critical requirement for search quality.

### Removal from Search

When a server is deleted via `remove_server()`, the search index entry is also removed via `search_repo.remove_entity(path)`. The `delete_with_versions()` repository method handles cascade deletion of all version documents (active + inactive) from MongoDB/DocumentDB.

---

## 6. Health Check Integration

Only the **active version** of each server is health-checked. The health check service filters out inactive versions:

```python
async def get_enabled_services(self) -> list[str]:
    for path, server_info in all_servers.items():
        if not server_info.get("is_enabled", False):
            continue
        # Skip inactive versions
        if server_info.get("version_group") and not server_info.get("is_active", True):
            continue
        enabled_paths.append(path)
```

When the active version is switched via `set_default_version()`, an immediate background health check is triggered for the newly active version:

```python
asyncio.create_task(health_service.perform_immediate_health_check(path))
```

This ensures the dashboard reflects the health status of the new active version promptly after a switch.

---

## 7. API Endpoints

All version management endpoints are under `/api/servers/{path}/versions`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/servers/{path}/versions` | List all versions of a server |
| `DELETE` | `/api/servers/{path}/versions/{version}` | Remove an inactive version |
| `PUT` | `/api/servers/{path}/versions/default` | Switch the active (default) version |

New versions are created by registering a server with the same path but a different `version` field. The `register_server()` method detects this and creates an inactive version document automatically.

### Version Creation via Registration

```bash
# First registration creates the server
POST /api/servers/register
{
  "server_name": "Context7",
  "path": "/context7",
  "version": "v1.0.0",
  "proxy_pass_url": "https://mcp.context7.com/mcp"
}

# Second registration with same path but different version creates an inactive version
POST /api/servers/register
{
  "server_name": "Context7",
  "path": "/context7",
  "version": "v2.0.0",
  "proxy_pass_url": "https://mcp-v2.context7.com/mcp"
}
```

The second call returns `is_new_version: true` to indicate a new version was added rather than a new server being created.

---

## 8. Version Swap Operation

Switching the active version (`set_default_version`) is the most complex operation in the versioning system. It performs a document swap:

1. Read the current active document at path `_id` (e.g., `/context7`)
2. Read the target inactive version document (e.g., `/context7:v2.0.0`)
3. Build a new active document from the target, assigning it the original path `_id`
4. Build a new inactive document from the current active, assigning it a compound `_id`
5. Delete the old active and target inactive documents
6. Insert the new active and new inactive documents
7. Update the `other_version_ids` array (remove target, add old active)
8. Re-index the search entry with the new active version's data
9. Regenerate nginx configuration and reload
10. Trigger an immediate background health check for the newly active version

This is an infrequent admin operation. The trade-off of complexity here versus simpler listing/search queries is acceptable.

---

## 9. Cascade Deletion

When a server is deleted via `remove_server()`, all version documents are deleted together using `delete_with_versions()`:

```python
filter_query = {
    "$or": [
        {"_id": path},                          # Active document
        {"_id": {"$regex": f"^{path}:"}},       # All inactive version documents
    ]
}
result = await collection.delete_many(filter_query)
```

This prevents orphaned version documents from remaining in the database after a server is removed.

---

## 10. Frontend Components

### Version Badge

A clickable badge on the ServerCard that shows the current active version (e.g., `v2.0.0`). Only visible when the server has multiple versions (`versions.length > 1`). Single-version servers show no badge.

### Version Selector Modal

Opened by clicking the version badge. Displays all versions as individual cards with:

- Version number and status badge (`ACTIVE`, `stable`, `beta`, `deprecated`)
- Backend URL
- Release and sunset dates
- "Set Active" button (disabled for the already-active version)

An info footer explains the `X-MCP-Server-Version` header usage for clients that want to pin to a specific version.

### MCP Server Version Display

A separate, smaller badge below the routing version badge shows the MCP server-reported version (e.g., `srv 2.14.5`). If the version changed within the last 24 hours, a small green dot indicator appears. Hovering shows the previous version in a tooltip.

---

## 11. Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Existing single-version servers | Work unchanged. No `version_group`, no map entries, direct `proxy_pass` |
| No `X-MCP-Server-Version` header | Routes to active version (same as before versioning existed) |
| `version` field missing on legacy document | Defaults to `v1.0.0` |
| Client sends header for single-version server | Map returns empty string, falls back to default `proxy_pass` |

---

## 12. Index Strategy

```javascript
// Primary filter for all listing operations
db.mcp_servers.createIndex({ "is_active": 1 })

// For version group lookups (modal population)
db.mcp_servers.createIndex({ "version_group": 1 })

// Compound index for dashboard queries
db.mcp_servers.createIndex({ "is_active": 1, "is_enabled": 1 })
```

---

## 13. Future: Traffic Splitting (Phase 2)

Not yet implemented. Phase 2 will use nginx `split_clients` directive to route a percentage of traffic to different versions for gradual rollouts:

```nginx
split_clients "${remote_addr}${request_uri}" $canary_backend {
    10%     "http://server-v2:8000/";
    *       "http://server-v1:8000/";
}
```

| Condition | Routing |
|-----------|---------|
| `X-MCP-Server-Version: v2.0.0` | Force v2.0.0 (explicit header takes precedence) |
| No header + traffic split enabled | Percentage-based routing |
| No header + no traffic split | Route to active version |
