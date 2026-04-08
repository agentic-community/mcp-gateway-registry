# What filtering options are available for agents in the registry?

The registry provides several API endpoints for discovering and filtering agents, each with different filtering capabilities.

## Agent List Endpoint

`GET /api/agents`

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string (optional) | Substring search across agent name, description, tags, and skill names (case-insensitive) |
| `visibility` | string (optional) | Exact match filter: `public`, `private`, or `group-restricted` |
| `enabled_only` | boolean (optional) | When `true`, returns only enabled agents |

Example:
```bash
# List agents matching "internal" in name, description, or tags
curl "https://your-registry/api/agents?query=internal" \
  -H "Authorization: Bearer $TOKEN"

# List only public, enabled agents
curl "https://your-registry/api/agents?visibility=public&enabled_only=true" \
  -H "Authorization: Bearer $TOKEN"
```

## Semantic Search Endpoint

`POST /api/search/semantic`

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string (required, can be empty) | Natural language semantic + lexical search (max 512 chars) |
| `entity_types` | list (optional) | Filter by type: `a2a_agent`, `mcp_server`, `tool`, `skill`, `virtual_server` |
| `tags` | list (optional) | Exact tag filter with AND logic -- all specified tags must be present (case-insensitive) |
| `max_results` | integer (optional) | Limit per entity type, 1-50 (default: 10) |

Tags can also be specified as `#hashtags` inside the query string. They are extracted and merged with the explicit `tags` list.

Example:
```bash
# Search for agents tagged "internal"
curl -X POST "https://your-registry/api/search/semantic" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "",
    "entity_types": ["a2a_agent"],
    "tags": ["internal"]
  }'

# Hashtag syntax works too -- these are equivalent
curl -X POST "https://your-registry/api/search/semantic" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "#internal",
    "entity_types": ["a2a_agent"]
  }'

# Combine semantic search with tag filtering
curl -X POST "https://your-registry/api/search/semantic" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quiz generation",
    "entity_types": ["a2a_agent"],
    "tags": ["internal", "hr"]
  }'
```

## Discover Agents by Skills

`POST /api/agents/discover`

| Parameter | Type | Description |
|-----------|------|-------------|
| `skills` | list (required) | Skill names or IDs to match (partial matching -- returns agents with at least one match) |
| `tags` | list (optional) | Tag filter (case-insensitive) |
| `max_results` | integer (optional) | Limit, 1-100 (default: 10) |

Results are ranked by a weighted score: 60% skill match, 20% tag match, 20% trust level boost.

## Semantic Agent Discovery

`POST /api/agents/discover/semantic`

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string (required) | Natural language query describing needed capabilities |
| `max_results` | integer (optional) | Limit, 1-100 (default: 10) |

## Using Tags for Classification

Since the registry does not have a dedicated "agent type" or "classification" field, tags are the recommended way to categorize agents for filtering. For example:

- Tag internal agents with `internal` and vendor agents with `vendor`
- Filter via the semantic search endpoint: `"tags": ["internal"]`
- Filter via the agent list endpoint: `?query=internal` (searches tags among other fields)

The semantic search `tags` parameter provides the most precise filtering because it performs exact tag matching (all specified tags must be present). The agent list `query` parameter is a broader substring search that also matches against name, description, and skill names.

## Fields Not Available as Direct Filters

The following agent fields exist but are not exposed as filter parameters on any endpoint today:

- `trust_level` (used for ranking in discover endpoint, but not as a filter)
- `status` (lifecycle: active, deprecated, draft, beta)
- `supported_protocol` (a2a, other)
- `provider` / `provider_organization`
- `registered_by`
- `health_status`
- `metadata` keys (searchable in full-text, but no field-level filter)

If you need filtering by any of these fields, please open a feature request on [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues).

## Related Documentation

- [API Reference](../api-reference.md) -- full API documentation
- [A2A Agent Management](../a2a-agent-management.md) -- agent registration and management guide
- [Custom Metadata](../custom-metadata.md) -- using metadata fields for organization and compliance
