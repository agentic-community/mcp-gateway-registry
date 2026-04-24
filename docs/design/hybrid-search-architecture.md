# Hybrid Search Architecture

This document describes the hybrid search design for MCP servers and A2A agents in the registry.

## Overview

The registry implements hybrid search that combines semantic (vector) search with lexical (keyword) matching. This approach provides both conceptual understanding of queries and precise matching when users reference entities by name.

## Architecture Diagram

```
                              +-------------------+
                              |   Search Query    |
                              |  "context7 docs"  |
                              +--------+----------+
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
           +------------------+               +-------------------+
           |  Query Embedding |               |  Query Tokenizer  |
           |  (Vector Model)  |               |  (Keyword Extract)|
           +--------+---------+               +---------+---------+
                    |                                   |
                    | [0.12, -0.34, ...]               | ["context7", "docs"]
                    |                                   |
                    v                                   v
           +------------------+               +-------------------+
           |  Vector Search   |               |  Keyword Match    |
           |  (Cosine Sim)    |               |  (Regex on path,  |
           |                  |               |   name, desc,     |
           |                  |               |   tags, metadata, |
           |                  |               |   tools)          |
           +--------+---------+               +---------+---------+
                    |                                   |
                    | semantic_score                    | text_boost
                    |                                   |
                    +----------------+------------------+
                                     |
                                     v
                          +---------------------+
                          |  Score Combination  |
                          |  relevance_score =  |
                          |  semantic + boost   |
                          +----------+----------+
                                     |
                                     v
                          +---------------------+
                          |  Result Distribution|
                          |  Global ranking     |
                          |  with competitive   |
                          |  soft caps (60%)    |
                          |  up to max_results  |
                          +----------+----------+
                                     |
                                     v
                          +---------------------+
                          |  Result Grouping    |
                          |  - servers          |
                          |  - agents           |
                          |  - virtual_servers  |
                          |  - skills           |
                          +----------+----------+
                                     |
                                     v
                          +---------------------+
                          |  Tool Extraction    |
                          |  Extract matching   |
                          |  tools from servers |
                          |  -> tools[]         |
                          +---------------------+
```

## Search Flow

### 1. Query Processing

When a search query arrives:

1. **Embedding Generation**: Query is converted to a vector embedding using the configured model (Amazon Bedrock, OpenAI, or local sentence-transformers)

2. **Tokenization**: Query is split into meaningful keywords
   - Non-word characters are removed
   - Stopwords filtered (a, the, is, are, etc.)
   - Tokens shorter than 3 characters removed

### 2. Dual Search Strategy

**Vector Search (Semantic)**
- Uses HNSW index on DocumentDB (production) or application-level cosine similarity on MongoDB CE
- Finds conceptually similar content even with different wording
- Returns results sorted by cosine similarity
- DocumentDB uses configurable `efSearch` parameter (default 100) for HNSW recall quality
- Minimum `k=50` ensures small collections are fully covered

**Keyword Search (Lexical)**
- Regex matching on path, name, description, tags, metadata_text, and tool names/descriptions
- Catches explicit references that semantic search might miss
- Runs as separate query due to DocumentDB limitations (no `$unionWith` support)
- Each query keyword is matched independently using case-insensitive regex
- Keyword matches from both vector results and separate keyword query are merged, with the highest boost per document kept

### 3. Score Combination

The final relevance score combines both approaches:

```
normalized_vector_score = (cosine_similarity + 1.0) / 2.0   # Map [-1,1] to [0,1]
text_boost_contribution = text_boost * 0.1                   # Scale boost down
relevance_score = normalized_vector_score + text_boost_contribution
relevance_score = clamp(relevance_score, 0.0, 1.0)
```

The multiplier `0.1` is consistent across both DocumentDB and MongoDB CE search paths.

Text boost values (cumulative per keyword match):
| Match Location | Boost Value |
|----------------|-------------|
| Path           | +5.0        |
| Name           | +3.0        |
| Description    | +2.0        |
| Tags           | +1.5        |
| Metadata       | +1.0        |
| Tool (each)    | +1.0        |

### 4. Score-Before-Filter Pattern

All candidate results are scored before applying the distribution filter. This ensures the highest-scoring documents are selected:

1. Vector search returns candidates (up to `k` results)
2. Keyword search returns additional matches (merged by highest boost per document)
3. Every candidate receives a hybrid score (vector + text boost)
4. All candidates are sorted by hybrid score descending
5. The `_distribute_results()` function selects up to `max_results` items using global ranking with competitive soft caps (see [Result Distribution](#result-distribution) below)

This prevents lower-scoring documents from consuming a slot before higher-scoring documents are evaluated.

### 5. Diagnostic Logging

Both search paths emit a `Score for` log line for every candidate, enabling search quality debugging:

```
Score for 'Context7' (type=mcp_server): vector=0.3412, normalized_vector=0.6706,
  text_boost=8.0, boost_contrib=0.8000, final=1.0000
```

### 6. Result Distribution

The `max_results` parameter (range 1-50, default 10) controls how many total results are returned. Results are distributed across entity types using **global ranking with competitive soft caps**.

#### Algorithm

The `_distribute_results()` function in `search_repository.py` implements a two-pass approach:

**Pass 1 -- Pick with soft caps:**
1. Sort all scored candidates by `relevance_score` descending (all entity types on the same 0-1 scale)
2. Walk the sorted list, picking items up to `max_results`
3. If a type reaches its soft cap (`ceil(max_results * 0.6)`), check whether other entity types still have results remaining below in the ranking
4. If other types are waiting: skip this item (enforce cap for diversity)
5. If no other types remain: lift the cap (no point leaving slots empty)

**Pass 2 -- Backfill:**
6. If pass 1 didn't fill all `max_results` slots (because some items were skipped), backfill from the skipped items in score order

#### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SOFT_CAP_RATIO` | `0.6` | No single entity type can claim more than 60% of slots when other types are competing |
| Tool extraction limit | `max(3, ceil(max_results * 0.6))` | Scales tool extraction with `max_results`, minimum 3 for backward compatibility |
| Pipeline candidate limit | `max(max_results * 3, 50)` | Fetch enough candidates for global ranking |

#### Examples

**Example 1: Only servers exist (max_results=10)**

A registry with 20 MCP servers and no agents, tools, or skills.

```
Candidates (sorted by relevance_score):
  S(0.95), S(0.93), S(0.91), S(0.89), S(0.87), S(0.85),
  S(0.83), S(0.81), S(0.79), S(0.77), S(0.75), ...

soft_cap = ceil(10 * 0.6) = 6

Pass 1:
  Pick S(0.95) ... S(0.85) -> 6 servers (cap reached)
  S(0.83): cap hit, check remaining types -> only mcp_server left
           -> no competition, cap lifted
  Pick S(0.83) ... S(0.77) -> 4 more servers

Result: 10 servers (no artificial limit when only one type exists)
```

**Example 2: Mixed types (max_results=10)**

A registry with servers, agents, and tools.

```
Candidates (sorted by relevance_score):
  S(0.95), S(0.93), S(0.91), A(0.88), S(0.87), T(0.85),
  S(0.83), A(0.80), S(0.78), T(0.75), A(0.72), S(0.70)

soft_cap = ceil(10 * 0.6) = 6

Pass 1:
  Pick S(0.95), S(0.93), S(0.91)          -> 3 servers
  Pick A(0.88)                              -> 1 agent
  Pick S(0.87), T(0.85), S(0.83), A(0.80) -> 2 more servers, 1 tool, 1 agent
  Pick S(0.78)                              -> 6th server (cap reached)
  T(0.75): pick                             -> 2nd tool
  A(0.72): pick                             -> 10th total (done)

Result: 6 servers, 3 agents, 1 tool = 10 total
  (diverse results, highest relevance wins, cap prevents server dominance)
```

**Example 3: Small max_results (max_results=5)**

```
soft_cap = ceil(5 * 0.6) = 3

With mixed types, the dominant type gets at most 3 slots,
leaving 2 for other types. Similar diversity to the previous
default behavior of 3 per type.
```

**Example 4: Large max_results with one dominant type (max_results=50)**

A registry with 40 servers, 3 agents, and 2 tools.

```
soft_cap = ceil(50 * 0.6) = 30

Pass 1:
  Servers fill 30 slots (cap reached while agents/tools still available)
  3 agents and 2 tools fill 5 slots
  Cap lifted for servers (no more agents/tools)
  12 more servers fill remaining slots

Result: 42 servers, 3 agents, 2 tools = 47 total
  (all available entities returned, servers got the rest)
```

#### Backward Compatibility

With the default `max_results=10`, the soft cap is 6. In a typical registry with multiple entity types, results look similar to the previous 3-per-type behavior: the dominant type gets 5-6 results, others share the rest. The key difference is that `max_results=50` now actually returns up to 50 results instead of being capped at 15 (3 per type * 5 types).

#### Applies to All Search Paths

The same `_distribute_results()` function is used by all three search code paths:

| Search Path | When Used | Integration |
|-------------|-----------|-------------|
| Hybrid (DocumentDB) | Production with vector index | Scored tuples fed directly to `_distribute_results()` |
| Client-side (MongoDB CE) | Local dev without vector search | Dict results converted to tuples, then distributed |
| Lexical-only | When embedding model unavailable | Scores computed from `text_boost / MAX_LEXICAL_BOOST`, then distributed |

### 7. Result Structure

Search returns grouped results (up to `max_results` total, distributed across entity types):

```json
{
  "servers": [
    {
      "path": "/context7",
      "server_name": "Context7 MCP Server",
      "relevance_score": 1.0,
      "matching_tools": [
        {"tool_name": "query-docs", "description": "..."}
      ]
    }
  ],
  "tools": [
    {
      "server_path": "/context7",
      "tool_name": "query-docs",
      "inputSchema": {...}
    }
  ],
  "agents": [...],
  "virtual_servers": [
    {
      "path": "/virtual/dev-tools",
      "server_name": "Dev Tools",
      "relevance_score": 0.85,
      "backend_paths": ["/github", "/jira"],
      "tool_count": 5
    }
  ],
  "skills": [...]
}
```

## Entity Types

### MCP Servers

**What's included in the embedding:**
- Server name
- Server description
- Tags (prefixed with "Tags: ")
- Metadata text (flattened key-value pairs from server metadata)
- Tool names (each tool's name)
- Tool descriptions (each tool's description)

**What's NOT included in the embedding:**
- Tool inputSchema (JSON schema is stored but not embedded)
- Server path

**Stored document fields:**
- `path`, `name`, `description`, `tags`, `is_enabled`
- `metadata_text` (flattened metadata for keyword search)
- `tools[]` array with `name`, `description`, `inputSchema` per tool
- `embedding` vector
- `metadata` (full server info for reference)

### A2A Agents

**What's included in the embedding:**
- Agent name
- Agent description
- Tags (prefixed with "Tags: ")
- Capabilities (prefixed with "Capabilities: ")
- Metadata text (flattened key-value pairs from agent card metadata)
- Skill names (each skill's name)
- Skill descriptions (each skill's description)

**What's NOT included in the embedding:**
- Agent path
- Skill IDs, tags, and examples

**Stored document fields:**
- `path`, `name`, `description`, `tags`, `is_enabled`
- `metadata_text` (flattened metadata for keyword search)
- `capabilities[]` array
- `embedding` vector
- `metadata` (full agent card for reference)

### Agent Skills

**What's included in the embedding:**
- Skill name
- Skill description
- Tags (prefixed with "Tags: ")
- Metadata text (author, version, custom extra key-value pairs)

**Stored document fields:**
- `path`, `name`, `description`, `tags`, `is_enabled`
- `metadata_text` (author, version, flattened `extra` dict, registry_name for keyword search)
- `embedding` vector
- `metadata` (skill metadata for reference)

### Tools

- Not indexed separately - extracted from parent server documents
- When a server matches, its tools are checked for keyword matches
- Top-level `tools[]` array contains full schema (inputSchema)
- `matching_tools` in server results is a lightweight reference (no schema)

### Virtual MCP Servers

Virtual MCP Servers are indexed in the unified `mcp_embeddings_{dimensions}` collection (e.g., `mcp_embeddings_384` for 384-dimension models) alongside regular servers and agents, distinguished by `entity_type: "virtual_server"`.

**What's included in the embedding:**
- Server name
- Server description
- Tags (prefixed with "Tags: ")
- Tool names (alias or original name from each tool mapping)
- Tool description overrides (if specified in mappings)

**What's NOT included in the embedding:**
- Virtual server path
- Backend server paths
- Required scopes
- Tool input schemas

**Stored document fields:**
- `path`, `name`, `description`, `tags`, `is_enabled`
- `entity_type`: `"virtual_server"`
- `metadata_text` (created_by for keyword search)
- `tools[]` array with `name` (alias or original) per tool mapping
- `embedding` vector
- `metadata` object containing:
  - `server_name`, `num_tools`, `backend_count`
  - `backend_paths[]` (list of backend server paths)
  - `required_scopes[]`, `supported_transports[]`
  - `created_by`

**Search result structure:**
```json
{
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
  ]
}
```

## Metadata in Search

Custom metadata from servers, agents, skills, and virtual servers is included in semantic embeddings, hybrid/keyword search, and the REST API list endpoint keyword filters. Metadata is flattened to a text string using `flatten_metadata_to_text()` (defined in `registry/utils/metadata.py`):

- Each key name is included as a token
- Scalar values are converted to strings
- List values have each item converted to a string
- Nested dict values have each value converted to a string

For example, a server with metadata `{"source": "agentcore-sync", "region": "us-east-1"}` produces the metadata text: `source agentcore-sync region us-east-1`.

### Hybrid / DocumentDB Search

The flattened metadata text is:
1. Appended to `text_for_embedding` so semantic search captures metadata meaning
2. Stored in `metadata_text` field for keyword/regex matching
3. Matched in the `$or` keyword filter alongside path, name, description, tags, and tools
4. Scored with +1.0 text boost when matched in the `_build_text_boost_stage` pipeline

### REST API List Endpoint Keyword Search (Pure Lexical, No Vectors)

The REST API list endpoints below are **pure lexical search**. They do not use embeddings, vector similarity, or the DocumentDB search index. They load all items from storage, build a searchable text string per item in Python, and perform a case-insensitive substring match. No hybrid or semantic search is involved.

The same `flatten_metadata_to_text()` utility is used to include metadata in these filters:

| Endpoint | Parameter | Search Type | Metadata Handling |
|----------|-----------|-------------|-------------------|
| `GET /api/agents?query=` | `query` | Substring match (lexical only) | Metadata appended to `searchable_text` |
| `GET /api/servers?query=` | `query` | Substring match (lexical only) | Metadata appended to `searchable_text` |
| `GET /api/skills/search?q=` | `q` | Scored substring match (lexical only) | Metadata matched with +0.1 relevance score (author, version, extra) |

For hybrid (vector + keyword) search, use `POST /api/search/semantic` instead.

### Metadata Sources

| Entity Type    | Metadata Source |
|----------------|-----------------|
| MCP Server     | `server_info.get("metadata", {})` |
| A2A Agent      | `agent_card.metadata` |
| Agent Skill    | Author, version, `extra` dict (custom key-value pairs), registry_name |
| Virtual Server | `created_by` field |

## Backend Implementations

### DocumentDB (Production)
- Native HNSW vector index with `$search` aggregation pipeline
- Keyword query runs separately and merges results (no `$unionWith` support)
- Text boost calculated in aggregation pipeline using `$regexMatch`

### MongoDB CE (Development/Local)
- No native vector search support (`$vectorSearch` not available)
- Falls back to application-level search (in Python backend, not the calling agent):
  1. Fetch all documents with embeddings from collection
  2. Calculate cosine similarity in Python code
  3. Apply keyword matching and text boost in application
  4. Sort and limit results
- Same API contract as DocumentDB implementation

## Lexical Fallback Mode

When the embedding model is unavailable (misconfigured, network issues, API key expired, model not found), the search system automatically degrades to **lexical-only mode** instead of failing entirely.

### How It Works

1. **Detection**: On the first search request, if the embedding model fails to generate a query vector, the `_embedding_unavailable` flag is set in `DocumentDBSearchRepository`
2. **Fallback**: All subsequent searches skip embedding generation and use `_lexical_only_search()` instead
3. **Error Caching**: The `SentenceTransformersClient` caches load errors in `_load_error` to avoid repeated download attempts (e.g., hitting HuggingFace on every call)
4. **Indexing**: When the model is unavailable during startup, servers and agents are indexed without embeddings. Documents are stored with empty embedding vectors
5. **Response**: The API response includes a `search_mode` field set to `"lexical-only"` (instead of the normal `"hybrid"`) so callers know the search quality is reduced

### Lexical-Only Search Flow

```
                          +-------------------+
                          |   Search Query    |
                          |  "context7 docs"  |
                          +--------+----------+
                                   |
                                   v
                       +-----------------------+
                       | Embedding Model Check |
                       | _embedding_unavailable|
                       | == True?              |
                       +-----------+-----------+
                                   |
                          Yes (fallback)
                                   |
                                   v
                       +-----------------------+
                       |  Keyword Tokenization |
                       |  ["context7", "docs"] |
                       +-----------+-----------+
                                   |
                                   v
                       +-----------------------+
                       |  MongoDB Aggregation  |
                       |  $regexMatch on path, |
                       |  name, description,   |
                       |  tags, metadata,      |
                       |  tools                |
                       +-----------+-----------+
                                   |
                                   v
                       +-----------------------+
                       |  Text Boost Scoring   |
                       |  Normalized by         |
                       |  MAX_LEXICAL_BOOST     |
                       |  (12.5)               |
                       +-----------+-----------+
                                   |
                                   v
                       +-----------------------+
                       |  Result Grouping      |
                       |  search_mode:         |
                       |  "lexical-only"       |
                       +-----------------------+
```

### Scoring in Lexical-Only Mode

In lexical-only mode, the text boost score is normalized to a 0-1 range using a fixed denominator (`MAX_LEXICAL_BOOST = 13.5`):

```
relevance_score = text_boost / MAX_LEXICAL_BOOST
```

The same boost weights from hybrid mode apply:

| Match Location | Boost Value |
|----------------|-------------|
| Path           | +5.0        |
| Name           | +3.0        |
| Description    | +2.0        |
| Tags           | +1.5        |
| Metadata       | +1.0        |
| Tool (each)    | +1.0        |

### Recovery

When the embedding model becomes available again (e.g., after a restart with correct configuration), the system automatically returns to full hybrid search mode. The `_embedding_unavailable` flag and `_load_error` cache are per-process and reset on restart.

## HNSW Tuning (DocumentDB)

The DocumentDB `$search` pipeline includes two tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | `max(max_results * 3, 50)` | Number of nearest neighbors to retrieve. Minimum 50 ensures small collections are fully covered. |
| `efSearch` | `100` (configurable via `VECTOR_SEARCH_EF_SEARCH`) | Controls HNSW recall quality. Higher values improve recall at the cost of query latency. Default DocumentDB value is ~40, which can miss documents in small collections. |

The `efSearch` setting is configured in `registry/core/config.py` as `vector_search_ef_search`.

## Lifecycle Status Filtering

Search results respect the lifecycle status of assets (servers, agents, skills). By default, **deprecated** and **draft** assets are excluded from search results. Only **active** and **beta** assets appear.

### How It Works

1. **Index-Time**: When an asset is indexed for search, its `status` field is stored in the search document alongside other fields (`path`, `name`, `description`, `tags`, `is_enabled`, etc.)

2. **Query-Time**: The `_build_status_filter()` function constructs a MongoDB `$match` filter that excludes assets by lifecycle status:

```python
# Default behavior: exclude deprecated and draft
{
    "$or": [
        {"status": {"$nin": ["deprecated", "draft"]}},
        {"status": {"$exists": False}}  # Treat missing field as active
    ]
}
```

3. **Opt-In Inclusion**: Callers can include filtered assets using request parameters:
   - `include_deprecated: true` -- Include deprecated assets in results
   - `include_draft: true` -- Include draft assets in results
   - `include_disabled: true` -- Include disabled assets (is_enabled=False) in results

### Search Request Example

```json
{
    "query": "feature flags",
    "entity_types": ["skill"],
    "max_results": 10,
    "include_deprecated": true,
    "include_draft": false
}
```

### Status Values

| Status | Default in Search | Description |
|--------|-------------------|-------------|
| `active` | Included | Asset is active and ready for use |
| `beta` | Included | Asset is in beta testing phase |
| `deprecated` | **Excluded** | Asset is deprecated and may be removed |
| `draft` | **Excluded** | Asset is in draft mode, not ready for production |

### Indexed Document Fields

The `status` field is stored in the search document for all entity types:

| Entity Type | Status Source |
|-------------|--------------|
| MCP Server | `server_info.get("status", "active")` |
| A2A Agent | `agent_card.status` (default: `"active"`) |
| Agent Skill | `skill.status` (default: `"active"`) |
| Virtual Server | Not applicable (always active) |

Documents indexed before this feature (without a `status` field) are treated as `active` by the `$exists: False` fallback in the filter.

### Filter Application

The status filter is applied consistently across all three search code paths:

| Search Path | Filter Location |
|-------------|-----------------|
| Hybrid (DocumentDB) | Pre-filter in `$search` pipeline via `_build_status_filter()` |
| Client-side (MongoDB CE) | Query filter in `collection.find()` |
| Lexical-only | Aggregation `$match` stage |

### Re-indexing

When an asset's lifecycle status changes (e.g., from `active` to `deprecated`), the asset is re-indexed via the normal update flow. The search document's `status` field is updated, and subsequent searches will respect the new status.

## Performance Considerations

1. **Result Distribution**: Global ranking with competitive soft caps limits results to `max_results` (default 10, max 50). The distribution algorithm is O(n) where n is the candidate set size (at most 150 documents).
2. **Score-Before-Filter**: All candidates scored and sorted before applying the distribution filter
3. **Index Reuse**: HNSW index parameters (m=16, efConstruction=128) optimized for recall
4. **efSearch Tuning**: Set to 100 for near-exact recall in typical deployments
5. **Embedding Caching**: Lazy-loaded model with singleton pattern
6. **Keyword Fallback**: Separate query ensures explicit matches are not missed
7. **Error Caching**: Failed model loads are cached to avoid repeated download/API attempts

## Example: Why Hybrid Matters

Query: "context7"

- **Vector-only**: Might return documentation servers with similar semantic content
- **Keyword-only**: Finds exact match but misses related servers
- **Hybrid**: Ranks /context7 at top (keyword boost) while including semantically similar alternatives
