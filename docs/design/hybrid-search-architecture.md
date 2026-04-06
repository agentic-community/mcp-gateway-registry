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
                          |  Result Grouping    |
                          |  - servers (top 3)  |
                          |  - agents (top 3)   |
                          |  - virtual_servers  |
                          |    (top 3)          |
                          |  - skills (top 3)   |
                          +----------+----------+
                                     |
                                     v
                          +---------------------+
                          |  Tool Extraction    |
                          |  Extract matching   |
                          |  tools from servers |
                          |  -> tools[] (top 3) |
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

All candidate results are scored before applying the top-3-per-entity-type filter. This ensures the highest-scoring documents are selected:

1. Vector search returns candidates (up to `k` results)
2. Keyword search returns additional matches (merged by highest boost per document)
3. Every candidate receives a hybrid score (vector + text boost)
4. All candidates are sorted by hybrid score descending
5. Top 3 per entity type (servers, agents, tools) are selected from the sorted list

This prevents lower-scoring documents from consuming a top-3 slot before higher-scoring documents are evaluated.

### 5. Diagnostic Logging

Both search paths emit a `Score for` log line for every candidate, enabling search quality debugging:

```
Score for 'Context7' (type=mcp_server): vector=0.3412, normalized_vector=0.6706,
  text_boost=8.0, boost_contrib=0.8000, final=1.0000
```

### 6. Result Structure

Search returns grouped results (top 3 per entity type):

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

Custom metadata from servers, agents, skills, and virtual servers is included in both semantic embeddings and keyword search. Metadata is flattened to a text string using `_flatten_metadata_to_text()`:

- Each key name is included as a token
- Scalar values are converted to strings
- List values have each item converted to a string
- Nested dict values have each value converted to a string

For example, a server with metadata `{"source": "agentcore-sync", "region": "us-east-1"}` produces the metadata text: `source agentcore-sync region us-east-1`.

This flattened text is:
1. Appended to `text_for_embedding` so semantic search captures metadata meaning
2. Stored in `metadata_text` field for keyword/regex matching
3. Matched in the `$or` keyword filter alongside path, name, description, tags, and tools
4. Scored with +1.0 text boost when matched in the `_build_text_boost_stage` pipeline

Metadata sources per entity type:
| Entity Type    | Metadata Source |
|----------------|-----------------|
| MCP Server     | `server_info.get("metadata", {})` |
| A2A Agent      | `agent_card.get("metadata", {})` |
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

## Performance Considerations

1. **Result Limiting**: Top 3 per entity type to reduce payload size
2. **Score-Before-Filter**: All candidates scored and sorted before applying top-3 filter
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
