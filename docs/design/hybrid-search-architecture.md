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
           |                  |               |   tags, tools)    |
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

**Keyword Search (Lexical)**
- Regex matching on path, name, description, tags, and tool names/descriptions
- Catches explicit references that semantic search might miss
- Runs as separate query due to DocumentDB limitations (no `$unionWith` support)

### 3. Score Combination

The final relevance score combines both approaches:

```
relevance_score = normalized_vector_score + (text_boost * 0.1)
```

Text boost values:
| Match Location | Boost Value |
|----------------|-------------|
| Path           | +5.0        |
| Name           | +3.0        |
| Description    | +2.0        |
| Tags           | +1.5        |
| Tool (each)    | +1.0        |

### 4. Result Structure

Search returns grouped results:

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
  "agents": [...]
}
```

## Entity Types

### MCP Servers

**What's included in the embedding:**
- Server name
- Server description
- Tags (prefixed with "Tags: ")
- Tool names (each tool's name)
- Tool descriptions (each tool's description)

**What's NOT included in the embedding:**
- Tool inputSchema (JSON schema is stored but not embedded)
- Server path
- Server metadata

**Stored document fields:**
- `path`, `name`, `description`, `tags`, `is_enabled`
- `tools[]` array with `name`, `description`, `inputSchema` per tool
- `embedding` vector
- `metadata` (full server info for reference)

### A2A Agents

**What's included in the embedding:**
- Agent name
- Agent description
- Tags (prefixed with "Tags: ")
- Capabilities (prefixed with "Capabilities: ")
- Skill names (each skill's name)
- Skill descriptions (each skill's description)

**What's NOT included in the embedding:**
- Agent path
- Agent card metadata (stored but not embedded)
- Skill IDs, tags, and examples

**Stored document fields:**
- `path`, `name`, `description`, `tags`, `is_enabled`
- `capabilities[]` array
- `embedding` vector
- `metadata` (full agent card for reference)

### Tools

- Not indexed separately - extracted from parent server documents
- When a server matches, its tools are checked for keyword matches
- Top-level `tools[]` array contains full schema (inputSchema)
- `matching_tools` in server results is a lightweight reference (no schema)

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

## Performance Considerations

1. **Result Limiting**: Top 3 per entity type to reduce payload size
2. **Index Reuse**: HNSW index parameters (m=16, efConstruction=128) optimized for recall
3. **Embedding Caching**: Lazy-loaded model with singleton pattern
4. **Keyword Fallback**: Separate query ensures explicit matches are not missed

## Example: Why Hybrid Matters

Query: "context7"

- **Vector-only**: Might return documentation servers with similar semantic content
- **Keyword-only**: Finds exact match but misses related servers
- **Hybrid**: Ranks /context7 at top (keyword boost) while including semantically similar alternatives
