# Why are some of my assets not showing up in semantic search?

## Question

I have servers, agents, or skills registered in the registry, but they do not appear when I search for them using semantic search. Keyword search on the list pages finds them fine, but the semantic search endpoint returns no results for these assets. What's happening and how do I fix it?

## Answer

This happens when embedding generation fails during registration. The asset gets stored in the source collection (servers, agents, or skills) but never makes it to the embeddings index. Common causes:

- Embedding model was temporarily unavailable (network timeout, cold start)
- Model initialization failed at the time of registration
- Asset was imported via federation sync without triggering re-indexing

The registry provides admin APIs to detect and fix this.

## How to detect missing embeddings

### Via CLI (registry_management.py)

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    embeddings-missing
```

Example output:

```
Embeddings Index Status:
  Source documents:  380
  Indexed:           378
  Missing:           2

Missing documents (2):

  Path                                               Type            Name
  -------------------------------------------------- --------------- ------------------------------
  /atlassian/                                        mcp_server      Atlassian
  /my-new-agent                                     a2a_agent       My New Agent
```

### Via REST API

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
    https://your-registry/api/admin/embeddings/missing | jq .
```

## How to fix missing embeddings

### Re-index all missing documents

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    embeddings-reindex --all-missing
```

This finds all missing documents and generates their embeddings in batches of 100.

### Re-index specific paths

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    embeddings-reindex --paths /atlassian/ /my-new-agent
```

### Via REST API

```bash
curl -s -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"paths": ["/atlassian/", "/my-new-agent"]}' \
    https://your-registry/api/admin/embeddings/reindex | jq .
```

## When should I run this?

- After upgrading the registry (some documents may not have been re-indexed)
- After seeing "Embedding model unavailable" warnings in logs
- After federation sync imports new assets
- As a periodic health check (schedule weekly or after deployments)

## Requirements

- Admin permissions required (use the "Get JWT Token" button in the registry UI)
- The embedding model must be available and healthy for re-indexing to succeed
- Batch limit: 100 paths per API call (the CLI handles batching automatically)
