# Why do I sometimes see search results for assets that no longer exist?

## Question

Semantic search (or the registration duplicate-check) sometimes returns a server, agent, or skill
that I already deleted. Clicking through to it 404s or shows an empty page, because the underlying
record is gone. Why does the deleted asset still show up, and how do I clean it up?

## Answer

This is the inverse of [missing embeddings](fix-missing-search-embeddings.md). Here the source
record was deleted but its **embedding vector was left behind in the search index** (issue #1145).
Semantic search and the dedup check query the embedding index, so an orphaned vector surfaces as a
"phantom" result that no longer exists in the database.

### Why this normally should not happen anymore

Deleting a server, agent, skill, or virtual server now removes its embedding as part of the delete
flow. The removal is retried (up to 3 attempts with backoff), and the source record is **not**
deleted unless the embedding removal succeeds first, so a delete cannot silently leave an orphan.
If the removal ever fails after all retries, the registry increments the
`mcp_registry_embedding_removal_failures_total` metric so operators can alert on it.

Orphans you might still see come from one of:

- Embeddings created before this safeguard shipped (a pre-existing backlog).
- A rare delete where the search backend was unavailable through all retries (the failure metric
  fires in that case).
- Records removed out-of-band (directly in the database, bypassing the service layer).

The registry provides admin APIs and CLI commands to detect and remove these.

## How to detect stale embeddings

### Via CLI (registry_management.py)

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    embeddings-stale
```

Example output:

```
Embeddings Index Status:
  Source documents:  364
  Indexed:           487
  Stale (orphaned):  123

Stale embeddings (123):

  Path                                               Type            Name
  -------------------------------------------------- --------------- ------------------------------
  /llm_prompt/157ffff8-...                           skill           Some Prompt
  /agents/agentcore-record_6d0je                     a2a_agent       Stale Agent
```

### Via REST API

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
    https://your-registry/api/admin/embeddings/stale | jq .
```

## How to fix stale embeddings

### Remove all stale embeddings

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    embeddings-stale-cleanup --all-stale
```

This finds every orphaned embedding and removes it in batches of 100.

### Remove specific paths

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    embeddings-stale-cleanup --paths /llm_prompt/157ffff8-... /agents/agentcore-record_6d0je
```

### Via REST API

```bash
curl -s -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"paths": ["/llm_prompt/157ffff8-...", "/agents/agentcore-record_6d0je"]}' \
    https://your-registry/api/admin/embeddings/stale/cleanup | jq .
```

### Reading the cleanup result

Each path is reported as one of:

- `removed` - an orphaned embedding existed and was deleted.
- `not_found` - nothing was indexed at that path (e.g. a typo, or it was already cleaned). This is
  a no-op, not an error, and is counted separately from `removed` so you can tell a real cleanup
  from one that matched nothing.
- `failed` - the delete raised an error (the path is left in place; retry later).

```json
{
  "removed": 2,
  "not_found": 0,
  "failed": 0,
  "total": 2,
  "details": [
    {"path": "/llm_prompt/157ffff8-...", "status": "removed", "error": null},
    {"path": "/agents/agentcore-record_6d0je", "status": "removed", "error": null}
  ]
}
```

## When should I run this?

- After upgrading to a version that adds this safeguard, to clear any pre-existing orphan backlog.
- If `mcp_registry_embedding_removal_failures_total` is non-zero (a delete-time cleanup failed).
- After deleting records out-of-band (directly in the database).
- As a periodic reconciliation check, alongside `embeddings-missing`.

## Requirements

- Admin permissions required (use the "Get JWT Token" button in the registry UI).
- Batch limit: 100 paths per API call (the CLI handles batching automatically).
- Supported on the DocumentDB/MongoDB search backend.
