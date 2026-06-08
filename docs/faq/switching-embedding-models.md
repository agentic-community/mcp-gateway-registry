# How do I switch to a different embedding model?

## Question

I want to change the embedding model used for semantic search (e.g., from the default `all-MiniLM-L6-v2` to Amazon Bedrock Titan or OpenAI). How do I switch models, and what happens to my existing search functionality during the transition?

## Answer

The registry supports multiple embedding providers via the `EMBEDDINGS_PROVIDER` setting. When you switch models, the registry automatically uses a new collection (keyed by vector dimension), so the old embeddings remain intact for rollback. You then re-index all documents using the admin APIs.

Search remains operational during the transition. Documents not yet re-indexed will still be found via keyword matching (they just won't have vector similarity until re-indexed).

## Step-by-step guide

### 1. Update configuration

Set the new embedding provider in your environment or `.env` file:

```bash
# For Amazon Bedrock Titan v2 (1024 dimensions)
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=amazon.titan-embed-text-v2:0
EMBEDDINGS_MODEL_DIMENSIONS=1024
EMBEDDINGS_AWS_REGION=us-east-1

# For OpenAI text-embedding-3-small (1536 dimensions)
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=text-embedding-3-small
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_API_KEY=sk-...
```

### 2. Restart the registry

After restarting, the registry uses the new collection (e.g., `mcp_embeddings_1024` instead of `mcp_embeddings_384`). This collection starts empty.

### 3. Check what needs indexing

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    embeddings-missing
```

Since the new collection is empty, all assets will be reported as missing:

```
Embeddings Index Status:
  Source documents:  380
  Indexed:           0
  Missing:           380
```

### 4. Re-index all documents with the new model

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    embeddings-reindex --all-missing
```

```
Found 380 missing documents. Reindexing...
  Batch 1: 100 success, 0 failed
  Batch 2: 100 success, 0 failed
  Batch 3: 100 success, 0 failed
  Batch 4: 80 success, 0 failed

Reindex complete: 380 success, 0 failed
```

### 5. Verify search works

```bash
uv run python api/registry_management.py \
    --registry-url https://your-registry-url --token-file .token \
    server-search --query "documentation search"
```

## How do I roll back?

Revert the environment variables to the previous model settings and restart. The old collection (e.g., `mcp_embeddings_384`) still has all its data intact. No re-indexing needed for rollback.

## Using the REST API directly

```bash
TOKEN=$(cat .token | jq -r '.tokens.access_token')

# Check how many documents need indexing
curl -s -H "Authorization: Bearer $TOKEN" \
    https://your-registry/api/admin/embeddings/missing | jq '.total_missing'

# Reindex in batches (max 100 per call)
curl -s -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"paths": ["/server-1", "/server-2"]}' \
    https://your-registry/api/admin/embeddings/reindex | jq .
```

## Performance expectations

- Local sentence-transformers (all-MiniLM-L6-v2): ~100 documents in 5-10 seconds
- LiteLLM via Amazon Bedrock: ~100 documents in 15-30 seconds (API rate limited)
- LiteLLM via OpenAI: ~100 documents in 10-20 seconds (API rate limited)
- The registry remains fully operational during re-indexing
- Documents not yet re-indexed are findable via keyword search only

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDINGS_PROVIDER` | `sentence-transformers` | Provider: `sentence-transformers` or `litellm` |
| `EMBEDDINGS_MODEL_NAME` | `all-MiniLM-L6-v2` | Model identifier |
| `EMBEDDINGS_MODEL_DIMENSIONS` | `384` | Vector dimensions (must match model output) |
| `EMBEDDINGS_API_KEY` | (none) | API key for LiteLLM providers |
| `EMBEDDINGS_API_BASE` | (none) | Custom API base URL |
| `EMBEDDINGS_AWS_REGION` | `us-east-1` | AWS region for Bedrock models |
