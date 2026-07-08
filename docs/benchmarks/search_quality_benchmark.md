# Search Quality Benchmark (Reference)

These results are from our development registry and are provided as sample reference output. Your registry will have different assets and different results. To generate your own benchmark, follow the instructions in [tests/fixtures/search_dataset/README.md](https://github.com/agentic-community/mcp-gateway-registry/blob/main/tests/fixtures/search_dataset/README.md).

## Deployment Under Test

| Parameter | Value |
|-----------|-------|
| Version | 1.24.3 (fix/hybrid-search-rrf-scoring) |
| Database | mongodb-ce |
| Embedding model | all-MiniLM-L6-v2 (384 dimensions) |
| Scoring method | Reciprocal Rank Fusion (k=60) |
| Registry contents | 134 servers, 119 agents, 109 skills |
| Ground truth | 100 hand-curated queries, 10 categories |

## Quality Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| NDCG@10 (avg) | 0.686 | Relevant results appear in top 10, mostly well-ordered |
| MRR (avg) | 0.717 | First relevant result typically at position 1-2 |
| Recall@10 (avg) | 0.755 | 75% of expected documents found in top 10 |
| Perfect queries (NDCG=1.0) | 42/95 | 44% of queries return ideal ranking |
| Zero-hit queries (NDCG=0.0) | 14/95 | 15% miss all expected results |

## Score Health

| Metric | Value |
|--------|-------|
| Saturated at 1.0 | 11% (down from 48% with legacy scoring) |
| Unique score values | 435 (out of 979 total scores) |
| Score range | 0.25 to 1.00 |

## Quality by Category

| Category | Queries | NDCG@10 | MRR | Recall@10 |
|----------|---------|---------|-----|-----------|
| tool-precision | 10 | 1.000 | 1.000 | 1.000 |
| exact-name | 10 | 0.900 | 0.900 | 0.900 |
| tricky | 19 | 0.808 | 0.820 | 0.833 |
| multi-entity | 10 | 0.783 | 0.917 | 0.758 |
| conflict-ambiguous | 4 | 0.729 | 0.875 | 0.750 |
| semantic | 10 | 0.661 | 0.612 | 0.850 |
| conflict-vector-vs-lexical | 6 | 0.645 | 0.833 | 0.556 |
| no-answer | 6 | 0.564 | 0.533 | 0.667 |
| agent-focused | 10 | 0.431 | 0.514 | 0.647 |
| skill-focused | 10 | 0.186 | 0.143 | 0.400 |

## Sample Queries and Results

### "cloudflare" (exact-name)

| # | Type | Name | Score |
|---|------|------|-------|
| 1 | Server | Cloudflare Documentation MCP Server | 1.00 |
| 2 | Server | cloudflare-api | 0.99 |
| 3 | Server | Cloudflare Documentation MCP Server | 0.97 |
| 4 | Tool | search_cloudflare_documentation | 0.65 |
| 5 | Tool | search | 0.51 |

### "travel booking flights hotels activities" (multi-entity)

| # | Type | Name | Score |
|---|------|------|-------|
| 1 | Server | ai.autonomad/travel | 1.00 |
| 2 | Agent | Flight Booking Agent | 0.86 |
| 3 | Agent | Travel Assistant Agent | 0.79 |
| 4 | Server | dev-essentials | 0.55 |
| 5 | Agent | Travel Assistant Agent | 0.42 |

### "current_time_by_timezone" (tool-precision)

| # | Type | Name | Score |
|---|------|------|-------|
| 1 | Server | Current Time API | 1.00 |
| 2 | Server | Current Time API | 0.98 |
| 3 | Tool | current_time_by_timezone | 0.65 |
| 4 | Tool | current_time_by_timezone | 0.65 |

### "documentation" (conflict-ambiguous)

| # | Type | Name | Score |
|---|------|------|-------|
| 1 | Skill | documentation-lookup | 1.00 |
| 2 | Server | aws-kb | 0.85 |
| 3 | Server | Cloudflare Documentation MCP Server | 0.78 |
| 4 | Skill | new-feature-design | 0.79 |
| 5 | Tool | aws___read_documentation | 0.65 |

### "Italian paint finishes catalog" (tricky, niche product)

| # | Type | Name | Score |
|---|------|------|-------|
| 1 | Server | record_novacolor_italian_finishes | 1.00 |

### "send slack message notification" (no-answer)

No results returned (correct: no Slack integration exists in this registry).

## How to Generate Your Own Results

```bash
# Step 1: Generate ground truth from your registry
uv run python scripts/benchmark_search.py \
    --url https://your-registry.example.com \
    --token-file .token \
    --generate-ground-truth

# Step 2: Run benchmark
uv run python scripts/benchmark_search.py \
    --url https://your-registry.example.com \
    --token-file .token \
    --queries tests/fixtures/search_dataset/generated_ground_truth.json
```

See [tests/fixtures/search_dataset/README.md](https://github.com/agentic-community/mcp-gateway-registry/blob/main/tests/fixtures/search_dataset/README.md) for full documentation on the evaluation harness, metrics explained, and how to add custom queries.
