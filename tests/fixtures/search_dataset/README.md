# Search Evaluation Test Harness

## Quick Start with Claude Code

If you use Claude Code, the entire pipeline (dataset creation, benchmarking, and reporting) is available as an interactive skill:

```
/search-benchmark https://your-registry-url /path/to/your/token-file
```

You can get a token from the "Get JWT Token" button in the top-left corner of the registry UI.

The [search-benchmark skill](../../.claude/skills/search-benchmark/SKILL.md) will guide you step by step through generating ground truth, running the benchmark, and interpreting results. Alternatively, follow the manual steps below.

## Dataset

The evaluation datasets are not checked into this repository (they contain deployment-specific content and are large). Generate them using the instructions below. The two files you will create are:

- **ground_truth.json** - Search queries with expected results (generated from your registry's assets, or hand-curated)
- **unified_dataset.json** - Document embeddings dump (for offline evaluation only)

The assets in a typical registry can come from these sources:

1. **[Anthropic MCP Server Registry](https://github.com/modelcontextprotocol/servers)**: Federated MCP servers from the public Anthropic registry (context7, exa, hydrata, strava, linkedin, petstore, etc.)
2. **[Anthropic Skills Registry](https://github.com/anthropics/skills)**: Skills from the public Anthropic skills repo (pr-review, mcp-builder, claude-api, etc.)
3. **[GoDaddy ANS (Agent Name Service)](https://www.godaddy.com/ans)**: Agents verified via the Agent Name Service identity protocol
4. **Custom entries**: Servers, agents, and skills registered during development and testing of this registry (travel agents, support agents, AWS KB, SRE gateway, etc.)

## Testing Approaches

Two ways to test search quality:

1. **Against a live deployment** (default): Point at your registry URL with a token file and run queries against the real API
2. **Standalone offline**: Run scoring locally using a dataset dump with the same embedding model as production

## Files

| File | Description |
|------|-------------|
| `unified_dataset.json` | Generated: document embeddings dump from your local MongoDB (for offline mode) |
| `ground_truth.json` | Generated: search queries with expected results (from your registry's assets) |
| `generated_ground_truth.json` | Generated: programmatic ground truth (alternative to hand-curated) |
| `benchmark_results.json` | Generated: raw benchmark results from a live run |
| `benchmark_results.md` | Generated: markdown report with NDCG/MRR/Recall metrics |
| `scripts/benchmark_search.py` | Script: live deployment testing and report generation |
| `scripts/evaluate_search.py` | Script: offline evaluation comparing RRF vs legacy scoring |

## Method 1: Test Against Your Deployed Registry (Recommended)

The standard workflow for benchmarking search quality on any registry deployment. The script generates a ground truth dataset from your registry's own assets, runs the queries, and produces a quality report.

### Step 1: Get a Token

Get a short-lived JWT token from the "Get JWT Token" button in the top-left corner of the registry UI. Save it to a file:

```bash
# Paste the full JSON response or just the token into a file
echo "eyJhbGciOi..." > .token
```

Note: This is a short-lived token. If tests fail with 401 errors, generate a fresh one.

### Step 2: Generate Ground Truth From Your Registry

This pulls all servers, agents, and skills from your registry and auto-generates test queries from asset names, tags, and descriptions:

```bash
uv run python scripts/benchmark_search.py \
    --url https://your-registry.example.com \
    --token-file .token \
    --generate-ground-truth
```

Output: `tests/fixtures/search_dataset/generated_ground_truth.json`

The generated queries cover:
- **exact-name:** Asset names as queries (should find the asset directly)
- **tag-based:** Tag values as queries (should find all assets with that tag)
- **description-derived:** Key phrases from descriptions (tests semantic matching)

These are simple programmatically generated queries using keyword extraction from your asset catalog. They provide a reasonable baseline for measuring search quality but are not a substitute for human-curated queries. You are encouraged to review the generated file, remove low-quality entries, and add your own queries that reflect how your users actually search. The format is straightforward (see [Ground Truth Format](#ground-truth-format) below).

### Step 3: Run Benchmark

```bash
uv run python scripts/benchmark_search.py \
    --url https://your-registry.example.com \
    --token-file .token \
    --queries tests/fixtures/search_dataset/generated_ground_truth.json
```

This runs all queries against the live API, evaluates results against your ground truth, and generates:
- `tests/fixtures/search_dataset/benchmark_results.json` (raw data)
- `tests/fixtures/search_dataset/benchmark_results.md` (markdown report)

### What the Report Contains

- **Quality Metrics:** NDCG@10, MRR, Recall@10 (evaluated against your ground truth)
- **Quality by Category:** breakdown by query type
- **Score Health:** saturation count, unique score values, score range
- **Per-query results:** ranked results with scores, found/missing from expected

### Additional Options

```bash
# Compare two runs (e.g., before and after upgrade)
uv run python scripts/benchmark_search.py \
    --compare results_before.json results_after.json

# Regenerate report from existing results
uv run python scripts/benchmark_search.py --report tests/fixtures/search_dataset/benchmark_results.json

# Use our bundled ground truth (specific to development registry)
uv run python scripts/benchmark_search.py \
    --url http://localhost --token-file .token
```

## Method 2: Standalone Offline Evaluation

Runs both scoring methods (RRF and legacy) locally against the document dataset using the same embedding model as production. No server, Docker, or network required.

### Embedding Model Compatibility

The offline evaluation uses `all-MiniLM-L6-v2` (384 dimensions), which is the default embedding model for the registry. The stored document embeddings in `unified_dataset.json` were generated with this model, and the evaluation script encodes queries with the same model so that cosine similarity is meaningful.

If your deployment uses a different embedding model (e.g., OpenAI `text-embedding-3-small` via LiteLLM, or Amazon Bedrock Titan), the offline evaluation is not supported for that configuration. Use Method 1 (live deployment testing) instead, which queries your real API where both query and document embeddings use whatever model your registry is configured with.

### Prerequisites: Generate the Embedding Dataset

The embedding dataset (`unified_dataset.json`) is not checked into git (it is 4.3MB and may contain deployment-specific content). You must generate it from a running local MongoDB first:

```bash
# Make sure MongoDB is running (docker ps | grep mongo)
docker exec mcp-mongodb mongosh --quiet mcp_registry --eval "
const col = db.mcp_embeddings_384_default;
print(JSON.stringify(col.find({}).toArray()));
" > tests/fixtures/search_dataset/unified_dataset.json
```

This dumps all 378 documents with their 384-dimensional embeddings from the `mcp_embeddings_384_default` collection. The file is gitignored so it won't accidentally be committed.

### Run

```bash
cd /path/to/mcp-gateway-registry

# Full evaluation (both methods compared)
uv run python scripts/evaluate_search.py

# Per-query breakdown
uv run python scripts/evaluate_search.py --verbose

# Save detailed JSON for analysis
uv run python scripts/evaluate_search.py --output results.json

# Single method only
uv run python scripts/evaluate_search.py --method rrf
uv run python scripts/evaluate_search.py --method legacy
```

First run downloads the embedding model (~80MB from HuggingFace). Subsequent runs use the cached model and complete in ~20 seconds.

### What evaluate_search.py Does

1. Loads `unified_dataset.json` (378 documents with embeddings)
2. Loads `ground_truth.json` (100 queries with expected results)
3. For each query, encodes it using the `all-MiniLM-L6-v2` model (same model that created the document embeddings)
4. Scores all 378 documents using both RRF and legacy methods
5. Compares the ranked results against ground truth using NDCG@10, MRR, and Recall@10
6. Prints a summary table comparing both methods

### What benchmark_search.py Does

1. Reads 100 queries from `ground_truth.json`
2. Sends each query to your live registry's `POST /api/search/semantic` endpoint
3. Captures the response (scores, ranking, entity types)
4. Saves raw results to `benchmark_results.json`
5. Evaluates results against ground truth (NDCG@10, MRR, Recall@10)
6. Generates `benchmark_results.md` report with metrics, score health, and per-query breakdowns
7. Supports `--compare` for side-by-side diff of two runs and `--report` to regenerate a report from existing results

## How the Evaluation Works

### Step 1: Load dataset and ground truth

The script loads all 378 documents (with their stored 384-dim embeddings) and the 100 annotated queries.

### Step 2: For each query, encode it

The query is encoded using the same `all-MiniLM-L6-v2` model that produced the document embeddings. This gives us a real query vector (not an approximation).

### Step 3: Score documents using both methods

**RRF method:**
1. Compute cosine similarity between query vector and every document embedding
2. Sort by similarity to get the vector-ranked list
3. Compute text_boost for every document (keyword matching on name, path, tags, tools, etc.)
4. Sort by text_boost to get the keyword-ranked list
5. Merge using RRF formula: `score(doc) = 1/(60 + vector_rank) + 1/(60 + keyword_rank)`

**Legacy method:**
1. Same cosine similarity computation
2. Same text_boost computation
3. Combine: `score = (cosine + 1.0) / 2.0 + text_boost * 0.1`
4. Clamp to [0, 1]

### Step 4: Compare against ground truth using NDCG@10

For each query, we know which documents should appear and how relevant they are (grade 1-3). We check where those documents actually landed in the ranked results and compute NDCG (Normalized Discounted Cumulative Gain).

## Understanding the Metrics

### NDCG@10 (Normalized Discounted Cumulative Gain at position 10)

The primary metric for search quality. NDCG measures whether the right documents appear in the right order within the top 10 results. It gives more credit for placing a highly relevant document at position #1 than at position #8, using a logarithmic discount. The score is normalized against the ideal ranking so that 1.0 means the system produced the perfect ordering.

- **1.0** = perfect (all expected documents appear in ideal order)
- **0.0** = none of the expected documents appear in top 10
- **0.5** = expected documents appear but not in ideal positions

Reference: Jarvelin, K. and Kekalainen, J. (2002). "Cumulated gain-based evaluation of IR techniques." ACM Transactions on Information Systems, 20(4), 422-446.

### Recall@10

Measures completeness: what fraction of the known relevant documents actually appear somewhere in the top 10. A system can have high recall but low NDCG if it finds the right documents but ranks them poorly. Conversely, low recall means relevant documents are being missed entirely.

- **1.0** = all expected documents found
- **0.5** = half of the expected documents found

### MRR (Mean Reciprocal Rank)

Measures how quickly a user finds their first useful result. In practice, users often only look at the first few results. MRR captures whether the system puts at least one relevant document near the top. Averaged across all queries, it reflects the typical user experience of "how many results do I have to scan before finding something useful."

- **1.0** = first result is relevant
- **0.5** = second result is the first relevant one
- **0.1** = tenth result is the first relevant one

Reference: Voorhees, E. M. (1999). "The TREC-8 Question Answering Track Report." Proceedings of TREC-8.

### RRF (Reciprocal Rank Fusion)

The scoring method used by our hybrid search. RRF combines two independently-ranked lists (vector similarity and keyword matching) into a single ranking using only ordinal positions, not raw scores. This avoids the normalization problem that arises when combining scores from different scales (cosine similarity is bounded 0-1, while keyword boost scores are unbounded). The formula `1/(k + rank)` with k=60 gives a smooth decay where the top-ranked document in each list contributes most, but lower-ranked documents still participate.

Reference: Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 758-759.

Production implementations:
- Elasticsearch: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
- OpenSearch: https://opensearch.org/blog/hybrid-search/
- MongoDB Atlas: https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/
- Weaviate: https://weaviate.io/blog/hybrid-search-fusion-algorithms

## Ground Truth Format

Each entry in `ground_truth.json`:

```json
{
  "query": "cloudflare",
  "category": "exact-name",
  "description": "Exact product name in server names and tags",
  "expected": [
    {"path": "/ai-registry/cloudflare-docs", "grade": 3, "reason": "Cloudflare in name and tags"},
    {"path": "/cloudflare-docs", "grade": 3, "reason": "Cloudflare in name and tags"},
    {"path": "/cloudflare-api", "grade": 3, "reason": "Cloudflare in name and tags"}
  ]
}
```

**Fields:**
- `query`: The search string
- `category`: One of the 10 test categories (see below)
- `description`: What this query tests
- `expected`: List of documents that should appear, with relevance grades
  - `path`: The document `_id` in the dataset
  - `grade`: 3 = perfect match, 2 = highly relevant, 1 = somewhat relevant
  - `reason`: Why this document is expected (for human reviewers)

## Query Categories

| Category | Count | Tests |
|----------|-------|-------|
| `exact-name` | 10 | Product/tool names as queries (lexical precision) |
| `semantic` | 10 | Natural language with no keyword overlap (vector quality) |
| `agent-focused` | 10 | Queries targeting agent assets |
| `skill-focused` | 10 | Queries targeting skill assets |
| `tool-precision` | 10 | Exact tool names (should find parent server) |
| `multi-entity` | 10 | Correct answers span multiple entity types |
| `conflict-ambiguous` | 4 | Generic words matching many documents |
| `conflict-vector-vs-lexical` | 6 | Vector and keyword signals disagree |
| `no-answer` | 10 | Nothing in the dataset truly matches |
| `tricky` | 20 | Edge cases, adversarial, non-English, empty queries |

## How to Add New Queries

1. Open `ground_truth.json`
2. Add a new entry with `query`, `category`, `description`, and `expected`
3. Make sure `path` values exist in `unified_dataset.json` (check the `_id` field)
4. Run the evaluation to see how both methods handle your new query

To validate paths exist:
```bash
python3 -c "
import json
with open('tests/fixtures/search_dataset/unified_dataset.json') as f:
    ids = {d['_id'] for d in json.load(f)}
with open('tests/fixtures/search_dataset/ground_truth.json') as f:
    for q in json.load(f):
        for exp in q['expected']:
            if exp['path'] not in ids:
                print(f'MISSING: {exp[\"path\"]} in query \"{q[\"query\"]}\"')
"
```

## How to Update the Dataset

If the registry content changes and you want a fresh dump:

```bash
docker exec mcp-mongodb mongosh --quiet mcp_registry --eval "
const col = db.mcp_embeddings_384_default;
print(JSON.stringify(col.find({}).toArray()));
" > tests/fixtures/search_dataset/unified_dataset.json
```

Before committing, check for customer data:
```bash
grep -i "tiaa\|expedia\|ericsson" tests/fixtures/search_dataset/unified_dataset.json
```

## How to Test a New Scoring Method

1. Add your scoring function to `scripts/evaluate_search.py` (follow the pattern of `_score_rrf` and `_score_legacy`)
2. Add it to the `--method` choices in the argparser
3. Wire it into `_run_evaluation`
4. Run: `uv run python scripts/evaluate_search.py --method your_method`
5. Compare NDCG, recall, MRR against existing methods

The harness is designed to be extended. Each scoring method is a function that takes `(docs, query_embedding, query_tokens)` and returns a ranked list of `(doc, score)` tuples.
