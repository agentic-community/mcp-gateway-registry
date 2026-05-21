# MCP Gateway Registry Benchmark Report

*Generated: 2026-05-21 02:30 UTC*
*Backend: documentdb, Corpus size: 100 entities per type*

---

## Deployment Configuration

| Parameter | Value |
|-----------|-------|
| Registry Version | `1.24.1-25-gc144741c-stress-test-pr1023` |
| Cloud Provider | aws |
| Compute Platform | ecs |
| Architecture | x86_64 |
| Storage Backend | documentdb |
| Search Backend | documentdb |
| Auth Provider | keycloak |
| Embeddings Provider | litellm |
| Embeddings Backend | openai |
| Python Version | 3.14 |
| OS | linux |
| Deployment Mode | with-gateway |
| Federation Enabled | True |
| Backend Instances (detected) | 2 |

### Corpus Size at Test Time

| Entity | Count |
|--------|-------|
| MCP Servers | 117 |
| Agents | 110 |
| Skills | 100 |

## Registration Throughput (Phase 1)

Bulk registration of 100 entities per type with concurrency=3.
Total wall clock: 787.6s.

| Entity Type | Target | Registered | Skipped | Failed | Failure Rate | p50 | p95 | p99 |
|-------------|--------|------------|---------|--------|--------------|-----|-----|-----|
| servers | 100 | 99 | 1 | 0 | 0.0% | 1.05s | 3.33s | 4.26s |
| agents | 100 | 57 | 10 | 33 | 33.0% | 7.21s | 19.70s | 22.25s |
| skills | 100 | 99 | 1 | 0 | 0.0% | 1.06s | 2.01s | 2.41s |

## API Latency, Serial (Phase 2a)

Steady-state per-request latency. Each operation measured 50 times (first iteration discarded as warmup).
Total wall clock: 1321.9s.

### List Endpoints

| Operation | p50 | p95 | p99 | Max |
|-----------|-----|-----|-----|-----|
| list_servers_first_page | 133ms | 182ms | 293ms | 364ms |
| list_servers_max_page | 282ms | 578ms | 1.13s | 1.14s |
| list_servers_paginated | 136ms | 205ms | 306ms | 352ms |
| list_agents_first_page | 178ms | 236ms | 312ms | 313ms |
| list_agents_max_page | 320ms | 438ms | 501ms | 507ms |
| list_agents_paginated | 175ms | 318ms | 934ms | 955ms |
| list_skills_first_page | 53ms | 67ms | 90ms | 109ms |
| list_skills_max_page | 54ms | 81ms | 107ms | 129ms |
| list_skills_paginated | 51ms | 93ms | 136ms | 237ms |

### Semantic Search (Serial)

| k | Queries | p50 | p95 | p99 | Max |
|---|---------|-----|-----|-----|-----|
| k=5 | 20 | 309ms | 793ms | 1.94s | 5.29s |
| k=10 | 20 | 311ms | 650ms | 1.58s | 7.02s |
| k=50 | 20 | 334ms | 724ms | 1.50s | 3.75s |

## Semantic Search Concurrency Scaling (Phase 2b)

Concurrent search load test using 20 curated queries at k=5. Each concurrency level ran 50 iterations (first discarded as warmup).
Total wall clock: 920.8s.

| Concurrency | Requests | Throughput (rps) | p50 | p90 | p95 | p99 | Max |
|-------------|----------|-----------------|-----|-----|-----|-----|-----|
| 1 | 50 | 2.1 | 356ms | 621ms | 716ms | 3.47s | 3.47s |
| 10 | 500 | 7.3 | 1.32s | 1.68s | 1.83s | 3.78s | 4.48s |
| 100 | 5000 | 7.8 | 13.46s | 29.43s | 30.39s | 30.73s | 31.28s |

### Scaling Analysis

- Baseline p99 (concurrency=1): 3.47s
- Peak p99 (concurrency=100): 30.73s
- Degradation ratio: 8.9x

Moderate degradation under peak concurrent load. Acceptable for most production workloads.

## Methodology

- **Registration (Phase 1):** Async bulk registration of generated payloads sourced from the Anthropic MCP registry and GoDaddy ANS catalog.
- **API Latency (Phase 2a):** Serial requests, each operation measured N+1 times (first iteration discarded as warmup). Reports steady-state per-request latency.
- **Search Concurrency (Phase 2b):** Concurrent batches of semantic search requests at increasing parallelism levels. Reports aggregate latency and throughput.
- **Warmup:** First iteration at each level/operation is always discarded. Covers embedding model lazy-load, connection pool establishment, and DB working-set warmup.
- **All result JSON files** include a `registry_info` snapshot of the deployment configuration at test time, captured from `GET /api/registry-management/telemetry/info`.

## Reproducing These Results

```bash
# 1. Register entities
bash tests/stress/run_stress_test.sh 100 \
    --base-url <REGISTRY_URL> --token-file .token --skip-generate

# 2. Measure API latency (serial)
uv run python -m tests.stress.measure_api_performance \
    --size 100 --base-url <REGISTRY_URL> --iterations 50 --token-file .token

# 3. Measure search concurrency
uv run python -m tests.stress.measure_search_concurrency \
    --base-url <REGISTRY_URL> --token-file .token --iterations 50
```

See `tests/stress/README.md` for full documentation.
