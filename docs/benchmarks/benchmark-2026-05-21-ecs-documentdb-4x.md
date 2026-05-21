# MCP Gateway Registry Benchmark Report

*Generated: 2026-05-21 22:41 UTC*
*Backend: documentdb, Corpus size: 1000 entities per type*

---

## Deployment Configuration

| Parameter | Value |
|-----------|-------|
| Registry Version | `1.24.1-31-g45efe101-stress-test-pr1023` |
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
| Backend Instances (detected) | 4 |

### Corpus Size at Test Time

| Entity | Count |
|--------|-------|
| MCP Servers | 1016 |
| Agents | 904 |
| Skills | 1002 |

## Registration Throughput (Phase 1)

Bulk registration of 1000 entities per type with concurrency=3.
Total wall clock: 5391.7s.

| Entity Type | Target | Registered | Skipped | Failed | Failure Rate | p50 | p95 | p99 |
|-------------|--------|------------|---------|--------|--------------|-----|-----|-----|
| servers | 1000 | 951 | 49 | 0 | 0.0% | 794ms | 2.77s | 4.40s |
| agents | 1000 | 878 | 116 | 6 | 0.6% | 6.82s | 17.47s | 21.42s |
| skills | 1000 | 999 | 1 | 0 | 0.0% | 1.16s | 3.01s | 4.67s |

## API Latency, Serial (Phase 2a)

Steady-state per-request latency. Each operation measured 50 times (first iteration discarded as warmup).
Total wall clock: 2590.1s.

### List Endpoints

| Operation | p50 | p95 | p99 | Max |
|-----------|-----|-----|-----|-----|
| list_servers_first_page | 142ms | 264ms | 347ms | 389ms |
| list_servers_max_page | 1.03s | 1.87s | 2.89s | 3.01s |
| list_servers_paginated | 141ms | 401ms | 582ms | 1.41s |
| list_agents_first_page | 145ms | 248ms | 312ms | 320ms |
| list_agents_max_page | 1.07s | 1.70s | 2.99s | 3.87s |
| list_agents_paginated | 141ms | 397ms | 642ms | 1.56s |
| list_skills_first_page | 150ms | 331ms | 432ms | 462ms |
| list_skills_max_page | 174ms | 613ms | 830ms | 968ms |
| list_skills_paginated | 158ms | 475ms | 778ms | 4.70s |

### Semantic Search (Serial)

| k | Queries | p50 | p95 | p99 | Max |
|---|---------|-----|-----|-----|-----|
| k=5 | 20 | 436ms | 1000ms | 3.36s | 17.04s |
| k=10 | 20 | 446ms | 1.09s | 2.50s | 18.84s |
| k=50 | 20 | 526ms | 1.13s | 3.17s | 16.28s |

## Semantic Search Concurrency Scaling (Phase 2b)

Concurrent search load test using 20 curated queries at k=5. Each concurrency level ran 50 iterations (first discarded as warmup).
Total wall clock: 1116.2s.

| Concurrency | Requests | Throughput (rps) | p50 | p90 | p95 | p99 | Max |
|-------------|----------|-----------------|-----|-----|-----|-----|-----|
| 1 | 50 | 2.3 | 375ms | 661ms | 857ms | 1.02s | 1.02s |
| 10 | 500 | 7.0 | 1.13s | 1.72s | 1.96s | 14.53s | 21.29s |
| 100 | 5000 | 7.0 | 13.16s | 20.69s | 27.85s | 30.54s | 41.11s |

### Scaling Analysis

- Baseline p99 (concurrency=1): 1.02s
- Peak p99 (concurrency=100): 30.54s
- Degradation ratio: 29.8x

Significant degradation under peak concurrent load. Consider horizontal scaling (more ECS tasks) for high-concurrency workloads.

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
