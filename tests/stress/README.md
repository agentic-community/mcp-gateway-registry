# Registry Stress Test Harness

Tracks [Issue #997](https://github.com/agentic-community/mcp-gateway-registry/issues/997). Implementation guide: [`lld-stress-test.md`](../../lld-stress-test.md).

This directory contains the registry stress test harness. The goal is to register 100/500/1000 MCP servers, A2A agents, and Agent Skills against a running registry and measure API + UI performance on both `mongodb-ce` and DocumentDB backends.

**Current status: Phase 1 + Phase 2a + Phase 2b** — data generators, bulk registration, serial API latency measurement, and concurrent search scaling. UI performance measurement and report builder are tracked separately (Phases 3-4).

## What ships in Phase 1

| Script | Purpose |
|---|---|
| `generators/generate_servers.py` | Page the Anthropic MCP Registry and write per-server payload JSONs. |
| `generators/generate_agents.py` | Page the GoDaddy ANS catalog and write per-agent payload JSONs. |
| `generators/generate_skills.py` | Walk the `anthropics/skills` repo via GitHub trees API and write per-skill payload JSONs. |
| `register_entities.py` | Async bulk-register the generated payloads against a running registry. |
| `measure_api_performance.py` | Phase 2a: measure steady-state per-request latency for list endpoints + semantic search (serial). |
| `measure_search_concurrency.py` | Phase 2b: measure semantic search latency under concurrent load (1, 10, 100 parallel). |
| `queries.json` | Curated 20-query set used by the Phase 2a/2b semantic-search measurements. |
| `run_stress_test.sh` | Orchestrator that runs all three generators then the loader. Optionally chains Phase 2a via `STRESS_MEASURE_API=1`. |
| `cleanup.py` | Delete all stress-test entities (identified by `stress-test` tag) from the registry. |

Generated payloads land under `tests/stress/data/<entity>/<count>/`. Registration aggregates land under `tests/stress/results/<backend>/size-<count>/registration.json`. Both paths are already in `.gitignore` (lines 431-432) and are **not** committed.

## Prerequisites

### Environment variables

| Variable | Required for | Notes |
|---|---|---|
| `ANS_API_KEY`, `ANS_API_SECRET` | `generate_agents.py` | GoDaddy ANS credentials, per the variable names documented in `docs/design/ans-integration.md`. |
| `ANS_API_ENDPOINT` | optional | Defaults to `https://api.godaddy.com` (production). **For customer-tier credentials, set this to `https://api.ote-godaddy.com`** (production's `/v1/agents` is gated behind GoDaddy's internal SSO and only accepts internally-provisioned keys, while OTE accepts customer-issued partner keys against the same API shape). |
| `GITHUB_TOKEN` or `GITHUB_PAT` | optional | Avoids the 60 req/hr anonymous rate limit when fetching `anthropics/skills`. Either name works; `GITHUB_PAT` matches the project's existing convention used elsewhere in `.env`. |
| `STRESS_BASE_URL` | optional | Registry base URL (defaults to `http://localhost`). |
| `STRESS_TOKEN_FILE` | optional | Path to the JWT token file (defaults to `.token` in the repo root). |
| `STRESS_RESULTS_DIR` | optional | Override the results directory. |

### Getting a JWT token

The stress-test runner does **not** auto-generate tokens. Get one yourself and put it where the script can find it:

1. Open the registry UI and click **Get JWT Token**.
2. Save the downloaded file as `.token` in the repo root (the default location), or save the raw JWT string (just the `eyJ...` token, nothing else) to `.token`. Both formats are accepted.
3. Override the path with `--token-file /path/to/your-token-file` or `STRESS_TOKEN_FILE=/path/to/your-token-file` if you want to keep it elsewhere.

Accepted file shapes:

- Nested JSON (what the **Get JWT Token** button produces): one of `{"access_token": "..."}`, `{"tokens": {"access_token": "..."}}`, or `{"token_data": {"access_token": "..."}}`.
- Plain text: the file contains nothing but the raw JWT string.

## Quick start

### 1. Start the registry

| Environment | How to start |
|---|---|
| Local (Docker Compose) | `./build_and_run.sh` |
| EKS (Kubernetes) | Deploy via `charts/` Helm charts |
| ECS (AWS Fargate) | Deploy via `terraform/aws-ecs/` |

### 2. Run the stress test

```bash
bash tests/stress/run_stress_test.sh 100 \
    --base-url http://localhost \
    --token-file .token

# Results land at:
#   tests/stress/data/{servers,agents,skills}/100/*.json
#   tests/stress/results/<detected-backend>/size-100/registration.json
```

For a friction-free demo, scope the run to a single entity type via the optional 2nd positional argument (defaults to `all`):

```bash
# ~80 seconds, 98-99/100 registered, no server-side wedging:
bash tests/stress/run_stress_test.sh 100 skills \
    --base-url http://localhost \
    --token-file .token

# Other supported values: servers, agents, all (default)
```

`servers` and `agents` reliably wedge MongoDB CE on local Docker at any concurrency we've tried; they are useful for surfacing registry bottlenecks (which is the harness's job) but `skills` is the type to demo to a reviewer who wants to see a clean end-to-end run.

## Phase 2 — API performance measurement

After the registry is populated (Phase 1), measure steady-state per-request latency for the list endpoints and semantic search:

```bash
# Stand-alone:
uv run python -m tests.stress.measure_api_performance \
    --backend mongodb-ce --size 100 \
    --iterations 50 \
    --base-url http://localhost \
    --token-file .token

# Chained from the orchestrator (after Phase 1 finishes):
STRESS_MEASURE_API=1 bash tests/stress/run_stress_test.sh 100 skills \
    --base-url http://localhost \
    --token-file .token
```

What it measures (per `lld-stress-test.md` §5):

- **List endpoints** (`/api/{servers,agents,skills}`) in three modes per type: first page (`limit=50`), max page (`limit=500`, the API's hard cap), and a pagination walkthrough (`offset=0,50,...`).
- **Semantic search** (`POST /api/search/semantic`) for each of the 20 curated queries in `queries.json`, at `k ∈ {5, 10, 50}`, with `include_draft: true` so the `status: draft` stress corpus is visible.

All requests are issued serially — we want steady-state per-request latency, not concurrent-load throughput.

### Warmup discard

Every operation runs `iterations + 1` times. The first iteration is timed and checked for a non-error status (so we don't silently mask a broken endpoint) but its latency is **discarded** before percentile math. The output JSON records `warmup_strategy: "discard_first_iteration"` at the top level.

Why: the first request to a freshly-booted registry consistently runs 5–10× slower than steady state due to (1) sentence-transformers embedding model lazy-load (~80 MB), (2) MongoDB working-set warmup, (3) HTTP connection-pool establishment, (4) FastAPI/Pydantic warm-paths. Including iteration 0 in percentile math means `p99` and `max` are dominated by warmup, not by real tail behavior.

### Output

Two files per `(backend, size)` run, alongside the Phase 1 `registration.json`:

```
tests/stress/results/<backend>/size-<N>/
  registration.json          # Phase 1
  api_perf.json              # Phase 2a (machine-readable)
  api_perf.md                # Phase 2a (human-readable)
  search_concurrency.json    # Phase 2b (machine-readable)
  search_concurrency.md      # Phase 2b (human-readable)
```

`api_perf.md` and `search_concurrency.md` are what you give a reviewer. All JSON files include a `registry_info` block with the deployment snapshot (version, cloud, compute, storage, auth, embeddings config) captured from `GET /api/registry/telemetry/info` at test start.

## Phase 2b — Search concurrency measurement

After Phase 2a (or independently), measure semantic search latency under concurrent load:

```bash
uv run python -m tests.stress.measure_search_concurrency \
    --base-url https://d2xl2zfuhgc4l0.cloudfront.net \
    --token-file .token \
    --iterations 50
```

What it measures:

- **Concurrency=1**: Baseline single-user search latency (should match Phase 2a serial results).
- **Concurrency=10**: 10 simultaneous search requests per iteration. Simulates a small team using search at the same time.
- **Concurrency=100**: 100 simultaneous search requests per iteration. Simulates burst load from agent-driven discovery workflows.

Each level runs `iterations + 1` batches (first discarded as warmup). Each batch fires `concurrency` simultaneous `POST /api/search/semantic` requests using the 20 curated queries from `queries.json` with `k=5`. The output reports p50/p90/p95/p99 latency and throughput (requests per second) per concurrency level.

The key scaling metric: if p99 at concurrency=100 is less than 2x the p99 at concurrency=1, the search backend scales well under concurrent load.

## Full Benchmark Sequence

The complete benchmark flow against a deployed registry (3 commands):

```bash
# 1. Register 100 servers + agents + skills (skip generation if data exists)
bash tests/stress/run_stress_test.sh 100 \
    --base-url https://d2xl2zfuhgc4l0.cloudfront.net \
    --token-file .token \
    --skip-generate

# 2. Measure API list + serial search latency
uv run python -m tests.stress.measure_api_performance \
    --size 100 \
    --base-url https://d2xl2zfuhgc4l0.cloudfront.net \
    --iterations 50 \
    --token-file .token

# 3. Measure search concurrency scaling
uv run python -m tests.stress.measure_search_concurrency \
    --base-url https://d2xl2zfuhgc4l0.cloudfront.net \
    --token-file .token \
    --iterations 50
```

After all three tests complete, generate the benchmark report using the `/benchmark-report` Claude skill (or run the script directly):

```bash
# 4. Generate benchmark report (reads all JSON results, outputs to docs/benchmarks/)
/usr/bin/python3 .claude/skills/benchmark-report/generate_benchmark_report.py \
    --results-dir tests/stress/results/documentdb/size-100
```

The report is written to `docs/benchmarks/benchmark-{date}-{compute}-{backend}-{instances}x.md` and includes the deployment configuration, all latency tables, and scaling analysis. Commit this file to the repo as a historical benchmark record.

To clean up after testing:

```bash
uv run python -m tests.stress.cleanup \
    --base-url https://d2xl2zfuhgc4l0.cloudfront.net \
    --token-file .token
```

`api_perf.md` and `search_concurrency.md` are quick human-readable summaries. The `docs/benchmarks/` report is the canonical record that combines all three phases with the deployment configuration.

### Token refresh

On a 401, the script invokes `keycloak/setup/generate-agent-token.sh` once and retries. Subsequent 401s fail the operation. No pre-emptive refresh.

Against a deployed instance, either set the env vars or pass the flags:

```bash
# Env-var form
STRESS_BASE_URL=https://your-registry.example.com \
STRESS_TOKEN_FILE=/path/to/that-deployment-token.json \
  bash tests/stress/run_stress_test.sh 100

# Flag form (CLI takes precedence over env vars)
bash tests/stress/run_stress_test.sh 100 skills \
    --base-url https://your-registry.example.com \
    --token-file /path/to/that-deployment-token.json
```

## Running scripts individually

### Generators

Each generator caches upstream API responses under `tests/stress/data/.cache/` so re-runs are fast. Pass `--force` to overwrite existing payload JSONs in the output dir.

```bash
uv run python -m tests.stress.generators.generate_servers --count 1000
uv run python -m tests.stress.generators.generate_agents  --count 1000
uv run python -m tests.stress.generators.generate_skills  --count 1000
```

If the upstream returns fewer unique records than the target count, the generator augments with `-stress-{i:05d}` suffixes on `name`/`path` and reports `source_records` vs `augmented_records` honestly in its summary. Downstream analysis must discount duplicate-embedding effects when augmentation kicks in.

### Bulk registration

```bash
uv run python -m tests.stress.register_entities \
    --entity-type all \
    --count 100 \
    --backend mongodb-ce \
    --base-url http://localhost \
    --concurrency 3
```

#### A note on `--concurrency`

The default is **3**. Going higher overwhelms the registry quickly: every `POST /api/servers/register` triggers synchronous embedding compute, a full nginx config regeneration, and a security scan, so even at concurrency=10 we observed MongoDB `Connection reset by peer` cascades after the first ~30-50 successful registrations on a local `mongodb-ce` stack. Until those server-side bottlenecks are addressed (tracked as separate issues), keep concurrency low and let the run take longer. If the stack is sized for higher throughput (e.g. DocumentDB on real infra), raise the flag explicitly.

Output schema (`tests/stress/results/<backend>/size-<count>/registration.json`):

```json
{
  "backend": "mongodb-ce",
  "size": 100,
  "wall_clock_seconds": 12.4,
  "entity_types": {
    "servers": {
      "entity_type": "servers",
      "target_count": 100,
      "registered": 99,
      "skipped": 1,
      "failed": 0,
      "failure_rate": 0.0,
      "wall_clock_seconds": 4.2,
      "latency_ms": {"p50": 38, "p95": 210, "p99": 480, "min": 12, "max": 510, "mean": 52},
      "failures": []
    },
    "agents": {"...": "..."},
    "skills": {"...": "..."}
  }
}
```

Re-running the loader against an already-populated registry marks each existing entity as `skipped` (not `failed`) — the script is idempotent. A run is considered successful when every entity type's `failure_rate < 0.01`.

## Recommended runtime environment knobs

The registry's continual MCP-server health-check loop (default 30 s) keeps auth-server and MongoDB busy at idle. With ~50+ registered servers this can be enough to starve the registration request path and produce nginx 504s on `/validate` subrequests. Two options before a stress run:

- Set `HEALTH_CHECK_INTERVAL_SECONDS` to something large (e.g. `86400`) in `.env` to effectively disable the loop for the duration of the run, then `docker compose up -d registry` to pick it up.
- Or accept the noise; the loader's `failures[]` array will capture the 504s with their payload filenames so you can re-run those specifically.

## Notes on data fidelity

- **Servers** are registered with `status: draft` so the registry's health-check loop does not spam unreachable synthetic URLs.
- **Skills** point at real `SKILL.md` URLs in `anthropics/skills`; the registry will fetch and embed them, but the SKILL.md content is real (not synthetic).
- **Agents** carry a `stress-test` tag and synthetic URLs (`stress-test-*.invalid`). Filter by this tag to clean up after a run.

## Cleanup

```bash
# Delete generated data and results for one (backend, size) pair
rm -rf tests/stress/data/{servers,agents,skills}/100/
rm -rf tests/stress/results/mongodb-ce/size-100/

# Delete upstream API caches (forces re-fetch on next generator run)
rm -rf tests/stress/data/.cache/
```

To remove the registered entities themselves, use the registry's existing CLI:

```bash
uv run python -m api.registry_management remove-by-tag stress-test \
    --token-file .oauth-tokens/ingress.json \
    --registry-url http://localhost
```

(See `api/registry_management.py --help` for the exact subcommand name in your version.)

## What's next (Phases 3-5)

- **Phase 3**: `measure_ui_performance.py` — Playwright-driven UI metrics (TTFB, FCP, TTI, search interaction). The UI uses a single Dashboard page with a `viewFilter` state, so scenarios click the viewFilter selector rather than navigating to separate URLs.
- **Phase 4**: `report_builder.py` + `run_all.sh` — cross-size and cross-backend comparison reports.
- **Phase 5**: `docs/performance-baselines.md` — committed baseline numbers produced from Phase 4 runs.

See [`lld-stress-test.md`](../../lld-stress-test.md) for the full design.
