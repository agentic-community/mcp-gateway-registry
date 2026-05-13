# Registry Stress Test Harness

Tracks [Issue #997](https://github.com/agentic-community/mcp-gateway-registry/issues/997). Implementation guide: [`lld-stress-test.md`](../../lld-stress-test.md).

This directory contains the registry stress test harness. The goal is to register 100/500/1000 MCP servers, A2A agents, and Agent Skills against a running registry and measure API + UI performance on both `mongodb-ce` and DocumentDB backends.

**Current status: Phase 1** — data generators and bulk registration only. API/UI performance measurement and report building are tracked separately (Phases 2-4).

## What ships in Phase 1

| Script | Purpose |
|---|---|
| `generators/generate_servers.py` | Page the Anthropic MCP Registry and write per-server payload JSONs. |
| `generators/generate_agents.py` | Page the GoDaddy ANS catalog and write per-agent payload JSONs. |
| `generators/generate_skills.py` | Walk the `anthropics/skills` repo via GitHub trees API and write per-skill payload JSONs. |
| `register_entities.py` | Async bulk-register the generated payloads against a running registry. |
| `run_stress_test.sh` | Orchestrator that runs all three generators then the loader. |

Generated payloads land under `tests/stress/data/<entity>/<count>/`. Registration aggregates land under `tests/stress/results/<backend>/size-<count>/registration.json`. Both paths are already in `.gitignore` (lines 431-432) and are **not** committed.

## Prerequisites

### Environment variables

| Variable | Required for | Notes |
|---|---|---|
| `ANS_API_KEY`, `ANS_API_SECRET` | `generate_agents.py` | GoDaddy ANS credentials, per the variable names documented in `docs/design/ans-integration.md`. |
| `ANS_API_ENDPOINT` | optional | Defaults to `https://api.godaddy.com` (production). **For customer-tier credentials, set this to `https://api.ote-godaddy.com`** — production's `/v1/agents` is gated behind GoDaddy's internal SSO and only accepts internally-provisioned keys, while OTE accepts customer-issued partner keys against the same API shape. |
| `GITHUB_TOKEN` | optional | Avoids the 60 req/hr anonymous rate limit when fetching `anthropics/skills`. |
| `STRESS_BASE_URL` | optional | Registry base URL (defaults to `http://localhost`). |
| `STRESS_TOKEN_FILE` | optional | Path to the JWT token file (defaults to `.oauth-tokens/ingress.json`). |
| `STRESS_RESULTS_DIR` | optional | Override the results directory. |

### Getting a JWT token

For local stacks with the Keycloak setup that ships in this repo, you do not need to fetch a token manually — `run_stress_test.sh` checks for an existing JWT under `.oauth-tokens/` and regenerates one via `keycloak/setup/generate-agent-token.sh` when none is valid.

For deployed stacks (or any setup where the bundled Keycloak script can't reach the IdP), grab a token from the registry UI's **Get JWT Token** button, save it under `.oauth-tokens/`, and pass its path via `STRESS_TOKEN_FILE`. The file is the nested-token JSON the CLI already consumes; the script's auto-regenerate step is skipped when `STRESS_TOKEN_FILE` points at a valid file.

Set `STRESS_SKIP_TOKEN_REFRESH=1` to disable the auto-regenerate step entirely (useful for CI / non-local stacks).

## Quick start

```bash
# 1) Bring up the registry
docker compose up -d

# 2) Generate + register at size=100 against mongodb-ce.
#    The script auto-fetches a JWT if none is found under .oauth-tokens/.
bash tests/stress/run_stress_test.sh mongodb-ce 100

# Results land at:
#   tests/stress/data/{servers,agents,skills}/100/*.json
#   tests/stress/results/mongodb-ce/size-100/registration.json
```

For a friction-free demo on a local stack, scope the run to a single entity type via the optional 3rd positional argument (defaults to `all`):

```bash
# ~80 seconds, 98-99/100 registered, no server-side wedging:
bash tests/stress/run_stress_test.sh mongodb-ce 100 skills

# Other supported values: servers, agents, all (default)
```

`servers` and `agents` reliably wedge MongoDB CE on local Docker at any concurrency we've tried; they are useful for surfacing registry bottlenecks (which is the harness's job) but `skills` is the type to demo to a reviewer who wants to see a clean end-to-end run.

Against a deployed instance:

```bash
STRESS_BASE_URL=https://your-registry.example.com \
STRESS_TOKEN_FILE=/path/to/that-deployment-token.json \
  bash tests/stress/run_stress_test.sh mongodb-ce 100
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

## What's next (Phases 2-5)

- **Phase 2**: `measure_api_performance.py` + `queries.json` — p50/p95/p99 latency for list endpoints and semantic search at k=5/10/50.
- **Phase 3**: `measure_ui_performance.py` — Playwright-driven UI metrics (TTFB, FCP, TTI, search interaction). The UI uses a single Dashboard page with a `viewFilter` state, so scenarios click the viewFilter selector rather than navigating to separate URLs.
- **Phase 4**: `report_builder.py` + `run_all.sh` — cross-size and cross-backend comparison reports.
- **Phase 5**: `docs/performance-baselines.md` — committed baseline numbers produced from Phase 4 runs.

See [`lld-stress-test.md`](../../lld-stress-test.md) for the full design.
