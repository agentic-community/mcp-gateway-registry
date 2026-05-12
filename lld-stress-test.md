# LLD: Registry Scalability Stress Test Harness (Issue #997)

*Created: 2026-05-09*
*Status: Draft for review*
*Tracks: [Issue #997](https://github.com/agentic-community/mcp-gateway-registry/issues/997)*

## 1. Overview

This document specifies the low-level design for the stress test harness introduced by issue #997. The harness generates synthetic-but-realistic registration payloads from public data sources, registers them against a running registry, measures API and UI performance at 100 / 500 / 1000 entities, and produces per-backend and cross-backend reports.

The harness lives entirely under `tests/stress/`. All scripts commit to the repo. Generated data and result files do not (already listed in `.gitignore`).

## 2. Directory Layout

```
tests/stress/
  README.md                       # How to run the harness
  __init__.py
  config.py                       # Shared settings (paths, endpoints, timeouts)
  constants.py                    # ENTITY_TYPES, TARGET_SIZES, QUERY_SET
  data/                           # GITIGNORED: generated payloads
    servers/{100,500,1000}/*.json
    agents/{100,500,1000}/*.json
    skills/{100,500,1000}/*.json
  results/                        # GITIGNORED: measurement outputs
    {mongodb-ce,documentdb}/size-{100,500,1000}/
      api_perf.json
      api_perf.md
      ui_perf.json
      ui_perf.md
      registration.json
      summary.md
    comparison.md                 # cross-backend / cross-size roll-up
  generators/
    __init__.py
    _base.py                      # shared validation + write helpers
    generate_servers.py
    generate_agents.py
    generate_skills.py
  register_entities.py
  measure_api_performance.py
  measure_ui_performance.py
  report_builder.py               # aggregates per-run JSON into summary.md
  run_stress_test.sh              # single (backend, size) run
  run_all.sh                      # wrapper for 2 backends x 3 sizes
  queries.json                    # curated query set for semantic search
```

## 3. Data Generators

### 3.1 Shared contract (`generators/_base.py`)

```python
class GeneratorResult(BaseModel):
    entity_type: Literal["servers", "agents", "skills"]
    target_count: int
    actual_count: int
    source_records: int         # unique records fetched from upstream
    augmented_records: int      # synthesized duplicates to reach target
    elapsed_seconds: float
    output_dir: Path
    errors: list[str]


def _write_payload(
    output_dir: Path,
    payload: dict,
    filename: str,
) -> None:
    """Write a single payload JSON. Overwrites on re-run (idempotent)."""


def _unique_suffix(
    base_id: str,
    index: int,
) -> str:
    """Return f'{base_id}-stress-{index:05d}' for augmented copies."""
```

Every generator CLI accepts:

- `--count {100,500,1000}` (required)
- `--output-dir PATH` (default `tests/stress/data/<entity>/<count>/`)
- `--force` (overwrite existing)
- `--cache-dir PATH` (cache upstream API responses locally to avoid repeat pulls)

### 3.2 `generate_servers.py`

```python
def _fetch_anthropic_catalog(
    cache_dir: Path,
    api_version: str,
) -> list[dict]:
    """
    Page through https://registry.modelcontextprotocol.io/v0/servers
    with limit=100, following `metadata.nextCursor`.
    Cache raw responses under cache_dir/anthropic-pages/.
    """


def _transform_to_registry_shape(
    anthropic_record: dict,
    suffix_index: int | None = None,
) -> dict:
    """
    Reuse cli/anthropic_transformer.py:transform_server.
    When suffix_index is not None, append `-stress-{i}` to `path` and
    `server_name` so the augmented copy is distinct.
    """


def _build_payload(
    record: dict,
    index: int,
) -> dict:
    """
    Produce a registry-format JSON matching cli/examples/complete-server-example.json:
    - server_name (unique)
    - description
    - path (unique, starts with /)
    - proxy_pass_url (synthetic: http://stress-test-{i}.invalid:8100)
    - supported_transports: ["streamable-http"]
    - tags (from upstream categories + "stress-test")
    - status: "draft"     # ensures no health checks run
    - visibility: "public"
    """


def main() -> int:
    # 1. Parse args
    # 2. Fetch (cached) Anthropic catalog -> list[dict] (unique)
    # 3. If len(unique) >= count: take first `count`; augmented = 0
    #    else: repeat-with-suffix until count reached
    # 4. For each selected record: _transform_to_registry_shape -> _build_payload
    # 5. Validate against registry schema (import ServerCreate from registry.schemas.server)
    # 6. Write one JSON per payload to output_dir
    # 7. Print GeneratorResult as JSON
```

**Validation**: Each payload is passed through `registry.schemas.server.ServerCreate(**payload)` before writing. Failures short-circuit with a clear error so we do not pollute the data dir with invalid files.

### 3.3 `generate_agents.py`

```python
def _fetch_ans_agents(
    api_key: str,
    api_secret: str,
    cache_dir: Path,
    endpoint: str = "https://api.godaddy.com",
) -> list[dict]:
    """
    Page through ANS GET /v1/agents with Authorization: sso-key {key}:{secret}.
    Respect 429 with exponential backoff (max 5 retries, base 2s).
    Cache raw JSON per page under cache_dir/ans-pages/.
    """


def _transform_to_agent_card(
    ans_record: dict,
    suffix_index: int | None = None,
) -> dict:
    """
    Map ANS record -> A2A Agent Card shape (see cli/examples/code_reviewer_agent.json):
      protocolVersion: "1.0"
      name: ans_record["name"] + optional "-stress-{i}"
      description: ans_record.get("description", "ANS-sourced stress test agent")
      url: ans_record.get("endpoint") or "https://stress-test-{i}.invalid/agent"
      version: ans_record.get("version", "1.0.0")
      capabilities: {"streaming": True}
      defaultInputModes / defaultOutputModes: ["text/plain", "application/json"]
      skills: [] (empty for agents w/o declared skills)
      preferredTransport: "JSONRPC"
      provider: ans_record.get("organization", "Stress Test Org")
      path: "/" + slugify(name)
      tags: ["stress-test", *ans_record.get("tags", [])]
      visibility: "public"
      isEnabled: True
    """


def main() -> int:
    # 1. Load ANS_API_KEY / ANS_API_SECRET from env (fail-fast if missing)
    # 2. _fetch_ans_agents (cached)
    # 3. Augment to target count (vary version + suffix name+path)
    # 4. Validate each via registry.schemas.agent.AgentCreate
    # 5. Write per-payload JSONs
```

Env requirement documented in `tests/stress/README.md`: `ANS_API_KEY`, `ANS_API_SECRET` must be set.

### 3.4 `generate_skills.py`

```python
def _enumerate_skills_repo(
    cache_dir: Path,
    repo: str = "anthropics/skills",
    branch: str = "main",
) -> list[dict]:
    """
    Use GitHub tree API:
      GET /repos/anthropics/skills/git/trees/{branch}?recursive=1
    Find every blob path ending with /SKILL.md. Return records of shape:
      {
        "skill_name": <parent-folder>,
        "skill_md_url": "https://raw.githubusercontent.com/.../{path}",
        "description": None
      }
    Cache tree response to cache_dir/github-tree.json.
    """


def _transform_to_skill_payload(
    record: dict,
    suffix_index: int | None = None,
) -> dict:
    """
    Produce CLI-compatible payload for skill-register:
      name: slug(skill_name) + optional "-stress-{i:05d}"
      skill_md_url: record["skill_md_url"]
      description: record["description"] or f"Skill sourced from anthropics/skills: {name}"
      version: "1.0.0"
      tags: ["stress-test", "anthropic-skills"]
      target_agents: ["claude-code"]
      visibility: "public"
    """


def main() -> int:
    # Same flow: fetch (cached) -> augment (vary name + version) -> validate -> write
```

## 4. Bulk Registration (`register_entities.py`)

### 4.1 CLI

```
uv run python tests/stress/register_entities.py \
    --entity-type {servers,agents,skills,all} \
    --count {100,500,1000} \
    --backend {mongodb-ce,documentdb} \
    --base-url http://localhost \
    --concurrency 10 \
    --data-dir tests/stress/data \
    --results-dir tests/stress/results \
    --token-file .oauth-tokens/ingress.token
```

### 4.2 Control flow

```python
async def _register_one(
    client: RegistryClient,
    entity_type: str,
    payload: dict,
    sem: asyncio.Semaphore,
) -> RegistrationRecord:
    """
    Acquire semaphore, time the call, return RegistrationRecord with
    outcome='success'|'skipped'|'failed' and latency_ms.
    Skipped = entity already exists (409 or duplicate-key from registry).
    """


async def _register_all(
    entity_type: str,
    payloads: list[dict],
    concurrency: int,
) -> list[RegistrationRecord]:
    """Create asyncio.Semaphore(concurrency) and gather _register_one tasks."""


def _aggregate(
    records: list[RegistrationRecord],
) -> RegistrationAggregate:
    """Compute counts + p50/p95/p99 latency_ms."""


def main() -> int:
    # 1. Parse args
    # 2. Read payloads from data-dir (one glob per requested entity type)
    # 3. For each entity_type: asyncio.run(_register_all(...))
    # 4. Aggregate and write:
    #    results/<backend>/size-<count>/registration.json
```

### 4.3 Client choice

Reuse `api/registry_client.py::RegistryClient` via its async methods. For entity types where the client only has sync methods, wrap in `asyncio.to_thread`.

Idempotency:

- **servers / skills**: match by `path`. Catch 409 Conflict -> mark `skipped`.
- **agents**: match by `name`. Catch 409 / validation error "already exists".

### 4.4 Output schema

```json
{
  "backend": "mongodb-ce",
  "entity_type": "servers",
  "target_count": 1000,
  "registered": 994,
  "skipped": 4,
  "failed": 2,
  "failure_rate": 0.002,
  "wall_clock_seconds": 72.4,
  "latency_ms": {"p50": 38, "p95": 210, "p99": 480},
  "failures": [
    {"payload": "server-0123.json", "error": "..."}
  ]
}
```

## 5. API Performance Measurement (`measure_api_performance.py`)

### 5.1 CLI

```
uv run python tests/stress/measure_api_performance.py \
    --backend {mongodb-ce,documentdb} \
    --size {100,500,1000} \
    --base-url http://localhost \
    --iterations 50 \
    --queries-file tests/stress/queries.json \
    --token-file .oauth-tokens/ingress.token
```

### 5.2 Endpoints under test

| Operation | Method | Path |
|---|---|---|
| list servers | GET | `/api/servers?include_disabled=false&limit={50,all}` |
| list agents | GET | `/api/agents?limit={50,all}` |
| list skills | GET | `/api/skills?limit={50,all}` |
| semantic search | POST | `/api/search/semantic` |

For list endpoints: run with `limit=50` (first page), with `limit=0` or omitted (all rows), and with pagination walkthrough (`offset=0,50,100,...`).

For semantic search: iterate `queries.json` x `k in [5, 10, 50]`. Body shape is the `SemanticSearchRequest` model already defined in [registry/api/search_routes.py:197](../../registry/api/search_routes.py#L197):

```json
{
  "query": "<text>",
  "entity_types": ["mcp_server", "agent", "skill"],
  "max_results": 5,
  "include_disabled": false
}
```

### 5.3 Curated query set (`queries.json`)

20 queries total, 5 per entity category plus 5 mixed queries:

```json
[
  {"id": "server-01", "query": "github repository access", "expected_entity_types": ["mcp_server"]},
  {"id": "server-02", "query": "postgres database query", "expected_entity_types": ["mcp_server"]},
  {"id": "agent-01", "query": "code review assistant", "expected_entity_types": ["agent"]},
  {"id": "skill-01", "query": "markdown pdf conversion", "expected_entity_types": ["skill"]},
  {"id": "mixed-01", "query": "secure my container", "expected_entity_types": ["mcp_server", "agent", "skill"]}
]
```

Final query set TBD during implementation; the 20-query target is a floor, not a cap.

### 5.4 Control flow

```python
async def _measure_call(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    payload: dict | None,
) -> CallRecord:
    """Time a single HTTP request. Include status, latency_ms, response_bytes."""


async def _run_iterations(
    session: aiohttp.ClientSession,
    op: Operation,
    iterations: int,
) -> list[CallRecord]:
    """Run a single operation N times serially (serial is intentional: we want
    steady-state per-request latency, not concurrent-load throughput)."""


def _summarize(
    records: list[CallRecord],
) -> OperationSummary:
    """Return p50/p95/p99, min, max, mean, error_count, expected_hits (for search)."""


async def main_async(args: argparse.Namespace) -> int:
    # 1. Build Operation list from endpoints table + queries x k values
    # 2. For each operation: await _run_iterations -> _summarize
    # 3. Write:
    #    results/<backend>/size-<size>/api_perf.json
    #    results/<backend>/size-<size>/api_perf.md
```

### 5.5 Output schema (`api_perf.json`)

```json
{
  "backend": "mongodb-ce",
  "size": 1000,
  "iterations": 50,
  "collected_at": "2026-05-09T12:00:00Z",
  "operations": [
    {
      "name": "list_servers_paginated",
      "method": "GET",
      "url_pattern": "/api/servers?limit=50&offset={offset}",
      "latency_ms": {"p50": 45, "p95": 120, "p99": 180, "min": 30, "max": 220, "mean": 52},
      "error_count": 0
    },
    {
      "name": "semantic_search",
      "query_id": "server-01",
      "k": 10,
      "latency_ms": {"p50": 140, "p95": 380, "p99": 520, "min": 110, "max": 600, "mean": 160},
      "error_count": 0,
      "expected_hits": 48
    }
  ]
}
```

## 6. UI Performance Measurement (`measure_ui_performance.py`)

### 6.1 CLI

```
uv run python tests/stress/measure_ui_performance.py \
    --backend {mongodb-ce,documentdb} \
    --size {100,500,1000} \
    --base-url http://localhost \
    --headless
```

### 6.2 Tool: Playwright

Playwright config already exists at [frontend/playwright.config.ts](../../frontend/playwright.config.ts). This script uses the Python `playwright` package (add to dev deps if not present) rather than reusing the TS runner, because the measurement primitives we need (DOM node count, PerformanceObserver) are cleaner via Python + CDP.

### 6.3 Scenarios

| Scenario | Action | Metrics |
|---|---|---|
| `list_servers_load` | navigate to servers list page | time-to-first-byte, first-contentful-paint, time-to-interactive, DOM node count |
| `list_agents_load` | navigate to agents list page | same |
| `list_skills_load` | navigate to skills list page | same |
| `search_interaction` | focus search box, type query, press Enter, wait for results render | keystroke-to-render ms, result count |
| `pagination_scroll` | scroll to bottom (or click next page N times) | per-page render ms, total scroll duration |
| `filter_by_tag` | click "stress-test" tag filter | click-to-filtered-render ms, filtered count |

For `search_interaction`, run once per `k in [5, 10, 50]` by setting the UI's `max_results` selector (if exposed) or by calling the API-level proxy endpoint the UI uses. Document which approach was chosen in the output report.

### 6.4 Control flow

```python
async def _launch_browser(
    headless: bool,
) -> tuple[Browser, BrowserContext]:
    """Launch Chromium with a clean context. Set viewport 1440x900."""


async def _measure_navigation(
    page: Page,
    url: str,
) -> NavigationMetrics:
    """
    Use page.evaluate() to pull:
      performance.timing (legacy) or performance.getEntriesByType('navigation')[0]
      document.querySelectorAll('*').length
    Return NavigationMetrics with ttfb, fcp, tti, dom_nodes.
    """


async def _measure_search(
    page: Page,
    query: str,
    k: int,
) -> SearchMetrics:
    """Inject query, measure from keydown to results-rendered via MutationObserver."""


async def main_async(args: argparse.Namespace) -> int:
    # 1. Launch browser
    # 2. Log in (use storage state from .oauth-tokens or cookie injection)
    # 3. Run each scenario, collect metrics
    # 4. Screenshot each terminal state to results/<backend>/size-<size>/screenshots/
    # 5. Write ui_perf.json and ui_perf.md
```

### 6.5 Output schema (`ui_perf.json`)

```json
{
  "backend": "mongodb-ce",
  "size": 1000,
  "scenarios": [
    {
      "name": "list_servers_load",
      "ttfb_ms": 80, "fcp_ms": 300, "tti_ms": 1250, "dom_nodes": 9843
    },
    {
      "name": "search_interaction",
      "query": "github repository access",
      "k": 10,
      "keystroke_to_render_ms": 420,
      "result_count": 10
    }
  ]
}
```

## 7. Report Builder (`report_builder.py`)

Consumes the JSON outputs and produces three markdown files:

1. `results/<backend>/size-<N>/summary.md` - single run summary (registration + API + UI tables).
2. `results/<backend>/cross-size.md` - per-backend comparison across 100/500/1000.
3. `results/comparison.md` - cross-backend comparison at each size (mongodb-ce vs documentdb).

```python
def _load_run(
    backend: str,
    size: int,
    results_dir: Path,
) -> RunData:
    """Load registration.json, api_perf.json, ui_perf.json for a (backend, size) pair."""


def _build_summary_md(run: RunData) -> str: ...


def _build_cross_size_md(
    backend: str,
    runs: list[RunData],
) -> str: ...


def _build_cross_backend_md(
    size: int,
    runs_by_backend: dict[str, RunData],
) -> str: ...
```

Table shape for cross-backend:

| Metric | 100 (mongodb-ce) | 100 (documentdb) | 500 (mongodb-ce) | 500 (documentdb) | 1000 (mongodb-ce) | 1000 (documentdb) |
|---|---|---|---|---|---|---|
| list_servers p95 ms | | | | | | |
| semantic_search k=10 p95 ms | | | | | | |
| UI list_servers tti ms | | | | | | |
| Registration failure rate | | | | | | |

## 8. Orchestration

### 8.1 `run_stress_test.sh`

```bash
#!/bin/bash
set -e

# Args
BACKEND="${1:?must pass backend (mongodb-ce|documentdb)}"
SIZE="${2:?must pass size (100|500|1000)}"
BASE_URL="${BASE_URL:-http://localhost}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

echo "[1/5] Generating data (size=$SIZE)..."
uv run python tests/stress/generators/generate_servers.py --count "$SIZE"
uv run python tests/stress/generators/generate_agents.py --count "$SIZE"
uv run python tests/stress/generators/generate_skills.py --count "$SIZE"

echo "[2/5] Registering entities against backend=$BACKEND..."
uv run python tests/stress/register_entities.py \
    --entity-type all --count "$SIZE" --backend "$BACKEND" --base-url "$BASE_URL"

echo "[3/5] Measuring API performance..."
uv run python tests/stress/measure_api_performance.py \
    --backend "$BACKEND" --size "$SIZE" --base-url "$BASE_URL"

echo "[4/5] Measuring UI performance..."
uv run python tests/stress/measure_ui_performance.py \
    --backend "$BACKEND" --size "$SIZE" --base-url "$BASE_URL" --headless

echo "[5/5] Building summary report..."
uv run python tests/stress/report_builder.py \
    --mode single --backend "$BACKEND" --size "$SIZE"

echo "Done. Results at tests/stress/results/$BACKEND/size-$SIZE/"
```

The script does **not** stand up the registry itself. Operators are expected to:

1. Set `STORAGE_BACKEND=<backend>` (plus the required connection env vars) in `.env`.
2. `docker compose up -d` (or deploy to ECS for DocumentDB runs).
3. Wait for registry to become healthy.
4. Run `run_stress_test.sh`.

This keeps the harness decoupled from deploy tooling and lets us run it equally against a local Docker stack and a remote ECS deployment.

### 8.2 `run_all.sh`

```bash
#!/bin/bash
set -e
for backend in mongodb-ce documentdb; do
  for size in 100 500 1000; do
    bash tests/stress/run_stress_test.sh "$backend" "$size"
  done
done

uv run python tests/stress/report_builder.py --mode cross
echo "Comparison report at tests/stress/results/comparison.md"
```

## 9. Configuration and Secrets

| Variable | Purpose | Source |
|---|---|---|
| `ANS_API_KEY`, `ANS_API_SECRET` | GoDaddy ANS agent catalog | `.env` |
| `GITHUB_TOKEN` | Raise GitHub API rate limit for tree enumeration (optional but recommended) | `.env` |
| `STRESS_BASE_URL` | Registry base URL | env or `--base-url` |
| `STRESS_TOKEN_FILE` | Path to an ingress `.token` file containing the nested token JSON (for authenticated endpoints) | env, default `.oauth-tokens/ingress.token` |

> **Getting a token**: open the registry UI, click the **Get JWT Token** button, and save the downloaded `.token` file to `.oauth-tokens/ingress.token` (or wherever `STRESS_TOKEN_FILE` points). The file is the nested-token format the registry already consumes; no transformation is needed.
>
> **Token lifetime**: the issued JWT is valid for 8 hours, which comfortably exceeds the wall-clock time of even the largest (backend, size=1000) run. Expiry during a run is therefore not a concern. Still, operators are recommended to grab a fresh token immediately before starting each stress run so the full 8-hour window is available and no retry-on-401 path needs to kick in.
| `STRESS_RESULTS_DIR` | Override results dir | env, default `tests/stress/results` |

No new secrets are introduced; all reuse existing conventions.

## 10. Dependencies

Add to `[tool.uv] dev-dependencies`:

- `playwright>=1.48.0` (already Playwright, but confirm Python bindings)
- `aiohttp>=3.9.0` (for async HTTP in API perf script)

Playwright browsers installed on demand:

```bash
uv run playwright install chromium
```

No runtime deps change. This harness is dev-only.

## 11. Validation & Edge Cases

- **Registry unreachable**: generators still work (they do not touch the registry). Registration and measurement scripts fail-fast with a clear error.
- **Partial data**: if the Anthropic registry returns only 300 unique records for a target of 1000, augmentation kicks in; the generator log clearly reports `source_records=300, augmented_records=700`. Result reports carry this metadata so we can discount duplicate-embedding effects.
- **Health check noise**: servers are registered with `status: draft` to suppress health checks during load tests. API perf script uses `include_draft: true` in the semantic search body to ensure those entities are searchable during measurement.
- **Token expiry during long runs**: registration script refreshes token via `RegistryClient` if a 401 is seen; retries once before marking failure.
- **DocumentDB network**: when running against DocumentDB, the operator must run the harness from a host inside the VPC (or via a bastion). The harness does not orchestrate network plumbing.
- **Concurrency limits**: default `--concurrency 10` for registration. Higher values risk hitting any rate limiting in the registry; document this in the README and leave tuning to the operator.
- **Result cleanup**: operators can delete `tests/stress/data/<entity>/<size>/` and `tests/stress/results/<backend>/size-<size>/` between runs. A `--clean` flag on each script is not provided; plain `rm -rf` works and is documented.

## 12. Acceptance Mapping (to Issue #997)

| Issue AC | Covered by |
|---|---|
| Generators produce valid payloads for 100/500/1000 | `generators/*.py` + schema validation step |
| `register_entities.py` registers on both backends with <1% failure | `register_entities.py`, idempotent, reports failure_rate |
| API p50/p95/p99 for list + semantic search at k=5,10,50 on both backends | `measure_api_performance.py` + `queries.json` |
| UI tti / DOM / search latency on both backends | `measure_ui_performance.py` |
| Cross-size and cross-backend reports | `report_builder.py` |
| Documented baselines | final `docs/performance-baselines.md` (produced by operator after first full run; not auto-generated) |

## 13. Out of Scope (for this LLD)

- Concurrent-client load testing (a separate issue will cover this).
- Automated regression detection on baselines (would require CI integration + stable infra; tracked separately once we have baselines).
- Migrating data/results to a companion repo (issue-level note, not an LLD concern).

## 14. Open Questions

1. **UI search `k` control**: is `max_results` exposed in the UI? If not, we need either a UI-only measurement (k=UI-default) plus a separate API-level k=5/10/50 measurement, or a short UI patch. Decide during Phase 1.
2. **DocumentDB test cluster**: use a dedicated stress-test cluster or piggyback on an existing non-prod cluster? Cost implications must be confirmed before the first DocumentDB run.
3. **Result retention**: how long do we keep results locally? Propose 30 days unless we move to a separate repo sooner.

## 15. Implementation Phases

**Phase 1** (generators + registration): deliver `generators/*.py`, `register_entities.py`, and `run_stress_test.sh` (stub that stops after step 2). Validate end-to-end by registering 100 entities on local mongodb-ce.

**Phase 2** (API measurement): deliver `measure_api_performance.py` and `queries.json`. Run against 100/500/1000 on mongodb-ce, confirm latencies are in a sane range.

**Phase 3** (UI measurement): deliver `measure_ui_performance.py`. Gate on resolving open question #1.

**Phase 4** (reporting + cross-backend): deliver `report_builder.py`, `run_all.sh`, and run the full matrix on both backends.

**Phase 5** (documentation): produce `docs/performance-baselines.md` from the Phase-4 results. This is the only artifact that commits non-script content to the repo.
