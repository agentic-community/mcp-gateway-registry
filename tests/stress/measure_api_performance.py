#!/usr/bin/env python3
"""Measure steady-state per-request API latency against a populated registry.

Reads a previously populated registry (one created by `register_entities.py`
at the requested `(backend, size)` pair), iterates each operation serially N
times, and writes per-operation latency percentiles to:

  tests/stress/results/<backend>/size-<N>/api_perf.json

Operations covered:

  - List endpoints (servers, agents, skills): first page (`limit=50`),
    max page (`limit=500`, the API's hard cap), pagination walkthrough.
  - Semantic search: each curated query iterated over k ∈ {5, 10, 50}
    against `POST /api/search/semantic` with `include_draft: true`.

Warmup handling
---------------
The first iteration of every operation is timed and checked for a non-error
status (so we don't silently mask a broken endpoint), but its `latency_ms`
is *discarded* before computing percentiles. The output JSON records
`warmup_strategy: "discard_first_iteration"` at the top level so consumers
can see why `iterations: 50` corresponds to 49 samples in the percentile
math. The rationale (embedding model lazy-load, MongoDB working-set warmup,
HTTP connection-pool establishment, FastAPI/Pydantic warm-paths) is
documented at length in `.scratchpad/phase2-plan.md`. See LLD §5 for the
overall measurement methodology.

Concurrency
-----------
All requests are issued serially -- we want steady-state per-request
latency, not concurrent-load throughput. LLD §5.4 calls this out
explicitly.

Token refresh
-------------
On 401 the loader re-resolves the JWT via the same convention as
`run_stress_test.sh` (regenerate via `keycloak/setup/generate-agent-token.sh`
when the file under `.oauth-tokens/` is expired) and retries once. No
pre-emptive refresh.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess  # nosec B404 - used only for the local keycloak token-regen script
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from tests.stress.config import (
    default_base_url,
    default_results_dir,
    default_token_file,
    fetch_registry_info,
    project_root,
    results_dir_for,
)
from tests.stress.constants import BACKENDS, HTTP_TIMEOUT_SECONDS, TARGET_SIZES
from tests.stress.generators._base import ensure_project_on_path
from tests.stress.queries import Query, default_queries_path, load_queries

ensure_project_on_path()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


WARMUP_STRATEGY: str = "discard_first_iteration"
SEMANTIC_K_VALUES: tuple[int, ...] = (5, 10, 50)
# Maps `queries.json` expected_entity_types values to the per-type result
# arrays the registry's semantic-search response uses.
EXPECTED_TYPE_TO_RESPONSE_KEY: dict[str, str] = {
    "mcp_server": "servers",
    "a2a_agent": "agents",
    "skill": "skills",
    "tool": "tools",
    "virtual_server": "virtual_servers",
}
LIST_PAGE_LIMIT: int = 50
# The registry's list endpoints enforce `limit <= 500` (Pydantic Field constraint
# on the route handler). LLD §5.2 calls for an "all rows" measurement; at sizes
# beyond 500 the operation effectively measures "as-many-as-the-API-allows".
# The paginated walkthrough is the operation that actually traverses the full
# corpus.
LIST_MAX_LIMIT: int = 500
LIST_ENTITY_TYPES: tuple[str, ...] = ("servers", "agents", "skills")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CallRecord(BaseModel):
    """One timed HTTP call."""

    iteration: int
    status_code: int | None
    latency_ms: float
    response_bytes: int = 0
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class OperationSummary(BaseModel):
    """Aggregate of N CallRecords for one operation."""

    name: str
    method: str
    url_pattern: str
    samples: int
    latency_ms: dict[str, float] = Field(default_factory=dict)
    error_count: int = 0
    notes: str | None = None
    # Populated only for semantic_search operations:
    query_id: str | None = None
    k: int | None = None
    expected_hits: int | None = None


# ---------------------------------------------------------------------------
# Token loading + 401-driven refresh
# ---------------------------------------------------------------------------


def _load_token(token_file: Path) -> str:
    if not token_file.exists():
        raise FileNotFoundError(
            f"Token file not found: {token_file}. "
            "Run keycloak/setup/generate-agent-token.sh first, "
            "or point STRESS_TOKEN_FILE at a valid JWT JSON."
        )
    raw = token_file.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        token = raw.strip()
        if not token:
            raise RuntimeError(f"Empty token file: {token_file}") from None
        return token

    token = data.get("access_token")
    if not token and "tokens" in data:
        token = data["tokens"].get("access_token")
    if not token and "token_data" in data:
        token = data["token_data"].get("access_token")
    if not token:
        raise RuntimeError(f"No 'access_token' field found in token file: {token_file}")
    return token


def _refresh_token_via_keycloak(token_file: Path) -> str:
    """Call the bundled keycloak generator script, then re-read the file.

    Returns the new access token. Raises if the script is missing or the
    file is still invalid afterwards.
    """
    generator = project_root() / "keycloak" / "setup" / "generate-agent-token.sh"
    if not generator.exists():
        raise RuntimeError(
            f"Cannot refresh token: keycloak generator script not found at {generator}"
        )

    logger.warning("Got 401; regenerating JWT via %s", generator)
    result = subprocess.run(  # nosec B603 - hardcoded internal script path
        ["bash", str(generator)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(project_root()),
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Token regen script exited {result.returncode}: {result.stderr[:500]}"
        )

    # The generator always writes the M2M token here regardless of caller's path.
    generated = project_root() / ".oauth-tokens" / "mcp-gateway-m2m-token.json"
    if generated.exists() and generated != token_file:
        logger.info("Using freshly-generated token from %s", generated)
        return _load_token(generated)
    return _load_token(token_file)


# ---------------------------------------------------------------------------
# Percentile math
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    return {
        "p50": _pct(sorted_vals, 50),
        "p95": _pct(sorted_vals, 95),
        "p99": _pct(sorted_vals, 99),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": statistics.mean(sorted_vals),
    }


def _pct(sorted_vals: list[float], pct: int) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct / 100
    lower = int(k)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = k - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * frac


# ---------------------------------------------------------------------------
# Core measurement: a single request with 401-driven refresh.
# ---------------------------------------------------------------------------


class TokenState:
    """Mutable container so refresh updates are visible to all callers."""

    def __init__(self, token: str, token_file: Path) -> None:
        self.token = token
        self.token_file = token_file
        self.refreshed = False  # we only refresh once per run

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


def _time_request(
    client: httpx.Client,
    method: str,
    url: str,
    token_state: TokenState,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> CallRecord:
    """Time a single HTTP call. Auto-refresh token once on 401."""
    return _time_request_inner(
        client, method, url, token_state, params, json_body, iteration_index=-1
    )


def _time_request_inner(
    client: httpx.Client,
    method: str,
    url: str,
    token_state: TokenState,
    params: dict[str, Any] | None,
    json_body: dict[str, Any] | None,
    iteration_index: int,
) -> CallRecord:
    start = time.perf_counter()
    try:
        resp = client.request(
            method,
            url,
            headers=token_state.headers(),
            params=params,
            json=json_body,
        )
    except httpx.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return CallRecord(
            iteration=iteration_index,
            status_code=None,
            latency_ms=elapsed_ms,
            error=f"http_error: {exc}",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    if resp.status_code == 401 and not token_state.refreshed:
        try:
            new_token = _refresh_token_via_keycloak(token_state.token_file)
        except Exception as exc:
            return CallRecord(
                iteration=iteration_index,
                status_code=401,
                latency_ms=elapsed_ms,
                error=f"token_refresh_failed: {exc}",
            )
        token_state.token = new_token
        token_state.refreshed = True
        return _time_request_inner(
            client, method, url, token_state, params, json_body, iteration_index
        )

    return CallRecord(
        iteration=iteration_index,
        status_code=resp.status_code,
        latency_ms=elapsed_ms,
        response_bytes=len(resp.content),
        error=None if resp.is_success else _truncate(resp.text, 300),
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


# ---------------------------------------------------------------------------
# Per-operation runners.
# ---------------------------------------------------------------------------


def _summarize(
    name: str,
    method: str,
    url_pattern: str,
    records: list[CallRecord],
    notes: str | None = None,
) -> OperationSummary:
    """Aggregate records, discarding the first one (warmup)."""
    if not records:
        return OperationSummary(
            name=name, method=method, url_pattern=url_pattern, samples=0, notes=notes
        )

    # Discard iteration 0 (warmup) before computing percentiles.
    countable = records[1:]
    successful_latencies = [
        r.latency_ms for r in countable if r.error is None and r.status_code is not None
    ]
    error_count = sum(1 for r in countable if r.error is not None)

    return OperationSummary(
        name=name,
        method=method,
        url_pattern=url_pattern,
        samples=len(countable),
        latency_ms=_percentiles(successful_latencies),
        error_count=error_count,
        notes=notes,
    )


def _run_list_first_page(
    client: httpx.Client,
    base_url: str,
    entity: str,
    iterations: int,
    token_state: TokenState,
) -> OperationSummary:
    url = f"{base_url}/api/{entity}"
    params = {"limit": LIST_PAGE_LIMIT, "offset": 0}
    name = f"list_{entity}_first_page"
    records: list[CallRecord] = []
    for i in range(iterations + 1):  # +1 to harvest the warmup sample
        r = _time_request_inner(client, "GET", url, token_state, params, None, i)
        records.append(r)
    return _summarize(name, "GET", f"/api/{entity}?limit={LIST_PAGE_LIMIT}&offset=0", records)


def _run_list_max(
    client: httpx.Client,
    base_url: str,
    entity: str,
    iterations: int,
    token_state: TokenState,
) -> OperationSummary:
    """List operation at the API's maximum allowed `limit` (500).

    At corpus sizes > 500 this captures only the first 500 rows; the
    paginated walkthrough operation is what traverses the full corpus.
    """
    url = f"{base_url}/api/{entity}"
    params = {"limit": LIST_MAX_LIMIT, "offset": 0}
    name = f"list_{entity}_max_page"
    records: list[CallRecord] = []
    for i in range(iterations + 1):
        r = _time_request_inner(client, "GET", url, token_state, params, None, i)
        records.append(r)
    return _summarize(
        name,
        "GET",
        f"/api/{entity}?limit={LIST_MAX_LIMIT}",
        records,
        notes="API caps limit at 500; this is `as much as a single request can return`.",
    )


def _run_list_paginated(
    client: httpx.Client,
    base_url: str,
    entity: str,
    size: int,
    iterations: int,
    token_state: TokenState,
) -> OperationSummary:
    """Walk pages 0..ceil(size/50)-1, one page at a time, repeated `iterations` times.

    Each page request is one timing sample; the first sample (page 0 of
    iteration 0) is the warmup. Total samples = iterations * pages_per_walk.
    """
    url = f"{base_url}/api/{entity}"
    pages_per_walk = max(1, math.ceil(size / LIST_PAGE_LIMIT))
    name = f"list_{entity}_paginated"
    records: list[CallRecord] = []
    sample_index = 0
    for _ in range(iterations + 1):
        for page in range(pages_per_walk):
            params = {"limit": LIST_PAGE_LIMIT, "offset": page * LIST_PAGE_LIMIT}
            r = _time_request_inner(client, "GET", url, token_state, params, None, sample_index)
            records.append(r)
            sample_index += 1
    return _summarize(
        name,
        "GET",
        f"/api/{entity}?limit={LIST_PAGE_LIMIT}&offset={{0..{pages_per_walk - 1}*{LIST_PAGE_LIMIT}}}",
        records,
        notes=f"each iteration walks {pages_per_walk} pages",
    )


def _count_expected_hits(
    response_json: dict[str, Any],
    expected_types: list[str],
) -> int:
    """Count entries in the response whose entity-type matches `expected_types`.

    The registry's `/api/search/semantic` response groups results by entity type
    into per-type arrays (`servers`, `agents`, `skills`, `tools`, `virtual_servers`).
    We sum the lengths of the arrays the caller cares about.
    """
    total = 0
    for et in expected_types:
        key = EXPECTED_TYPE_TO_RESPONSE_KEY.get(et)
        if not key:
            continue
        arr = response_json.get(key) or []
        if isinstance(arr, list):
            total += len(arr)
    return total


def _run_semantic_search(
    client: httpx.Client,
    base_url: str,
    query: Query,
    k: int,
    iterations: int,
    token_state: TokenState,
) -> OperationSummary:
    """Issue `iterations + 1` semantic-search requests for a (query, k) pair.

    The first iteration is the warmup discard. `expected_hits` is captured
    from the last successful response (it should be stable across iterations
    against an unchanging corpus).
    """
    url = f"{base_url}/api/search/semantic"
    body = {
        "query": query.query,
        "entity_types": query.expected_entity_types,
        "max_results": k,
        "include_disabled": False,
        "include_draft": True,
    }
    records: list[CallRecord] = []
    last_expected_hits: int | None = None

    for i in range(iterations + 1):
        start = time.perf_counter()
        try:
            resp = client.post(
                url,
                headers=token_state.headers(),
                json=body,
            )
        except httpx.HTTPError as exc:
            records.append(
                CallRecord(
                    iteration=i,
                    status_code=None,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    error=f"http_error: {exc}",
                )
            )
            continue

        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 401 and not token_state.refreshed:
            try:
                token_state.token = _refresh_token_via_keycloak(token_state.token_file)
                token_state.refreshed = True
                # retry once
                start = time.perf_counter()
                resp = client.post(url, headers=token_state.headers(), json=body)
                elapsed_ms = (time.perf_counter() - start) * 1000
            except Exception as exc:
                records.append(
                    CallRecord(
                        iteration=i,
                        status_code=401,
                        latency_ms=elapsed_ms,
                        error=f"token_refresh_failed: {exc}",
                    )
                )
                continue

        if resp.is_success:
            try:
                last_expected_hits = _count_expected_hits(resp.json(), query.expected_entity_types)
            except Exception:
                last_expected_hits = None
            records.append(
                CallRecord(
                    iteration=i,
                    status_code=resp.status_code,
                    latency_ms=elapsed_ms,
                    response_bytes=len(resp.content),
                )
            )
        else:
            records.append(
                CallRecord(
                    iteration=i,
                    status_code=resp.status_code,
                    latency_ms=elapsed_ms,
                    error=_truncate(resp.text, 300),
                )
            )

    summary = _summarize("semantic_search", "POST", "/api/search/semantic", records)
    summary.query_id = query.id
    summary.k = k
    summary.expected_hits = last_expected_hits
    return summary


# ---------------------------------------------------------------------------
# Markdown writer.
# ---------------------------------------------------------------------------


def _fmt_ms(value: float | None) -> str:
    """Format a latency_ms value for the Markdown report."""
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _build_api_perf_md(report: dict[str, Any]) -> str:
    """Render the JSON report as a human-readable Markdown document.

    The Markdown is the artifact reviewers will eyeball; the JSON is the
    machine-readable source of truth that report_builder.py will consume
    in Phase 4 to produce cross-(backend, size) and cross-backend tables.
    """
    operations: list[dict[str, Any]] = report.get("operations", [])
    list_ops = [op for op in operations if op["name"].startswith("list_")]
    search_ops = [op for op in operations if op["name"] == "semantic_search"]

    lines: list[str] = []
    lines.append(f"# API performance — {report['backend']} @ size={report['size']}")
    lines.append("")
    lines.append(f"- Iterations: **{report['iterations']}** "
                 f"(samples per op after warmup discard)")
    lines.append(f"- Warmup strategy: `{report['warmup_strategy']}` "
                 "(first iteration timed but excluded from percentile math)")
    lines.append(f"- Base URL: `{report['base_url']}`")
    lines.append(f"- Collected at: `{report['collected_at']}`")
    lines.append(f"- Wall clock: **{report['wall_clock_seconds']:.1f}** s")
    lines.append("")

    lines.append("## List endpoints")
    lines.append("")
    lines.append("| Operation | Samples | p50 ms | p95 ms | p99 ms | min ms | max ms | mean ms | errors |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for op in list_ops:
        lat = op.get("latency_ms") or {}
        lines.append(
            "| `{name}` | {samples} | {p50} | {p95} | {p99} | {mn} | {mx} | {mean} | {err} |".format(
                name=op["name"],
                samples=op["samples"],
                p50=_fmt_ms(lat.get("p50")),
                p95=_fmt_ms(lat.get("p95")),
                p99=_fmt_ms(lat.get("p99")),
                mn=_fmt_ms(lat.get("min")),
                mx=_fmt_ms(lat.get("max")),
                mean=_fmt_ms(lat.get("mean")),
                err=op.get("error_count", 0),
            )
        )
    lines.append("")

    notes = [op for op in list_ops if op.get("notes")]
    if notes:
        lines.append("### Notes")
        for op in notes:
            lines.append(f"- **`{op['name']}`**: {op['notes']}")
        lines.append("")

    lines.append("## Semantic search")
    lines.append("")
    lines.append("Query body sets `include_draft: true`; Phase 1 entities are registered as "
                 "`status: draft`.")
    lines.append("")
    lines.append("| Query | k | Samples | p50 ms | p95 ms | p99 ms | mean ms | hits | errors |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for op in search_ops:
        lat = op.get("latency_ms") or {}
        lines.append(
            "| `{qid}` | {k} | {samples} | {p50} | {p95} | {p99} | {mean} | {hits} | {err} |".format(
                qid=op.get("query_id", "?"),
                k=op.get("k", "?"),
                samples=op["samples"],
                p50=_fmt_ms(lat.get("p50")),
                p95=_fmt_ms(lat.get("p95")),
                p99=_fmt_ms(lat.get("p99")),
                mean=_fmt_ms(lat.get("mean")),
                hits=op.get("expected_hits", "?"),
                err=op.get("error_count", 0),
            )
        )
    lines.append("")

    bad = [op for op in operations if op.get("error_count", 0) > 0]
    if bad:
        lines.append("## Operations with errors")
        lines.append("")
        for op in bad:
            lines.append(f"- `{op['name']}`"
                         + (f" query=`{op['query_id']}` k={op['k']}" if op.get("query_id") else "")
                         + f": error_count={op['error_count']}, samples={op['samples']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def _detect_backend(
    base_url: str,
    token: str,
) -> str:
    """Auto-detect storage backend from the registry's /api/stats endpoint."""
    import httpx as _httpx

    headers = {"Authorization": f"Bearer {token}"}
    resp = _httpx.get(f"{base_url.rstrip('/')}/api/stats", headers=headers, timeout=10)
    resp.raise_for_status()
    backend = resp.json()["database_status"]["backend"]
    logger.info("Auto-detected backend: %s", backend)
    return backend


def _main(args: argparse.Namespace) -> int:
    token_file: Path = args.token_file
    try:
        token = _load_token(token_file)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    token_state = TokenState(token, token_file)

    # Auto-detect backend if not provided
    if args.backend is None:
        try:
            args.backend = _detect_backend(args.base_url, token)
        except Exception as exc:
            logger.error("Failed to auto-detect backend: %s", exc)
            logger.error("Provide --backend explicitly or ensure the registry is reachable.")
            return 1

    output_dir = results_dir_for(args.backend, args.size, args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        queries = load_queries(args.queries_file)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Loaded %d curated queries from %s", len(queries), args.queries_file)

    operations: list[OperationSummary] = []
    overall_start = time.time()

    with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for entity in LIST_ENTITY_TYPES:
            logger.info("Measuring list_%s (iterations=%d)", entity, args.iterations)
            operations.append(
                _run_list_first_page(client, args.base_url, entity, args.iterations, token_state)
            )
            operations.append(
                _run_list_max(client, args.base_url, entity, args.iterations, token_state)
            )
            operations.append(
                _run_list_paginated(
                    client, args.base_url, entity, args.size, args.iterations, token_state
                )
            )

        for q in queries:
            for k in SEMANTIC_K_VALUES:
                logger.info(
                    "Measuring semantic_search query_id=%s k=%d (iterations=%d)",
                    q.id,
                    k,
                    args.iterations,
                )
                operations.append(
                    _run_semantic_search(
                        client, args.base_url, q, k, args.iterations, token_state
                    )
                )

    registry_info = fetch_registry_info(args.base_url, token_state.token)

    overall = {
        "backend": args.backend,
        "size": args.size,
        "iterations": args.iterations,
        "warmup_strategy": WARMUP_STRATEGY,
        "base_url": args.base_url,
        "collected_at": datetime.now(UTC).isoformat(),
        "wall_clock_seconds": time.time() - overall_start,
        "registry_info": registry_info,
        "operations": [op.model_dump() for op in operations],
    }

    out_file = output_dir / "api_perf.json"
    out_file.write_text(json.dumps(overall, indent=2, default=str))
    logger.info("Wrote api_perf report: %s", out_file)

    md_file = output_dir / "api_perf.md"
    md_file.write_text(_build_api_perf_md(overall))
    logger.info("Wrote api_perf markdown: %s", md_file)

    # Surface obvious problems as non-zero exit code so CI can react.
    bad_ops = [op for op in operations if op.error_count > 0 or not op.latency_ms]
    if bad_ops:
        for op in bad_ops:
            logger.warning(
                "Operation %s had error_count=%d, samples=%d",
                op.name,
                op.error_count,
                op.samples,
            )
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure steady-state per-request latency against a populated registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="Storage backend. Auto-detected from /api/stats if not provided.",
    )
    parser.add_argument("--size", type=int, required=True, choices=TARGET_SIZES)
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Steady-state samples per operation (default: 50). "
        "The script issues `iterations + 1` requests per operation and discards the first.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=default_queries_path(),
        help="Curated query set (default: tests/stress/queries.json).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir(),
        help="Root results directory (default: tests/stress/results/).",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=default_token_file(),
        help="JWT token file (default: .oauth-tokens/ingress.json).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    return _main(args)


if __name__ == "__main__":
    sys.exit(main())
