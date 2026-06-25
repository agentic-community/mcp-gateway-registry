#!/usr/bin/env python3
"""Concurrent load test for the auth-server ``/validate`` hot path (issue #1316).

``/validate`` is the nginx ``auth_request`` subrequest that runs on every
proxied request. It is CPU-bound (JWT verify + scope/tool authz), so it is the
endpoint the free-threaded Python 3.14t trial is meant to speed up under
concurrency. This script drives concurrent load directly at the auth-server
``/validate`` endpoint (bypassing nginx) on both the success and error paths,
then reads latency percentiles from Prometheus so a before/after comparison can
be made across interpreter builds.

Two request mixes are exercised:
  - ``success``: a valid Bearer token -> HTTP 200
  - ``error``: the valid token with its signature corrupted -> HTTP 401
    (fail-closed path)

Both are real ``/validate`` work: the 401 path still parses headers and
attempts JWT validation, so its latency is meaningful.

Workflow for a before/after comparison (GIL on vs off, same image)
------------------------------------------------------------------
The free-threaded image runs with the GIL re-enabled via PYTHON_GIL=1, so no
rebuild is needed. Toggle it through the auth-server extra_env file:
1. GIL ON (baseline):
     printf 'PYTHON_GIL=1\n' > extra_env/auth-server.env
     docker compose up -d --no-deps auth-server   # log: GIL enabled at runtime=True
     uv run python -m tests.stress.measure_validate_latency \
         --label gil-on --concurrency 100 --out tests/stress/results/validate/gil-on.json
2. GIL OFF (free-threaded), then compare:
     rm -f extra_env/auth-server.env
     docker compose up -d --no-deps auth-server   # log: GIL enabled at runtime=False
     uv run python -m tests.stress.measure_validate_latency \
         --label gil-off --free-threaded --concurrency 100 \
         --out tests/stress/results/validate/gil-off.json \
         --compare-to tests/stress/results/validate/gil-on.json

Latency source
--------------
Server-side percentiles come from the ``http_server_duration_milliseconds``
histogram (FastAPI auto-instrumentation), filtered to ``http_target="/validate"``
and split by ``http_status_code``. To isolate exactly one run's requests, the
script snapshots the cumulative histogram buckets *before* and *after* the load
and computes the quantile from the (after - before) delta. This is immune to the
rate-window bleed that contaminates back-to-back runs, so the server-side
numbers are accurate even with no gap between runs. The client-observed wall
time per request is also recorded as a sanity cross-check.

Note on the Prometheus scrape gap: the script waits one scrape interval
(``--scrape-wait``) after the load finishes so the final samples are visible to
Prometheus before the "after" snapshot is taken.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_AUTH_URL: str = "http://localhost:8888"
DEFAULT_PROM_URL: str = "http://localhost:9090"
DEFAULT_TOKEN_FILE: str = ".token"
DEFAULT_ORIGINAL_URL: str = "http://localhost/api/servers"
DEFAULT_REQUESTS: int = 2000
DEFAULT_CONCURRENCY: int = 50
DEFAULT_SCRAPE_WAIT_S: int = 20


class PathResult(BaseModel):
    """Client-side timing for one request mix (success or error)."""

    path: str = Field(..., description="'success' or 'error'")
    expected_status: int = Field(..., description="HTTP status the path should return")
    total_requests: int = Field(..., description="Requests issued")
    status_counts: dict[str, int] = Field(default_factory=dict)
    client_p50_ms: float = 0.0
    client_p95_ms: float = 0.0
    client_p99_ms: float = 0.0
    throughput_rps: float = 0.0
    wall_seconds: float = 0.0


class PromLatency(BaseModel):
    """Server-side /validate latency read from Prometheus, by status code."""

    status_code: str = Field(..., description="HTTP status code label")
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    avg_ms: float | None = None
    request_count: float | None = None


class BenchmarkReport(BaseModel):
    """Top-level result, serialized to JSON for before/after comparison."""

    label: str
    timestamp: str
    auth_url: str
    prometheus_url: str
    concurrency: int
    requests_per_path: int
    free_threaded: bool | None = None
    client_results: list[PathResult] = Field(default_factory=list)
    server_latency: list[PromLatency] = Field(default_factory=list)


class SweepCell(BaseModel):
    """Aggregated (median over repeats) result for one (concurrency) cell."""

    concurrency: int
    repeats: int
    # Server-side success-path percentiles (the CPU-bound JWT-verify work).
    server_200_p50_ms: float | None = None
    server_200_p95_ms: float | None = None
    server_200_p99_ms: float | None = None
    # Client-side success-path throughput and tail.
    client_200_p95_ms: float | None = None
    client_200_rps: float | None = None
    # Per-repeat raw reports kept for auditability.
    runs: list[BenchmarkReport] = Field(default_factory=list)


class SweepReport(BaseModel):
    """A full concurrency sweep at a fixed GIL state."""

    label: str
    timestamp: str
    free_threaded: bool | None = None
    concurrency_levels: list[int] = Field(default_factory=list)
    repeats: int = 1
    requests_per_path: int = 0
    cells: list[SweepCell] = Field(default_factory=list)


def _read_token(
    token_file: Path,
) -> str:
    """Read a JWT from a token file (raw token or JSON with tokens.access_token)."""
    if not token_file.is_file():
        raise FileNotFoundError(
            f"Token file not found: {token_file}. Provide a valid JWT file via "
            f"--token-file (default: {DEFAULT_TOKEN_FILE})."
        )
    raw = token_file.read_text().strip()
    if not raw:
        raise ValueError(f"Token file is empty: {token_file}")
    if raw.startswith("{"):
        data = json.loads(raw)
        token = data.get("tokens", {}).get("access_token") or data.get("access_token")
        if not token:
            raise ValueError(f"No access_token found in JSON token file: {token_file}")
        return token
    return raw


def _derive_error_token(
    valid_token: str,
) -> str:
    """Derive a guaranteed-invalid JWT from a valid one by corrupting its signature.

    Keeps the real ``header.payload`` so the error path still runs the full JWT
    decode + signature-verification work (the CPU cost #1316 targets) before
    failing closed with HTTP 401, rather than rejecting on a malformed-structure
    shortcut. The signature segment is replaced with a fixed bogus value that
    cannot verify against any key.
    """
    parts = valid_token.split(".")
    if len(parts) != 3:
        # Not a standard 3-segment JWT; fall back to appending a bad segment so
        # the result is still structurally a JWT that fails verification.
        return f"{valid_token}.aW52YWxpZA"
    header, payload, _signature = parts
    return f"{header}.{payload}.aW52YWxpZHNpZ25hdHVyZQ"


def _percentile(
    samples: list[float],
    pct: float,
) -> float:
    """Return the pct (0-1) percentile of samples in milliseconds."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(0, min(len(ordered) - 1, int(round(pct * (len(ordered) - 1)))))
    return ordered[rank]


async def _fire_one(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
) -> tuple[int, float]:
    """Issue one GET /validate, returning (status_code, latency_ms)."""
    start = time.perf_counter()
    try:
        resp = await client.get(url, headers=headers)
        status = resp.status_code
    except httpx.HTTPError:
        status = 0
    latency_ms = (time.perf_counter() - start) * 1000.0
    return status, latency_ms


async def _run_path(
    path_name: str,
    expected_status: int,
    auth_url: str,
    headers: dict[str, str],
    requests_count: int,
    concurrency: int,
) -> PathResult:
    """Drive `requests_count` concurrent GET /validate calls for one mix."""
    url = f"{auth_url.rstrip('/')}/validate"
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    status_counts: dict[str, int] = {}

    async def _bounded(client: httpx.AsyncClient) -> None:
        async with semaphore:
            status, latency_ms = await _fire_one(client, url, headers)
            latencies.append(latency_ms)
            key = str(status)
            status_counts[key] = status_counts.get(key, 0) + 1

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    wall_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        tasks = [_bounded(client) for _ in range(requests_count)]
        await asyncio.gather(*tasks)
    wall_seconds = time.perf_counter() - wall_start

    logger.info(
        "Path '%s' done: %d requests in %.2fs, status counts=%s",
        path_name,
        requests_count,
        wall_seconds,
        status_counts,
    )
    return PathResult(
        path=path_name,
        expected_status=expected_status,
        total_requests=requests_count,
        status_counts=status_counts,
        client_p50_ms=round(_percentile(latencies, 0.50), 3),
        client_p95_ms=round(_percentile(latencies, 0.95), 3),
        client_p99_ms=round(_percentile(latencies, 0.99), 3),
        throughput_rps=round(requests_count / wall_seconds, 1) if wall_seconds else 0.0,
        wall_seconds=round(wall_seconds, 3),
    )


def _prom_query(
    prom_url: str,
    query: str,
) -> float | None:
    """Run an instant PromQL query, returning the first scalar value or None."""
    try:
        resp = httpx.get(
            f"{prom_url.rstrip('/')}/api/v1/query",
            params={"query": query},
            timeout=15.0,
        )
        resp.raise_for_status()
        results = resp.json().get("data", {}).get("result", [])
        if not results:
            return None
        return float(results[0]["value"][1])
    except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
        logger.warning("Prometheus query failed (%s): %s", query, exc)
        return None


def _prom_query_buckets(
    prom_url: str,
    query: str,
) -> dict[str, float]:
    """Run an instant query and return a {le_label: value} map for histogram buckets."""
    try:
        resp = httpx.get(
            f"{prom_url.rstrip('/')}/api/v1/query",
            params={"query": query},
            timeout=15.0,
        )
        resp.raise_for_status()
        out: dict[str, float] = {}
        for r in resp.json().get("data", {}).get("result", []):
            le = r.get("metric", {}).get("le")
            if le is not None:
                out[le] = float(r["value"][1])
        return out
    except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
        logger.warning("Prometheus bucket query failed (%s): %s", query, exc)
        return {}


def _le_value(
    le: str,
) -> float:
    """Parse an ``le`` bucket-boundary label to a float (handles +Inf)."""
    if le in ("+Inf", "Inf"):
        return float("inf")
    return float(le)


def _histogram_quantile(
    buckets: dict[str, float],
    pct: float,
) -> float | None:
    """Compute the pct (0-1) quantile from cumulative {le: count} buckets, in ms.

    Mirrors Prometheus ``histogram_quantile``: linear interpolation within the
    bucket that contains the target rank. ``buckets`` must be the *delta* over
    one run so the quantile reflects only that run's requests.
    """
    if not buckets:
        return None
    items = sorted(buckets.items(), key=lambda kv: _le_value(kv[0]))
    total = items[-1][1]  # the +Inf bucket holds the full count
    if total <= 0:
        return None

    rank = pct * total
    prev_le = 0.0
    prev_count = 0.0
    for le, cum in items:
        if cum >= rank:
            upper = _le_value(le)
            if upper == float("inf"):
                # Rank falls in the open-ended top bucket: best estimate is its
                # lower (last finite) boundary.
                return prev_le
            if cum == prev_count:
                return upper
            frac = (rank - prev_count) / (cum - prev_count)
            return prev_le + frac * (upper - prev_le)
        prev_le = _le_value(le) if _le_value(le) != float("inf") else prev_le
        prev_count = cum
    return prev_le


def _snapshot_histogram(
    prom_url: str,
    status_code: str,
) -> tuple[dict[str, float], float, float]:
    """Snapshot cumulative /validate histogram buckets, sum, and count for a status.

    Returns ``(buckets_by_le, sum_ms, count)`` at the latest Prometheus scrape.
    Two snapshots (before/after a run) are subtracted to isolate that run's
    latency, which is immune to the rate-window bleed that affects back-to-back
    runs.
    """
    labels = f'{{http_target="/validate",http_status_code="{status_code}"}}'
    base = "http_server_duration_milliseconds"
    buckets = _prom_query_buckets(prom_url, f"sum by (le) ({base}_bucket{labels})")
    total_sum = _prom_query(prom_url, f"sum({base}_sum{labels})") or 0.0
    total_count = _prom_query(prom_url, f"sum({base}_count{labels})") or 0.0
    return buckets, total_sum, total_count


def _delta_latency(
    status_code: str,
    before: tuple[dict[str, float], float, float],
    after: tuple[dict[str, float], float, float],
) -> PromLatency:
    """Compute per-run latency percentiles from before/after histogram snapshots."""
    before_buckets, before_sum, before_count = before
    after_buckets, after_sum, after_count = after

    delta_buckets = {
        le: after_buckets.get(le, 0.0) - before_buckets.get(le, 0.0) for le in after_buckets
    }
    delta_count = after_count - before_count
    delta_sum = after_sum - before_sum
    avg = (delta_sum / delta_count) if delta_count > 0 else None

    def _round(v: float | None) -> float | None:
        return round(v, 3) if v is not None else None

    return PromLatency(
        status_code=status_code,
        p50_ms=_round(_histogram_quantile(delta_buckets, 0.50)),
        p95_ms=_round(_histogram_quantile(delta_buckets, 0.95)),
        p99_ms=_round(_histogram_quantile(delta_buckets, 0.99)),
        avg_ms=_round(avg),
        request_count=round(delta_count) if delta_count > 0 else 0,
    )


async def _run_single(
    auth_url: str,
    prometheus_url: str,
    token: str,
    original_url: str,
    concurrency: int,
    requests_count: int,
    scrape_wait: int,
    label: str,
    free_threaded: bool,
) -> BenchmarkReport:
    """Run both request mixes once at a fixed concurrency and read server latency.

    Snapshots the cumulative histograms before the load, runs both paths, waits
    one scrape interval so the final samples are visible, then snapshots again.
    Per-run latency is the (after - before) delta -- immune to rate-window bleed
    between back-to-back runs, so the server-side numbers are accurate even with
    no gap between runs.
    """
    success_headers = {
        "Authorization": f"Bearer {token}",
        "X-Original-URL": original_url,
    }
    error_headers = {
        "Authorization": f"Bearer {_derive_error_token(token)}",
        "X-Original-URL": original_url,
    }

    logger.info("Run '%s': %d requests/path at concurrency %d", label, requests_count, concurrency)

    before_200 = _snapshot_histogram(prometheus_url, "200")
    before_401 = _snapshot_histogram(prometheus_url, "401")

    success = await _run_path(
        "success", 200, auth_url, success_headers, requests_count, concurrency
    )
    error = await _run_path("error", 401, auth_url, error_headers, requests_count, concurrency)

    logger.info("Waiting %ds for Prometheus to scrape final samples...", scrape_wait)
    await asyncio.sleep(scrape_wait)

    after_200 = _snapshot_histogram(prometheus_url, "200")
    after_401 = _snapshot_histogram(prometheus_url, "401")
    server_latency = [
        _delta_latency("200", before_200, after_200),
        _delta_latency("401", before_401, after_401),
    ]

    return BenchmarkReport(
        label=label,
        timestamp=datetime.now(UTC).isoformat(),
        auth_url=auth_url,
        prometheus_url=prometheus_url,
        concurrency=concurrency,
        requests_per_path=requests_count,
        free_threaded=free_threaded,
        client_results=[success, error],
        server_latency=server_latency,
    )


async def _run_benchmark(
    args: argparse.Namespace,
) -> BenchmarkReport:
    """Single-run entry point: drive one concurrency level once."""
    token = _read_token(Path(args.token_file))
    return await _run_single(
        auth_url=args.auth_url,
        prometheus_url=args.prometheus_url,
        token=token,
        original_url=args.original_url,
        concurrency=args.concurrency,
        requests_count=args.requests,
        scrape_wait=args.scrape_wait,
        label=args.label,
        free_threaded=args.free_threaded,
    )


def _median(
    values: list[float],
) -> float | None:
    """Median of non-None values, or None if empty."""
    clean = [v for v in values if v is not None]
    return round(statistics.median(clean), 3) if clean else None


def _server_200(
    report: BenchmarkReport,
) -> PromLatency | None:
    """Return the success-path (HTTP 200) server latency from a report."""
    for s in report.server_latency:
        if s.status_code == "200":
            return s
    return None


def _client_200(
    report: BenchmarkReport,
) -> PathResult | None:
    """Return the success-path client result from a report."""
    for r in report.client_results:
        if r.path == "success":
            return r
    return None


def _aggregate_cell(
    concurrency: int,
    runs: list[BenchmarkReport],
) -> SweepCell:
    """Aggregate repeats for one concurrency level using the median per metric."""
    server = [_server_200(r) for r in runs]
    client = [_client_200(r) for r in runs]
    return SweepCell(
        concurrency=concurrency,
        repeats=len(runs),
        server_200_p50_ms=_median([s.p50_ms for s in server if s]),
        server_200_p95_ms=_median([s.p95_ms for s in server if s]),
        server_200_p99_ms=_median([s.p99_ms for s in server if s]),
        client_200_p95_ms=_median([c.client_p95_ms for c in client if c]),
        client_200_rps=_median([c.throughput_rps for c in client if c]),
        runs=runs,
    )


async def _run_sweep(
    args: argparse.Namespace,
    levels: list[int],
) -> SweepReport:
    """Run every concurrency level `repeats` times and aggregate by median."""
    token = _read_token(Path(args.token_file))
    cells: list[SweepCell] = []
    for concurrency in levels:
        runs: list[BenchmarkReport] = []
        for i in range(args.repeats):
            report = await _run_single(
                auth_url=args.auth_url,
                prometheus_url=args.prometheus_url,
                token=token,
                original_url=args.original_url,
                concurrency=concurrency,
                requests_count=args.requests,
                scrape_wait=args.scrape_wait,
                label=f"{args.label}-c{concurrency}-r{i + 1}",
                free_threaded=args.free_threaded,
            )
            runs.append(report)
        cell = _aggregate_cell(concurrency, runs)
        cells.append(cell)
        logger.info(
            "Cell c=%d done: server200 p95=%s p99=%s (median of %d)",
            concurrency,
            cell.server_200_p95_ms,
            cell.server_200_p99_ms,
            cell.repeats,
        )

    return SweepReport(
        label=args.label,
        timestamp=datetime.now(UTC).isoformat(),
        free_threaded=args.free_threaded,
        concurrency_levels=levels,
        repeats=args.repeats,
        requests_per_path=args.requests,
        cells=cells,
    )


def _fmt(
    value: float | None,
) -> str:
    """Format an optional millisecond value for a table cell."""
    return f"{value:.2f}" if value is not None else "-"


def _print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    """Print a left-titled, right-aligned ASCII table to stdout."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: list[str]) -> str:
        # First column left-aligned (labels), the rest right-aligned (numbers).
        out = [cells[0].ljust(widths[0])]
        out += [cells[i].rjust(widths[i]) for i in range(1, len(cells))]
        return "  ".join(out)

    total = sum(widths) + 2 * (len(headers) - 1)
    print(f"\n{title}")
    print("-" * total)
    print(_line(headers))
    print("-" * total)
    for row in rows:
        print(_line(row))
    print("-" * total)


def _print_report(
    report: BenchmarkReport,
) -> None:
    """Print a readable table summary of one benchmark run."""
    print(f"\n{'=' * 72}")
    print(f"Benchmark '{report.label}'  (concurrency={report.concurrency})")

    client_rows = [
        [
            r.path,
            str(sum(r.status_counts.values())),
            ",".join(f"{k}={v}" for k, v in sorted(r.status_counts.items())),
            f"{r.client_p50_ms:.2f}",
            f"{r.client_p95_ms:.2f}",
            f"{r.client_p99_ms:.2f}",
            f"{r.throughput_rps:.1f}",
        ]
        for r in report.client_results
    ]
    _print_table(
        "CLIENT (wall-clock, as observed by the load generator)",
        ["path", "requests", "status", "p50 ms", "p95 ms", "p99 ms", "rps"],
        client_rows,
    )

    server_rows = [
        [
            s.status_code,
            _fmt(s.p50_ms),
            _fmt(s.p95_ms),
            _fmt(s.p99_ms),
            _fmt(s.avg_ms),
            f"{int(s.request_count)}" if s.request_count is not None else "-",
        ]
        for s in report.server_latency
    ]
    _print_table(
        "SERVER (/validate latency from Prometheus, by status code)",
        ["status", "p50 ms", "p95 ms", "p99 ms", "avg ms", "count"],
        server_rows,
    )
    print(f"{'=' * 72}\n")


def _print_comparison(
    current: BenchmarkReport,
    baseline_path: Path,
) -> None:
    """Print a table comparing current server-side latency against a baseline."""
    baseline = BenchmarkReport(**json.loads(baseline_path.read_text()))
    base_by_status = {s.status_code: s for s in baseline.server_latency}

    rows: list[list[str]] = []
    for s in current.server_latency:
        b = base_by_status.get(s.status_code)
        if not b:
            continue
        for metric in ("p50_ms", "p95_ms", "p99_ms"):
            cur = getattr(s, metric)
            old = getattr(b, metric)
            if cur is None or old is None or old == 0:
                continue
            delta_pct = (cur - old) / old * 100.0
            verdict = "improved" if delta_pct < 0 else "regressed"
            rows.append(
                [
                    s.status_code,
                    metric.replace("_ms", ""),
                    f"{old:.2f}",
                    f"{cur:.2f}",
                    f"{delta_pct:+.1f}%",
                    verdict,
                ]
            )

    _print_table(
        f"SERVER COMPARISON: '{baseline.label}' (baseline) -> '{current.label}' (current)",
        ["status", "pctl", "baseline ms", "current ms", "delta", "verdict"],
        rows,
    )
    print(f"{'=' * 72}\n")


def _print_sweep(
    sweep: SweepReport,
) -> None:
    """Print a sweep as a concurrency-by-metric table (median over repeats)."""
    print(f"\n{'=' * 72}")
    print(
        f"SWEEP '{sweep.label}'  (repeats={sweep.repeats}, "
        f"requests/path={sweep.requests_per_path}, free_threaded={sweep.free_threaded})"
    )
    rows = [
        [
            str(c.concurrency),
            _fmt(c.server_200_p50_ms),
            _fmt(c.server_200_p95_ms),
            _fmt(c.server_200_p99_ms),
            _fmt(c.client_200_p95_ms),
            f"{c.client_200_rps:.1f}" if c.client_200_rps is not None else "-",
        ]
        for c in sweep.cells
    ]
    _print_table(
        "SUCCESS path, median over repeats (server-side unless noted)",
        ["concur", "srv p50", "srv p95", "srv p99", "cli p95", "cli rps"],
        rows,
    )
    print(f"{'=' * 72}\n")


def _print_sweep_comparison(
    current: SweepReport,
    baseline_path: Path,
) -> None:
    """Print a per-concurrency baseline-vs-current sweep comparison (server p95/p99)."""
    baseline = SweepReport(**json.loads(baseline_path.read_text()))
    base_by_c = {c.concurrency: c for c in baseline.cells}

    rows: list[list[str]] = []
    for c in current.cells:
        b = base_by_c.get(c.concurrency)
        if not b:
            continue
        for metric in ("server_200_p95_ms", "server_200_p99_ms"):
            cur = getattr(c, metric)
            old = getattr(b, metric)
            if cur is None or old is None or old == 0:
                continue
            delta_pct = (cur - old) / old * 100.0
            verdict = "improved" if delta_pct < 0 else "regressed"
            rows.append(
                [
                    str(c.concurrency),
                    metric.replace("server_200_", "").replace("_ms", ""),
                    f"{old:.2f}",
                    f"{cur:.2f}",
                    f"{delta_pct:+.1f}%",
                    verdict,
                ]
            )

    _print_table(
        f"SWEEP COMPARISON (success p95/p99): '{baseline.label}' -> '{current.label}'",
        ["concur", "pctl", "baseline ms", "current ms", "delta", "verdict"],
        rows,
    )
    print(f"{'=' * 72}\n")


def _parse_concurrency_list(
    raw: str,
) -> list[int]:
    """Parse a comma-separated concurrency list like '50,75,100' into ints."""
    levels = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            levels.append(int(part))
    if not levels:
        raise ValueError(f"No valid concurrency levels parsed from: {raw!r}")
    return levels


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Concurrent load test for the auth-server /validate hot path (#1316).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage (GIL on vs off A/B, no rebuild -- toggle PYTHON_GIL via extra_env):
    # GIL ON baseline
    printf 'PYTHON_GIL=1\\n' > extra_env/auth-server.env
    docker compose up -d --no-deps auth-server
    uv run python -m tests.stress.measure_validate_latency \\
        --label gil-on --concurrency 100 \\
        --out tests/stress/results/validate/gil-on.json

    # GIL OFF (free-threaded), then compare
    rm -f extra_env/auth-server.env
    docker compose up -d --no-deps auth-server
    uv run python -m tests.stress.measure_validate_latency \\
        --label gil-off --free-threaded --concurrency 100 \\
        --out tests/stress/results/validate/gil-off.json \\
        --compare-to tests/stress/results/validate/gil-on.json
""",
    )
    parser.add_argument("--label", default="run", help="Label for this run (e.g. baseline)")
    parser.add_argument("--auth-url", default=DEFAULT_AUTH_URL, help="Auth-server base URL")
    parser.add_argument("--prometheus-url", default=DEFAULT_PROM_URL, help="Prometheus base URL")
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE, help="JWT token file path")
    parser.add_argument(
        "--original-url", default=DEFAULT_ORIGINAL_URL, help="X-Original-URL header value"
    )
    parser.add_argument(
        "--requests", type=int, default=DEFAULT_REQUESTS, help="Requests per path (success/error)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent requests"
    )
    parser.add_argument(
        "--scrape-wait",
        type=int,
        default=DEFAULT_SCRAPE_WAIT_S,
        help="Seconds to wait for Prometheus scrape after load",
    )
    parser.add_argument(
        "--free-threaded",
        action="store_true",
        help="Tag this run as a free-threaded interpreter build (metadata only)",
    )
    parser.add_argument(
        "--sweep",
        help=(
            "Comma-separated concurrency levels to sweep, e.g. '50,75,100,125,150'. "
            "Each level runs --repeats times; results are aggregated by median. "
            "Overrides --concurrency when set."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeats per concurrency level in --sweep mode (default: 1)",
    )
    parser.add_argument("--out", help="Write the JSON report to this path")
    parser.add_argument(
        "--compare-to",
        help="Compare against this saved JSON (single report, or sweep report in --sweep mode)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def _run_sweep_mode(
    args: argparse.Namespace,
) -> int:
    """Run a multi-concurrency sweep, print the table, optionally save/compare."""
    levels = _parse_concurrency_list(args.sweep)
    logger.info("Sweep over concurrency=%s, repeats=%d", levels, args.repeats)
    sweep = asyncio.run(_run_sweep(args, levels))
    _print_sweep(sweep)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sweep.model_dump(), indent=2))
        logger.info("Wrote sweep report to %s", out_path)

    if args.compare_to:
        _print_sweep_comparison(sweep, Path(args.compare_to))

    return 0


def _run_single_mode(
    args: argparse.Namespace,
) -> int:
    """Run one concurrency level, print the tables, optionally save/compare."""
    report = asyncio.run(_run_benchmark(args))
    _print_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.model_dump(), indent=2))
        logger.info("Wrote report to %s", out_path)

    if args.compare_to:
        _print_comparison(report, Path(args.compare_to))

    return 0


def main() -> int:
    """Control flow: parse args, then dispatch to sweep or single-run mode."""
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.sweep:
        return _run_sweep_mode(args)
    return _run_single_mode(args)


if __name__ == "__main__":
    sys.exit(main())
