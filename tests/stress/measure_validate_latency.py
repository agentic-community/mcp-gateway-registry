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

Workflow for a before/after comparison
---------------------------------------
1. Run against the baseline (standard GIL) image, saving the result:
     uv run python -m tests.stress.measure_validate_latency \
         --label baseline --out tests/stress/results/validate/baseline.json
2. Rebuild auth-server on the free-threaded image, then run again:
     uv run python -m tests.stress.measure_validate_latency \
         --label freethreaded --out tests/stress/results/validate/freethreaded.json
3. Compare the two JSON files (or pass --compare-to):
     uv run python -m tests.stress.measure_validate_latency \
         --label freethreaded --compare-to tests/stress/results/validate/baseline.json

Latency source
--------------
Percentiles are computed by Prometheus from the
``http_server_duration_milliseconds`` histogram (FastAPI auto-instrumentation),
filtered to ``http_target="/validate"`` and split by ``http_status_code``. This
is end-to-end server-side latency as the auth-server measures it, independent of
client-side network noise. The script also records the client-observed wall
time per request for a sanity cross-check.

Note on the Prometheus scrape gap: the script waits one scrape interval after
the load finishes so the final samples are visible to Prometheus before
querying. Counters are cumulative, so percentiles are taken over a rate window
that covers the load burst.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
DEFAULT_WINDOW: str = "5m"


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


def _query_server_latency(
    prom_url: str,
    status_code: str,
    window: str,
) -> PromLatency:
    """Read /validate server-side latency percentiles for one status code.

    The histogram metric exposes ``_bucket`` / ``_sum`` / ``_count`` suffixed
    series. The label selector must follow the full metric name (suffix
    included), e.g. ``metric_sum{labels}`` -- not ``metric{labels}_sum`` (the
    latter is a PromQL parse error).
    """
    labels = f'{{http_target="/validate",http_status_code="{status_code}"}}'
    base = "http_server_duration_milliseconds"

    def _pct(p: float) -> float | None:
        q = f"histogram_quantile({p}, sum by (le) (rate({base}_bucket{labels}[{window}])))"
        return _prom_query(prom_url, q)

    avg = _prom_query(
        prom_url,
        f"sum(rate({base}_sum{labels}[{window}])) / sum(rate({base}_count{labels}[{window}]))",
    )
    count = _prom_query(prom_url, f"sum({base}_count{labels})")

    def _round(v: float | None) -> float | None:
        return round(v, 3) if v is not None else None

    return PromLatency(
        status_code=status_code,
        p50_ms=_round(_pct(0.50)),
        p95_ms=_round(_pct(0.95)),
        p99_ms=_round(_pct(0.99)),
        avg_ms=_round(avg),
        request_count=_round(count),
    )


async def _run_benchmark(
    args: argparse.Namespace,
) -> BenchmarkReport:
    """Drive both request mixes, then read server-side latency from Prometheus."""
    token = _read_token(Path(args.token_file))
    success_headers = {
        "Authorization": f"Bearer {token}",
        "X-Original-URL": args.original_url,
    }
    error_headers = {
        "Authorization": f"Bearer {_derive_error_token(token)}",
        "X-Original-URL": args.original_url,
    }

    logger.info(
        "Starting /validate load: %d requests/path at concurrency %d (label=%s)",
        args.requests,
        args.concurrency,
        args.label,
    )
    success = await _run_path(
        "success", 200, args.auth_url, success_headers, args.requests, args.concurrency
    )
    error = await _run_path(
        "error", 401, args.auth_url, error_headers, args.requests, args.concurrency
    )

    logger.info("Waiting %ds for Prometheus to scrape final samples...", args.scrape_wait)
    await asyncio.sleep(args.scrape_wait)

    server_latency = [
        _query_server_latency(args.prometheus_url, "200", args.window),
        _query_server_latency(args.prometheus_url, "401", args.window),
    ]

    return BenchmarkReport(
        label=args.label,
        timestamp=datetime.now(UTC).isoformat(),
        auth_url=args.auth_url,
        prometheus_url=args.prometheus_url,
        concurrency=args.concurrency,
        requests_per_path=args.requests,
        free_threaded=args.free_threaded,
        client_results=[success, error],
        server_latency=server_latency,
    )


def _print_report(
    report: BenchmarkReport,
) -> None:
    """Log a human-readable summary of one benchmark run."""
    logger.info("=" * 70)
    logger.info("Benchmark '%s' (concurrency=%d)", report.label, report.concurrency)
    logger.info("-" * 70)
    for r in report.client_results:
        logger.info(
            "CLIENT %-8s status=%s p50=%.2f p95=%.2f p99=%.2f throughput=%.1f rps",
            r.path,
            r.status_counts,
            r.client_p50_ms,
            r.client_p95_ms,
            r.client_p99_ms,
            r.throughput_rps,
        )
    for s in report.server_latency:
        logger.info(
            "SERVER status=%s p50=%s p95=%s p99=%s avg=%s ms (count=%s)",
            s.status_code,
            s.p50_ms,
            s.p95_ms,
            s.p99_ms,
            s.avg_ms,
            s.request_count,
        )
    logger.info("=" * 70)


def _print_comparison(
    current: BenchmarkReport,
    baseline_path: Path,
) -> None:
    """Compare current server-side latency against a saved baseline JSON."""
    baseline = BenchmarkReport(**json.loads(baseline_path.read_text()))
    base_by_status = {s.status_code: s for s in baseline.server_latency}

    logger.info("COMPARISON vs baseline '%s'", baseline.label)
    logger.info("-" * 70)
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
            arrow = "improved" if delta_pct < 0 else "regressed"
            logger.info(
                "status=%s %s: %.2f -> %.2f ms (%+.1f%%, %s)",
                s.status_code,
                metric,
                old,
                cur,
                delta_pct,
                arrow,
            )
    logger.info("=" * 70)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Concurrent load test for the auth-server /validate hot path (#1316).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Baseline run (standard GIL image), save result
    uv run python -m tests.stress.measure_validate_latency \\
        --label baseline --out tests/stress/results/validate/baseline.json

    # After rebuilding on the free-threaded image
    uv run python -m tests.stress.measure_validate_latency \\
        --label freethreaded --free-threaded \\
        --out tests/stress/results/validate/freethreaded.json \\
        --compare-to tests/stress/results/validate/baseline.json
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
        "--window", default=DEFAULT_WINDOW, help="PromQL rate window for percentiles"
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
    parser.add_argument("--out", help="Write the JSON report to this path")
    parser.add_argument("--compare-to", help="Compare server latency against this saved JSON")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """Control flow: parse args, run the benchmark, report, optionally compare."""
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

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


if __name__ == "__main__":
    sys.exit(main())
