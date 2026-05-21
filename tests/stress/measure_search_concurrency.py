"""Measure semantic search latency under concurrent load.

Fires all 20 curated queries simultaneously at varying parallelism levels
(1, 10, 100) and reports per-query and aggregate latency percentiles.
This complements the serial measure_api_performance.py by answering:
"What happens to search latency when N users search at the same time?"

Usage:
    uv run python -m tests.stress.measure_search_concurrency \
        --base-url https://d2xl2zfuhgc4l0.cloudfront.net \
        --token-file .token \
        --iterations 50

Output:
    tests/stress/results/<backend>/size-<N>/search_concurrency.json
    tests/stress/results/<backend>/size-<N>/search_concurrency.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from tests.stress.config import (
    default_base_url,
    default_token_file,
    fetch_registry_info,
)
from tests.stress.constants import (
    HTTP_TIMEOUT_SECONDS,
    TARGET_SIZES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

CONCURRENCY_LEVELS = [1, 10, 100]
SEARCH_K = 5
DEFAULT_ITERATIONS = 50


def _load_token(token_file: Path) -> str:
    """Load JWT token from file (JSON or plain text)."""
    raw = token_file.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        token = raw.strip()
        if not token:
            raise RuntimeError(f"Empty token file: {token_file}") from None
        return token

    token = (
        data.get("access_token")
        or data.get("tokens", {}).get("access_token")
        or data.get("token_data", {}).get("access_token")
    )
    if not token:
        raise RuntimeError(f"No 'access_token' field found in token file: {token_file}")
    return token


def _load_queries(queries_file: Path) -> list[dict[str, Any]]:
    """Load curated queries from JSON file."""
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")
    data = json.loads(queries_file.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty list in {queries_file}")
    return data


def _detect_backend(
    base_url: str,
    token: str,
) -> str:
    """Auto-detect storage backend from the registry's /api/stats endpoint."""
    headers = {"Authorization": f"Bearer {token}"}
    resp = httpx.get(f"{base_url.rstrip('/')}/api/stats", headers=headers, timeout=10)
    resp.raise_for_status()
    backend = resp.json()["database_status"]["backend"]
    logger.info("Auto-detected backend: %s", backend)
    return backend


def _compute_percentiles(samples: list[float]) -> dict[str, float]:
    """Compute p50, p90, p95, p99, mean, min, max from a list of latencies."""
    if not samples:
        return {}
    sorted_s = sorted(samples)
    n = len(sorted_s)
    return {
        "count": n,
        "mean_ms": round(statistics.mean(sorted_s), 2),
        "min_ms": round(sorted_s[0], 2),
        "p50_ms": round(sorted_s[int(n * 0.50)], 2),
        "p90_ms": round(sorted_s[int(n * 0.90)], 2),
        "p95_ms": round(sorted_s[int(n * 0.95)], 2),
        "p99_ms": round(sorted_s[min(int(n * 0.99), n - 1)], 2),
        "max_ms": round(sorted_s[-1], 2),
    }


async def _search_once(
    client: httpx.AsyncClient,
    base_url: str,
    query_obj: dict[str, Any],
    token: str,
) -> float:
    """Execute a single semantic search and return latency in ms."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "query": query_obj["query"],
        "entity_types": query_obj.get("expected_entity_types", []),
        "max_results": SEARCH_K,
        "include_disabled": False,
        "include_draft": True,
    }
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url.rstrip('/')}/api/search/semantic",
        headers=headers,
        json=payload,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if resp.status_code != 200:
        logger.warning(
            "Search failed (status=%d, query=%s): %s",
            resp.status_code,
            query_obj.get("query", "")[:30],
            resp.text[:100],
        )
    return elapsed_ms


async def _run_concurrent_batch(
    base_url: str,
    queries: list[dict[str, Any]],
    concurrency: int,
    token: str,
) -> list[float]:
    """Run `concurrency` search requests simultaneously, return all latencies."""
    sem = asyncio.Semaphore(concurrency)

    async def _bounded_search(
        client: httpx.AsyncClient,
        query_obj: dict[str, Any],
    ) -> float:
        async with sem:
            return await _search_once(client, base_url, query_obj, token)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        # Build a task list: cycle through queries to fill concurrency slots
        tasks = []
        for i in range(concurrency):
            query_obj = queries[i % len(queries)]
            tasks.append(_bounded_search(client, query_obj))

        latencies = await asyncio.gather(*tasks)

    return list(latencies)


async def _run_level(
    base_url: str,
    queries: list[dict[str, Any]],
    concurrency: int,
    iterations: int,
    token: str,
) -> dict[str, Any]:
    """Run multiple iterations at a given concurrency level.

    Each iteration fires `concurrency` simultaneous requests.
    First iteration is discarded as warmup.
    """
    all_latencies: list[float] = []
    iteration_stats: list[dict[str, float]] = []

    total_iterations = iterations + 1

    logger.info(
        "Concurrency=%d: running %d iterations (%d warmup + %d measured)",
        concurrency,
        total_iterations,
        1,
        iterations,
    )

    for i in range(total_iterations):
        batch_latencies = await _run_concurrent_batch(
            base_url, queries, concurrency, token
        )

        if i == 0:
            logger.debug("Concurrency=%d: warmup iteration discarded", concurrency)
            continue

        all_latencies.extend(batch_latencies)
        batch_stats = _compute_percentiles(batch_latencies)
        iteration_stats.append(batch_stats)

        if (i) % 10 == 0:
            logger.info(
                "Concurrency=%d: iteration %d/%d, batch p50=%.1fms p99=%.1fms",
                concurrency,
                i,
                iterations,
                batch_stats.get("p50_ms", 0),
                batch_stats.get("p99_ms", 0),
            )

    aggregate = _compute_percentiles(all_latencies)
    total_requests = concurrency * iterations
    elapsed_per_iteration_avg = aggregate.get("mean_ms", 0)

    return {
        "concurrency": concurrency,
        "iterations": iterations,
        "total_requests": total_requests,
        "k": SEARCH_K,
        "warmup_strategy": "discard_first_iteration",
        "aggregate_latency": aggregate,
        "throughput_rps": round(
            (total_requests / (sum(s.get("mean_ms", 0) for s in iteration_stats) / 1000))
            if iteration_stats
            else 0,
            1,
        ),
    }


def _generate_markdown(report: dict[str, Any]) -> str:
    """Generate a human-readable markdown report."""
    lines = []
    lines.append(f"# Search Concurrency Report")
    lines.append(f"")
    lines.append(f"- **Backend:** {report['backend']}")
    lines.append(f"- **Base URL:** {report['base_url']}")
    lines.append(f"- **Queries:** {report['num_queries']} curated queries")
    lines.append(f"- **k:** {SEARCH_K}")
    lines.append(f"- **Iterations per level:** {report['iterations']}")
    lines.append(f"- **Warmup:** first iteration discarded")
    lines.append(f"- **Timestamp:** {report['timestamp']}")
    lines.append(f"")
    lines.append(f"## Results")
    lines.append(f"")
    lines.append(
        "| Concurrency | Requests | Throughput (rps) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Max (ms) |"
    )
    lines.append(
        "|-------------|----------|-----------------|----------|----------|----------|----------|----------|"
    )

    for level in report["levels"]:
        agg = level["aggregate_latency"]
        lines.append(
            f"| {level['concurrency']:>11} "
            f"| {level['total_requests']:>8} "
            f"| {level['throughput_rps']:>15.1f} "
            f"| {agg['p50_ms']:>8.1f} "
            f"| {agg['p90_ms']:>8.1f} "
            f"| {agg['p95_ms']:>8.1f} "
            f"| {agg['p99_ms']:>8.1f} "
            f"| {agg['max_ms']:>8.1f} |"
        )

    lines.append(f"")
    lines.append(f"## Interpretation")
    lines.append(f"")
    lines.append(
        f"- **Concurrency=1** is the baseline single-user latency (should match Phase 2 serial results)."
    )
    lines.append(
        f"- **Concurrency=10** simulates a small team using search simultaneously."
    )
    lines.append(
        f"- **Concurrency=100** simulates burst load from agent-driven discovery workflows."
    )
    lines.append(f"")
    lines.append(
        f"If p99 at concurrency=100 is less than 2x the p99 at concurrency=1, "
        f"the search backend scales well under concurrent load."
    )
    lines.append(f"")
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> int:
    token = _load_token(args.token_file)

    # Auto-detect backend
    if args.backend is None:
        try:
            args.backend = _detect_backend(args.base_url, token)
        except Exception as exc:
            logger.error("Failed to auto-detect backend: %s", exc)
            return 1

    registry_info = fetch_registry_info(args.base_url, token)

    queries = _load_queries(args.queries_file)
    logger.info("Loaded %d curated queries from %s", len(queries), args.queries_file)

    output_dir = Path(args.results_dir) / args.backend / f"size-{args.size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels: list[dict[str, Any]] = []
    overall_start = time.time()

    for concurrency in CONCURRENCY_LEVELS:
        level_result = await _run_level(
            args.base_url,
            queries,
            concurrency,
            args.iterations,
            token,
        )
        levels.append(level_result)
        logger.info(
            "Concurrency=%d complete: p50=%.1fms, p99=%.1fms, throughput=%.1f rps",
            concurrency,
            level_result["aggregate_latency"]["p50_ms"],
            level_result["aggregate_latency"]["p99_ms"],
            level_result["throughput_rps"],
        )

    elapsed = time.time() - overall_start
    from datetime import datetime, timezone

    report = {
        "backend": args.backend,
        "base_url": args.base_url,
        "size": args.size,
        "iterations": args.iterations,
        "num_queries": len(queries),
        "k": SEARCH_K,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "registry_info": registry_info,
        "levels": levels,
    }

    json_path = output_dir / "search_concurrency.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("JSON report written to %s", json_path)

    md_path = output_dir / "search_concurrency.md"
    md_path.write_text(_generate_markdown(report))
    logger.info("Markdown report written to %s", md_path)

    # Print summary table
    print()
    print(_generate_markdown(report))
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure semantic search latency under concurrent load.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run against remote registry (auto-detects backend)
    uv run python -m tests.stress.measure_search_concurrency \\
        --base-url https://d2xl2zfuhgc4l0.cloudfront.net \\
        --token-file .token

    # Run locally with fewer iterations for quick check
    uv run python -m tests.stress.measure_search_concurrency \\
        --base-url http://localhost \\
        --token-file .token \\
        --iterations 10
""",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Storage backend. Auto-detected from /api/stats if not provided.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        choices=TARGET_SIZES,
        help="Corpus size (must match Phase 1 registration). Default: 100.",
    )
    parser.add_argument("--base-url", default=default_base_url())
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Iterations per concurrency level (default: 50). "
        "Each iteration fires `concurrency` simultaneous requests.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path("tests/stress/queries.json"),
        help="Curated query set (default: tests/stress/queries.json).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("tests/stress/results"),
        help="Root results directory.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=default_token_file(),
        help="JWT token file (default: .token).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    import sys

    sys.exit(main())
