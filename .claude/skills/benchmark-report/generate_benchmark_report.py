"""Generate a markdown benchmark report from stress test result files.

Reads registration.json, api_perf.json, and search_concurrency.json from
the results directory and produces a structured markdown report.

Usage:
    /usr/bin/python3 .claude/skills/benchmark-report/generate_benchmark_report.py \
        --results-dir tests/stress/results/documentdb/size-100 \
        --output docs/benchmarks/benchmark-report.md
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, return None if missing."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    return json.loads(path.read_text())


def _format_ms(val: float | None) -> str:
    """Format milliseconds for display."""
    if val is None:
        return "N/A"
    if val >= 1000:
        return f"{val / 1000:.2f}s"
    return f"{val:.0f}ms"


def _build_deployment_section(registry_info: dict[str, Any]) -> str:
    """Build the deployment configuration section."""
    lines = []
    lines.append("## Deployment Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Registry Version | `{registry_info.get('v', 'unknown')}` |")
    lines.append(f"| Cloud Provider | {registry_info.get('cloud', 'unknown')} |")
    lines.append(f"| Compute Platform | {registry_info.get('compute', 'unknown')} |")
    lines.append(f"| Architecture | {registry_info.get('arch', 'unknown')} |")
    lines.append(f"| Storage Backend | {registry_info.get('storage', 'unknown')} |")
    lines.append(f"| Search Backend | {registry_info.get('search_backend', 'unknown')} |")
    lines.append(f"| Auth Provider | {registry_info.get('auth', 'unknown')} |")
    lines.append(f"| Embeddings Provider | {registry_info.get('embeddings_provider', 'unknown')} |")
    lines.append(f"| Embeddings Backend | {registry_info.get('embeddings_backend_kind', 'unknown')} |")
    lines.append(f"| Python Version | {registry_info.get('py', 'unknown')} |")
    lines.append(f"| OS | {registry_info.get('os', 'unknown')} |")
    lines.append(f"| Deployment Mode | {registry_info.get('mode', 'unknown')} |")
    lines.append(f"| Federation Enabled | {registry_info.get('federation', 'unknown')} |")
    if registry_info.get("instance_count"):
        lines.append(f"| Backend Instances (detected) | {registry_info['instance_count']} |")
    lines.append("")
    lines.append("### Corpus Size at Test Time")
    lines.append("")
    lines.append("| Entity | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| MCP Servers | {registry_info.get('servers_count', 'N/A')} |")
    lines.append(f"| Agents | {registry_info.get('agents_count', 'N/A')} |")
    lines.append(f"| Skills | {registry_info.get('skills_count', 'N/A')} |")
    lines.append("")
    return "\n".join(lines)


def _build_registration_section(reg: dict[str, Any]) -> str:
    """Build the registration throughput section."""
    lines = []
    lines.append("## Registration Throughput (Phase 1)")
    lines.append("")
    lines.append(f"Bulk registration of {reg.get('size', '?')} entities per type "
                 f"with concurrency={reg.get('concurrency', '?')}.")
    lines.append(f"Total wall clock: {reg.get('wall_clock_seconds', 0):.1f}s.")
    lines.append("")
    lines.append("| Entity Type | Target | Registered | Skipped | Failed | Failure Rate | p50 | p95 | p99 |")
    lines.append("|-------------|--------|------------|---------|--------|--------------|-----|-----|-----|")

    for entity_type, data in reg.get("entity_types", {}).items():
        latency = data.get("latency_ms", {})
        lines.append(
            f"| {entity_type} "
            f"| {data.get('target_count', '?')} "
            f"| {data.get('registered', '?')} "
            f"| {data.get('skipped', '?')} "
            f"| {data.get('failed', '?')} "
            f"| {data.get('failure_rate', 0) * 100:.1f}% "
            f"| {_format_ms(latency.get('p50'))} "
            f"| {_format_ms(latency.get('p95'))} "
            f"| {_format_ms(latency.get('p99'))} |"
        )

    lines.append("")
    return "\n".join(lines)


def _build_api_perf_section(perf: dict[str, Any]) -> str:
    """Build the API performance section."""
    lines = []
    lines.append("## API Latency, Serial (Phase 2a)")
    lines.append("")
    lines.append(f"Steady-state per-request latency. Each operation measured "
                 f"{perf.get('iterations', '?')} times (first iteration discarded as warmup).")
    lines.append(f"Total wall clock: {perf.get('wall_clock_seconds', 0):.1f}s.")
    lines.append("")

    # Group operations by category
    list_ops = []
    search_ops = []
    for op in perf.get("operations", []):
        name = op.get("name", op.get("operation_name", ""))
        if "list_" in name:
            list_ops.append(op)
        elif "semantic_search" in name:
            search_ops.append(op)

    # List operations table
    lines.append("### List Endpoints")
    lines.append("")
    lines.append("| Operation | p50 | p95 | p99 | Max |")
    lines.append("|-----------|-----|-----|-----|-----|")

    for op in list_ops:
        lat = op.get("latency_ms", {})
        lines.append(
            f"| {op.get('name', op.get('operation_name', '?'))} "
            f"| {_format_ms(lat.get('p50'))} "
            f"| {_format_ms(lat.get('p95'))} "
            f"| {_format_ms(lat.get('p99'))} "
            f"| {_format_ms(lat.get('max'))} |"
        )

    lines.append("")

    # Semantic search table (aggregate by k value)
    if search_ops:
        lines.append("### Semantic Search (Serial)")
        lines.append("")
        lines.append("| k | Queries | p50 | p95 | p99 | Max |")
        lines.append("|---|---------|-----|-----|-----|-----|")

        # Group by k value and compute aggregate
        by_k: dict[int, list[dict]] = {}
        for op in search_ops:
            k = op.get("k", 0)
            if k not in by_k:
                by_k[k] = []
            by_k[k].append(op)

        for k in sorted(by_k.keys()):
            ops = by_k[k]
            p50s = [o["latency_ms"]["p50"] for o in ops if o.get("latency_ms", {}).get("p50")]
            p95s = [o["latency_ms"]["p95"] for o in ops if o.get("latency_ms", {}).get("p95")]
            p99s = [o["latency_ms"]["p99"] for o in ops if o.get("latency_ms", {}).get("p99")]
            maxs = [o["latency_ms"]["max"] for o in ops if o.get("latency_ms", {}).get("max")]

            def _avg(lst: list[float]) -> float | None:
                return sum(lst) / len(lst) if lst else None

            lines.append(
                f"| k={k} "
                f"| {len(ops)} "
                f"| {_format_ms(_avg(p50s))} "
                f"| {_format_ms(_avg(p95s))} "
                f"| {_format_ms(_avg(p99s))} "
                f"| {_format_ms(max(maxs) if maxs else None)} |"
            )

        lines.append("")

    return "\n".join(lines)


def _build_concurrency_section(conc: dict[str, Any]) -> str:
    """Build the search concurrency section."""
    lines = []
    lines.append("## Semantic Search Concurrency Scaling (Phase 2b)")
    lines.append("")
    lines.append(f"Concurrent search load test using {conc.get('num_queries', '?')} curated queries "
                 f"at k={conc.get('k', '?')}. "
                 f"Each concurrency level ran {conc.get('iterations', '?')} iterations "
                 f"(first discarded as warmup).")
    lines.append(f"Total wall clock: {conc.get('elapsed_seconds', 0):.1f}s.")
    lines.append("")
    lines.append("| Concurrency | Requests | Throughput (rps) | p50 | p90 | p95 | p99 | Max |")
    lines.append("|-------------|----------|-----------------|-----|-----|-----|-----|-----|")

    for level in conc.get("levels", []):
        agg = level.get("aggregate_latency", {})
        lines.append(
            f"| {level.get('concurrency', '?')} "
            f"| {level.get('total_requests', '?')} "
            f"| {level.get('throughput_rps', 0):.1f} "
            f"| {_format_ms(agg.get('p50_ms'))} "
            f"| {_format_ms(agg.get('p90_ms'))} "
            f"| {_format_ms(agg.get('p95_ms'))} "
            f"| {_format_ms(agg.get('p99_ms'))} "
            f"| {_format_ms(agg.get('max_ms'))} |"
        )

    lines.append("")

    # Scaling analysis
    levels = conc.get("levels", [])
    if len(levels) >= 2:
        baseline = levels[0].get("aggregate_latency", {}).get("p99_ms", 0)
        highest = levels[-1].get("aggregate_latency", {}).get("p99_ms", 0)
        if baseline > 0:
            ratio = highest / baseline
            lines.append("### Scaling Analysis")
            lines.append("")
            lines.append(f"- Baseline p99 (concurrency=1): {_format_ms(baseline)}")
            lines.append(f"- Peak p99 (concurrency={levels[-1].get('concurrency', '?')}): {_format_ms(highest)}")
            lines.append(f"- Degradation ratio: {ratio:.1f}x")
            lines.append("")
            if ratio <= 2:
                lines.append("The search backend scales well under concurrent load (ratio <= 2x).")
            elif ratio <= 10:
                lines.append("Moderate degradation under peak concurrent load. "
                             "Acceptable for most production workloads.")
            else:
                lines.append("Significant degradation under peak concurrent load. "
                             "Consider horizontal scaling (more ECS tasks) for high-concurrency workloads.")
            lines.append("")

    return "\n".join(lines)


def _build_report(
    results_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Build the full benchmark report from result files."""
    reg = _load_json(results_dir / "registration.json")
    perf = _load_json(results_dir / "api_perf.json")
    conc = _load_json(results_dir / "search_concurrency.json")

    # Get registry_info from whichever file has it
    registry_info: dict[str, Any] = {}
    for source in [conc, perf, reg]:
        if source and source.get("registry_info"):
            registry_info = source["registry_info"]
            break

    backend = (reg or perf or conc or {}).get("backend", "unknown")
    size = (reg or perf or conc or {}).get("size", "unknown")

    lines = []
    lines.append("# MCP Gateway Registry Benchmark Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append(f"*Backend: {backend}, Corpus size: {size} entities per type*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Deployment configuration
    if registry_info:
        lines.append(_build_deployment_section(registry_info))

    # Registration
    if reg:
        lines.append(_build_registration_section(reg))
    else:
        lines.append("## Registration Throughput (Phase 1)\n\nNo registration.json found.\n")

    # API performance
    if perf:
        lines.append(_build_api_perf_section(perf))
    else:
        lines.append("## API Latency (Phase 2a)\n\nNo api_perf.json found.\n")

    # Search concurrency
    if conc:
        lines.append(_build_concurrency_section(conc))
    else:
        lines.append("## Search Concurrency (Phase 2b)\n\nNo search_concurrency.json found.\n")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Registration (Phase 1):** Async bulk registration of generated payloads "
                 "sourced from the Anthropic MCP registry and GoDaddy ANS catalog.")
    lines.append("- **API Latency (Phase 2a):** Serial requests, each operation measured N+1 times "
                 "(first iteration discarded as warmup). Reports steady-state per-request latency.")
    lines.append("- **Search Concurrency (Phase 2b):** Concurrent batches of semantic search "
                 "requests at increasing parallelism levels. Reports aggregate latency and throughput.")
    lines.append("- **Warmup:** First iteration at each level/operation is always discarded. "
                 "Covers embedding model lazy-load, connection pool establishment, and DB working-set warmup.")
    lines.append("- **All result JSON files** include a `registry_info` snapshot of the deployment "
                 "configuration at test time, captured from `GET /api/registry-management/telemetry/info`.")
    lines.append("")

    # Reproduction
    lines.append("## Reproducing These Results")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Register entities")
    lines.append("bash tests/stress/run_stress_test.sh 100 \\")
    lines.append("    --base-url <REGISTRY_URL> --token-file .token --skip-generate")
    lines.append("")
    lines.append("# 2. Measure API latency (serial)")
    lines.append("uv run python -m tests.stress.measure_api_performance \\")
    lines.append("    --size 100 --base-url <REGISTRY_URL> --iterations 50 --token-file .token")
    lines.append("")
    lines.append("# 3. Measure search concurrency")
    lines.append("uv run python -m tests.stress.measure_search_concurrency \\")
    lines.append("    --base-url <REGISTRY_URL> --token-file .token --iterations 50")
    lines.append("```")
    lines.append("")
    lines.append("See `tests/stress/README.md` for full documentation.")
    lines.append("")

    return "\n".join(lines), registry_info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a benchmark report from stress test results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("tests/stress/results/documentdb/size-100"),
        help="Path to results directory containing the JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/benchmark-report.md"),
        help="Output path for the markdown report.",
    )
    args = parser.parse_args()

    report, registry_info = _build_report(args.results_dir)

    # Build a descriptive filename if output is the default
    if args.output == Path("docs/benchmarks/benchmark-report.md"):
        from datetime import datetime, timezone

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        compute = registry_info.get("compute", "unknown")
        backend = registry_info.get("storage", "unknown")
        instances = registry_info.get("instance_count", "x")
        filename = f"benchmark-{date_str}-{compute}-{backend}-{instances}x.md"
        args.output = args.output.parent / filename

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    logger.info("Benchmark report written to %s", args.output)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
