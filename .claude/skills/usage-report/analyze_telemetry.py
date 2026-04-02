"""Analyze telemetry CSV and output pre-formatted markdown tables.

Reads registry_metrics.csv, computes all distributions, instance
timelines, search stats, and version adoption. Writes a JSON file
with raw metrics and a markdown file with pre-formatted tables
ready to embed in the usage report.
"""
import argparse
import csv
import json
import logging
import os
from collections import Counter
from collections import defaultdict
from datetime import datetime


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def _read_csv(
    csv_path: str,
) -> list[dict[str, str]]:
    """Read the telemetry CSV and return rows as list of dicts."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    logger.info(f"Read {len(rows)} rows from {csv_path}")
    return rows


def _safe_int(
    value: str,
) -> int:
    """Convert string to int, defaulting to 0 for empty/invalid."""
    if not value or not value.strip():
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _classify_version(
    version: str,
) -> str:
    """Classify a version string as release or dev."""
    if not version:
        return "unknown"
    if version.startswith("v1.0.17") or version.startswith("v0."):
        return "dev"
    return "release"


def _extract_version_branch(
    version: str,
) -> str:
    """Extract branch name from dev version string."""
    if not version:
        return "unknown"
    # e.g. v1.0.17-45-gdc7fbe6-main -> main
    parts = version.split("-")
    if len(parts) >= 4:
        # Skip version, count, hash -- rest is branch
        return "-".join(parts[3:])
    return version


def _format_pct(
    count: int,
    total: int,
) -> str:
    """Format count as percentage string."""
    if total == 0:
        return "0%"
    return f"{count / total * 100:.0f}%"


def _md_distribution_table(
    title: str,
    counter: Counter,
    total: int,
    col_name: str = "Value",
) -> str:
    """Generate a markdown table for a distribution."""
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"| {col_name} | Events | Percentage |")
    lines.append(f"|{'---' * 5}|--------|------------|")

    for value, count in counter.most_common():
        pct = _format_pct(count, total)
        lines.append(f"| {value} | {count} | {pct} |")

    lines.append("")
    return "\n".join(lines)


def _compute_key_metrics(
    rows: list[dict[str, str]],
) -> dict:
    """Compute top-level key metrics."""
    total = len(rows)
    startup_count = sum(1 for r in rows if r.get("event") == "startup")
    heartbeat_count = sum(1 for r in rows if r.get("event") == "heartbeat")

    # Unique identified instances
    registry_ids = {
        r["registry_id"]
        for r in rows
        if r.get("registry_id", "").strip()
    }
    identified_count = len(registry_ids)

    # Null registry_id count
    null_id_count = sum(
        1 for r in rows if not r.get("registry_id", "").strip()
    )

    # Collection period
    timestamps = []
    for r in rows:
        ts = r.get("ts", "")
        if ts:
            timestamps.append(ts)
    timestamps.sort()
    earliest = timestamps[0] if timestamps else "N/A"
    latest = timestamps[-1] if timestamps else "N/A"

    return {
        "total_events": total,
        "startup_events": startup_count,
        "heartbeat_events": heartbeat_count,
        "identified_instances": identified_count,
        "null_registry_id_count": null_id_count,
        "null_registry_id_pct": _format_pct(null_id_count, total),
        "earliest_ts": earliest,
        "latest_ts": latest,
    }


def _compute_distributions(
    rows: list[dict[str, str]],
) -> dict[str, Counter]:
    """Compute value counts for each dimension."""
    dims = {
        "cloud": Counter(),
        "compute": Counter(),
        "storage": Counter(),
        "auth": Counter(),
        "version": Counter(),
        "version_type": Counter(),
        "mode": Counter(),
        "arch": Counter(),
        "federation": Counter(),
    }

    for row in rows:
        dims["cloud"][row.get("cloud") or "unknown"] += 1
        dims["compute"][row.get("compute") or "unknown"] += 1
        dims["storage"][row.get("storage") or "unknown"] += 1
        dims["auth"][row.get("auth") or "none"] += 1
        dims["mode"][row.get("mode") or "unknown"] += 1
        dims["arch"][row.get("arch") or "unknown"] += 1

        version = row.get("v") or ""
        dims["version"][version] += 1
        dims["version_type"][_classify_version(version)] += 1

        fed = row.get("federation", "").strip().lower()
        dims["federation"]["enabled" if fed == "true" else "disabled"] += 1

    return dims


def _compute_instance_table(
    rows: list[dict[str, str]],
) -> list[dict]:
    """Compute per-instance summary for identified instances."""
    instances = defaultdict(list)
    for row in rows:
        rid = row.get("registry_id", "").strip()
        if rid:
            instances[rid].append(row)

    result = []
    for rid, events in instances.items():
        events.sort(key=lambda r: r.get("ts", ""))

        # Latest event for current state
        latest = events[-1]

        # Track max servers/agents/skills across events
        max_servers = max(_safe_int(e.get("servers_count", "")) for e in events)
        max_agents = max(_safe_int(e.get("agents_count", "")) for e in events)
        max_skills = max(_safe_int(e.get("skills_count", "")) for e in events)

        # Track max search queries
        max_search = max(
            _safe_int(e.get("search_queries_total", "")) for e in events
        )

        first_ts = events[0].get("ts", "")[:10]
        latest_ts = events[-1].get("ts", "")[:10]

        result.append({
            "registry_id": rid[:12] + "...",
            "registry_id_full": rid,
            "cloud": latest.get("cloud") or "unknown",
            "compute": latest.get("compute") or "unknown",
            "storage": latest.get("storage") or "unknown",
            "auth": latest.get("auth") or "none",
            "federation": latest.get("federation", "").strip().lower() == "true",
            "arch": latest.get("arch") or "unknown",
            "mode": latest.get("mode") or "unknown",
            "version": latest.get("v") or "unknown",
            "events": len(events),
            "max_servers": max_servers,
            "max_agents": max_agents,
            "max_skills": max_skills,
            "max_search_queries": max_search,
            "first_seen": first_ts,
            "latest_seen": latest_ts,
        })

    result.sort(key=lambda x: x["events"], reverse=True)
    return result


def _compute_unidentified_profiles(
    rows: list[dict[str, str]],
) -> list[dict]:
    """Group unidentified events into distinct deployment profiles."""
    unidentified = [
        r for r in rows if not r.get("registry_id", "").strip()
    ]

    # Group by (cloud, compute, arch, storage, auth, mode)
    profiles = defaultdict(list)
    for row in unidentified:
        key = (
            row.get("cloud") or "unknown",
            row.get("compute") or "unknown",
            row.get("arch") or "unknown",
            row.get("storage") or "unknown",
            row.get("auth") or "none",
            row.get("mode") or "unknown",
        )
        profiles[key].append(row)

    result = []
    for key, events in profiles.items():
        cloud, compute, arch, storage, auth, mode = key

        max_servers = max(_safe_int(e.get("servers_count", "")) for e in events)
        max_agents = max(_safe_int(e.get("agents_count", "")) for e in events)
        max_skills = max(_safe_int(e.get("skills_count", "")) for e in events)
        max_search = max(
            _safe_int(e.get("search_queries_total", "")) for e in events
        )

        events.sort(key=lambda r: r.get("ts", ""))
        first_ts = events[0].get("ts", "")[:10]
        latest_ts = events[-1].get("ts", "")[:10]

        result.append({
            "cloud": cloud,
            "compute": compute,
            "arch": arch,
            "storage": storage,
            "auth": auth,
            "mode": mode,
            "events": len(events),
            "max_servers": max_servers,
            "max_agents": max_agents,
            "max_skills": max_skills,
            "max_search_queries": max_search,
            "first_seen": first_ts,
            "latest_seen": latest_ts,
        })

    result.sort(key=lambda x: x["events"], reverse=True)
    return result


def _compute_instance_timeline(
    rows: list[dict[str, str]],
    registry_id: str | None = None,
    cloud: str | None = None,
    compute: str | None = None,
) -> list[dict]:
    """Compute daily timeline for a specific instance or profile.

    Filter by registry_id for identified instances, or by
    cloud+compute for unidentified profiles.
    """
    if registry_id:
        filtered = [
            r for r in rows
            if r.get("registry_id", "").strip() == registry_id
        ]
    elif cloud and compute:
        filtered = [
            r for r in rows
            if not r.get("registry_id", "").strip()
            and (r.get("cloud") or "unknown") == cloud
            and (r.get("compute") or "unknown") == compute
        ]
    else:
        return []

    # Group by date
    daily = defaultdict(list)
    for row in filtered:
        ts = row.get("ts", "")
        date = ts[:10] if ts else "unknown"
        daily[date].append(row)

    result = []
    for date in sorted(daily.keys()):
        events = daily[date]
        max_servers = max(_safe_int(e.get("servers_count", "")) for e in events)
        max_agents = max(_safe_int(e.get("agents_count", "")) for e in events)
        max_skills = max(_safe_int(e.get("skills_count", "")) for e in events)
        max_search = max(
            _safe_int(e.get("search_queries_total", "")) for e in events
        )

        result.append({
            "date": date,
            "events": len(events),
            "max_servers": max_servers,
            "max_agents": max_agents,
            "max_skills": max_skills,
            "max_search_queries": max_search,
        })

    return result


def _compute_version_table(
    rows: list[dict[str, str]],
) -> list[dict]:
    """Compute version adoption table."""
    total = len(rows)
    version_counts = Counter()
    for row in rows:
        version_counts[row.get("v") or "unknown"] += 1

    result = []
    for version, count in version_counts.most_common():
        vtype = _classify_version(version)
        branch = _extract_version_branch(version) if vtype == "dev" else "--"

        result.append({
            "version": version,
            "type": "**Release**" if vtype == "release" else f"Dev ({branch})",
            "events": count,
            "percentage": _format_pct(count, total),
        })

    return result


def _compute_search_stats(
    rows: list[dict[str, str]],
) -> dict:
    """Compute search usage statistics."""
    total_sum = sum(_safe_int(r.get("search_queries_total", "")) for r in rows)
    total_24h = sum(_safe_int(r.get("search_queries_24h", "")) for r in rows)
    total_1h = sum(_safe_int(r.get("search_queries_1h", "")) for r in rows)

    max_total = max(
        (_safe_int(r.get("search_queries_total", "")) for r in rows),
        default=0,
    )

    # Instances with any search activity
    active_instances = set()
    for r in rows:
        if _safe_int(r.get("search_queries_total", "")) > 0:
            rid = r.get("registry_id", "").strip()
            if rid:
                active_instances.add(rid[:12] + "...")
            else:
                # Use profile key for unidentified
                key = f"{r.get('cloud')}/{r.get('compute')}"
                active_instances.add(key)

    total = len(rows)
    avg = total_sum / total if total > 0 else 0

    return {
        "instances_with_search": len(active_instances),
        "active_instance_names": sorted(active_instances),
        "lifetime_sum": total_sum,
        "lifetime_avg": round(avg, 1),
        "lifetime_max": max_total,
        "sum_24h": total_24h,
        "sum_1h": total_1h,
    }


def _compute_feature_adoption(
    rows: list[dict[str, str]],
) -> list[dict]:
    """Compute feature adoption rates."""
    total = len(rows)

    fed_enabled = sum(
        1 for r in rows
        if r.get("federation", "").strip().lower() == "true"
    )
    with_gw = sum(
        1 for r in rows if r.get("mode") == "with-gateway"
    )
    reg_only = sum(
        1 for r in rows if r.get("mode") == "registry-only"
    )

    return [
        {
            "feature": "Federation",
            "enabled": fed_enabled,
            "disabled": total - fed_enabled,
            "rate": _format_pct(fed_enabled, total),
        },
        {
            "feature": "with-gateway mode",
            "enabled": with_gw,
            "disabled": total - with_gw,
            "rate": _format_pct(with_gw, total),
        },
        {
            "feature": "registry-only mode",
            "enabled": reg_only,
            "disabled": total - reg_only,
            "rate": _format_pct(reg_only, total),
        },
        {
            "feature": "Heartbeat opt-in",
            "enabled": sum(
                1 for r in rows if r.get("event") == "heartbeat"
            ),
            "disabled": total,
            "rate": "0%",
        },
    ]


def _build_markdown_tables(
    metrics: dict,
    distributions: dict[str, Counter],
    instances: list[dict],
    unidentified: list[dict],
    versions: list[dict],
    search: dict,
    features: list[dict],
    rows: list[dict[str, str]],
) -> str:
    """Build all markdown tables as a single string."""
    total = metrics["total_events"]
    lines = []

    # Key Metrics
    lines.append("## Key Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Events | {metrics['total_events']} |")
    lines.append(f"| Startup Events | {metrics['startup_events']} |")
    lines.append(f"| Heartbeat Events | {metrics['heartbeat_events']} |")
    lines.append(
        f"| Unique Registry Instances (identified) "
        f"| {metrics['identified_instances']} |"
    )
    lines.append(
        f"| Events with null registry_id "
        f"| {metrics['null_registry_id_count']} "
        f"({metrics['null_registry_id_pct']}) |"
    )
    lines.append(
        f"| Collection Period "
        f"| {metrics['earliest_ts'][:10]} to {metrics['latest_ts'][:10]} |"
    )
    lines.append("")

    # Identified Instances
    lines.append("## Deployment Landscape")
    lines.append("")
    lines.append("### Registry Instances (Identified)")
    lines.append("")
    lines.append(
        "| Registry ID | Cloud | Compute | Storage | Auth "
        "| Federation | Arch | Servers | Agents | Skills "
        "| Search | Events | First Seen |"
    )
    lines.append(
        "|-------------|-------|---------|---------|------"
        "|------------|------|---------|--------|--------"
        "|--------|--------|------------|"
    )
    for inst in instances:
        fed = "Yes" if inst["federation"] else "No"
        lines.append(
            f"| `{inst['registry_id']}` "
            f"| {inst['cloud']} "
            f"| {inst['compute']} "
            f"| {inst['storage']} "
            f"| {inst['auth']} "
            f"| {fed} "
            f"| {inst['arch']} "
            f"| {inst['max_servers']} "
            f"| {inst['max_agents']} "
            f"| {inst['max_skills']} "
            f"| {inst['max_search_queries']} "
            f"| {inst['events']} "
            f"| {inst['first_seen']} |"
        )
    lines.append("")

    # Unidentified Profiles
    lines.append("### Unidentified Instances (null registry_id)")
    lines.append("")
    lines.append(
        "| Cloud | Compute | Arch | Storage | Auth "
        "| Mode | Servers | Agents | Skills "
        "| Search | Events | Period |"
    )
    lines.append(
        "|-------|---------|------|---------|------"
        "|------|---------|--------|--------"
        "|--------|--------|--------|"
    )
    for prof in unidentified:
        period = prof["first_seen"]
        if prof["first_seen"] != prof["latest_seen"]:
            period = f"{prof['first_seen']} - {prof['latest_seen']}"
        lines.append(
            f"| {prof['cloud']} "
            f"| {prof['compute']} "
            f"| {prof['arch']} "
            f"| {prof['storage']} "
            f"| {prof['auth']} "
            f"| {prof['mode']} "
            f"| {prof['max_servers']} "
            f"| {prof['max_agents']} "
            f"| {prof['max_skills']} "
            f"| {prof['max_search_queries']} "
            f"| {prof['events']} "
            f"| {period} |"
        )
    lines.append("")

    # Distribution tables
    dim_config = [
        ("Cloud Provider", "cloud", "Cloud"),
        ("Compute Platform", "compute", "Compute"),
        ("Architecture", "arch", "Architecture"),
        ("Storage Backend", "storage", "Storage"),
        ("Auth Provider", "auth", "Auth Provider"),
    ]
    for title, key, col_name in dim_config:
        lines.append(_md_distribution_table(
            title, distributions[key], total, col_name,
        ))

    # Version Adoption
    lines.append("## Version Adoption")
    lines.append("")
    lines.append("| Version | Type | Events | Percentage |")
    lines.append("|---------|------|--------|------------|")
    for v in versions:
        lines.append(
            f"| `{v['version']}` "
            f"| {v['type']} "
            f"| {v['events']} "
            f"| {v['percentage']} |"
        )
    lines.append("")

    # Feature Adoption
    lines.append("## Feature Adoption")
    lines.append("")
    lines.append("| Feature | Enabled | Disabled | Rate |")
    lines.append("|---------|---------|----------|------|")
    for feat in features:
        lines.append(
            f"| {feat['feature']} "
            f"| {feat['enabled']} "
            f"| {feat['disabled']} "
            f"| {feat['rate']} |"
        )
    lines.append("")

    # Search Usage
    lines.append("## Search Usage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(
        f"| Instances with search activity "
        f"| {search['instances_with_search']} "
        f"({', '.join(search['active_instance_names'])}) |"
    )
    lines.append(
        f"| Total search queries (lifetime sum) "
        f"| {search['lifetime_sum']} |"
    )
    lines.append(
        f"| Average per event | {search['lifetime_avg']} |"
    )
    lines.append(
        f"| Max from single event | {search['lifetime_max']} |"
    )
    lines.append("")

    # Instance Timelines
    lines.append("## Instance Timelines")
    lines.append("")

    # Identified instances
    for inst in instances:
        timeline = _compute_instance_timeline(
            rows, registry_id=inst["registry_id_full"],
        )
        if not timeline:
            continue
        lines.append(f"### `{inst['registry_id']}` ({inst['cloud']}/{inst['compute']})")
        lines.append("")
        lines.append("| Date | Events | Servers | Agents | Skills | Search Queries |")
        lines.append("|------|--------|---------|--------|--------|----------------|")
        for day in timeline:
            lines.append(
                f"| {day['date']} "
                f"| {day['events']} "
                f"| {day['max_servers']} "
                f"| {day['max_agents']} "
                f"| {day['max_skills']} "
                f"| {day['max_search_queries']} |"
            )
        lines.append("")

    # Unidentified profiles with notable activity
    for prof in unidentified:
        if prof["max_servers"] > 0 or prof["max_search_queries"] > 0 or prof["events"] >= 5:
            timeline = _compute_instance_timeline(
                rows, cloud=prof["cloud"], compute=prof["compute"],
            )
            if not timeline:
                continue
            label = f"{prof['cloud']}/{prof['compute']}/{prof['auth']}"
            lines.append(f"### Unidentified: {label}")
            lines.append("")
            lines.append(
                "| Date | Events | Servers | Agents "
                "| Skills | Search Queries |"
            )
            lines.append(
                "|------|--------|---------|--------"
                "|--------|----------------|"
            )
            for day in timeline:
                lines.append(
                    f"| {day['date']} "
                    f"| {day['events']} "
                    f"| {day['max_servers']} "
                    f"| {day['max_agents']} "
                    f"| {day['max_skills']} "
                    f"| {day['max_search_queries']} |"
                )
            lines.append("")

    return "\n".join(lines)


def _write_outputs(
    md_content: str,
    metrics_json: dict,
    output_dir: str,
    date_str: str,
) -> None:
    """Write markdown tables and JSON metrics to files."""
    md_path = os.path.join(output_dir, f"tables-{date_str}.md")
    json_path = os.path.join(output_dir, f"metrics-{date_str}.json")

    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Markdown tables written to {md_path}")

    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    logger.info(f"JSON metrics written to {json_path}")


def main() -> None:
    """Parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze telemetry CSV and generate markdown tables",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to registry_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Report date (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        raise SystemExit(1)

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    rows = _read_csv(args.csv)
    if not rows:
        logger.error("No data in CSV file")
        raise SystemExit(1)

    metrics = _compute_key_metrics(rows)
    distributions = _compute_distributions(rows)
    instances = _compute_instance_table(rows)
    unidentified = _compute_unidentified_profiles(rows)
    versions = _compute_version_table(rows)
    search = _compute_search_stats(rows)
    features = _compute_feature_adoption(rows)

    md_content = _build_markdown_tables(
        metrics, distributions, instances, unidentified,
        versions, search, features, rows,
    )

    # Build JSON with all computed data
    metrics_json = {
        "report_date": date_str,
        "key_metrics": metrics,
        "distributions": {
            k: dict(v.most_common()) for k, v in distributions.items()
        },
        "identified_instances": instances,
        "unidentified_profiles": unidentified,
        "version_adoption": versions,
        "search_stats": search,
        "feature_adoption": features,
    }

    _write_outputs(md_content, metrics_json, args.output_dir, date_str)
    logger.info(f"Analysis complete: {metrics['total_events']} events, "
                f"{metrics['identified_instances']} identified instances")


if __name__ == "__main__":
    main()
