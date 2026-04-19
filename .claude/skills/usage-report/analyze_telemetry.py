"""Analyze telemetry CSV and output pre-formatted markdown tables.

Reads registry_metrics.csv, computes all distributions, instance
timelines, search stats, and version adoption. Writes a JSON file
with raw metrics and a markdown file with pre-formatted tables
ready to embed in the usage report.

Supports comparison with a previous metrics JSON to produce an
executive summary with deltas at the top of the report.
"""
import argparse
import csv
import glob
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


def _find_previous_metrics(
    output_dir: str,
    current_date: str,
) -> str | None:
    """Find the most recent metrics JSON file before the current date.

    Scans the output directory and dated subdirectories (YYYY-MM-DD/)
    for metrics-YYYY-MM-DD.json files and returns the path to the most
    recent one that predates current_date.
    """
    flat_pattern = os.path.join(output_dir, "metrics-*.json")
    dated_pattern = os.path.join(output_dir, "*", "metrics-*.json")
    candidates = glob.glob(flat_pattern) + glob.glob(dated_pattern)

    valid = []
    for path in candidates:
        basename = os.path.basename(path)
        date_part = basename.replace("metrics-", "").replace(".json", "")
        if len(date_part) == 10 and date_part < current_date:
            valid.append((date_part, path))

    if not valid:
        logger.info("No previous metrics file found for comparison")
        return None

    valid.sort(key=lambda x: x[0], reverse=True)
    selected = valid[0]
    logger.info(f"Found previous metrics: {selected[1]} (date: {selected[0]})")
    return selected[1]


def _load_previous_metrics(
    path: str,
) -> dict | None:
    """Load a previous metrics JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded previous metrics from {path}")
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load previous metrics: {e}")
        return None


def _compute_delta(
    current: int | float,
    previous: int | float,
) -> str:
    """Compute a delta string like '+5 (+25%)' or '-3 (-10%)'."""
    diff = current - previous
    if previous == 0:
        if diff == 0:
            return "no change"
        return f"+{diff}" if diff > 0 else f"{diff}"

    pct = diff / previous * 100
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff} ({sign}{pct:.0f}%)"


def _compute_per_cloud_unique_installs(
    rows: list[dict[str, str]],
) -> dict[str, int]:
    """Compute unique registry_id count per cloud provider."""
    cloud_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        rid = row.get("registry_id", "").strip()
        if not rid:
            continue
        cloud = row.get("cloud") or "unknown"
        cloud_ids[cloud].add(rid)

    return {cloud: len(ids) for cloud, ids in sorted(cloud_ids.items())}


def _compute_per_cloud_last_event(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Compute last startup and heartbeat timestamp per cloud provider.

    Returns dict mapping cloud -> {"last_startup": ts, "last_heartbeat": ts}.
    Timestamps are date strings (YYYY-MM-DD).
    """
    cloud_last_startup: dict[str, str] = {}
    cloud_last_heartbeat: dict[str, str] = {}

    for row in rows:
        cloud = row.get("cloud") or "unknown"
        ts = (row.get("ts") or "")[:10]
        if not ts:
            continue

        event_type = row.get("event", "")
        if event_type == "startup":
            if ts > cloud_last_startup.get(cloud, ""):
                cloud_last_startup[cloud] = ts
        elif event_type == "heartbeat":
            if ts > cloud_last_heartbeat.get(cloud, ""):
                cloud_last_heartbeat[cloud] = ts

    result: dict[str, dict[str, str]] = {}
    all_clouds = sorted(set(list(cloud_last_startup.keys()) + list(cloud_last_heartbeat.keys())))
    for cloud in all_clouds:
        result[cloud] = {
            "last_startup": cloud_last_startup.get(cloud, "--"),
            "last_heartbeat": cloud_last_heartbeat.get(cloud, "--"),
        }

    return result


def _compute_instance_lifetime(
    instances: list[dict],
) -> list[dict]:
    """Compute the lifetime (age in days) for each identified instance.

    Lifetime is the number of days between the first event and the
    last event for that instance. A single-day instance has age 0.

    Returns a list sorted by age descending.
    """
    result = []
    for inst in instances:
        first = inst.get("first_seen", "")
        latest = inst.get("latest_seen", "")
        if not first or not latest:
            continue

        first_date = datetime.strptime(first, "%Y-%m-%d")
        latest_date = datetime.strptime(latest, "%Y-%m-%d")
        age_days = (latest_date - first_date).days

        result.append({
            "registry_id": inst["registry_id"],
            "cloud": inst["cloud"],
            "compute": inst["compute"],
            "auth": inst["auth"],
            "first_seen": first,
            "latest_seen": latest,
            "age_days": age_days,
            "events": inst["events"],
        })

    result.sort(key=lambda x: x["age_days"], reverse=True)
    return result


def _build_exec_summary_md(
    current_metrics: dict,
    previous_metrics: dict | None,
    current_cloud_installs: dict[str, int],
    previous_cloud_installs: dict[str, int] | None,
    cloud_last_event: dict[str, dict[str, str]] | None = None,
) -> str:
    """Build the executive summary markdown section with comparison."""
    lines = []
    lines.append("## Executive Summary")
    lines.append("")

    curr = current_metrics

    # Lead with new installs if we have a previous report
    if previous_metrics:
        prev = previous_metrics.get("key_metrics", {})
        prev_identified = prev.get("identified_instances", 0)
        new_installs = curr["identified_instances"] - prev_identified
        prev_date = previous_metrics.get("report_date", "unknown")
        if new_installs > 0:
            lines.append(
                f"**{new_installs} new registry installs** since the last "
                f"report ({prev_date}), bringing the total to "
                f"**{curr['identified_instances']} unique identified "
                f"registry instances** across "
                f"**{curr['total_events']} events** "
                f"over the period {curr['earliest_ts'][:10]} to "
                f"{curr['latest_ts'][:10]}."
            )
        else:
            lines.append(
                f"This report covers **{curr['total_events']} events** "
                f"from **{curr['identified_instances']} unique identified "
                f"registry instances** "
                f"over the period {curr['earliest_ts'][:10]} to "
                f"{curr['latest_ts'][:10]}. No new installs since the "
                f"last report ({prev_date})."
            )
    else:
        lines.append(
            f"This report covers **{curr['total_events']} events** "
            f"from **{curr['identified_instances']} unique identified "
            f"registry instances** "
            f"over the period {curr['earliest_ts'][:10]} to "
            f"{curr['latest_ts'][:10]}."
        )
    lines.append("")

    if previous_metrics is None:
        lines.append("*No previous report available for comparison.*")
        lines.append("")
        lines.append("### Unique Registry Installs by Cloud Provider")
        lines.append("")
        lines.append("| Cloud Provider | Unique Installs | Last Startup | Last Heartbeat |")
        lines.append("|----------------|-----------------|--------------|----------------|")
        for cloud, count in current_cloud_installs.items():
            last_ev = (cloud_last_event or {}).get(cloud, {})
            last_startup = last_ev.get("last_startup", "--")
            last_heartbeat = last_ev.get("last_heartbeat", "--")
            lines.append(f"| {cloud} | {count} | {last_startup} | {last_heartbeat} |")
        lines.append("")
        return "\n".join(lines)

    prev = previous_metrics.get("key_metrics", {})
    prev_date = previous_metrics.get("report_date", "unknown")

    lines.append(f"### Comparison with Previous Report ({prev_date})")
    lines.append("")
    lines.append("| Metric | Previous | Current | Change |")
    lines.append("|--------|----------|---------|--------|")

    # Key metric comparisons
    comparisons = [
        ("Total Events", "total_events"),
        ("Startup Events", "startup_events"),
        ("Heartbeat Events", "heartbeat_events"),
        ("Unique Instances (identified)", "identified_instances"),
        ("Events with null registry_id", "null_registry_id_count"),
    ]
    for label, key in comparisons:
        prev_val = prev.get(key, 0)
        curr_val = curr.get(key, 0)
        delta = _compute_delta(curr_val, prev_val)
        lines.append(f"| {label} | {prev_val} | {curr_val} | {delta} |")

    lines.append("")

    lines.append("### Unique Registry Installs by Cloud Provider")
    lines.append("")
    if previous_cloud_installs:
        lines.append(
            "| Cloud Provider | Previous | Current | Change | Last Startup | Last Heartbeat |"
        )
        lines.append(
            "|----------------|----------|---------|--------|--------------|----------------|"
        )
        all_clouds = sorted(
            set(list(current_cloud_installs.keys()) + list(previous_cloud_installs.keys()))
        )
        for cloud in all_clouds:
            prev_count = previous_cloud_installs.get(cloud, 0)
            curr_count = current_cloud_installs.get(cloud, 0)
            delta = _compute_delta(curr_count, prev_count)
            last_ev = (cloud_last_event or {}).get(cloud, {})
            last_startup = last_ev.get("last_startup", "--")
            last_heartbeat = last_ev.get("last_heartbeat", "--")
            lines.append(
                f"| {cloud} | {prev_count} | {curr_count} | {delta} "
                f"| {last_startup} | {last_heartbeat} |"
            )
    else:
        lines.append("| Cloud Provider | Unique Installs | Last Startup | Last Heartbeat |")
        lines.append("|----------------|-----------------|--------------|----------------|")
        for cloud, count in current_cloud_installs.items():
            last_ev = (cloud_last_event or {}).get(cloud, {})
            last_startup = last_ev.get("last_startup", "--")
            last_heartbeat = last_ev.get("last_heartbeat", "--")
            lines.append(f"| {cloud} | {count} | {last_startup} | {last_heartbeat} |")
    lines.append("")

    # Distribution shift highlights
    prev_dists = previous_metrics.get("distributions", {})
    prev_cloud_dist = prev_dists.get("cloud", {})

    # Highlight new cloud providers
    prev_clouds = set(prev_cloud_dist.keys()) if prev_cloud_dist else set()
    curr_clouds = set(current_cloud_installs.keys())
    new_clouds = curr_clouds - prev_clouds
    if new_clouds:
        lines.append(
            f"**New cloud providers**: {', '.join(sorted(new_clouds))}"
        )
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

        # search_queries_total is already a lifetime cumulative counter
        # in each event, so take the max (latest value) not the sum
        total_search = max(
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
            "total_search_queries": total_search,
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
        # search_queries_total is already a lifetime cumulative counter
        # in each event, so take the max (latest value) not the sum
        total_search = max(
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
            "total_search_queries": total_search,
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
    """Compute search usage statistics.

    search_queries_total is a lifetime cumulative counter in each event,
    so we take the max per instance (latest reported value) then sum
    across instances to get the fleet-wide total.
    """
    # Group by instance to get the max (latest) lifetime count per instance
    instance_max_total: dict[str, int] = {}
    instance_max_24h: dict[str, int] = {}
    instance_max_1h: dict[str, int] = {}

    active_instances: set[str] = set()

    for r in rows:
        rid = r.get("registry_id", "").strip()
        if rid:
            instance_key = rid[:12] + "..."
        else:
            instance_key = f"{r.get('cloud')}/{r.get('compute')}"

        sq_total = _safe_int(r.get("search_queries_total", ""))
        sq_24h = _safe_int(r.get("search_queries_24h", ""))
        sq_1h = _safe_int(r.get("search_queries_1h", ""))

        instance_max_total[instance_key] = max(
            instance_max_total.get(instance_key, 0), sq_total
        )
        instance_max_24h[instance_key] = max(
            instance_max_24h.get(instance_key, 0), sq_24h
        )
        instance_max_1h[instance_key] = max(
            instance_max_1h.get(instance_key, 0), sq_1h
        )

        if sq_total > 0:
            active_instances.add(instance_key)

    # Fleet-wide totals: sum of per-instance max values
    total_sum = sum(instance_max_total.values())
    total_24h = sum(instance_max_24h.values())
    total_1h = sum(instance_max_1h.values())

    max_total = max(instance_max_total.values(), default=0)

    instance_count = len(instance_max_total)
    avg = total_sum / instance_count if instance_count > 0 else 0

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
            "feature": "Heartbeat (opt-out, on by default)",
            "enabled": len({
                r.get("registry_id", "").strip()
                for r in rows
                if r.get("event") == "heartbeat"
                and r.get("registry_id", "").strip()
            }),
            "disabled": total - sum(
                1 for r in rows if r.get("event") == "heartbeat"
            ),
            "rate": _format_pct(
                sum(1 for r in rows if r.get("event") == "heartbeat"),
                total,
            ),
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
    exec_summary_md: str | None = None,
    instance_lifetime: list[dict] | None = None,
) -> str:
    """Build all markdown tables as a single string."""
    total = metrics["total_events"]
    lines = []

    # Executive Summary (at the top if available)
    if exec_summary_md:
        lines.append(exec_summary_md)
        lines.append("")

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

    # Instance Lifetime / Age table
    if instance_lifetime:
        ages = [inst["age_days"] for inst in instance_lifetime]
        avg_age = sum(ages) / len(ages) if ages else 0
        max_age = max(ages) if ages else 0
        active_count = sum(1 for a in ages if a > 0)

        lines.append("## Registry Instance Lifetime")
        lines.append("")
        lines.append(
            f"Across {len(instance_lifetime)} identified instances, "
            f"the average lifetime is **{avg_age:.1f} days** "
            f"(max {max_age} days). "
            f"{active_count} instances have been seen across multiple days, "
            f"while {len(ages) - active_count} were only seen on a single day."
        )
        lines.append("")
        lines.append(
            "| Registry ID | Cloud | Compute | Auth "
            "| First Seen | Last Seen | Age (days) | Events |"
        )
        lines.append(
            "|-------------|-------|---------|------"
            "|------------|-----------|------------|--------|"
        )
        for inst in instance_lifetime:
            lines.append(
                f"| `{inst['registry_id']}` "
                f"| {inst['cloud']} "
                f"| {inst['compute']} "
                f"| {inst['auth']} "
                f"| {inst['first_seen']} "
                f"| {inst['latest_seen']} "
                f"| {inst['age_days']} "
                f"| {inst['events']} |"
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
        "| Search (Lifetime) | Events | First Seen |"
    )
    lines.append(
        "|-------------|-------|---------|---------|------"
        "|------------|------|---------|--------|--------"
        "|-------------------|--------|------------|"
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
            f"| {inst['total_search_queries']} "
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
        "| Search (Lifetime) | Events | Period |"
    )
    lines.append(
        "|-------|---------|------|---------|------"
        "|------|---------|--------|--------"
        "|-------------------|--------|--------|"
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
            f"| {prof['total_search_queries']} "
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
        f"| Total search queries (sum of per-instance lifetime counts) "
        f"| {search['lifetime_sum']} |"
    )
    lines.append(
        f"| Average per instance | {search['lifetime_avg']} |"
    )
    lines.append(
        f"| Max from single instance | {search['lifetime_max']} |"
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
    parser.add_argument(
        "--previous-metrics",
        default=None,
        help=(
            "Path to previous metrics JSON for comparison. "
            "If not provided, auto-detects the most recent one in output-dir."
        ),
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
    instance_lifetime = _compute_instance_lifetime(instances)
    unidentified = _compute_unidentified_profiles(rows)
    versions = _compute_version_table(rows)
    search = _compute_search_stats(rows)
    features = _compute_feature_adoption(rows)
    cloud_installs = _compute_per_cloud_unique_installs(rows)
    cloud_last_event = _compute_per_cloud_last_event(rows)

    # Load previous metrics for comparison
    prev_metrics_path = args.previous_metrics
    if not prev_metrics_path:
        prev_metrics_path = _find_previous_metrics(args.output_dir, date_str)

    previous_metrics = None
    previous_cloud_installs = None
    if prev_metrics_path:
        previous_metrics = _load_previous_metrics(prev_metrics_path)
        if previous_metrics:
            previous_cloud_installs = previous_metrics.get(
                "per_cloud_unique_installs", None
            )

    # Build executive summary
    exec_summary_md = _build_exec_summary_md(
        metrics,
        previous_metrics,
        cloud_installs,
        previous_cloud_installs,
        cloud_last_event=cloud_last_event,
    )

    md_content = _build_markdown_tables(
        metrics, distributions, instances, unidentified,
        versions, search, features, rows,
        exec_summary_md=exec_summary_md,
        instance_lifetime=instance_lifetime,
    )

    # Build JSON with all computed data
    metrics_json = {
        "report_date": date_str,
        "key_metrics": metrics,
        "per_cloud_unique_installs": cloud_installs,
        "instance_lifetime": instance_lifetime,
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
