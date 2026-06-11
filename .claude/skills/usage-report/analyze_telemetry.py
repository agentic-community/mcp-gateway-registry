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
import re
from collections import Counter, defaultdict
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


def _parse_internal_instances(
    path: str,
) -> set[str]:
    """Parse known-internal-instances.md and return a set of registry IDs.

    The file contains a markdown table with registry IDs in backticks.
    Returns an empty set if the file does not exist.
    """
    if not path or not os.path.exists(path):
        logger.info("No known-internal-instances file found, treating all as external")
        return set()

    ids = set()
    with open(path) as f:
        for line in f:
            match = re.search(
                r"`([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})`", line
            )
            if match:
                ids.add(match.group(1))

    logger.info(f"Loaded {len(ids)} known internal instance IDs from {path}")
    return ids


def _is_internal(
    registry_id: str,
    internal_ids: set[str],
) -> bool:
    """Check if a registry_id (possibly truncated) matches a known internal ID."""
    if not internal_ids:
        return False
    truncated = registry_id.rstrip(".")
    for full_id in internal_ids:
        if full_id.startswith(truncated):
            return True
    return False


def _compute_stickiness(
    instance_lifetime: list[dict],
    internal_ids: set[str],
) -> dict:
    """Compute stickiness metrics for the customer (non-internal) fleet.

    Returns:
      sticky_3plus_days: count of instances with age_days >= 3 (legacy)
      one_day_wonders / one_day_wonder_pct: instances seen on exactly one
        calendar day, never seen again
      lifetime_bucket_counts / lifetime_bucket_pct: nested dict with the
        thresholds 3, 7, 14, 30 -> count and pct of customers whose
        age_days >= threshold. Used to track the customer-retention curve
        over time.
      longest_non_internal_*: the customer instance with the largest
        age_days value
    """
    non_internal = [
        inst for inst in instance_lifetime if not _is_internal(inst["registry_id"], internal_ids)
    ]
    sticky = [inst for inst in non_internal if inst["age_days"] >= 3]
    # One-day wonders: customer instances whose first_seen and last_seen fall on
    # the same calendar day (age_days == 0). These represent "tried it once and
    # never came back" deployments and are worth tracking as a leading
    # indicator of evaluation-to-persistence conversion.
    one_day_wonders = [inst for inst in non_internal if inst["age_days"] == 0]
    total_non_internal = len(non_internal)
    one_day_wonder_pct = (
        round(100.0 * len(one_day_wonders) / total_non_internal, 1)
        if total_non_internal
        else 0.0
    )
    # Lifetime buckets (cumulative thresholds): for each threshold, what
    # share of customer instances have an age_days >= that threshold? These
    # form the survival curve we want to chart over time. Capped at >= 30
    # days (no separate ">30" bucket).
    lifetime_thresholds = [3, 7, 14, 30]
    lifetime_bucket_counts = {
        threshold: sum(1 for inst in non_internal if inst["age_days"] >= threshold)
        for threshold in lifetime_thresholds
    }
    lifetime_bucket_pct = {
        threshold: (
            round(100.0 * count / total_non_internal, 1) if total_non_internal else 0.0
        )
        for threshold, count in lifetime_bucket_counts.items()
    }
    longest = max(non_internal, key=lambda x: x["age_days"]) if non_internal else None

    return {
        "sticky_3plus_days": len(sticky),
        "total_non_internal": total_non_internal,
        "one_day_wonders": len(one_day_wonders),
        "one_day_wonder_pct": one_day_wonder_pct,
        "lifetime_bucket_counts": lifetime_bucket_counts,
        "lifetime_bucket_pct": lifetime_bucket_pct,
        "longest_non_internal_id": longest["registry_id"] if longest else None,
        "longest_non_internal_days": longest["age_days"] if longest else 0,
    }


def _display_id(
    registry_id: str,
    internal_ids: set[str],
) -> str:
    """Return registry_id with ' (internal)' suffix if it matches a known internal ID."""
    if _is_internal(registry_id, internal_ids):
        return f"{registry_id} (internal)"
    return registry_id


def _latest_nonempty(
    events: list[dict],
    field: str,
) -> str:
    """Return the most recent non-empty value for a field across events.

    Events must be sorted by timestamp ascending. Heartbeat events populate
    fields like search_backend and embeddings_provider while startup events
    leave them empty, so the latest non-empty value reflects the current
    runtime configuration.
    """
    for event in reversed(events):
        value = (event.get(field) or "").strip()
        if value:
            return value
    return "unknown"


def _score_instances(
    instances: list[dict],
    internal_ids: set[str],
) -> list[dict]:
    """Score instances by feature usage and return sorted list (descending).

    Activity score = max_servers + max_agents + max_skills + total_search_queries.
    Instances with zero activity are excluded.
    """
    scored = []
    for inst in instances:
        servers = inst.get("max_servers", 0)
        agents = inst.get("max_agents", 0)
        skills = inst.get("max_skills", 0)
        search = inst.get("total_search_queries", 0)
        total = servers + agents + skills + search
        if total == 0:
            continue
        scored.append(
            {
                "registry_id": inst["registry_id"],
                "cloud": inst["cloud"],
                "compute": inst["compute"],
                "auth": inst["auth"],
                "version": inst.get("version", "unknown"),
                "servers": servers,
                "agents": agents,
                "skills": skills,
                "search": search,
                "total": total,
                "embeddings_provider": inst.get("embeddings_provider", "unknown"),
                "embeddings_backend_kind": inst.get("embeddings_backend_kind", "unknown"),
                "is_internal": _is_internal(inst["registry_id"], internal_ids),
            }
        )

    scored.sort(key=lambda x: x["total"], reverse=True)
    return scored


def _compute_embeddings_backend_breakdown(
    instances: list[dict],
) -> list[dict]:
    """Group unique instances by embeddings_backend_kind.

    Each instance contributes its latest non-empty backend_kind value. If an
    instance was observed only on schema v1 events (which did not carry the
    field), it lands in the "unknown" bucket. That bucket will shrink as
    operators upgrade to v1.0.22+.

    Returns:
        List of {"kind": str, "instances": int, "percentage": str} rows,
        sorted by instance count descending.
    """
    total = len(instances)
    if total == 0:
        return []

    counts: dict[str, int] = defaultdict(int)
    for inst in instances:
        kind = inst.get("embeddings_backend_kind") or "unknown"
        counts[kind] += 1

    rows = [
        {
            "kind": kind,
            "instances": count,
            "percentage": _format_pct(count, total),
        }
        for kind, count in counts.items()
    ]
    rows.sort(key=lambda r: r["instances"], reverse=True)
    return rows


def _build_embeddings_backend_breakdown_table(
    instances: list[dict],
) -> tuple[str, list[dict]]:
    """Build the Embeddings Backend Breakdown markdown section.

    Returns:
        Tuple of (markdown string, list of row dicts for JSON output).
    """
    rows = _compute_embeddings_backend_breakdown(instances)
    if not rows:
        return "", []

    total = sum(r["instances"] for r in rows)
    lines = []
    lines.append("## Embeddings Backend Breakdown")
    lines.append("")
    lines.append(
        f"Unique instances grouped by derived `embeddings_backend_kind`. Total instances: {total}."
    )
    lines.append("")
    lines.append("| Backend Kind | Unique Instances | % of Fleet |")
    lines.append("|--------------|------------------|------------|")
    for row in rows:
        lines.append(f"| `{row['kind']}` | {row['instances']} | {row['percentage']} |")
    lines.append("")

    # Reviewer D2: when "unknown" dominates, explain why. Pre-v1.0.22
    # registries did not emit the field, so they all bucket as unknown
    # during the rollout window.
    top_kind = rows[0]["kind"]
    top_count = rows[0]["instances"]
    if top_kind == "unknown" and top_count / total >= 0.5:
        lines.append(
            f"> **Note:** The `unknown` bucket dominates ({top_count}/{total} "
            f"instances) because the `embeddings_backend_kind` field was added "
            f"in schema v2 (registry v1.0.22+). Pre-v1.0.22 registries do not "
            f"emit this field. The bucket will shrink as operators upgrade."
        )
        lines.append("")

    return "\n".join(lines), rows


def _compute_cloud_detection_method_breakdown(
    instances: list[dict],
) -> list[dict]:
    """Group unique instances by cloud_detection_method.

    Each instance contributes its latest non-empty cloud_detection_method
    value. Pre-v3 events (before issue #986 shipped) did not carry the
    field, so those instances land in the "unknown" bucket until they
    emit a new event.

    Returns:
        List of {"method": str, "instances": int, "percentage": str} rows,
        sorted by instance count descending.
    """
    total = len(instances)
    if total == 0:
        return []

    counts: dict[str, int] = defaultdict(int)
    for inst in instances:
        method = inst.get("cloud_detection_method") or "unknown"
        counts[method] += 1

    rows = [
        {
            "method": method,
            "instances": count,
            "percentage": _format_pct(count, total),
        }
        for method, count in counts.items()
    ]
    rows.sort(key=lambda r: r["instances"], reverse=True)
    return rows


def _build_cloud_detection_method_breakdown_table(
    instances: list[dict],
) -> tuple[str, list[dict]]:
    """Build the Cloud Detection Method section markdown.

    Returns:
        Tuple of (markdown string, list of row dicts for JSON output).
    """
    rows = _compute_cloud_detection_method_breakdown(instances)
    if not rows:
        return "", []

    total = sum(r["instances"] for r in rows)
    lines = []
    lines.append("## Cloud Detection Method")
    lines.append("")
    lines.append(
        f"Unique instances grouped by `cloud_detection_method`. Total instances: {total}."
    )
    lines.append("")
    lines.append("| Detection Method | Unique Instances | % of Fleet |")
    lines.append("|------------------|------------------|------------|")
    for row in rows:
        lines.append(f"| `{row['method']}` | {row['instances']} | {row['percentage']} |")
    lines.append("")

    # Pre-v1.23 registries did not emit the field, so they all bucket as
    # unknown during the rollout window.
    top_method = rows[0]["method"]
    top_count = rows[0]["instances"]
    if top_method == "unknown" and top_count / total >= 0.5:
        lines.append(
            f"> **Note:** The `unknown` bucket dominates ({top_count}/{total} "
            f"instances) because the `cloud_detection_method` field was added "
            f"in schema v3 (registry v1.23.0+). Pre-v3 registries do not emit "
            f"this field. The bucket will shrink as operators upgrade."
        )
        lines.append("")

    return "\n".join(lines), rows


def _build_largest_catalogs_table(
    instances: list[dict],
    internal_ids: set[str],
    top_n: int = 10,
) -> str:
    """Build a markdown table ranking instances by registered-object count.

    Sort key is `max_servers + max_agents + max_skills` for the latest
    heartbeat per instance. This surfaces the comprehensive-catalog
    deployments (lots of registered MCP servers/agents/skills) which can
    rank low on the activity-score leaderboard if they have not generated
    much search volume yet.
    """
    enriched: list[dict] = []
    for inst in instances:
        if _is_internal(inst.get("registry_id_full", inst["registry_id"]), internal_ids):
            continue
        servers = inst.get("max_servers", 0)
        agents = inst.get("max_agents", 0)
        skills = inst.get("max_skills", 0)
        catalog = servers + agents + skills
        if catalog == 0:
            continue
        enriched.append({**inst, "catalog_total": catalog})
    enriched.sort(key=lambda x: x["catalog_total"], reverse=True)
    top = enriched[:top_n]

    lines: list[str] = []
    lines.append("## Largest Catalogs (by Registered Servers + Agents + Skills)")
    lines.append("")
    if not top:
        lines.append("*No customer instances with registered objects in this report.*")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "| Rank | Registry ID | Cloud/Compute/Auth "
        "| Version | Servers | Agents | Skills | Catalog Total |"
    )
    lines.append(
        "|------|-------------|-------------------"
        "|---------|---------|--------|--------|---------------|"
    )
    for i, inst in enumerate(top, 1):
        label = f"{inst['cloud']}/{inst['compute']}/{inst['auth']}"
        lines.append(
            f"| {i} "
            f"| `{inst['registry_id']}` "
            f"| {label} "
            f"| `{inst.get('version', 'unknown')}` "
            f"| {inst.get('max_servers', 0)} "
            f"| {inst.get('max_agents', 0)} "
            f"| {inst.get('max_skills', 0)} "
            f"| {inst['catalog_total']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_most_engaged_operators_table(
    upgrade_trajectories: list[dict],
    instance_lifetime: list[dict],
    instances: list[dict],
    internal_ids: set[str],
    top_n: int = 10,
) -> str:
    """Build a markdown table of customers with the longest upgrade chains.

    Activity scoreboards reward feature volume; persistence scoreboards
    reward staying alive. Neither rewards the operator who tracks the
    project closely enough to upgrade across multiple versions. This
    table ranks non-internal customer instances by `version_changes`
    (length of distinct-version trajectory minus one), with a tiebreaker
    on lifetime age_days.
    """
    if not upgrade_trajectories:
        return ""

    # Index instance metadata for fast lookup
    age_by_id: dict[str, int] = {}
    for li in instance_lifetime or []:
        rid_short = li.get("registry_id", "")
        if rid_short:
            age_by_id[rid_short] = li.get("age_days", 0)

    inst_meta: dict[str, dict] = {}
    for inst in instances:
        rid_short = inst.get("registry_id", "")
        if rid_short:
            inst_meta[rid_short] = inst

    enriched: list[dict] = []
    for traj in upgrade_trajectories:
        rid_short = traj.get("registry_id", "")
        rid_full = traj.get("registry_id_full", rid_short)
        if _is_internal(rid_full, internal_ids):
            continue
        meta = inst_meta.get(rid_short, {})
        enriched.append(
            {
                "registry_id": rid_short,
                "version_changes": traj.get("version_changes", 0),
                "trajectory": traj.get("trajectory", []),
                "age_days": age_by_id.get(rid_short, 0),
                "cloud": meta.get("cloud", "unknown"),
                "compute": meta.get("compute", "unknown"),
                "auth": meta.get("auth", "unknown"),
            }
        )
    enriched.sort(key=lambda x: (-x["version_changes"], -x["age_days"]))
    top = enriched[:top_n]

    lines: list[str] = []
    lines.append("## Most Engaged Operators (by Upgrade-Chain Length)")
    lines.append("")
    lines.append(
        "Customer instances ranked by the number of distinct versions they have "
        "ever reported. Length 7 means the operator upgraded six times. "
        "Tiebreaker: longest lifetime."
    )
    lines.append("")
    lines.append(
        "| Rank | Registry ID | Cloud/Compute/Auth | Version Changes | Age (days) | Trajectory |"
    )
    lines.append(
        "|------|-------------|-------------------|----------------:|-----------:|------------|"
    )
    for i, row in enumerate(top, 1):
        label = f"{row['cloud']}/{row['compute']}/{row['auth']}"
        path = " -> ".join(f"`{v}`" for v in row["trajectory"])
        lines.append(
            f"| {i} | `{row['registry_id']}` | {label} | {row['version_changes']} | "
            f"{row['age_days']} | {path} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_most_active_table(
    instances: list[dict],
    internal_ids: set[str],
    top_n: int = 10,
) -> str:
    """Build a markdown table of the most active non-internal instances."""
    scored = _score_instances(instances, internal_ids)
    non_internal = [inst for inst in scored if not inst["is_internal"]]
    top = non_internal[:top_n]

    lines = []
    lines.append("## Most Active Instances (by Feature Usage)")
    lines.append("")
    lines.append(
        "| Rank | Registry ID | Cloud/Compute/Auth "
        "| Version | Embeddings | Servers | Agents | Skills | Search | Total |"
    )
    lines.append(
        "|------|-------------|-------------------"
        "|---------|------------|---------|--------|--------|--------|-------|"
    )
    for i, inst in enumerate(top, 1):
        label = f"{inst['cloud']}/{inst['compute']}/{inst['auth']}"
        # Prefer the derived backend_kind when available (schema v2+). Fall
        # back to the raw provider string for pre-v1.0.22 instances so the
        # column still says something useful during the rollout window.
        backend_kind = inst.get("embeddings_backend_kind", "unknown")
        if backend_kind == "unknown":
            embeddings_label = inst.get("embeddings_provider", "unknown")
        else:
            embeddings_label = backend_kind
        lines.append(
            f"| {i} "
            f"| `{inst['registry_id']}` "
            f"| {label} "
            f"| `{inst['version']}` "
            f"| {embeddings_label} "
            f"| {inst['servers']} "
            f"| {inst['agents']} "
            f"| {inst['skills']} "
            f"| {inst['search']} "
            f"| {inst['total']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _get_sticky_instances(
    instances: list[dict],
    instance_lifetime: list[dict],
    internal_ids: set[str],
    min_age_days: int = 3,
) -> list[dict]:
    """Return non-internal instances with age >= min_age_days."""
    age_lookup = {inst["registry_id"]: inst["age_days"] for inst in instance_lifetime}

    sticky = []
    for inst in instances:
        rid = inst["registry_id"]
        if _is_internal(rid, internal_ids):
            continue
        age = age_lookup.get(rid, 0)
        if age >= min_age_days:
            sticky.append(inst)
    return sticky


def _compute_sticky_profile_counts(
    sticky_instances: list[dict],
) -> dict[tuple[str, str, str, str], int]:
    """Count instances per (cloud, compute, storage, auth) combination."""
    counter: dict[tuple[str, str, str, str], int] = {}
    for inst in sticky_instances:
        key = (inst["cloud"], inst["compute"], inst["storage"], inst["auth"])
        counter[key] = counter.get(key, 0) + 1
    return counter


def _build_sticky_breakdown_table(
    instances: list[dict],
    instance_lifetime: list[dict],
    internal_ids: set[str],
    previous_sticky_profiles: dict[str, int] | None = None,
) -> tuple[str, dict[str, int]]:
    """Build a single table of sticky instances grouped by deployment profile.

    Each row is a unique (cloud, compute, storage, auth) combination with
    instance count, percentage, and change vs the previous report.

    Returns:
        Tuple of (markdown_string, profile_counts_for_json).
    """
    sticky = _get_sticky_instances(instances, instance_lifetime, internal_ids)
    total_sticky = len(sticky)

    profile_counts = _compute_sticky_profile_counts(sticky)

    profile_counts_for_json = {
        f"{c}/{co}/{s}/{a}": count for (c, co, s, a), count in profile_counts.items()
    }

    if total_sticky == 0:
        return "", profile_counts_for_json

    sorted_profiles = sorted(
        profile_counts.items(),
        key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3]),
    )

    prev = previous_sticky_profiles or {}

    lines = []
    lines.append("## Sticky Instance Breakdown (3+ Days)")
    lines.append("")
    lines.append(
        f"**{total_sticky} non-internal instances** running for 3 or more days, "
        f"grouped by deployment profile."
    )
    lines.append("")
    lines.append("| Cloud | Compute | Storage | Auth | Instances | Percentage | Change |")
    lines.append("|-------|---------|---------|------|-----------|------------|--------|")
    for (cloud, compute, storage, auth), count in sorted_profiles:
        pct = _format_pct(count, total_sticky)
        profile_key = f"{cloud}/{compute}/{storage}/{auth}"
        prev_count = prev.get(profile_key, 0)
        if prev_count == 0 and count > 0:
            change = f"+{count} (new)"
        elif prev_count == count:
            change = "0"
        else:
            delta = count - prev_count
            sign = "+" if delta > 0 else ""
            change = f"{sign}{delta}"
        lines.append(f"| {cloud} | {compute} | {storage} | {auth} | {count} | {pct} | {change} |")
    lines.append("")

    return "\n".join(lines), profile_counts_for_json


def _build_sticky_cloud_compute_table(
    instances: list[dict],
    instance_lifetime: list[dict],
    internal_ids: set[str],
    previous_sticky_cloud_compute: dict[str, int] | None = None,
) -> tuple[str, dict[str, int]]:
    """Build a summary table of sticky instances grouped by cloud and compute.

    Returns:
        Tuple of (markdown_string, cloud_compute_counts_for_json).
    """
    sticky = _get_sticky_instances(instances, instance_lifetime, internal_ids)
    total_sticky = len(sticky)

    counter: dict[tuple[str, str], int] = {}
    for inst in sticky:
        key = (inst["cloud"], inst["compute"])
        counter[key] = counter.get(key, 0) + 1

    counts_for_json = {f"{cloud}/{compute}": count for (cloud, compute), count in counter.items()}

    if total_sticky == 0:
        return "", counts_for_json

    sorted_pairs = sorted(
        counter.items(),
        key=lambda x: (x[0][0], x[0][1]),
    )

    prev = previous_sticky_cloud_compute or {}

    lines = []
    lines.append("### By Cloud and Compute Platform")
    lines.append("")
    lines.append("| Cloud | Compute | Instances | Percentage | Change |")
    lines.append("|-------|---------|-----------|------------|--------|")
    for (cloud, compute), count in sorted_pairs:
        pct = _format_pct(count, total_sticky)
        pair_key = f"{cloud}/{compute}"
        prev_count = prev.get(pair_key, 0)
        if prev_count == 0 and count > 0:
            change = f"+{count} (new)"
        elif prev_count == count:
            change = "0"
        else:
            delta = count - prev_count
            sign = "+" if delta > 0 else ""
            change = f"{sign}{delta}"
        lines.append(f"| {cloud} | {compute} | {count} | {pct} | {change} |")
    lines.append("")

    return "\n".join(lines), counts_for_json


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

        result.append(
            {
                "registry_id": inst["registry_id"],
                "cloud": inst["cloud"],
                "compute": inst["compute"],
                "auth": inst["auth"],
                "first_seen": first,
                "latest_seen": latest,
                "age_days": age_days,
                "events": inst["events"],
            }
        )

    result.sort(key=lambda x: x["age_days"], reverse=True)
    return result


# Internal-install classification began shipping with this release (issue #1216).
# Deployments running an earlier version emit no classification fields, so any
# internal/workshop installs before this release are only visible via the manual
# known-internal-instances allowlist, not the telemetry signal.
INTERNAL_CLASSIFICATION_SINCE_RELEASE = "1.24.5"

# The internal deployment types we expect (matches InternalDeploymentType in
# registry/core/config.py). "none" means not an internal deployment.
INTERNAL_DEPLOYMENT_TYPES = ("dev", "workshop", "other")


def _compute_internal_installs(
    instances: list[dict],
    rows: list[dict[str, str]],
) -> dict:
    """Compute internal-install metrics from the telemetry classification fields.

    An instance counts as internal when its telemetry reports
    internal_only_deployment=true (issue #1216). Returns:
      total: number of distinct internal installs in the window
      by_type: count of internal installs per internal_deployment_type
      timeseries: month -> {type -> count of installs first seen that month},
        capturing all internal types over time
      since_release: the release classification capture began (for the note)
    """
    internal = [inst for inst in instances if inst.get("internal_only_deployment")]

    by_type: dict[str, int] = {t: 0 for t in INTERNAL_DEPLOYMENT_TYPES}
    for inst in internal:
        dtype = (inst.get("internal_deployment_type") or "").strip().lower()
        if dtype not in by_type:
            # Defensive: an unexpected/blank type still counts toward the total
            # under an "other" bucket so we never silently drop an internal install.
            by_type.setdefault("other", 0)
            dtype = "other"
        by_type[dtype] += 1

    # Monthly timeseries keyed by first-seen month (YYYY-MM). Each month maps to
    # a per-type count so the chart can stack workshop/dev/other over time.
    timeseries: dict[str, dict[str, int]] = {}
    for inst in internal:
        first_seen = inst.get("first_seen", "")
        if len(first_seen) < 7:
            continue
        month = first_seen[:7]
        dtype = (inst.get("internal_deployment_type") or "other").strip().lower()
        if dtype not in INTERNAL_DEPLOYMENT_TYPES:
            dtype = "other"
        bucket = timeseries.setdefault(month, {t: 0 for t in INTERNAL_DEPLOYMENT_TYPES})
        bucket[dtype] += 1

    return {
        "total": len(internal),
        "by_type": by_type,
        "timeseries": dict(sorted(timeseries.items())),
        "since_release": INTERNAL_CLASSIFICATION_SINCE_RELEASE,
    }


def _build_internal_installs_md(
    internal_installs: dict,
) -> str:
    """Build the 'Internal Installs' report section (issue #1216/#1217).

    Renders the total count, a per-type breakdown, a monthly timeseries table
    across all internal types, and the capture-start note for the release.
    """
    total = internal_installs.get("total", 0)
    by_type = internal_installs.get("by_type", {})
    timeseries = internal_installs.get("timeseries", {})
    since = internal_installs.get("since_release", INTERNAL_CLASSIFICATION_SINCE_RELEASE)

    lines: list[str] = []
    lines.append("## Internal Installs")
    lines.append("")
    lines.append(
        f"> **Note:** Internal/workshop install classification is captured from "
        f"telemetry starting with release **{since}** onwards. Deployments running "
        f"earlier versions report no classification, so internal installs before "
        f"**{since}** are undercounted here (they appear only if listed in the manual "
        f"known-internal-instances allowlist)."
    )
    lines.append("")

    # Summary sentence + per-type breakdown
    type_summary = ", ".join(
        f"{by_type.get(t, 0)} {t}" for t in INTERNAL_DEPLOYMENT_TYPES
    )
    lines.append(
        f"We had **{total}** internal install(s) in this window ({type_summary})."
    )
    lines.append("")
    lines.append("| Internal Deployment Type | Installs |")
    lines.append("|--------------------------|----------|")
    for t in INTERNAL_DEPLOYMENT_TYPES:
        lines.append(f"| {t} | {by_type.get(t, 0)} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    # Monthly timeseries across all internal types
    lines.append("### Internal Installs Over Time")
    lines.append("")
    if not timeseries:
        lines.append(
            f"_No internal installs recorded yet. Data begins with release {since}._"
        )
    else:
        header_types = " | ".join(t.capitalize() for t in INTERNAL_DEPLOYMENT_TYPES)
        divider = " | ".join("---" for _ in INTERNAL_DEPLOYMENT_TYPES)
        lines.append(f"| Month | {header_types} | Total |")
        lines.append(f"|-------|{divider}|-------|")
        for month, counts in timeseries.items():
            row_counts = " | ".join(str(counts.get(t, 0)) for t in INTERNAL_DEPLOYMENT_TYPES)
            month_total = sum(counts.get(t, 0) for t in INTERNAL_DEPLOYMENT_TYPES)
            lines.append(f"| {month} | {row_counts} | {month_total} |")
    lines.append("")

    return "\n".join(lines)


def _build_exec_summary_md(
    current_metrics: dict,
    previous_metrics: dict | None,
    current_cloud_installs: dict[str, int],
    previous_cloud_installs: dict[str, int] | None,
    cloud_last_event: dict[str, dict[str, str]] | None = None,
    stickiness: dict | None = None,
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

    # One-day-wonder line: customers who tried it once and never came back.
    # Always render this when stickiness data is available, regardless of
    # whether there is a previous report to compare against.
    if stickiness and stickiness.get("total_non_internal", 0) > 0:
        odw_count = stickiness.get("one_day_wonders", 0)
        odw_pct = stickiness.get("one_day_wonder_pct", 0.0)
        odw_total = stickiness.get("total_non_internal", 0)
        lines.append(
            f"**{odw_pct}% of customer instances ({odw_count} of {odw_total}) "
            f"are one-day wonders** -- their startup or heartbeat events fall "
            f"on a single calendar day and that registry_id is never seen "
            f"again. These represent operators trying out the solution once, "
            f"out of curiosity, and not coming back."
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
        lines.append(f"**New cloud providers**: {', '.join(sorted(new_clouds))}")
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
    registry_ids = {r["registry_id"] for r in rows if r.get("registry_id", "").strip()}
    identified_count = len(registry_ids)

    # Null registry_id count
    null_id_count = sum(1 for r in rows if not r.get("registry_id", "").strip())

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
        max_search = max(_safe_int(e.get("search_queries_total", "")) for e in events)

        # search_queries_total is already a lifetime cumulative counter
        # in each event, so take the max (latest value) not the sum
        total_search = max(_safe_int(e.get("search_queries_total", "")) for e in events)

        # Track search backend, embeddings provider, and backend kind
        # (most recent non-empty value across heartbeat + startup events).
        search_backend = _latest_nonempty(events, "search_backend")
        embeddings_provider = _latest_nonempty(events, "embeddings_provider")
        embeddings_backend_kind = _latest_nonempty(events, "embeddings_backend_kind")
        # cloud_detection_method added in schema v3 (issue #986). Pre-v3
        # instances produce empty values and land in the "unknown" bucket.
        cloud_detection_method = _latest_nonempty(events, "cloud_detection_method")

        first_ts = events[0].get("ts", "")[:10]
        latest_ts = events[-1].get("ts", "")[:10]

        # auth: heartbeat events don't carry the field, so picking
        # latest.get("auth") would be empty for any instance whose latest
        # event is a heartbeat -- mislabeling 80%+ of customers as "none"
        # auth provider when really we just lack the field on that event.
        # Use _latest_nonempty to walk back to the most recent startup event.
        auth = _latest_nonempty(events, "auth")

        # Internal/workshop deployment classification (schema v5+, issue #1216).
        # Pre-v5 events omit these fields, so they stay empty and the instance
        # is treated as external (unless matched by the manual allowlist).
        internal_only_raw = _latest_nonempty(events, "internal_only_deployment")
        internal_only_flag = internal_only_raw.strip().lower() == "true"
        internal_deployment_type = _latest_nonempty(events, "internal_deployment_type")

        result.append(
            {
                "registry_id": rid[:12] + "...",
                "registry_id_full": rid,
                "cloud": latest.get("cloud") or "unknown",
                "cloud_detection_method": cloud_detection_method,
                "compute": latest.get("compute") or "unknown",
                "storage": latest.get("storage") or "unknown",
                "auth": auth,
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
                "search_backend": search_backend,
                "embeddings_provider": embeddings_provider,
                "embeddings_backend_kind": embeddings_backend_kind,
                "internal_only_deployment": internal_only_flag,
                "internal_deployment_type": internal_deployment_type,
                "first_seen": first_ts,
                "latest_seen": latest_ts,
            }
        )

    result.sort(key=lambda x: x["events"], reverse=True)
    return result


def _compute_unidentified_profiles(
    rows: list[dict[str, str]],
) -> list[dict]:
    """Group unidentified events into distinct deployment profiles."""
    unidentified = [r for r in rows if not r.get("registry_id", "").strip()]

    # Group by (cloud, compute, arch, storage, auth, mode)
    # Note: heartbeat events don't carry the `auth` field, so an empty
    # auth here means "we don't know" not "no auth provider configured".
    # Use "unknown" rather than "none" so we don't mislabel.
    profiles = defaultdict(list)
    for row in unidentified:
        key = (
            row.get("cloud") or "unknown",
            row.get("compute") or "unknown",
            row.get("arch") or "unknown",
            row.get("storage") or "unknown",
            row.get("auth") or "unknown",
            row.get("mode") or "unknown",
        )
        profiles[key].append(row)

    result = []
    for key, events in profiles.items():
        cloud, compute, arch, storage, auth, mode = key

        max_servers = max(_safe_int(e.get("servers_count", "")) for e in events)
        max_agents = max(_safe_int(e.get("agents_count", "")) for e in events)
        max_skills = max(_safe_int(e.get("skills_count", "")) for e in events)
        max_search = max(_safe_int(e.get("search_queries_total", "")) for e in events)
        # search_queries_total is already a lifetime cumulative counter
        # in each event, so take the max (latest value) not the sum
        total_search = max(_safe_int(e.get("search_queries_total", "")) for e in events)

        events.sort(key=lambda r: r.get("ts", ""))
        first_ts = events[0].get("ts", "")[:10]
        latest_ts = events[-1].get("ts", "")[:10]

        result.append(
            {
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
            }
        )

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
        filtered = [r for r in rows if r.get("registry_id", "").strip() == registry_id]
    elif cloud and compute:
        filtered = [
            r
            for r in rows
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
        max_search = max(_safe_int(e.get("search_queries_total", "")) for e in events)

        result.append(
            {
                "date": date,
                "events": len(events),
                "max_servers": max_servers,
                "max_agents": max_agents,
                "max_skills": max_skills,
                "max_search_queries": max_search,
            }
        )

    return result


def _compute_version_table(
    rows: list[dict[str, str]],
) -> list[dict]:
    """Compute version adoption table with event counts and unique-instance counts."""
    total_events = len(rows)
    version_events: Counter = Counter()
    version_instances: dict[str, set[str]] = {}
    all_instances: set[str] = set()
    for row in rows:
        version = row.get("v") or "unknown"
        version_events[version] += 1
        rid = (row.get("registry_id") or "").strip()
        if rid:
            version_instances.setdefault(version, set()).add(rid)
            all_instances.add(rid)

    total_instances = len(all_instances)

    result = []
    for version, count in version_events.most_common():
        vtype = _classify_version(version)
        branch = _extract_version_branch(version) if vtype == "dev" else "--"
        instance_count = len(version_instances.get(version, set()))

        result.append(
            {
                "version": version,
                "type": "**Release**" if vtype == "release" else f"Dev ({branch})",
                "events": count,
                "percentage": _format_pct(count, total_events),
                "instances": instance_count,
                "instance_percentage": _format_pct(instance_count, total_instances),
            }
        )

    return result


def _compute_upgrade_trajectories(
    rows: list[dict[str, str]],
    internal_ids: set[str],
) -> list[dict]:
    """Compute per-instance version-upgrade trajectories.

    For each non-internal registry_id, collects the set of DISTINCT version
    strings the instance has ever reported, ordered by first-seen timestamp.
    This matches "what version numbers have we seen from this instance"
    rather than the noisy every-event sequence (HA replicas with the same
    registry_id can flap between versions, which is not an upgrade).

    Excludes:
      - internal instances (known UUIDs from known-internal-instances.md)
      - rows with no registry_id or no version
      - instances that only ever reported one distinct version (no upgrade)

    Returns a list of dicts sorted descending by version_changes:
      [{
         "registry_id": "9c02f727-51a...",
         "version_changes": 6,
         "trajectory": ["v1.0.18", "v1.0.19", "v1.0.20", "v1.0.21",
                        "v1.0.22", "v1.0.22-p1", "1.23.0"],
      }, ...]
    """
    # Track per-instance: {version: earliest_ts_seen}
    per_instance_first_seen: dict[str, dict[str, str]] = {}
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid or rid in internal_ids:
            continue
        version = (row.get("v") or "").strip()
        if not version:
            continue
        ts = (row.get("ts") or "").strip()
        if not ts:
            continue
        first_seen = per_instance_first_seen.setdefault(rid, {})
        prior = first_seen.get(version)
        if prior is None or ts < prior:
            first_seen[version] = ts

    trajectories: list[dict] = []
    for rid, first_seen_map in per_instance_first_seen.items():
        # Order distinct versions by first-seen timestamp
        ordered = sorted(first_seen_map.items(), key=lambda kv: kv[1])
        sequence = [v for v, _ts in ordered]

        # Skip instances that only ever reported one distinct version
        if len(sequence) < 2:
            continue

        trajectories.append(
            {
                "registry_id": rid[:12] + "...",
                "registry_id_full": rid,
                "version_changes": len(sequence) - 1,
                "trajectory": sequence,
            }
        )

    # Sort descending by version_changes, then by registry_id for stable output
    trajectories.sort(key=lambda t: (-t["version_changes"], t["registry_id"]))
    return trajectories


def _build_upgrade_trajectory_table(
    trajectories: list[dict],
) -> str:
    """Build the upgrade-trajectory markdown section.

    Returns an empty string if no instances have upgraded.
    """
    if not trajectories:
        return ""

    lines: list[str] = []
    lines.append("## Version Upgrade Trajectories")
    lines.append("")
    lines.append(
        f"**{len(trajectories)} non-internal customer instances** have reported "
        f"more than one version string over their lifetime, indicating at least "
        f"one upgrade. Sorted descending by number of version changes."
    )
    lines.append("")
    lines.append("| Registry ID | Version Changes | Trajectory |")
    lines.append("|-------------|----------------:|------------|")
    for t in trajectories:
        path = " -> ".join(f"`{v}`" for v in t["trajectory"])
        lines.append(f"| `{t['registry_id']}` | {t['version_changes']} | {path} |")
    lines.append("")
    return "\n".join(lines)


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

        instance_max_total[instance_key] = max(instance_max_total.get(instance_key, 0), sq_total)
        instance_max_24h[instance_key] = max(instance_max_24h.get(instance_key, 0), sq_24h)
        instance_max_1h[instance_key] = max(instance_max_1h.get(instance_key, 0), sq_1h)

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

    fed_enabled = sum(1 for r in rows if r.get("federation", "").strip().lower() == "true")
    with_gw = sum(1 for r in rows if r.get("mode") == "with-gateway")
    reg_only = sum(1 for r in rows if r.get("mode") == "registry-only")

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
            "enabled": len(
                {
                    r.get("registry_id", "").strip()
                    for r in rows
                    if r.get("event") == "heartbeat" and r.get("registry_id", "").strip()
                }
            ),
            "disabled": total - sum(1 for r in rows if r.get("event") == "heartbeat"),
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
    internal_ids: set[str] | None = None,
    previous_sticky_profiles: dict[str, int] | None = None,
    previous_sticky_cloud_compute: dict[str, int] | None = None,
    upgrade_trajectories: list[dict] | None = None,
) -> tuple[str, dict[str, int], dict[str, int], list[dict]]:
    """Build all markdown tables as a single string.

    Returns:
        Tuple of (
            markdown_content,
            sticky_profile_counts,
            sticky_cloud_compute_counts,
            embeddings_backend_rows,
        ).
    """
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
    lines.append(f"| Unique Registry Instances (identified) | {metrics['identified_instances']} |")
    lines.append(
        f"| Events with null registry_id "
        f"| {metrics['null_registry_id_count']} "
        f"({metrics['null_registry_id_pct']}) |"
    )
    lines.append(
        f"| Collection Period | {metrics['earliest_ts'][:10]} to {metrics['latest_ts'][:10]} |"
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
            rid_display = _display_id(inst["registry_id"], internal_ids or set())
            lines.append(
                f"| `{rid_display}` "
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
        rid_display = _display_id(inst["registry_id"], internal_ids or set())
        lines.append(
            f"| `{rid_display}` "
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
        lines.append(
            _md_distribution_table(
                title,
                distributions[key],
                total,
                col_name,
            )
        )

    # Version Adoption
    lines.append("## Version Adoption")
    lines.append("")
    lines.append("| Version | Type | Events | % Events | Instances | % Instances |")
    lines.append("|---------|------|--------|----------|-----------|-------------|")
    for v in versions:
        lines.append(
            f"| `{v['version']}` | {v['type']} | {v['events']} | {v['percentage']} | "
            f"{v['instances']} | {v['instance_percentage']} |"
        )
    lines.append("")

    # Version Upgrade Trajectories (instances that reported >1 distinct version)
    upgrade_md = _build_upgrade_trajectory_table(upgrade_trajectories or [])
    if upgrade_md:
        lines.append(upgrade_md)

    # Feature Adoption
    lines.append("## Feature Adoption")
    lines.append("")
    lines.append("| Feature | Enabled | Disabled | Rate |")
    lines.append("|---------|---------|----------|------|")
    for feat in features:
        lines.append(
            f"| {feat['feature']} | {feat['enabled']} | {feat['disabled']} | {feat['rate']} |"
        )
    lines.append("")

    # Embeddings Backend Breakdown (schema v2+)
    embeddings_md, embeddings_backend_rows = _build_embeddings_backend_breakdown_table(instances)
    if embeddings_md:
        lines.append(embeddings_md)

    # Cloud Detection Method Breakdown (schema v3+, issue #986)
    cloud_detection_md, cloud_detection_rows = _build_cloud_detection_method_breakdown_table(
        instances
    )
    if cloud_detection_md:
        lines.append(cloud_detection_md)

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
        f"| Total search queries (sum of per-instance lifetime counts) | {search['lifetime_sum']} |"
    )
    lines.append(f"| Average per instance | {search['lifetime_avg']} |")
    lines.append(f"| Max from single instance | {search['lifetime_max']} |")
    lines.append("")

    # Sticky Instance Breakdown
    sticky_profile_counts: dict[str, int] = {}
    sticky_cloud_compute_counts: dict[str, int] = {}
    if instance_lifetime:
        sticky_md, sticky_profile_counts = _build_sticky_breakdown_table(
            instances,
            instance_lifetime,
            internal_ids or set(),
            previous_sticky_profiles=previous_sticky_profiles,
        )
        if sticky_md:
            lines.append(sticky_md)

        cc_md, sticky_cloud_compute_counts = _build_sticky_cloud_compute_table(
            instances,
            instance_lifetime,
            internal_ids or set(),
            previous_sticky_cloud_compute=previous_sticky_cloud_compute,
        )
        if cc_md:
            lines.append(cc_md)

    # Most Active Instances (search-volume + feature-volume blend)
    most_active_md = _build_most_active_table(instances, internal_ids or set())
    lines.append(most_active_md)

    # Largest Catalogs (servers + agents + skills, regardless of search)
    largest_catalogs_md = _build_largest_catalogs_table(instances, internal_ids or set())
    lines.append(largest_catalogs_md)

    # Most Engaged Operators (by upgrade-chain length)
    most_engaged_md = _build_most_engaged_operators_table(
        upgrade_trajectories or [],
        instance_lifetime or [],
        instances,
        internal_ids or set(),
    )
    if most_engaged_md:
        lines.append(most_engaged_md)

    # Instance Timelines
    lines.append("## Instance Timelines")
    lines.append("")

    # Identified instances
    for inst in instances:
        timeline = _compute_instance_timeline(
            rows,
            registry_id=inst["registry_id_full"],
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
                rows,
                cloud=prof["cloud"],
                compute=prof["compute"],
            )
            if not timeline:
                continue
            label = f"{prof['cloud']}/{prof['compute']}/{prof['auth']}"
            lines.append(f"### Unidentified: {label}")
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

    return (
        "\n".join(lines),
        sticky_profile_counts,
        sticky_cloud_compute_counts,
        embeddings_backend_rows,
    )


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
            "If not provided, auto-detects the most recent one in search-dir."
        ),
    )
    parser.add_argument(
        "--search-dir",
        default=None,
        help=(
            "Directory to search for previous metrics files. "
            "Defaults to the parent of output-dir (useful when output-dir "
            "is a dated subfolder like reports/2026-04-19/)."
        ),
    )
    parser.add_argument(
        "--internal-instances",
        default=None,
        help=(
            "Path to known-internal-instances.md file listing internal "
            "registry instance IDs. If provided, internal instances are "
            "labeled in tables and excluded from stickiness metrics."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        raise SystemExit(1)

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    internal_ids = _parse_internal_instances(args.internal_instances)

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
        search_dir = args.search_dir or os.path.dirname(os.path.abspath(args.output_dir))
        prev_metrics_path = _find_previous_metrics(search_dir, date_str)

    previous_metrics = None
    previous_cloud_installs = None
    if prev_metrics_path:
        previous_metrics = _load_previous_metrics(prev_metrics_path)
        if previous_metrics:
            previous_cloud_installs = previous_metrics.get("per_cloud_unique_installs", None)

    # Compute stickiness BEFORE the exec-summary build so the one-day-wonder
    # line can be emitted in the lead paragraph.
    stickiness = _compute_stickiness(instance_lifetime, internal_ids)

    # Build executive summary
    exec_summary_md = _build_exec_summary_md(
        metrics,
        previous_metrics,
        cloud_installs,
        previous_cloud_installs,
        cloud_last_event=cloud_last_event,
        stickiness=stickiness,
    )

    upgrade_trajectories = _compute_upgrade_trajectories(rows, internal_ids)

    previous_sticky_profiles = None
    previous_sticky_cloud_compute = None
    if previous_metrics:
        previous_sticky_profiles = previous_metrics.get("sticky_profiles", None)
        previous_sticky_cloud_compute = previous_metrics.get("sticky_cloud_compute", None)

    (
        md_content,
        sticky_profile_counts,
        sticky_cc_counts,
        embeddings_backend_breakdown,
    ) = _build_markdown_tables(
        metrics,
        distributions,
        instances,
        unidentified,
        versions,
        search,
        features,
        rows,
        exec_summary_md=exec_summary_md,
        instance_lifetime=instance_lifetime,
        internal_ids=internal_ids,
        previous_sticky_profiles=previous_sticky_profiles,
        previous_sticky_cloud_compute=previous_sticky_cloud_compute,
        upgrade_trajectories=upgrade_trajectories,
    )

    # Internal installs section (issue #1216/#1217): count + per-type breakdown +
    # monthly timeseries, with the capture-start-at-release note. Appended after
    # the main tables so it has its own clearly labeled section.
    internal_installs = _compute_internal_installs(instances, rows)
    md_content = md_content + "\n\n" + _build_internal_installs_md(internal_installs)

    # Build JSON with all computed data
    metrics_json = {
        "report_date": date_str,
        "key_metrics": metrics,
        "per_cloud_unique_installs": cloud_installs,
        "instance_lifetime": instance_lifetime,
        "internal_installs": internal_installs,
        "stickiness": stickiness,
        "sticky_profiles": sticky_profile_counts,
        "sticky_cloud_compute": sticky_cc_counts,
        "embeddings_backend_breakdown": embeddings_backend_breakdown,
        "internal_instance_ids": sorted(internal_ids),
        "distributions": {k: dict(v.most_common()) for k, v in distributions.items()},
        "identified_instances": instances,
        "unidentified_profiles": unidentified,
        "version_adoption": versions,
        "upgrade_trajectories": upgrade_trajectories,
        "search_stats": search,
        "feature_adoption": features,
    }

    _write_outputs(md_content, metrics_json, args.output_dir, date_str)
    logger.info(
        f"Analysis complete: {metrics['total_events']} events, "
        f"{metrics['identified_instances']} identified instances"
    )


if __name__ == "__main__":
    main()
