"""Analyze registry instance liveness from telemetry CSV.

Heartbeats are sent once per day by default (configurable via
MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES, default 1440). This script uses
that cadence to classify each non-internal registry instance into one of
three tiers based on recent activity:

- CONFIRMED ALIVE  (leading indicator)  - >= 5 heartbeats in the last 7 days
- STRONGER ALIVE   (trailing indicator) - >= 10 heartbeats in the last 14 days
- LIKELY ALIVE     (loose signal)       - any event (startup or heartbeat)
                                          in the last 7 days

Dormant instances (no event in the last 14 days) are reported separately.

The script reads the telemetry CSV plus the metrics JSON produced by
analyze_telemetry.py (for per-instance cloud/compute/auth metadata), writes
a markdown section ready to paste into the report, and a JSON file with
the raw counts for delta tracking.
"""

import argparse
import csv
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from internal_instances import load_internal_instance_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


CONFIRMED_WINDOW_DAYS: int = 7
CONFIRMED_MIN_HEARTBEATS: int = 5
STRONGER_WINDOW_DAYS: int = 14
STRONGER_MIN_HEARTBEATS: int = 10
LIKELY_WINDOW_DAYS: int = 7
DORMANT_WINDOW_DAYS: int = 14


def _parse_ts(
    ts_str: str,
) -> datetime | None:
    """Parse an ISO-8601 timestamp string. Returns None on failure."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_events(
    csv_path: str,
) -> tuple[dict[str, list[datetime]], dict[str, list[datetime]], datetime]:
    """Read the CSV and return per-instance event timestamps.

    Returns:
        (heartbeats_by_id, all_events_by_id, latest_ts)
        - heartbeats_by_id: maps full registry_id -> list of heartbeat timestamps
        - all_events_by_id: maps full registry_id -> list of all event timestamps
        - latest_ts: the most recent timestamp seen in the CSV
    """
    heartbeats: dict[str, list[datetime]] = defaultdict(list)
    all_events: dict[str, list[datetime]] = defaultdict(list)
    latest_ts: datetime | None = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("registry_id") or "").strip()
            if not rid:
                continue
            ts = _parse_ts((row.get("ts") or "").strip())
            if ts is None:
                continue
            all_events[rid].append(ts)
            event_type = (row.get("event") or "").strip()
            if event_type == "heartbeat":
                heartbeats[rid].append(ts)
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
    if latest_ts is None:
        raise ValueError(f"No valid timestamps found in {csv_path}")
    logger.info(
        f"Loaded {sum(len(v) for v in all_events.values())} events for "
        f"{len(all_events)} instances; latest ts={latest_ts}"
    )
    return dict(heartbeats), dict(all_events), latest_ts


def _count_in_window(
    timestamps: list[datetime],
    cutoff: datetime,
) -> int:
    """Count timestamps at or after the cutoff."""
    return sum(1 for t in timestamps if t >= cutoff)


def _load_instance_info(
    metrics_json_path: str,
) -> dict[str, dict]:
    """Load per-instance metadata from metrics JSON, keyed by full registry_id."""
    with open(metrics_json_path) as f:
        metrics = json.load(f)
    info = {}
    for inst in metrics.get("identified_instances", []):
        full_id = inst.get("registry_id_full")
        if full_id:
            info[full_id] = inst
    logger.info(f"Loaded metadata for {len(info)} instances from {metrics_json_path}")
    return info


def _classify_instances(
    heartbeats: dict[str, list[datetime]],
    all_events: dict[str, list[datetime]],
    latest_ts: datetime,
    internal_ids: set[str],
) -> dict:
    """Classify each customer instance into liveness tiers.

    Returns a dict with keys: confirmed, stronger, likely, dormant, silent,
    each mapping to a list of registry_ids.
    """
    cutoff_confirmed = latest_ts - timedelta(days=CONFIRMED_WINDOW_DAYS)
    cutoff_stronger = latest_ts - timedelta(days=STRONGER_WINDOW_DAYS)
    cutoff_likely = latest_ts - timedelta(days=LIKELY_WINDOW_DAYS)
    cutoff_dormant = latest_ts - timedelta(days=DORMANT_WINDOW_DAYS)

    customer_ids = [rid for rid in all_events if rid not in internal_ids]

    confirmed: list[str] = []
    stronger: list[str] = []
    likely: list[str] = []
    dormant: list[str] = []

    for rid in customer_ids:
        hbs = heartbeats.get(rid, [])
        evts = all_events.get(rid, [])
        hb_count_7d = _count_in_window(hbs, cutoff_confirmed)
        hb_count_14d = _count_in_window(hbs, cutoff_stronger)
        evts_7d = _count_in_window(evts, cutoff_likely)
        last_event = max(evts)

        if hb_count_7d >= CONFIRMED_MIN_HEARTBEATS:
            confirmed.append(rid)
        if hb_count_14d >= STRONGER_MIN_HEARTBEATS:
            stronger.append(rid)
        if evts_7d > 0:
            likely.append(rid)
        if last_event < cutoff_dormant:
            dormant.append(rid)

    silent_but_recent = sorted(set(likely) - set(confirmed))

    return {
        "customer_total": len(customer_ids),
        "confirmed": confirmed,
        "stronger": stronger,
        "likely": likely,
        "dormant": dormant,
        "silent_but_recent": silent_but_recent,
    }


def _short_id(
    full_id: str,
) -> str:
    """Return the truncated registry_id format used in the report."""
    return full_id[:11] + "..."


def _breakdown_by(
    ids: list[str],
    info_by_id: dict[str, dict],
    key: str,
) -> list[tuple[str, int]]:
    """Count instances grouped by a single metadata key."""
    counter: Counter = Counter()
    for rid in ids:
        info = info_by_id.get(rid, {})
        counter[info.get(key, "unknown")] += 1
    return counter.most_common()


def _breakdown_by_pair(
    ids: list[str],
    info_by_id: dict[str, dict],
    key_a: str,
    key_b: str,
) -> list[tuple[tuple[str, str], int]]:
    """Count instances grouped by two metadata keys."""
    counter: Counter = Counter()
    for rid in ids:
        info = info_by_id.get(rid, {})
        counter[(info.get(key_a, "unknown"), info.get(key_b, "unknown"))] += 1
    return counter.most_common()


def _format_confirmed_table(
    confirmed: list[str],
    info_by_id: dict[str, dict],
    heartbeats: dict[str, list[datetime]],
    all_events: dict[str, list[datetime]],
    latest_ts: datetime,
) -> str:
    """Format a markdown table of confirmed-alive instances.

    Mirrors the style of the Registry Instance Lifetime table: one row per
    instance, sorted by heartbeat count in the last 7 days (descending).
    """
    cutoff_confirmed = latest_ts - timedelta(days=CONFIRMED_WINDOW_DAYS)
    cutoff_stronger = latest_ts - timedelta(days=STRONGER_WINDOW_DAYS)

    rows = []
    for rid in confirmed:
        info = info_by_id.get(rid, {})
        hbs = heartbeats.get(rid, [])
        evts = all_events.get(rid, [])
        hb_7d = _count_in_window(hbs, cutoff_confirmed)
        hb_14d = _count_in_window(hbs, cutoff_stronger)
        last_event = max(evts) if evts else latest_ts
        days_since_last = (latest_ts - last_event).days
        rows.append(
            {
                "rid": rid,
                "cloud": info.get("cloud", "?"),
                "compute": info.get("compute", "?"),
                "auth": info.get("auth", "?"),
                "version": info.get("version", "?"),
                "hb_7d": hb_7d,
                "hb_14d": hb_14d,
                "days_since_last": days_since_last,
            }
        )
    rows.sort(key=lambda r: (-r["hb_7d"], -r["hb_14d"], r["days_since_last"]))

    lines = [
        "| Registry ID | Cloud | Compute | Auth | Version | Heartbeats (7d) | Heartbeats (14d) | Days Since Last Event |",
        "|-------------|-------|---------|------|---------|-----------------|------------------|-----------------------|",
    ]
    for r in rows:
        lines.append(
            f"| `{_short_id(r['rid'])}` | {r['cloud']} | {r['compute']} | {r['auth']} | "
            f"`{r['version']}` | {r['hb_7d']} | {r['hb_14d']} | {r['days_since_last']} |"
        )
    return "\n".join(lines)


def _format_breakdown_table(
    title: str,
    header: tuple[str, ...],
    breakdown: list,
    total: int,
    heading_level: int = 3,
) -> str:
    """Format a single-key or two-key breakdown as a markdown table."""
    prefix = "#" * heading_level
    lines = [f"{prefix} {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("-" * (len(h) + 2) for h in header) + "|")
    for key, count in breakdown:
        pct = f"{count / total * 100:.0f}%" if total else "0%"
        if isinstance(key, tuple):
            key_cols = " | ".join(key)
        else:
            key_cols = key
        lines.append(f"| {key_cols} | {count} | {pct} |")
    return "\n".join(lines)


def _build_markdown_section(
    classified: dict,
    info_by_id: dict[str, dict],
    heartbeats: dict[str, list[datetime]],
    all_events: dict[str, list[datetime]],
    latest_ts: datetime,
    previous: dict | None,
) -> str:
    """Build the full Liveness markdown section."""
    total = classified["customer_total"]
    confirmed = classified["confirmed"]
    stronger = classified["stronger"]
    likely = classified["likely"]
    dormant = classified["dormant"]
    silent = classified["silent_but_recent"]

    def _delta(
        current: int,
        previous_key: str,
    ) -> str:
        if not previous:
            return "n/a"
        prev = previous.get("counts", {}).get(previous_key)
        if prev is None:
            return "n/a"
        diff = current - prev
        sign = "+" if diff > 0 else ""
        return f"{prev} -> {current} ({sign}{diff})"

    def _pct(
        n: int,
    ) -> str:
        return f"{n / total * 100:.0f}%" if total else "0%"

    lines = []
    lines.append("## Liveness (Currently Active Instances)")
    lines.append("")
    lines.append(
        f"Heartbeats are sent once per 24 hours (default interval). "
        f"Classifying {total} customer instances into tiers based on recent telemetry:"
    )
    lines.append("")
    lines.append("| Tier | Definition | Instances | % of Customer Fleet | vs Previous |")
    lines.append("|------|------------|-----------|---------------------|-------------|")
    lines.append(
        f"| **Confirmed Alive** (leading) | >= {CONFIRMED_MIN_HEARTBEATS} heartbeats in last "
        f"{CONFIRMED_WINDOW_DAYS} days | {len(confirmed)} | {_pct(len(confirmed))} | "
        f"{_delta(len(confirmed), 'confirmed')} |"
    )
    lines.append(
        f"| **Stronger Alive** (trailing) | >= {STRONGER_MIN_HEARTBEATS} heartbeats in last "
        f"{STRONGER_WINDOW_DAYS} days | {len(stronger)} | {_pct(len(stronger))} | "
        f"{_delta(len(stronger), 'stronger')} |"
    )
    lines.append(
        f"| Likely Alive | Any event in last {LIKELY_WINDOW_DAYS} days | {len(likely)} | "
        f"{_pct(len(likely))} | {_delta(len(likely), 'likely')} |"
    )
    lines.append(
        f"| Silent-but-recent | Event in last {LIKELY_WINDOW_DAYS} days but < "
        f"{CONFIRMED_MIN_HEARTBEATS} heartbeats | {len(silent)} | {_pct(len(silent))} | "
        f"{_delta(len(silent), 'silent_but_recent')} |"
    )
    lines.append(
        f"| Dormant | No event in last {DORMANT_WINDOW_DAYS} days | {len(dormant)} | "
        f"{_pct(len(dormant))} | {_delta(len(dormant), 'dormant')} |"
    )
    lines.append("")
    lines.append(
        f"> **Interpretation:** Confirmed-Alive is the leading signal for revenue-countable "
        f"deployments -- a registry that has phoned home every day for a week. Stronger-Alive "
        f"requires the same behavior over two weeks, a more durable trailing signal. Silent-but-"
        f"recent instances sent a startup event or a partial heartbeat stream in the last "
        f"{LIKELY_WINDOW_DAYS} days but have not yet hit the daily-heartbeat bar -- these are "
        f"either very new installs or instances with heartbeats disabled."
    )
    lines.append("")

    # Confirmed-alive detail
    if confirmed:
        lines.append("### Confirmed-Alive Instances")
        lines.append("")
        lines.append(
            "One row per confirmed-alive instance, sorted by heartbeats in the last 7 days."
        )
        lines.append("")
        lines.append(
            _format_confirmed_table(confirmed, info_by_id, heartbeats, all_events, latest_ts)
        )
        lines.append("")

    # Breakdowns for confirmed-alive
    if confirmed:
        lines.append("### Confirmed-Alive Breakdown")
        lines.append("")
        lines.append(
            _format_breakdown_table(
                "By Cloud",
                ("Cloud", "Instances", "% of Confirmed"),
                _breakdown_by(confirmed, info_by_id, "cloud"),
                len(confirmed),
                heading_level=4,
            )
        )
        lines.append("")
        lines.append(
            _format_breakdown_table(
                "By Cloud + Compute",
                ("Cloud", "Compute", "Instances", "% of Confirmed"),
                _breakdown_by_pair(confirmed, info_by_id, "cloud", "compute"),
                len(confirmed),
                heading_level=4,
            )
        )
        lines.append("")
        lines.append(
            _format_breakdown_table(
                "By Auth Provider",
                ("Auth", "Instances", "% of Confirmed"),
                _breakdown_by(confirmed, info_by_id, "auth"),
                len(confirmed),
                heading_level=4,
            )
        )
        lines.append("")

    return "\n".join(lines)


def _build_json_output(
    classified: dict,
    latest_ts: datetime,
) -> dict:
    """Build the JSON artifact for delta tracking."""
    return {
        "latest_ts": latest_ts.isoformat(),
        "thresholds": {
            "confirmed_window_days": CONFIRMED_WINDOW_DAYS,
            "confirmed_min_heartbeats": CONFIRMED_MIN_HEARTBEATS,
            "stronger_window_days": STRONGER_WINDOW_DAYS,
            "stronger_min_heartbeats": STRONGER_MIN_HEARTBEATS,
            "likely_window_days": LIKELY_WINDOW_DAYS,
            "dormant_window_days": DORMANT_WINDOW_DAYS,
        },
        "counts": {
            "customer_total": classified["customer_total"],
            "confirmed": len(classified["confirmed"]),
            "stronger": len(classified["stronger"]),
            "likely": len(classified["likely"]),
            "silent_but_recent": len(classified["silent_but_recent"]),
            "dormant": len(classified["dormant"]),
        },
        "confirmed_instance_ids": sorted(classified["confirmed"]),
        "stronger_instance_ids": sorted(classified["stronger"]),
    }


def _find_previous_liveness(
    search_dir: str,
    current_output: str,
) -> dict | None:
    """Find the most recent previous liveness JSON file in search_dir."""
    if not search_dir or not os.path.isdir(search_dir):
        return None
    candidates = []
    for root, _dirs, files in os.walk(search_dir):
        for name in files:
            if name.startswith("liveness-") and name.endswith(".json"):
                path = os.path.join(root, name)
                if os.path.abspath(path) == os.path.abspath(current_output):
                    continue
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort()
    previous_path = candidates[-1]
    with open(previous_path) as f:
        previous = json.load(f)
    logger.info(f"Loaded previous liveness metrics from {previous_path}")
    return previous


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify registry instances by liveness (heartbeat-based).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python3 analyze_liveness.py \\
        --csv $DATE_DIR/registry_metrics.csv \\
        --metrics-json $DATE_DIR/metrics-YYYY-MM-DD.json \\
        --output-dir $DATE_DIR \\
        --search-dir OUTPUT_DIR \\
        --date YYYY-MM-DD \\
        --internal-instances .claude/skills/usage-report/known-internal-instances.md
""",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to registry_metrics.csv",
    )
    parser.add_argument(
        "--metrics-json",
        required=True,
        help="Path to metrics-YYYY-MM-DD.json (for per-instance cloud/compute metadata)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write liveness-YYYY-MM-DD.md and liveness-YYYY-MM-DD.json",
    )
    parser.add_argument(
        "--search-dir",
        default=None,
        help="Directory to search for previous liveness JSON files (default: parent of output-dir)",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Report date (YYYY-MM-DD), used in output filenames",
    )
    parser.add_argument(
        "--internal-instances",
        default=None,
        help="Path to known-internal-instances.md",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.csv, newline="") as f:
        internal_ids = load_internal_instance_ids(args.internal_instances, csv.DictReader(f))
    heartbeats, all_events, latest_ts = _load_events(args.csv)
    info_by_id = _load_instance_info(args.metrics_json)

    classified = _classify_instances(heartbeats, all_events, latest_ts, internal_ids)

    md_output_path = os.path.join(args.output_dir, f"liveness-{args.date}.md")
    json_output_path = os.path.join(args.output_dir, f"liveness-{args.date}.json")

    search_dir = args.search_dir or os.path.dirname(os.path.abspath(args.output_dir))
    previous = _find_previous_liveness(search_dir, json_output_path)

    md_section = _build_markdown_section(
        classified, info_by_id, heartbeats, all_events, latest_ts, previous
    )
    json_data = _build_json_output(classified, latest_ts)

    with open(md_output_path, "w") as f:
        f.write(md_section + "\n")
    logger.info(f"Liveness markdown written to {md_output_path}")

    with open(json_output_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"Liveness JSON written to {json_output_path}")

    counts = json_data["counts"]
    logger.info(
        f"Liveness summary: confirmed={counts['confirmed']}, "
        f"stronger={counts['stronger']}, likely={counts['likely']}, "
        f"silent_but_recent={counts['silent_but_recent']}, dormant={counts['dormant']} "
        f"(customer_total={counts['customer_total']})"
    )


if __name__ == "__main__":
    main()
