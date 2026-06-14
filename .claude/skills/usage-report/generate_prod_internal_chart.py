"""Generate a community-vs-internal deployment timeseries chart + JSON summary.

Reads ALL CSV files in a given directory (and dated subdirectories), classifies
each registry instance as a COMMUNITY or INTERNAL deployment based on the version
string it reports, and produces:

  1. A PNG with two panels (cumulative + daily unique installs, community vs
     internal), same shape as generate_compute_timeseries_chart.py.
  2. A JSON summary the renderer reads to populate the
     "Community vs Internal Deployments" report section.

Classification rule (per SKILL.md step 5h):
  - community: a clean release tag matching ^v?MAJOR.MINOR.PATCH(-pN)?$
    (e.g. 1.0.0, 1.24.4, v1.0.20, v1.0.22-p1).
  - internal: anything else (git-describe builds like 1.24.1-25-g5b4b2a30-main,
    bare commit hashes, dev, sha-..., branch-suffixed builds). Instances in the
    known-internal allowlist are ALWAYS internal regardless of version.
  - unknown: an empty version string.

Each instance is attributed to the LATEST version it reported (by ts).
"""

import argparse
import csv
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import sys as _sys  # noqa: E402

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_style import apply_tufte_style, tufte_axes  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

CHART_TITLE: str = "AI Registry -- Community vs Internal Deployments"
FIGURE_WIDTH: int = 14
FIGURE_HEIGHT: int = 7
LINE_PALETTE: str = "Set2"

# A clean release tag: optional leading v, MAJOR.MINOR.PATCH, optional -pN suffix.
# Anything else (git-describe, commit hashes, dev, branch-suffixed) is internal.
_RELEASE_RE = re.compile(r"^v?\d+\.\d+\.\d+(-p\d+)?$")

CLASS_COMMUNITY: str = "community"
CLASS_INTERNAL: str = "internal"
CLASS_UNKNOWN: str = "unknown"


def _load_internal_allowlist(
    path: str | None,
) -> set[str]:
    """Read known-internal registry instance IDs from the allowlist markdown.

    The file lists instance IDs in a markdown table with backtick-quoted IDs
    (e.g. `| \\`<uuid>\\` | Internal |`). We extract every backtick-quoted token
    that looks like an instance id. Returns an empty set if no path/file.
    """
    if not path or not os.path.isfile(path):
        return set()
    ids: set[str] = set()
    with open(path) as f:
        content = f.read()
    for match in re.findall(r"`([^`]+)`", content):
        token = match.strip()
        if token:
            ids.add(token)
    logger.info(f"Loaded {len(ids)} known-internal instance IDs from {path}")
    return ids


def _classify_version(
    version: str,
) -> str:
    """Classify a version string as community / internal / unknown."""
    v = (version or "").strip()
    if not v:
        return CLASS_UNKNOWN
    if _RELEASE_RE.match(v):
        return CLASS_COMMUNITY
    return CLASS_INTERNAL


def _find_csv_files(
    directory: str,
) -> list[str]:
    """Find all CSV files in the directory and dated subdirectories."""
    csv_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".csv"):
            csv_files.append(filepath)
        elif os.path.isdir(filepath):
            for subfile in os.listdir(filepath):
                if subfile.endswith(".csv"):
                    csv_files.append(os.path.join(filepath, subfile))
    csv_files.sort()
    logger.info(f"Found {len(csv_files)} CSV files in {directory}")
    return csv_files


def _read_all_csvs(
    csv_files: list[str],
) -> list[dict[str, str]]:
    """Read and combine all CSV files into a single list of rows."""
    all_rows = []
    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        logger.info(f"Read {len(rows)} rows from {csv_path}")
        all_rows.extend(rows)
    logger.info(f"Total rows across all CSVs: {len(all_rows)}")
    return all_rows


def _extract_date(
    ts: str,
) -> str | None:
    """Extract YYYY-MM-DD date from an ISO timestamp string."""
    if not ts or len(ts) < 10:
        return None
    return ts[:10]


def _latest_version_per_instance(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    """Map each registry_id to the version it reported in its LATEST event."""
    latest_ts: dict[str, str] = {}
    latest_version: dict[str, str] = {}
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid or rid.lower() == "null":
            continue
        ts = (row.get("ts") or "").strip()
        if rid not in latest_ts or ts > latest_ts[rid]:
            latest_ts[rid] = ts
            latest_version[rid] = (row.get("v") or "").strip()
    return latest_version


def _classify_instance(
    rid: str,
    version: str,
    internal_ids: set[str],
) -> str:
    """Classify one instance: allowlist forces internal, else by version."""
    if rid in internal_ids:
        return CLASS_INTERNAL
    return _classify_version(version)


def _compute_summary(
    rows: list[dict[str, str]],
    internal_ids: set[str],
    yesterday: str | None,
) -> dict:
    """Build the JSON summary: yesterday, cumulative, per_version, timeseries.

    - cumulative: all-time unique instances per class (by latest version).
    - yesterday: per-class unique instances active on the `yesterday` date.
    - per_version: cumulative unique instances per version within each class,
      sorted descending by count.
    - timeseries: per-day community/internal counts (daily active uniques).
    """
    latest_version = _latest_version_per_instance(rows)
    instance_class: dict[str, str] = {
        rid: _classify_instance(rid, ver, internal_ids) for rid, ver in latest_version.items()
    }

    # Cumulative per-class counts.
    cumulative = {CLASS_COMMUNITY: 0, CLASS_INTERNAL: 0, CLASS_UNKNOWN: 0}
    for cls in instance_class.values():
        cumulative[cls] += 1
    cumulative["total"] = sum(cumulative.values())

    # Per-version unique-instance counts within each class.
    per_version_counts: dict[str, dict[str, int]] = {
        CLASS_COMMUNITY: defaultdict(int),
        CLASS_INTERNAL: defaultdict(int),
    }
    for rid, ver in latest_version.items():
        cls = instance_class[rid]
        if cls in per_version_counts:
            per_version_counts[cls][ver or "(none)"] += 1
    per_version = {
        cls: [
            {"version": v, "instances": n}
            for v, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        for cls, counts in per_version_counts.items()
    }

    # Daily active uniques per class, and the `yesterday` headline.
    daily_ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid or rid.lower() == "null":
            continue
        date = _extract_date(row.get("ts") or "")
        if not date:
            continue
        daily_ids[date][instance_class.get(rid, CLASS_UNKNOWN)].add(rid)

    timeseries = []
    for date in sorted(daily_ids.keys()):
        timeseries.append(
            {
                "date": date,
                CLASS_COMMUNITY: len(daily_ids[date].get(CLASS_COMMUNITY, set())),
                CLASS_INTERNAL: len(daily_ids[date].get(CLASS_INTERNAL, set())),
            }
        )

    yday = {CLASS_COMMUNITY: 0, CLASS_INTERNAL: 0, CLASS_UNKNOWN: 0}
    if yesterday and yesterday in daily_ids:
        for cls in (CLASS_COMMUNITY, CLASS_INTERNAL, CLASS_UNKNOWN):
            yday[cls] = len(daily_ids[yesterday].get(cls, set()))
    yday["total"] = sum(yday.values())

    return {
        "yesterday_date": yesterday,
        "yesterday": yday,
        "cumulative": cumulative,
        "per_version": per_version,
        "timeseries": timeseries,
    }


def _compute_cumulative_series(
    rows: list[dict[str, str]],
    instance_class: dict[str, str],
) -> dict[str, list[tuple[str, int]]]:
    """Cumulative unique installs per class per date."""
    class_events: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid or rid.lower() == "null":
            continue
        date = _extract_date(row.get("ts") or "")
        if not date:
            continue
        cls = instance_class.get(rid, CLASS_UNKNOWN)
        class_events[cls].append((date, rid))

    result: dict[str, list[tuple[str, int]]] = {}
    for cls, events in class_events.items():
        events.sort(key=lambda x: x[0])
        seen_ids: set[str] = set()
        daily_cumulative: dict[str, int] = {}
        for date, rid in events:
            seen_ids.add(rid)
            daily_cumulative[date] = len(seen_ids)
        result[cls] = [(d, daily_cumulative[d]) for d in sorted(daily_cumulative.keys())]
    return result


def _compute_daily_series(
    rows: list[dict[str, str]],
    instance_class: dict[str, str],
) -> dict[str, list[tuple[str, int]]]:
    """Daily unique installs per class per date."""
    class_date_ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid or rid.lower() == "null":
            continue
        date = _extract_date(row.get("ts") or "")
        if not date:
            continue
        cls = instance_class.get(rid, CLASS_UNKNOWN)
        class_date_ids[cls][date].add(rid)

    result: dict[str, list[tuple[str, int]]] = {}
    for cls, date_ids in class_date_ids.items():
        result[cls] = [(d, len(date_ids[d])) for d in sorted(date_ids.keys())]
    return result


def _generate_chart(
    cumulative_data: dict[str, list[tuple[str, int]]],
    daily_data: dict[str, list[tuple[str, int]]],
    output_path: str,
) -> None:
    """Generate and save the timeseries chart with two subplots."""
    apply_tufte_style()

    fig, (ax_cumulative, ax_daily) = plt.subplots(
        2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True
    )
    fig.suptitle(CHART_TITLE, fontsize=14, fontweight="bold", y=0.98)

    colors = sns.color_palette(LINE_PALETTE, max(len(cumulative_data), 1))

    for idx, (cls, series) in enumerate(sorted(cumulative_data.items())):
        dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in series]
        counts = [c for _, c in series]
        ax_cumulative.plot(
            dates, counts, marker="o", markersize=5, linewidth=2, label=cls, color=colors[idx]
        )
    ax_cumulative.set_ylabel("Cumulative Unique Installs")
    ax_cumulative.set_title("Cumulative Unique Registry Installs", fontsize=11)
    ax_cumulative.legend(title="Deployment Type", loc="upper left")
    ax_cumulative.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for idx, (cls, series) in enumerate(sorted(daily_data.items())):
        dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in series]
        counts = [c for _, c in series]
        ax_daily.plot(
            dates, counts, marker="s", markersize=4, linewidth=2, label=cls, color=colors[idx]
        )
    ax_daily.set_ylabel("Daily Unique Installs")
    ax_daily.set_title("Daily Active Registry Installs", fontsize=11)
    ax_daily.legend(title="Deployment Type", loc="upper left")
    ax_daily.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_daily.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax_daily.xaxis.get_majorticklabels(), rotation=45, ha="right")

    for _ax in fig.axes:
        tufte_axes(_ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Community-vs-internal chart saved to {output_path}")


def main() -> None:
    """Parse arguments, classify instances, write the chart + JSON summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify registry instances as community vs internal by version "
            "string and produce a timeseries chart + JSON breakdown."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing CSV files to read (scans subdirectories too)",
    )
    parser.add_argument("--output", required=True, help="Path to save the output PNG")
    parser.add_argument(
        "--summary-json", required=True, help="Path to write the JSON summary the renderer reads"
    )
    parser.add_argument(
        "--internal-instances",
        default=None,
        help="Optional markdown file of known-internal instance IDs (always counted internal)",
    )
    parser.add_argument(
        "--yesterday",
        default=None,
        help="YYYY-MM-DD of the previous complete day, for the headline breakdown",
    )
    parser.add_argument(
        "--exclude-incomplete-day",
        default=None,
        help=(
            "Optional YYYY-MM-DD. Events on this date are dropped from the chart "
            "series (but kept in cumulative/summary) so the chart doesn't show a "
            "misleading trailing dip from a still-in-progress day."
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        logger.error(f"Directory not found: {args.csv_dir}")
        raise SystemExit(1)

    csv_files = _find_csv_files(args.csv_dir)
    if not csv_files:
        logger.error(f"No CSV files found in {args.csv_dir}")
        raise SystemExit(1)

    all_rows = _read_all_csvs(csv_files)
    if not all_rows:
        logger.error("No data found in CSV files")
        raise SystemExit(1)

    internal_ids = _load_internal_allowlist(args.internal_instances)

    # Summary uses ALL rows (cumulative + yesterday headline are accurate).
    summary = _compute_summary(all_rows, internal_ids, args.yesterday)
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON written to {args.summary_json}")

    # Chart classification map (latest version per instance), then optionally
    # drop the incomplete day from the plotted series only.
    latest_version = _latest_version_per_instance(all_rows)
    instance_class = {
        rid: _classify_instance(rid, ver, internal_ids) for rid, ver in latest_version.items()
    }
    chart_rows = all_rows
    if args.exclude_incomplete_day:
        chart_rows = [
            r for r in all_rows if (r.get("ts") or "")[:10] != args.exclude_incomplete_day
        ]
        logger.info(
            f"Excluded incomplete day {args.exclude_incomplete_day}: "
            f"{len(all_rows)} -> {len(chart_rows)} events (chart only)"
        )

    cumulative_data = _compute_cumulative_series(chart_rows, instance_class)
    daily_data = _compute_daily_series(chart_rows, instance_class)
    if not cumulative_data:
        logger.error("No identified registry instances found in data")
        raise SystemExit(1)

    _generate_chart(cumulative_data, daily_data, args.output)


if __name__ == "__main__":
    main()
