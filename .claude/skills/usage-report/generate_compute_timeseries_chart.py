"""Generate a timeseries chart of unique registry installs per compute platform.

Reads ALL CSV files in a given directory (and dated subdirectories),
deduplicates events, and produces a PNG line chart with two subplots:

  1. Cumulative unique registry_id count per compute platform over time
  2. Daily unique registry_id count per compute platform

Compute platform values: ecs, eks, kubernetes, docker, ec2, vm, unknown.
"""

import argparse
import csv
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

CHART_TITLE: str = "AI Registry -- Unique Registry Installs per Compute Platform"
FIGURE_WIDTH: int = 14
FIGURE_HEIGHT: int = 7
LINE_PALETTE: str = "Set2"
DIMENSION_COLUMN: str = "compute"
DIMENSION_LABEL: str = "Compute Platform"


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
            reader = csv.DictReader(f)
            rows = list(reader)
        logger.info(f"Read {len(rows)} rows from {csv_path}")
        all_rows.extend(rows)
    logger.info(f"Total rows across all CSVs: {len(all_rows)}")
    return all_rows


def _deduplicate_events(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Deduplicate rows by (registry_id, ts) to avoid double-counting."""
    seen = set()
    unique_rows = []
    for row in rows:
        rid = row.get("registry_id", "").strip()
        ts = row.get("ts", "").strip()
        key = (rid, ts)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    logger.info(f"Deduplicated: {len(rows)} -> {len(unique_rows)} unique events")
    return unique_rows


def _extract_date(
    ts: str,
) -> str | None:
    """Extract YYYY-MM-DD date from an ISO timestamp string."""
    if not ts or len(ts) < 10:
        return None
    return ts[:10]


def _compute_cumulative_installs(
    rows: list[dict[str, str]],
) -> dict[str, list[tuple[str, int]]]:
    """Compute cumulative unique registry installs per compute platform per date.

    Returns a dict keyed by compute platform, each value is a sorted list
    of (date_str, cumulative_unique_count) tuples.
    """
    platform_events: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        rid = row.get("registry_id", "").strip()
        if not rid:
            continue
        platform = row.get(DIMENSION_COLUMN) or "unknown"
        ts = row.get("ts", "")
        date = _extract_date(ts)
        if not date:
            continue
        platform_events[platform].append((date, rid))

    result: dict[str, list[tuple[str, int]]] = {}
    for platform, events in platform_events.items():
        events.sort(key=lambda x: x[0])
        seen_ids: set[str] = set()
        daily_cumulative: dict[str, int] = {}
        for date, rid in events:
            seen_ids.add(rid)
            daily_cumulative[date] = len(seen_ids)
        sorted_dates = sorted(daily_cumulative.keys())
        result[platform] = [(d, daily_cumulative[d]) for d in sorted_dates]
    return result


def _compute_daily_unique_installs(
    rows: list[dict[str, str]],
) -> dict[str, list[tuple[str, int]]]:
    """Compute daily unique registry installs per compute platform.

    Returns a dict keyed by compute platform, each value is a sorted list
    of (date_str, unique_count_that_day) tuples.
    """
    platform_date_ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        rid = row.get("registry_id", "").strip()
        if not rid:
            continue
        platform = row.get(DIMENSION_COLUMN) or "unknown"
        ts = row.get("ts", "")
        date = _extract_date(ts)
        if not date:
            continue
        platform_date_ids[platform][date].add(rid)

    result: dict[str, list[tuple[str, int]]] = {}
    for platform, date_ids in platform_date_ids.items():
        sorted_dates = sorted(date_ids.keys())
        result[platform] = [(d, len(date_ids[d])) for d in sorted_dates]
    return result


def _compute_snapshot_table(
    csv_files: list[str],
) -> tuple[list[str], list[tuple[str, dict[str, int]]]]:
    """Build per-snapshot unique-instance counts grouped by compute platform.

    For each CSV file whose parent directory is named YYYY-MM-DD, read the file
    and count unique registry_ids per compute value. Returns the ordered list
    of platform columns and a list of (snapshot_date, counts_by_platform)
    tuples sorted descending by date (newest first).
    """
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    per_date: list[tuple[str, dict[str, int]]] = []
    all_platforms: set[str] = set()
    for csv_path in csv_files:
        parent = os.path.basename(os.path.dirname(csv_path))
        if not date_re.match(parent):
            continue
        seen: dict[str, str] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = (row.get("registry_id") or "").strip()
                if not rid or rid.lower() == "null":
                    continue
                platform = row.get("compute") or "unknown"
                seen[rid] = platform
        counts: dict[str, int] = defaultdict(int)
        for platform in seen.values():
            counts[platform] += 1
            all_platforms.add(platform)
        per_date.append((parent, dict(counts)))

    preferred_order = ["docker", "kubernetes", "ecs", "ec2", "unknown"]
    columns = [p for p in preferred_order if p in all_platforms]
    columns += sorted(p for p in all_platforms if p not in preferred_order)

    per_date.sort(key=lambda item: item[0], reverse=True)
    return columns, per_date


def _write_snapshot_table(
    columns: list[str],
    rows: list[tuple[str, dict[str, int]]],
    output_path: str,
) -> None:
    """Write a markdown per-platform snapshot table, sorted descending by date.

    The newest row is bolded so future reports can drop it straight into the
    "Per-Platform Growth (Unique Installs)" section.
    """
    header_cells = ["Date"] + columns + ["Total"]
    align_cells = ["------"] + ["-----:"] * len(columns) + ["-----:"]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(align_cells) + "|",
    ]
    for idx, (date_str, counts) in enumerate(rows):
        total = sum(counts.values())
        if idx == 0:
            values = [f"**{counts.get(c, 0)}**" for c in columns]
            lines.append(
                f"| **{date_str}** | " + " | ".join(values) + f" | **{total}** |"
            )
        else:
            values = [str(counts.get(c, 0)) for c in columns]
            lines.append(f"| {date_str} | " + " | ".join(values) + f" | {total} |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Per-platform snapshot table written to {output_path}")


def _generate_chart(
    cumulative_data: dict[str, list[tuple[str, int]]],
    daily_data: dict[str, list[tuple[str, int]]],
    output_path: str,
) -> None:
    """Generate and save the timeseries chart with two subplots."""
    sns.set_theme(style="whitegrid")

    fig, (ax_cumulative, ax_daily) = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        sharex=True,
    )
    fig.suptitle(
        CHART_TITLE,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    colors = sns.color_palette(LINE_PALETTE, len(cumulative_data))

    for idx, (platform, series) in enumerate(sorted(cumulative_data.items())):
        dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in series]
        counts = [c for _, c in series]
        ax_cumulative.plot(
            dates,
            counts,
            marker="o",
            markersize=5,
            linewidth=2,
            label=platform,
            color=colors[idx],
        )

    ax_cumulative.set_ylabel("Cumulative Unique Installs")
    ax_cumulative.set_title("Cumulative Unique Registry Installs", fontsize=11)
    ax_cumulative.legend(title=DIMENSION_LABEL, loc="upper left")
    ax_cumulative.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for idx, (platform, series) in enumerate(sorted(daily_data.items())):
        dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in series]
        counts = [c for _, c in series]
        ax_daily.plot(
            dates,
            counts,
            marker="s",
            markersize=4,
            linewidth=2,
            label=platform,
            color=colors[idx],
        )

    ax_daily.set_ylabel("Daily Unique Installs")
    ax_daily.set_title("Daily Active Registry Installs", fontsize=11)
    ax_daily.legend(title=DIMENSION_LABEL, loc="upper left")
    ax_daily.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_daily.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax_daily.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Timeseries chart saved to {output_path}")


def main() -> None:
    """Parse arguments and generate the compute-platform timeseries chart."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a timeseries chart of unique registry installs per compute platform "
            "(docker, kubernetes, ecs, ec2, etc.)."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing CSV files to read (scans subdirectories too)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG",
    )
    parser.add_argument(
        "--snapshots-table",
        default=None,
        help=(
            "Optional path to write a markdown per-platform snapshot table "
            "(unique instance counts by compute platform, sorted descending "
            "by date). Reads one CSV per dated subdirectory under --csv-dir."
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

    unique_rows = _deduplicate_events(all_rows)

    cumulative_data = _compute_cumulative_installs(unique_rows)
    daily_data = _compute_daily_unique_installs(unique_rows)

    if not cumulative_data:
        logger.error("No identified registry instances found in data")
        raise SystemExit(1)

    _generate_chart(cumulative_data, daily_data, args.output)

    if args.snapshots_table:
        columns, rows = _compute_snapshot_table(csv_files)
        if not rows:
            logger.warning(
                "No dated-subfolder CSVs found; skipping snapshot table output"
            )
        else:
            _write_snapshot_table(columns, rows, args.snapshots_table)


if __name__ == "__main__":
    main()
