"""Generate a timeseries chart of unique registry installs per cloud provider.

Reads ALL CSV files in a given directory, deduplicates events, and produces
a PNG line chart showing cumulative unique registry_id values per cloud
provider over time.
"""

import argparse
import csv
import logging
import os
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import sys as _sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_labels import cloud_label  # noqa: E402
from tufte_style import apply_tufte_style, tufte_axes  # noqa: E402

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

CHART_TITLE: str = "AI Registry -- Unique Registry Installs per Cloud Provider"
FIGURE_WIDTH: int = 14
FIGURE_HEIGHT: int = 7
LINE_PALETTE: str = "Set2"


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
    """Compute cumulative unique registry installs per cloud provider per date.

    Returns a dict keyed by cloud provider, each value is a sorted list
    of (date_str, cumulative_unique_count) tuples.
    """
    # Group events by cloud provider
    cloud_events = defaultdict(list)
    for row in rows:
        rid = row.get("registry_id", "").strip()
        if not rid:
            continue
        cloud = cloud_label(row)
        ts = row.get("ts", "")
        date = _extract_date(ts)
        if not date:
            continue
        cloud_events[cloud].append((date, rid))

    # For each cloud, compute cumulative unique registry_ids per date
    result = {}
    for cloud, events in cloud_events.items():
        events.sort(key=lambda x: x[0])

        # Collect all dates and track cumulative set
        seen_ids = set()
        daily_cumulative = {}
        for date, rid in events:
            seen_ids.add(rid)
            daily_cumulative[date] = len(seen_ids)

        # Convert to sorted list of tuples
        sorted_dates = sorted(daily_cumulative.keys())
        result[cloud] = [(d, daily_cumulative[d]) for d in sorted_dates]

    return result


def _compute_daily_unique_installs(
    rows: list[dict[str, str]],
) -> dict[str, list[tuple[str, int]]]:
    """Compute daily unique registry installs per cloud provider.

    Returns a dict keyed by cloud provider, each value is a sorted list
    of (date_str, unique_count_that_day) tuples.
    """
    # Group by (cloud, date) -> set of registry_ids
    cloud_date_ids = defaultdict(lambda: defaultdict(set))
    for row in rows:
        rid = row.get("registry_id", "").strip()
        if not rid:
            continue
        cloud = cloud_label(row)
        ts = row.get("ts", "")
        date = _extract_date(ts)
        if not date:
            continue
        cloud_date_ids[cloud][date].add(rid)

    result = {}
    for cloud, date_ids in cloud_date_ids.items():
        sorted_dates = sorted(date_ids.keys())
        result[cloud] = [(d, len(date_ids[d])) for d in sorted_dates]

    return result


def _compute_daily_new_installs(
    rows: list[dict[str, str]],
) -> dict[str, list[tuple[str, int]]]:
    """Compute daily NEW registry installs per cloud provider.

    A "new install" is the first calendar day a given registry_id is ever
    seen in the dataset. Each registry_id is counted exactly once, on its
    first-seen date, under the cloud provider it reported on that date.

    Distinct from _compute_daily_unique_installs which counts any instance
    that emitted an event on the day (returning visitors get re-counted).
    This function answers "how many brand-new deployments came online on
    each day?" rather than "how many were active that day?".

    Returns a dict keyed by cloud provider, each value is a sorted list
    of (date_str, new_install_count_that_day) tuples.
    """
    # First pass: find each registry_id's earliest (date, cloud) pair.
    earliest: dict[str, tuple[str, str]] = {}
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid:
            continue
        date = _extract_date(row.get("ts", ""))
        if not date:
            continue
        cloud = cloud_label(row)
        prior = earliest.get(rid)
        if prior is None or date < prior[0]:
            earliest[rid] = (date, cloud)

    # Second pass: bucket by (cloud, date).
    cloud_date_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _rid, (date, cloud) in earliest.items():
        cloud_date_counts[cloud][date] += 1

    result = {}
    for cloud, date_counts in cloud_date_counts.items():
        sorted_dates = sorted(date_counts.keys())
        result[cloud] = [(d, date_counts[d]) for d in sorted_dates]

    return result


def _generate_chart(
    cumulative_data: dict[str, list[tuple[str, int]]],
    daily_data: dict[str, list[tuple[str, int]]],
    new_data: dict[str, list[tuple[str, int]]],
    output_path: str,
) -> None:
    """Generate and save the timeseries chart with three subplots.

    Top panel: cumulative unique installs per cloud (ever-seen running total).
    Middle panel: daily unique active installs per cloud (any event that day).
    Bottom panel: daily NEW installs per cloud (first-seen on that day only).
    """
    apply_tufte_style()

    fig, (ax_cumulative, ax_daily, ax_new) = plt.subplots(
        3,
        1,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT * 1.4),
        sharex=True,
    )
    fig.suptitle(
        CHART_TITLE,
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    # Use a stable colour map keyed on cloud provider so the same cloud
    # gets the same colour in all three panels even if a panel's dict
    # iteration order differs.
    all_clouds = sorted(set(cumulative_data) | set(daily_data) | set(new_data))
    palette = sns.color_palette(LINE_PALETTE, len(all_clouds))
    cloud_color = {cloud: palette[i] for i, cloud in enumerate(all_clouds)}

    # Plot cumulative installs
    for cloud, series in sorted(cumulative_data.items()):
        dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in series]
        counts = [c for _, c in series]
        ax_cumulative.plot(
            dates,
            counts,
            marker="o",
            markersize=5,
            linewidth=2,
            label=cloud,
            color=cloud_color[cloud],
        )

    ax_cumulative.set_ylabel("Cumulative Unique Installs")
    ax_cumulative.set_title("Cumulative Unique Registry Installs", loc="left", fontsize=11)
    ax_cumulative.legend(title="Cloud Provider", loc="upper left")
    ax_cumulative.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    tufte_axes(ax_cumulative)

    # Plot daily active installs
    for cloud, series in sorted(daily_data.items()):
        dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in series]
        counts = [c for _, c in series]
        ax_daily.plot(
            dates,
            counts,
            marker="s",
            markersize=4,
            linewidth=2,
            label=cloud,
            color=cloud_color[cloud],
        )

    ax_daily.set_ylabel("Daily Active Installs")
    ax_daily.set_title("Daily Active Registry Installs", loc="left", fontsize=11)
    ax_daily.legend(title="Cloud Provider", loc="upper left")
    ax_daily.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    tufte_axes(ax_daily)

    # Plot daily NEW installs (first-seen per day)
    for cloud, series in sorted(new_data.items()):
        dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in series]
        counts = [c for _, c in series]
        ax_new.plot(
            dates,
            counts,
            marker="^",
            markersize=4,
            linewidth=2,
            label=cloud,
            color=cloud_color[cloud],
        )

    ax_new.set_ylabel("New Installs (first-seen)")
    ax_new.set_title(
        "Daily NEW Registry Installs (first-seen on that day)", loc="left", fontsize=11
    )
    ax_new.legend(title="Cloud Provider", loc="upper left")
    ax_new.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    tufte_axes(ax_new)

    # Format x-axis dates on the bottom panel
    ax_new.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_new.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax_new.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Timeseries chart saved to {output_path}")


def main() -> None:
    """Parse arguments and generate the timeseries chart."""
    parser = argparse.ArgumentParser(
        description="Generate timeseries chart of unique registry installs per cloud provider",
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing CSV files to read",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG",
    )
    parser.add_argument(
        "--exclude-incomplete-day",
        default=None,
        help=(
            "Optional YYYY-MM-DD. Events on this date are dropped before charting "
            "so the chart doesn't show a misleading dip from a still-in-progress day."
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
    if args.exclude_incomplete_day:
        kept = [r for r in unique_rows if (r.get("ts") or "")[:10] != args.exclude_incomplete_day]
        logger.info(
            f"Excluded incomplete day {args.exclude_incomplete_day}: "
            f"{len(unique_rows)} -> {len(kept)} events"
        )
        unique_rows = kept

    cumulative_data = _compute_cumulative_installs(unique_rows)
    daily_data = _compute_daily_unique_installs(unique_rows)
    new_data = _compute_daily_new_installs(unique_rows)

    if not cumulative_data:
        logger.error("No identified registry instances found in data")
        raise SystemExit(1)

    _generate_chart(cumulative_data, daily_data, new_data, args.output)


if __name__ == "__main__":
    main()
