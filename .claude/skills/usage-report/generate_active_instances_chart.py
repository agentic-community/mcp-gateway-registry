"""Generate a timeseries chart of customer-active registry instances over time.

Reads ALL CSV files in a given directory (and dated subdirectories),
deduplicates events, filters out known internal instances, and produces a
PNG line chart with three overlaid series:

  1. DAI  -- Daily Active Instances: count of unique registry_ids that sent
            at least one event (startup OR heartbeat) on that day.
  2. MA7  -- 7-day trailing moving average of DAI.
  3. S7   -- 7-day consistency streak: count of unique registry_ids that sent
            at least one event on EACH of the 7 days in the window [D-6..D].

All three signals are computed on the customer fleet (internal instances
loaded from known-internal-instances.md are excluded).

A CSV sidecar of the daily values is written alongside the chart so the
report can quote exact numbers and future reports can diff against it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

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

CHART_TITLE: str = "AI Registry -- Customer-Active Instances (rolling 7-day window)"
FIGURE_WIDTH: int = 14
FIGURE_HEIGHT: int = 6
MOVING_AVG_WINDOW: int = 7
STREAK_WINDOW: int = 7


def _find_csv_files(
    directory: str,
) -> list[str]:
    """Find all registry_metrics.csv files in the directory and dated subdirectories."""
    csv_files: list[str] = []
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


def _load_internal_instance_ids(
    path: str | None,
) -> set[str]:
    """Parse the known-internal-instances.md file into a set of full UUIDs."""
    if not path or not Path(path).exists():
        return set()
    pattern = re.compile(r"`([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})`")
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            for match in pattern.findall(line):
                ids.add(match)
    logger.info(f"Loaded {len(ids)} known internal instance IDs from {path}")
    return ids


def _read_all_csvs(
    csv_files: list[str],
) -> list[dict[str, str]]:
    """Read and concatenate all CSV files."""
    all_rows: list[dict[str, str]] = []
    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        logger.info(f"Read {len(rows)} rows from {csv_path}")
        all_rows.extend(rows)
    logger.info(f"Total rows across all CSVs: {len(all_rows)}")
    return all_rows


def _dedupe_by_id_ts(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Drop duplicate rows by (registry_id, ts)."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for r in rows:
        rid = (r.get("registry_id") or "").strip()
        ts = (r.get("ts") or "").strip()
        key = (rid, ts)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    logger.info(f"Deduplicated: {len(rows)} -> {len(out)} unique events")
    return out


def _extract_date(
    ts: str,
) -> str | None:
    """Return YYYY-MM-DD from an ISO timestamp, or None if unparseable."""
    if not ts or len(ts) < 10:
        return None
    return ts[:10]


def _build_daily_index(
    rows: list[dict[str, str]],
    internal_ids: set[str],
) -> dict[str, set[str]]:
    """Build {date_str: set(registry_id)} for customer instances only.

    Events from internal_ids are excluded. Null/empty registry_ids are
    excluded (we can't attribute them to a single instance).
    """
    by_day: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        rid = (r.get("registry_id") or "").strip()
        if not rid or rid in internal_ids:
            continue
        d = _extract_date(r.get("ts", ""))
        if not d:
            continue
        by_day[d].add(rid)
    return by_day


def _date_range(
    start: str,
    end: str,
) -> list[str]:
    """Return a contiguous list of YYYY-MM-DD strings from start..end inclusive."""
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    out: list[str] = []
    d = start_d
    while d <= end_d:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _compute_series(
    by_day: dict[str, set[str]],
) -> tuple[list[str], list[int], list[float | None], list[int | None]]:
    """Compute DAI, MA7, and 7-day consistency streak series.

    Returns a tuple of (dates, dai, ma7, streak7). ma7 and streak7 contain
    None for the first 6 days (insufficient window).
    """
    if not by_day:
        return [], [], [], []

    sorted_dates = sorted(by_day.keys())
    full_dates = _date_range(sorted_dates[0], sorted_dates[-1])

    dai = [len(by_day.get(d, set())) for d in full_dates]

    ma7: list[float | None] = []
    for i in range(len(full_dates)):
        if i < MOVING_AVG_WINDOW - 1:
            ma7.append(None)
        else:
            window = dai[i - MOVING_AVG_WINDOW + 1 : i + 1]
            ma7.append(sum(window) / MOVING_AVG_WINDOW)

    streak7: list[int | None] = []
    for i in range(len(full_dates)):
        if i < STREAK_WINDOW - 1:
            streak7.append(None)
        else:
            window_sets = [by_day.get(full_dates[j], set()) for j in range(i - STREAK_WINDOW + 1, i + 1)]
            common = set.intersection(*window_sets) if window_sets else set()
            streak7.append(len(common))

    return full_dates, dai, ma7, streak7


def _write_csv_sidecar(
    dates: list[str],
    dai: list[int],
    ma7: list[float | None],
    streak7: list[int | None],
    path: str,
) -> None:
    """Write the daily series as a CSV sidecar next to the chart."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "daily_active", "ma7", "streak7"])
        for i, d in enumerate(dates):
            ma_cell = "" if ma7[i] is None else f"{ma7[i]:.2f}"
            s_cell = "" if streak7[i] is None else str(streak7[i])
            w.writerow([d, dai[i], ma_cell, s_cell])
    logger.info(f"CSV sidecar written to {path}")


def _generate_chart(
    dates: list[str],
    dai: list[int],
    ma7: list[float | None],
    streak7: list[int | None],
    output_path: str,
) -> None:
    """Render the three-series line chart."""
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.suptitle(CHART_TITLE, fontsize=14, fontweight="bold", y=0.98)

    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    colors = sns.color_palette("Set2", 3)

    ax.plot(parsed, dai, marker="o", markersize=4, linewidth=1.5,
            color=colors[0], alpha=0.75, label="Daily Active Instances (DAI)")

    ma_dates = [p for p, v in zip(parsed, ma7) if v is not None]
    ma_vals = [v for v in ma7 if v is not None]
    ax.plot(ma_dates, ma_vals, linewidth=2.5, color=colors[1],
            label="7-day moving average (MA7)")

    s_dates = [p for p, v in zip(parsed, streak7) if v is not None]
    s_vals = [v for v in streak7 if v is not None]
    ax.plot(s_dates, s_vals, marker="s", markersize=4, linewidth=2.0,
            color=colors[2], label="7-day consistency streak (S7)")

    ax.set_ylabel("Customer instances")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(parsed) // 14)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved to {output_path}")


def main() -> None:
    """Parse arguments and generate the active-instances chart."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a chart of customer-active registry instances over time: "
            "Daily Active + 7-day moving average + 7-day consistency streak. "
            "Startup and heartbeat events are unioned per (instance, day)."
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
        "--internal-instances",
        default=None,
        help="Optional path to known-internal-instances.md. When provided, internal instance IDs are excluded from all three series.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write a CSV sidecar of the daily series (date, daily_active, ma7, streak7).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        logger.error(f"Directory not found: {args.csv_dir}")
        raise SystemExit(1)

    csv_files = _find_csv_files(args.csv_dir)
    if not csv_files:
        logger.error(f"No CSV files found in {args.csv_dir}")
        raise SystemExit(1)

    internal_ids = _load_internal_instance_ids(args.internal_instances)

    all_rows = _read_all_csvs(csv_files)
    unique_rows = _dedupe_by_id_ts(all_rows)
    by_day = _build_daily_index(unique_rows, internal_ids)

    dates, dai, ma7, streak7 = _compute_series(by_day)
    if not dates:
        logger.error("No customer events found after filtering")
        raise SystemExit(1)

    _generate_chart(dates, dai, ma7, streak7, args.output)

    if args.csv_out:
        _write_csv_sidecar(dates, dai, ma7, streak7, args.csv_out)


if __name__ == "__main__":
    main()
