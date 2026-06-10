"""Generate a timeseries chart of daily AWS customer instances reporting home.

Reads ALL CSV files in a given directory (and dated subdirectories),
deduplicates events, filters out known internal instances, and produces a PNG
line chart with three overlaid series, each counting unique AWS customer
registry_ids that sent at least one event (startup OR heartbeat) on that day:

  1. All reporters       -- every AWS customer instance that reported that day,
                            including instances that install and vanish the same
                            day (single-event installs).
  2. Persisted (>=2 evt) -- instances that emitted at least 2 events EVER. Drops
                            the true install-and-vanish-in-one-shot cohort.
  3. Persisted (>=2 day) -- instances that reported on at least 2 DISTINCT days
                            EVER. This is the "real running deployment"
                            definition used as the headline LTV counting rule:
                            an instance only counts if it came back on a
                            separate day rather than installing and deleting
                            within a single day.

This chart is the visual companion to the "persisted" counting rule in
generate_ltv_spend.py: the green (>=2 distinct days) line is the daily
instance count that drives the headline infra-spend number.

Filters: customer-only (internal UUIDs excluded), AWS-only (cloud == "aws"),
matching generate_ltv_spend.py so the counts reconcile.

A CSV sidecar of the daily values is written alongside the chart so the report
can quote exact numbers and future reports can diff against it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_style import apply_tufte_style, tufte_axes  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

CHART_TITLE: str = (
    "AI Registry -- Daily AWS customer instances reporting home\n"
    "(filtered to exclude install-and-vanish-within-a-day)"
)
FIGURE_WIDTH: int = 13
FIGURE_HEIGHT: int = 6
PERSIST_MIN_EVENTS: int = 2
PERSIST_MIN_DAYS: int = 2

ALL_COLOR: str = "#bbbbbb"
EVT_COLOR: str = "#1f77b4"
DAY_COLOR: str = "#2ca02c"


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


def _is_aws_customer(
    r: dict[str, str],
    internal_ids: set[str],
) -> bool:
    """Return True if this row is an AWS customer event (not internal)."""
    rid = (r.get("registry_id") or "").strip()
    if not rid or rid in internal_ids:
        return False
    cloud = (r.get("cloud") or "").strip()
    return cloud == "aws"


def _compute_daily_counts(
    rows: list[dict[str, str]],
    internal_ids: set[str],
    exclude_day: str | None = None,
) -> list[dict[str, int | str]]:
    """Compute per-day reporter counts under three persistence definitions.

    Returns a list of per-day dicts with keys: date, all_reporters,
    persisted_2events, persisted_2days.
    """
    by_day: dict[str, set[str]] = defaultdict(set)
    instance_event_count: dict[str, int] = defaultdict(int)
    instance_days: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        if not _is_aws_customer(r, internal_ids):
            continue
        rid = r["registry_id"].strip()
        d = (r.get("ts") or "")[:10]
        if len(d) < 10:
            continue
        # Event count is computed across ALL days (the persistence test looks at
        # an instance's whole history), but the per-day series and the
        # distinct-day test exclude the in-progress day.
        instance_event_count[rid] += 1
        if exclude_day and d == exclude_day:
            continue
        by_day[d].add(rid)
        instance_days[rid].add(d)

    persisted_events = {
        rid for rid, n in instance_event_count.items() if n >= PERSIST_MIN_EVENTS
    }
    persisted_days = {
        rid for rid, ds in instance_days.items() if len(ds) >= PERSIST_MIN_DAYS
    }

    logger.info(
        f"Unique AWS customer instances: {len(instance_event_count)} "
        f"(>= {PERSIST_MIN_EVENTS} events: {len(persisted_events)}, "
        f">= {PERSIST_MIN_DAYS} distinct days: {len(persisted_days)})"
    )

    out: list[dict[str, int | str]] = []
    for d in sorted(by_day.keys()):
        active = by_day[d]
        out.append(
            {
                "date": d,
                "all_reporters": len(active),
                "persisted_2events": len(active & persisted_events),
                "persisted_2days": len(active & persisted_days),
            }
        )
    return out


def _write_csv_sidecar(
    daily: list[dict[str, int | str]],
    path: str,
) -> None:
    """Write per-day reporter counts to a CSV sidecar."""
    if not daily:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(daily[0].keys()))
        w.writeheader()
        for row in daily:
            w.writerow(row)
    logger.info(f"CSV sidecar written to {path}")


def _generate_chart(
    daily: list[dict[str, int | str]],
    output_path: str,
) -> None:
    """Render the three-series daily-reporters line chart."""
    apply_tufte_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    dates = [datetime.strptime(str(r["date"]), "%Y-%m-%d") for r in daily]
    all_reporters = [r["all_reporters"] for r in daily]
    p_events = [r["persisted_2events"] for r in daily]
    p_days = [r["persisted_2days"] for r in daily]

    ax.plot(
        dates, all_reporters, lw=2.2, marker="o", ms=3, color=ALL_COLOR,
        label="All AWS reporters (incl. single-event installs)",
    )
    ax.plot(
        dates, p_events, lw=2.5, marker="s", ms=3, color=EVT_COLOR,
        label="Persisted: >= 2 events ever",
    )
    ax.plot(
        dates, p_days, lw=2.0, marker="^", ms=3, color=DAY_COLOR,
        label="Persisted: >= 2 distinct days (headline LTV rule)",
    )
    ax.set_title(CHART_TITLE, fontsize=12, fontweight="bold")
    ax.set_ylabel("unique instances reporting that day")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 14)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    tufte_axes(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved to {output_path}")


def main() -> None:
    """Parse arguments and generate the daily-reporters chart and CSV sidecar."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot daily AWS customer instances reporting home, with two "
            "persistence-filtered series that exclude install-and-vanish "
            "instances. The >= 2-distinct-days series is the headline LTV "
            "counting rule (see generate_ltv_spend.py)."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing CSV files (scans subdirectories too)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG chart",
    )
    parser.add_argument(
        "--internal-instances",
        default=None,
        help="Path to known-internal-instances.md. Internal IDs are excluded.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write per-day reporter counts CSV",
    )
    parser.add_argument(
        "--exclude-incomplete-day",
        default=None,
        help=(
            "Optional YYYY-MM-DD. Events on this date are dropped from the "
            "per-day series so a still-in-progress day doesn't show as a dip. "
            "An instance's events on this day still count toward its lifetime "
            "event total for the persistence test."
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

    internal_ids = _load_internal_instance_ids(args.internal_instances)
    all_rows = _read_all_csvs(csv_files)
    unique_rows = _dedupe_by_id_ts(all_rows)
    daily = _compute_daily_counts(
        unique_rows,
        internal_ids,
        args.exclude_incomplete_day,
    )
    if not daily:
        logger.error("No AWS customer events found after filtering")
        raise SystemExit(1)

    _generate_chart(daily, args.output)
    if args.csv_out:
        _write_csv_sidecar(daily, args.csv_out)


if __name__ == "__main__":
    main()
