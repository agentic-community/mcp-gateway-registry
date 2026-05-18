"""Plot lifetime-bucket retention percentages over time.

Reads every `metrics-*.json` file in the dated subdirectories under a base
output dir, retroactively computes per-snapshot lifetime-bucket percentages
for the customer (non-internal) fleet, and produces:

  1. A multi-line chart with one series per threshold:
       - >= 3 days
       - >= 7 days
       - >= 14 days
       - >= 30 days
     Each line is the percentage of non-internal customer instances whose
     age_days (last_seen - first_seen) was at least the threshold on that
     snapshot date. Plus a separate one-day-wonder line on a secondary axis.

  2. An optional CSV sidecar of the per-snapshot values so future reports
     can diff and the report narrative can quote exact numbers.

The percentages are recomputed from `instance_lifetime` + `internal_instance_ids`
so old snapshot files (written before lifetime_bucket_pct existed) are
handled transparently.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
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

CHART_TITLE: str = (
    "AI Registry -- Customer Lifetime Retention Over Time "
    "(% of customer instances surviving N days)"
)
FIGURE_WIDTH: int = 14
FIGURE_HEIGHT: int = 7
THRESHOLDS: list[int] = [3, 7, 14, 30]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _find_metrics_files(
    base_dir: str,
) -> list[tuple[str, str]]:
    """Return sorted list of (date_str, metrics_path) for each dated subfolder."""
    out: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full) or not DATE_RE.match(entry):
            continue
        candidate = os.path.join(full, f"metrics-{entry}.json")
        if os.path.exists(candidate):
            out.append((entry, candidate))
    logger.info(f"Found {len(out)} metrics files in {base_dir}")
    return out


def _is_internal_id(
    registry_id: str,
    internal_ids: set[str],
) -> bool:
    """Match an instance row's registry_id (which may be truncated like 'abc...') to internal IDs."""
    if not registry_id:
        return False
    if registry_id in internal_ids:
        return True
    if registry_id.endswith("..."):
        prefix = registry_id[:-3]
        for full in internal_ids:
            if full.startswith(prefix):
                return True
    return False


def _compute_bucket_pct_from_metrics(
    metrics_path: str,
) -> dict | None:
    """Re-derive lifetime-bucket percentages from a metrics-*.json file.

    Falls back to instance_lifetime + internal_instance_ids so old snapshots
    that predate the lifetime_bucket_pct field still produce numbers.
    Returns None if the file is missing the lifetime data.
    """
    with open(metrics_path) as f:
        data = json.load(f)

    lifetime = data.get("instance_lifetime") or []
    if not lifetime:
        return None
    internal_ids = set(data.get("internal_instance_ids") or [])

    non_internal = [
        inst for inst in lifetime if not _is_internal_id(inst.get("registry_id", ""), internal_ids)
    ]
    total = len(non_internal)
    if total == 0:
        return None

    one_day_wonders = sum(1 for inst in non_internal if inst.get("age_days", 0) == 0)
    one_day_pct = round(100.0 * one_day_wonders / total, 1)

    bucket_pct = {
        threshold: round(
            100.0 * sum(1 for inst in non_internal if inst.get("age_days", 0) >= threshold) / total,
            1,
        )
        for threshold in THRESHOLDS
    }

    return {
        "total_non_internal": total,
        "one_day_wonder_pct": one_day_pct,
        "bucket_pct": bucket_pct,
    }


def _build_timeseries(
    metrics_files: list[tuple[str, str]],
) -> list[dict]:
    """Build per-snapshot rows of bucket percentages."""
    out: list[dict] = []
    for date_str, path in metrics_files:
        derived = _compute_bucket_pct_from_metrics(path)
        if not derived:
            continue
        out.append(
            {
                "date": date_str,
                "total": derived["total_non_internal"],
                "one_day_wonder_pct": derived["one_day_wonder_pct"],
                **{f"pct_ge_{t}d": derived["bucket_pct"][t] for t in THRESHOLDS},
            }
        )
    return out


def _write_csv_sidecar(
    rows: list[dict],
    path: str,
) -> None:
    """Write the per-snapshot bucket-percentage rows to a CSV sidecar."""
    if not rows:
        return
    fieldnames = ["date", "total", "one_day_wonder_pct"] + [f"pct_ge_{t}d" for t in THRESHOLDS]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    logger.info(f"CSV sidecar written to {path}")


def _generate_chart(
    rows: list[dict],
    output_path: str,
) -> None:
    """Render the multi-line retention chart."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.suptitle(CHART_TITLE, fontsize=14, fontweight="bold", y=0.97)

    parsed = [datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
    palette = sns.color_palette("Blues_d", len(THRESHOLDS))[::-1]

    # Survival-curve series: percentage of customer fleet whose age_days >= threshold
    for idx, threshold in enumerate(THRESHOLDS):
        key = f"pct_ge_{threshold}d"
        values = [r[key] for r in rows]
        ax.plot(
            parsed,
            values,
            marker="o",
            markersize=5,
            linewidth=2,
            color=palette[idx],
            label=f">= {threshold} days",
        )

    ax.set_ylabel("% of customer instances", fontsize=11)
    ax.set_xlabel("Snapshot date", fontsize=11)
    ax.legend(
        title="Lifetime threshold",
        loc="upper left",
    )
    ax.set_ylim(0, max(35, max((r["pct_ge_3d"] for r in rows), default=35) + 5))

    # One-day-wonder line on a secondary axis (typically much higher than the
    # survival curves, so a shared axis would compress the signal).
    ax_odw = ax.twinx()
    odw_vals = [r["one_day_wonder_pct"] for r in rows]
    ax_odw.plot(
        parsed,
        odw_vals,
        marker="s",
        markersize=4,
        linewidth=2,
        linestyle="--",
        color=sns.color_palette("Reds_d")[2],
        label="one-day wonders",
    )
    ax_odw.set_ylabel("% one-day wonders", fontsize=11, color=sns.color_palette("Reds_d")[2])
    ax_odw.tick_params(axis="y", labelcolor=sns.color_palette("Reds_d")[2])
    ax_odw.set_ylim(0, 100)

    # Combined legend across both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_odw.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(parsed) // 14)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Lifetime-bucket chart saved to {output_path}")


def main() -> None:
    """Parse arguments and generate the lifetime-bucket retention chart."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot lifetime-bucket retention percentages over time. "
            "Reads metrics-*.json files from each dated subdirectory under "
            "--csv-dir and renders one line per >=N day threshold."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Base output directory (contains dated subfolders with metrics-*.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG chart",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write a per-snapshot CSV sidecar of the bucket percentages",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        logger.error(f"Directory not found: {args.csv_dir}")
        raise SystemExit(1)

    metrics_files = _find_metrics_files(args.csv_dir)
    if not metrics_files:
        logger.error(f"No metrics-*.json files found under {args.csv_dir}")
        raise SystemExit(1)

    rows = _build_timeseries(metrics_files)
    if not rows:
        logger.error("No bucket data could be computed from metrics files")
        raise SystemExit(1)

    _generate_chart(rows, args.output)
    if args.csv_out:
        _write_csv_sidecar(rows, args.csv_out)


if __name__ == "__main__":
    main()
