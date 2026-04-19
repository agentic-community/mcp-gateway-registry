"""Generate a faceted bar chart based on unique registry instance counts.

Unlike generate_charts.py which counts events, this chart counts unique
registry instances per dimension value. Each instance is counted once
using its latest reported value for each dimension.
"""
import argparse
import csv
import logging
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

CHART_TITLE: str = "AI Registry -- Deployment Distribution (Unique Instances)"
FIGURE_WIDTH: int = 16
FIGURE_HEIGHT: int = 10
BAR_COLOR_PALETTE: str = "Blues_d"


def _read_csv(
    csv_path: str,
) -> list[dict[str, str]]:
    """Read the telemetry CSV and return rows as list of dicts."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    logger.info(f"Read {len(rows)} rows from {csv_path}")
    return rows


def _get_latest_per_instance(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Deduplicate rows by registry_id, keeping the latest event per instance.

    Rows without a registry_id are excluded since they cannot be reliably
    deduplicated and would inflate the instance count.
    """
    instance_latest: dict[str, dict[str, str]] = {}

    skipped = 0
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid:
            skipped += 1
            continue

        ts = row.get("ts", "")
        existing = instance_latest.get(rid)
        if existing is None or ts > existing.get("ts", ""):
            instance_latest[rid] = row

    logger.info(
        f"Deduplicated {len(rows)} events to {len(instance_latest)} unique instances "
        f"(skipped {skipped} events with null registry_id)"
    )
    return list(instance_latest.values())


def _compute_distributions(
    instances: list[dict[str, str]],
) -> dict[str, Counter]:
    """Compute value counts for each dimension based on unique instances."""
    dimensions = {
        "Cloud Provider": Counter(),
        "Compute Platform": Counter(),
        "Storage Backend": Counter(),
        "Auth Provider": Counter(),
        "Architecture": Counter(),
        "Deployment Mode": Counter(),
    }

    for row in instances:
        cloud = row.get("cloud", "unknown") or "unknown"
        dimensions["Cloud Provider"][cloud] += 1

        compute = row.get("compute", "unknown") or "unknown"
        dimensions["Compute Platform"][compute] += 1

        storage = row.get("storage", "unknown") or "unknown"
        dimensions["Storage Backend"][storage] += 1

        auth = row.get("auth", "none") or "none"
        dimensions["Auth Provider"][auth] += 1

        arch = row.get("arch", "unknown") or "unknown"
        dimensions["Architecture"][arch] += 1

        mode = row.get("mode", "unknown") or "unknown"
        dimensions["Deployment Mode"][mode] += 1

    return dimensions


def _plot_single_facet(
    ax: plt.Axes,
    counter: Counter,
    title: str,
    total: int,
) -> None:
    """Plot a single horizontal bar chart with percentages."""
    items = counter.most_common()
    labels = [item[0] for item in items]
    counts = [item[1] for item in items]

    labels = labels[::-1]
    counts = counts[::-1]

    colors = sns.color_palette(BAR_COLOR_PALETTE, len(labels))
    bars = ax.barh(labels, counts, color=colors)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("")

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        label_text = f" {count} ({pct:.0f}%)"
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            fontsize=10,
        )

    max_count = max(counts) if counts else 1
    ax.set_xlim(0, max_count * 1.4)


def _generate_chart(
    rows: list[dict[str, str]],
    output_path: str,
) -> None:
    """Generate and save the faceted distribution chart."""
    instances = _get_latest_per_instance(rows)
    total = len(instances)

    distributions = _compute_distributions(instances)

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.suptitle(
        f"{CHART_TITLE}\n({total} unique instances)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    dimension_order = [
        "Cloud Provider",
        "Compute Platform",
        "Storage Backend",
        "Auth Provider",
        "Architecture",
        "Deployment Mode",
    ]

    for idx, dim_name in enumerate(dimension_order):
        row_idx = idx // 3
        col_idx = idx % 3
        ax = axes[row_idx][col_idx]
        _plot_single_facet(ax, distributions[dim_name], dim_name, total)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved to {output_path}")


def main() -> None:
    """Parse arguments and generate instance-based distribution chart."""
    parser = argparse.ArgumentParser(
        description="Generate deployment distribution chart based on unique registry instances",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to registry_metrics.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        raise SystemExit(1)

    rows = _read_csv(args.csv)

    if not rows:
        logger.error("No data in CSV file")
        raise SystemExit(1)

    _generate_chart(rows, args.output)


if __name__ == "__main__":
    main()
