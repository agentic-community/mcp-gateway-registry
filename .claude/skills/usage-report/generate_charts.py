"""Generate a single faceted bar chart from telemetry CSV data.

Reads registry_metrics.csv and produces a PNG with subplots showing
distributions across cloud, compute, storage, auth, version type,
and deployment mode.
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


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

CHART_TITLE: str = "AI Registry -- Deployment Distribution"
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


def _classify_version(
    version: str,
) -> str:
    """Classify a version string as Release or Dev."""
    if not version:
        return "unknown"
    if version.startswith("v1.0.17") or version.startswith("v0."):
        return "dev"
    return "release"


def _compute_distributions(
    rows: list[dict[str, str]],
) -> dict[str, Counter]:
    """Compute value counts for each dimension."""
    dimensions = {
        "Cloud Provider": Counter(),
        "Compute Platform": Counter(),
        "Storage Backend": Counter(),
        "Auth Provider": Counter(),
        "Version Type": Counter(),
        "Deployment Mode": Counter(),
    }

    for row in rows:
        cloud = row.get("cloud", "unknown") or "unknown"
        dimensions["Cloud Provider"][cloud] += 1

        compute = row.get("compute", "unknown") or "unknown"
        dimensions["Compute Platform"][compute] += 1

        storage = row.get("storage", "unknown") or "unknown"
        dimensions["Storage Backend"][storage] += 1

        auth = row.get("auth", "none") or "none"
        dimensions["Auth Provider"][auth] += 1

        version = row.get("v", "") or ""
        dimensions["Version Type"][_classify_version(version)] += 1

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
    # Sort by count descending
    items = counter.most_common()
    labels = [item[0] for item in items]
    counts = [item[1] for item in items]

    # Reverse so largest is on top
    labels = labels[::-1]
    counts = counts[::-1]

    colors = sns.color_palette(BAR_COLOR_PALETTE, len(labels))
    bars = ax.barh(labels, counts, color=colors)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("")

    # Add count and percentage labels on bars
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

    # Add some padding on the right for labels
    max_count = max(counts) if counts else 1
    ax.set_xlim(0, max_count * 1.4)


def _generate_chart(
    rows: list[dict[str, str]],
    output_path: str,
) -> None:
    """Generate and save the faceted distribution chart."""
    total = len(rows)
    distributions = _compute_distributions(rows)

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.suptitle(
        f"{CHART_TITLE}\n({total} events)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Plot each dimension
    dimension_order = [
        "Cloud Provider",
        "Compute Platform",
        "Storage Backend",
        "Auth Provider",
        "Version Type",
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
    """Parse arguments and generate charts."""
    parser = argparse.ArgumentParser(
        description="Generate telemetry distribution charts from CSV data",
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
