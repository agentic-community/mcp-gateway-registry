"""Generate a density plot showing registry instance age distribution.

Reads the metrics JSON (which contains instance_lifetime data) and
produces a PNG with a histogram + KDE overlay of instance ages in days.
"""

import argparse
import json
import logging
import os

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

CHART_TITLE: str = "AI Registry -- Instance Lifetime Distribution"
FIGURE_WIDTH: int = 12
FIGURE_HEIGHT: int = 6


def _load_lifetime_data(
    metrics_path: str,
) -> list[int]:
    """Load instance lifetime ages from metrics JSON."""
    with open(metrics_path) as f:
        data = json.load(f)

    lifetime_list = data.get("instance_lifetime", [])
    if not lifetime_list:
        logger.error("No instance_lifetime data in metrics JSON")
        return []

    ages = [inst["age_days"] for inst in lifetime_list]
    logger.info(f"Loaded {len(ages)} instance ages from {metrics_path}")
    return ages


def _generate_chart(
    ages: list[int],
    output_path: str,
) -> None:
    """Generate and save the lifetime density chart."""
    sns.set_theme(style="whitegrid")

    fig, (ax_hist, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    fig.suptitle(
        f"{CHART_TITLE}\n({len(ages)} instances)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Compute stats
    avg_age = sum(ages) / len(ages) if ages else 0
    max_age = max(ages) if ages else 0
    multi_day = sum(1 for a in ages if a > 0)
    single_day = sum(1 for a in ages if a == 0)

    # Left panel: histogram with KDE overlay
    # Use integer bins from 0 to max_age + 1
    bin_edges = list(range(0, max_age + 2))

    ax_hist.hist(
        ages,
        bins=bin_edges,
        color=sns.color_palette("Blues_d")[2],
        edgecolor="white",
        alpha=0.7,
        align="left",
        label="Count",
    )

    # Add KDE curve on secondary y-axis for density
    ax_kde = ax_hist.twinx()
    if len(set(ages)) > 1:
        sns.kdeplot(
            ages,
            ax=ax_kde,
            color=sns.color_palette("deep")[3],
            linewidth=2,
            bw_adjust=0.8,
            label="Density",
        )
    ax_kde.set_ylabel("Density", fontsize=10, color="gray")
    ax_kde.tick_params(axis="y", labelcolor="gray")

    ax_hist.set_xlabel("Instance Age (days)", fontsize=11)
    ax_hist.set_ylabel("Number of Instances", fontsize=11)
    ax_hist.set_title("Age Distribution", fontsize=12, fontweight="bold")

    # Set x-axis to integer ticks
    ax_hist.set_xticks(range(0, max_age + 1))

    # Add stats annotation
    stats_text = (
        f"Mean: {avg_age:.1f} days\n"
        f"Max: {max_age} days\n"
        f"Multi-day: {multi_day}\n"
        f"Single-day: {single_day}"
    )
    ax_hist.text(
        0.97,
        0.95,
        stats_text,
        transform=ax_hist.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.8},
    )

    # Right panel: horizontal bar showing age buckets
    buckets = {
        "0 days (single session)": single_day,
        "1-2 days": sum(1 for a in ages if 1 <= a <= 2),
        "3-5 days": sum(1 for a in ages if 3 <= a <= 5),
        "6-10 days": sum(1 for a in ages if 6 <= a <= 10),
        "11+ days": sum(1 for a in ages if a >= 11),
    }

    # Remove empty buckets
    buckets = {k: v for k, v in buckets.items() if v > 0}

    labels = list(buckets.keys())[::-1]
    counts = list(buckets.values())[::-1]
    total = len(ages)

    colors = sns.color_palette("Blues_d", len(labels))
    bars = ax_bar.barh(labels, counts, color=colors)

    ax_bar.set_title("Age Buckets", fontsize=12, fontweight="bold")
    ax_bar.set_xlabel("Number of Instances", fontsize=11)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        label_text = f" {count} ({pct:.0f}%)"
        ax_bar.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            fontsize=10,
        )

    max_count = max(counts) if counts else 1
    ax_bar.set_xlim(0, max_count * 1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Lifetime chart saved to {output_path}")


def main() -> None:
    """Parse arguments and generate the lifetime density chart."""
    parser = argparse.ArgumentParser(
        description="Generate registry instance lifetime density chart",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics-YYYY-MM-DD.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG",
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        logger.error(f"Metrics file not found: {args.metrics}")
        raise SystemExit(1)

    ages = _load_lifetime_data(args.metrics)

    if not ages:
        logger.error("No lifetime data available")
        raise SystemExit(1)

    _generate_chart(ages, args.output)


if __name__ == "__main__":
    main()
