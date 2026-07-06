"""Generate an instance-lifetime chart color-coded by compute platform.

Variant of generate_lifetime_chart.py. Reads the metrics JSON (which contains
per-instance lifetime + compute fields) and produces a PNG with three panels,
all split by compute platform (docker, kubernetes, ecs, ec2, podman, unknown):

  1. Age Distribution -- stacked histogram of instance ages, one colour per platform
  2. Age Spread       -- one boxplot per platform so lifetime spread is comparable
  3. Age Buckets      -- horizontal stacked bars per age bucket, split by platform

This answers "do managed-compute deployments (ecs/k8s) live longer than
single-shot docker installs?" directly from the lifetime distribution.
"""

import argparse
import json
import logging
import os
import statistics

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_style import apply_tufte_style, tufte_axes  # noqa: E402

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

CHART_TITLE: str = "AI Registry -- Instance Lifetime by Compute Platform"
FIGURE_WIDTH: int = 16
FIGURE_HEIGHT: int = 6
MAX_X_TICKS: int = 12
BOX_Y_TICK_STEP_DAYS: int = 5

# Stable platform ordering + colours so the legend is consistent across runs.
# Most-common managed platforms get distinct, high-contrast colours; the
# long-tail ("unknown", "podman", anything else) falls back to grey shades.
PLATFORM_ORDER: list[str] = ["docker", "kubernetes", "ecs", "ec2", "podman", "unknown"]
PLATFORM_COLORS: dict[str, str] = {
    "docker": "#1f77b4",      # blue
    "kubernetes": "#2ca02c",  # green
    "ecs": "#ff7f0e",         # orange
    "ec2": "#9467bd",         # purple
    "podman": "#8c564b",      # brown
    "unknown": "#9e9e9e",     # grey
}
FALLBACK_COLOR: str = "#cccccc"

AGE_BUCKETS: list[tuple[str, int, int]] = [
    ("0 days (single session)", 0, 0),
    ("1-2 days", 1, 2),
    ("3-5 days", 3, 5),
    ("6-10 days", 6, 10),
    ("11+ days", 11, 10**9),
]


def _choose_tick_step(
    max_age: int,
) -> int:
    """Pick a round tick step so the x-axis shows at most MAX_X_TICKS labels."""
    if max_age <= MAX_X_TICKS:
        return 1

    raw_step = max_age / MAX_X_TICKS
    nice_steps = [1, 2, 5, 10, 20, 25, 50, 100]
    for step in nice_steps:
        if step >= raw_step:
            return step
    return 200


def _load_lifetime_by_compute(
    metrics_path: str,
) -> dict[str, list[int]]:
    """Load instance ages grouped by compute platform from metrics JSON.

    Returns a dict mapping platform name -> list of ages (days). Platforms are
    ordered by PLATFORM_ORDER first, then any extras alphabetically.
    """
    with open(metrics_path) as f:
        data = json.load(f)

    lifetime_list = data.get("instance_lifetime", [])
    if not lifetime_list:
        logger.error("No instance_lifetime data in metrics JSON")
        return {}

    grouped: dict[str, list[int]] = {}
    for inst in lifetime_list:
        platform = (inst.get("compute") or "unknown").strip() or "unknown"
        grouped.setdefault(platform, []).append(inst["age_days"])

    # Order: known platforms first (in PLATFORM_ORDER), then extras alphabetically.
    extras = sorted(p for p in grouped if p not in PLATFORM_ORDER)
    ordered_names = [p for p in PLATFORM_ORDER if p in grouped] + extras
    ordered = {p: grouped[p] for p in ordered_names}

    total = sum(len(v) for v in ordered.values())
    summary = ", ".join(f"{p}={len(v)}" for p, v in ordered.items())
    logger.info(f"Loaded {total} instance ages across {len(ordered)} platforms: {summary}")
    return ordered


def _color_for(
    platform: str,
) -> str:
    """Return a stable colour for a platform name."""
    return PLATFORM_COLORS.get(platform, FALLBACK_COLOR)


def _plot_stacked_histogram(
    ax: plt.Axes,
    grouped: dict[str, list[int]],
    max_age: int,
) -> None:
    """Stacked histogram of ages, one colour per platform."""
    bin_edges = list(range(0, max_age + 2))
    platforms = list(grouped.keys())
    series = [grouped[p] for p in platforms]
    colors = [_color_for(p) for p in platforms]

    ax.hist(
        series,
        bins=bin_edges,
        stacked=True,
        color=colors,
        edgecolor="white",
        linewidth=0.3,
        align="left",
        label=platforms,
    )

    ax.set_xlabel("Instance Age (days)", fontsize=11)
    ax.set_ylabel("Number of Instances", fontsize=11)
    ax.set_title("Age Distribution (stacked by platform)", fontsize=12, fontweight="bold")

    tick_step = _choose_tick_step(max_age)
    ax.set_xticks(range(0, max_age + 1, tick_step))
    ax.legend(title="Compute", fontsize=8, title_fontsize=9, loc="upper right")


def _plot_per_platform_box(
    ax: plt.Axes,
    grouped: dict[str, list[int]],
) -> None:
    """One boxplot per platform so lifetime spread is directly comparable."""
    platforms = list(grouped.keys())
    # Build long-form data for seaborn: parallel lists of value + platform.
    values: list[int] = []
    cats: list[str] = []
    for p in platforms:
        for age in grouped[p]:
            values.append(age)
            cats.append(p)

    palette = {p: _color_for(p) for p in platforms}
    sns.boxplot(
        x=cats,
        y=values,
        order=platforms,
        hue=cats,
        hue_order=platforms,
        palette=palette,
        legend=False,
        ax=ax,
        width=0.6,
        fliersize=3,
    )

    ax.set_ylabel("Instance Age (days)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Age Spread by Platform", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30, labelsize=9)
    for _label in ax.get_xticklabels():
        _label.set_horizontalalignment("right")


def _plot_stacked_buckets(
    ax: plt.Axes,
    grouped: dict[str, list[int]],
) -> None:
    """Horizontal stacked bars: each age bucket split by platform."""
    platforms = list(grouped.keys())
    total = sum(len(v) for v in grouped.values())

    # bucket_counts[bucket_label] = {platform: count}
    bucket_labels = [b[0] for b in AGE_BUCKETS]
    per_bucket: dict[str, dict[str, int]] = {b: {p: 0 for p in platforms} for b in bucket_labels}
    bucket_totals: dict[str, int] = {b: 0 for b in bucket_labels}

    for platform, ages in grouped.items():
        for age in ages:
            for label, lo, hi in AGE_BUCKETS:
                if lo <= age <= hi:
                    per_bucket[label][platform] += 1
                    bucket_totals[label] += 1
                    break

    # Drop empty buckets, then reverse so the first bucket sits at the top.
    visible = [b for b in bucket_labels if bucket_totals[b] > 0][::-1]

    left = {b: 0 for b in visible}
    for platform in platforms:
        widths = [per_bucket[b][platform] for b in visible]
        ax.barh(
            visible,
            widths,
            left=[left[b] for b in visible],
            color=_color_for(platform),
            edgecolor="white",
            linewidth=0.3,
            label=platform,
        )
        for b, w in zip(visible, widths):
            left[b] += w

    ax.set_title("Age Buckets (stacked by platform)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Instances", fontsize=11)

    # Annotate each bar's grand total + percentage at the bar end.
    max_total = max((bucket_totals[b] for b in visible), default=1)
    for b in visible:
        count = bucket_totals[b]
        pct = count / total * 100 if total else 0
        ax.text(
            count + max_total * 0.01,
            b,
            f" {count} ({pct:.0f}%)",
            va="center",
            fontsize=10,
        )

    ax.set_xlim(0, max_total * 1.4)
    ax.legend(title="Compute", fontsize=8, title_fontsize=9, loc="lower right")


def _generate_density_chart(
    grouped: dict[str, list[int]],
    output_path: str,
) -> None:
    """Generate an overlaid KDE density plot, one curve per compute platform.

    Each platform's age distribution is drawn as its own density curve so the
    shapes are directly comparable: a curve that stays elevated further right
    means that platform's deployments live longer.
    """
    apply_tufte_style()

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 0.62, FIGURE_HEIGHT))

    all_ages = [a for ages in grouped.values() for a in ages]
    total = len(all_ages)

    fig.suptitle(
        f"{CHART_TITLE} (Density)\n({total} instances)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    for platform, ages in grouped.items():
        # KDE needs at least two distinct values to estimate a curve.
        if len(set(ages)) < 2:
            continue
        sns.kdeplot(
            ages,
            ax=ax,
            color=_color_for(platform),
            linewidth=2,
            bw_adjust=0.8,
            clip=(0, None),
            label=f"{platform} (n={len(ages)})",
        )

    ax.set_xlabel("Instance Age (days)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Age Density by Compute Platform", fontsize=12, fontweight="bold")
    ax.set_xlim(left=0)
    ax.legend(title="Compute", fontsize=9, title_fontsize=10, loc="upper right")

    tufte_axes(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Lifetime density-by-compute chart saved to {output_path}")


def _generate_box_chart(
    grouped: dict[str, list[int]],
    output_path: str,
) -> None:
    """Generate a standalone boxplot of instance age, one box per compute platform.

    Y-axis ticks are labelled every BOX_Y_TICK_STEP_DAYS days, and the legend
    calls out each platform's mean age so the reader can compare averages
    without eyeballing the boxes.
    """
    apply_tufte_style()

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 0.62, FIGURE_HEIGHT))

    all_ages = [a for ages in grouped.values() for a in ages]
    total = len(all_ages)
    max_age = max(all_ages) if all_ages else 0

    fig.suptitle(
        f"{CHART_TITLE} (Age Spread)\n({total} instances)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    platforms = list(grouped.keys())
    values: list[int] = []
    cats: list[str] = []
    for p in platforms:
        for age in grouped[p]:
            values.append(age)
            cats.append(p)

    palette = {p: _color_for(p) for p in platforms}
    sns.boxplot(
        x=cats,
        y=values,
        order=platforms,
        hue=cats,
        hue_order=platforms,
        palette=palette,
        legend=False,
        ax=ax,
        width=0.6,
        fliersize=3,
    )
    # Overlay individual points so dense clusters (e.g. the pile at 0) are visible.
    sns.stripplot(
        x=cats,
        y=values,
        order=platforms,
        color="#333333",
        size=2,
        alpha=0.25,
        jitter=0.2,
        ax=ax,
    )

    ax.set_ylabel("Instance Age (days)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Age Spread by Compute Platform", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)

    # Label y-ticks every BOX_Y_TICK_STEP_DAYS days.
    ax.yaxis.set_major_locator(mticker.MultipleLocator(BOX_Y_TICK_STEP_DAYS))
    ax.set_ylim(bottom=-1, top=max_age + 2)

    # Legend calls out both median (the box's centre line) and mean age per platform.
    # The median is 0 for most platforms because of the one-day-wonder cohort, so the
    # mean adds the differentiating signal that the long tails drive.
    legend_handles = []
    for p in platforms:
        ages = grouped[p]
        median_age = statistics.median(ages) if ages else 0
        mean_age = statistics.fmean(ages) if ages else 0
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=_color_for(p),
                markersize=10,
                label=f"{p}: median {median_age:.1f} d / mean {mean_age:.1f} d (n={len(ages)})",
            )
        )
    ax.legend(
        handles=legend_handles,
        title="Median / mean age by compute",
        fontsize=9,
        title_fontsize=10,
        loc="upper right",
    )

    tufte_axes(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Lifetime box-by-compute chart saved to {output_path}")


def _generate_chart(
    grouped: dict[str, list[int]],
    output_path: str,
) -> None:
    """Generate and save the compute-colored lifetime chart."""
    apply_tufte_style()

    all_ages = [a for ages in grouped.values() for a in ages]
    max_age = max(all_ages) if all_ages else 0
    total = len(all_ages)

    fig, (ax_hist, ax_box, ax_bar) = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": [3, 1.4, 2]},
    )

    fig.suptitle(
        f"{CHART_TITLE}\n({total} instances)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    _plot_stacked_histogram(ax_hist, grouped, max_age)
    _plot_per_platform_box(ax_box, grouped)
    _plot_stacked_buckets(ax_bar, grouped)

    for _ax in fig.axes:
        tufte_axes(_ax)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Lifetime-by-compute chart saved to {output_path}")


def main() -> None:
    """Parse arguments and generate the compute-colored lifetime chart."""
    parser = argparse.ArgumentParser(
        description="Generate registry instance lifetime chart colored by compute platform",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics-YYYY-MM-DD.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the three-panel output PNG",
    )
    parser.add_argument(
        "--density-output",
        help="Optional path to save a separate overlaid KDE density PNG (one curve per platform)",
    )
    parser.add_argument(
        "--box-output",
        help="Optional path to save a standalone boxplot PNG (one box per platform, "
        "y-ticks every 5 days, mean age per platform in the legend)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        logger.error(f"Metrics file not found: {args.metrics}")
        raise SystemExit(1)

    grouped = _load_lifetime_by_compute(args.metrics)

    if not grouped:
        logger.error("No lifetime data available")
        raise SystemExit(1)

    _generate_chart(grouped, args.output)

    if args.density_output:
        _generate_density_chart(grouped, args.density_output)

    if args.box_output:
        _generate_box_chart(grouped, args.box_output)


if __name__ == "__main__":
    main()
