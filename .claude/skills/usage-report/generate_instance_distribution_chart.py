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

    NOTE: Heartbeat events do not carry the `auth` field (the heartbeat
    schema only emits runtime metrics, not deployment shape for that
    column). For the latest-per-instance row we therefore copy the auth
    value forward from the most recent event that did populate it -- which
    is almost always a startup event. Without this patch any instance
    whose latest event is a heartbeat would get auth="" and end up
    mislabeled as "no auth provider".
    """
    rows_sorted = sorted(rows, key=lambda r: r.get("ts", ""))

    instance_latest: dict[str, dict[str, str]] = {}
    instance_latest_auth: dict[str, str] = {}
    skipped = 0
    for row in rows_sorted:
        rid = (row.get("registry_id") or "").strip()
        if not rid:
            skipped += 1
            continue

        # Track latest event in general (for cloud/compute/etc which heartbeat
        # events DO carry).
        existing = instance_latest.get(rid)
        if existing is None or row.get("ts", "") > existing.get("ts", ""):
            instance_latest[rid] = row

        # Separately track the most recent non-empty auth value, since
        # heartbeats leave it empty.
        auth = (row.get("auth") or "").strip()
        if auth:
            instance_latest_auth[rid] = auth

    # Stitch the latest auth back onto the latest-event row so downstream
    # code can keep using a single row per instance.
    for rid, latest in instance_latest.items():
        if rid in instance_latest_auth:
            latest = dict(latest)
            latest["auth"] = instance_latest_auth[rid]
            instance_latest[rid] = latest

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

        # auth comes from the most recent startup event for the instance;
        # see _get_latest_per_instance. If it is genuinely missing (e.g.
        # an instance that only ever sent heartbeats during the window)
        # bucket it as "unknown" rather than "none" so it does not get
        # confused with deployments that explicitly set auth_provider=none.
        auth = (row.get("auth") or "").strip() or "unknown"
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


def _filter_rows_active_on_date(
    rows: list[dict[str, str]],
    active_date: str,
) -> list[dict[str, str]]:
    """Restrict the row set to instances that had an event on active_date.

    Returns ALL events for the surviving registry_ids (not just the ones on
    that day) so the latest-per-instance and auth-stitching logic in
    _get_latest_per_instance still work. Without this, an instance that
    last sent a startup event two weeks ago and only sent heartbeats on
    active_date would lose its auth value.
    """
    active_ids: set[str] = set()
    for row in rows:
        if (row.get("ts") or "")[:10] != active_date:
            continue
        rid = (row.get("registry_id") or "").strip()
        if rid:
            active_ids.add(rid)

    if not active_ids:
        logger.warning(
            f"No instances reported on {active_date}; returning empty row set",
        )
        return []

    kept = [r for r in rows if (r.get("registry_id") or "").strip() in active_ids]
    logger.info(
        f"Active on {active_date}: {len(active_ids)} unique instances "
        f"(filtered {len(rows)} -> {len(kept)} events for those instances' full history)"
    )
    return kept


def _generate_chart(
    rows: list[dict[str, str]],
    output_path: str,
    active_on_date: str | None = None,
) -> None:
    """Generate and save the faceted distribution chart.

    When active_on_date (YYYY-MM-DD) is provided, the chart shows only
    those instances that reported at least one event on that date. The
    title is annotated to make this explicit.
    """
    instances = _get_latest_per_instance(rows)
    total = len(instances)

    distributions = _compute_distributions(instances)

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    if active_on_date:
        title_suffix = f"\n(Active on {active_on_date}: {total} unique instances)"
    else:
        title_suffix = f"\n({total} unique instances)"
    fig.suptitle(
        f"{CHART_TITLE}{title_suffix}",
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
    parser.add_argument(
        "--active-on-date",
        default=None,
        help=(
            "Optional YYYY-MM-DD. When provided, the chart is restricted to "
            "instances that had at least one event on that date. Use the last "
            "complete day (typically today - 1) to get a 'who is currently "
            "deployed' snapshot rather than 'who has ever been deployed'."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        raise SystemExit(1)

    rows = _read_csv(args.csv)

    if not rows:
        logger.error("No data in CSV file")
        raise SystemExit(1)

    if args.active_on_date:
        rows = _filter_rows_active_on_date(rows, args.active_on_date)
        if not rows:
            logger.error(f"No instances active on {args.active_on_date}")
            raise SystemExit(1)

    _generate_chart(rows, args.output, active_on_date=args.active_on_date)


if __name__ == "__main__":
    main()
