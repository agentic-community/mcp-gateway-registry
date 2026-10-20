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

import sys as _sys

import matplotlib.pyplot as plt
import seaborn as sns

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_labels import cloud_label  # noqa: E402
from tufte_style import apply_tufte_style, tufte_axes  # noqa: E402

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
    include_extended: bool = False,
) -> dict[str, Counter]:
    """Compute value counts for each dimension based on unique instances.

    When include_extended is True, three additional facets are included:
    Registry Version, Embeddings Backend, and Cloud Detection Method. These
    are most useful for the "new installs that day" view since they reveal
    what the latest installs are reporting.
    """
    dimensions: dict[str, Counter] = {
        "Cloud Provider": Counter(),
        "Compute Platform": Counter(),
        "Storage Backend": Counter(),
        "Auth Provider": Counter(),
        "Architecture": Counter(),
        "Deployment Mode": Counter(),
    }
    if include_extended:
        dimensions["Registry Version"] = Counter()
        dimensions["Embeddings Backend"] = Counter()
        dimensions["Cloud Detection Method"] = Counter()

    for row in instances:
        cloud = cloud_label(row)
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

        if include_extended:
            version = (row.get("v") or "").strip() or "unknown"
            dimensions["Registry Version"][version] += 1

            ebk = (row.get("embeddings_backend_kind") or "").strip() or "unknown"
            dimensions["Embeddings Backend"][ebk] += 1

            cdm = (row.get("cloud_detection_method") or "").strip() or "unknown"
            dimensions["Cloud Detection Method"][cdm] += 1

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

    for bar, count in zip(bars, counts, strict=False):
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


def _filter_rows_new_on_date(
    rows: list[dict[str, str]],
    new_date: str,
    range_start: str | None = None,
) -> list[dict[str, str]]:
    """Restrict the row set to instances first-seen on (or in a window ending on) new_date.

    "First-seen" means the instance's earliest event in the dataset has a
    timestamp date matching the filter. When range_start is None, the
    filter matches a single day. When range_start is provided, the filter
    matches first-seen dates in [range_start, new_date] inclusive. This
    answers "what are the brand-new installs from that period reporting?"
    rather than "who is still active?".

    Returns ALL events for the surviving registry_ids so the latest-event
    logic in _get_latest_per_instance can still resolve fields that vary
    by event type (e.g. auth from startups, embeddings from heartbeats).
    """
    earliest_per_id: dict[str, str] = {}
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid:
            continue
        ts = (row.get("ts") or "")[:10]
        if not ts:
            continue
        prior = earliest_per_id.get(rid)
        if prior is None or ts < prior:
            earliest_per_id[rid] = ts

    if range_start:
        new_ids = {rid for rid, ts in earliest_per_id.items() if range_start <= ts <= new_date}
        label = f"first-seen in {range_start}..{new_date}"
    else:
        new_ids = {rid for rid, ts in earliest_per_id.items() if ts == new_date}
        label = f"first-seen on {new_date}"

    if not new_ids:
        logger.warning(
            f"No instances {label}; returning empty row set",
        )
        return []

    kept = [r for r in rows if (r.get("registry_id") or "").strip() in new_ids]
    logger.info(
        f"New {label}: {len(new_ids)} unique instances "
        f"(filtered {len(rows)} -> {len(kept)} events for those instances' full history)"
    )
    return kept


def _generate_chart(
    rows: list[dict[str, str]],
    output_path: str,
    active_on_date: str | None = None,
    new_on_date: str | None = None,
    new_range_start: str | None = None,
    include_extended_facets: bool = False,
) -> None:
    """Generate and save the faceted distribution chart.

    When active_on_date (YYYY-MM-DD) is provided, the chart shows only
    those instances that reported at least one event on that date. When
    new_on_date is provided, the chart shows only instances first-seen
    on that date (brand-new installs).

    When include_extended_facets is True, the chart adds Registry Version,
    Embeddings Backend, and Cloud Detection Method panels (9 total facets
    instead of 6). This is the default for the new-installs view since
    those facets are most diagnostic for fresh deployments.
    """
    instances = _get_latest_per_instance(rows)
    total = len(instances)

    distributions = _compute_distributions(instances, include_extended=include_extended_facets)

    apply_tufte_style()

    if include_extended_facets:
        fig, axes = plt.subplots(3, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT * 1.4))
        cols = 3
    else:
        fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        cols = 3

    if new_on_date and new_range_start:
        title_suffix = (
            f"\n(New installs {new_range_start} to {new_on_date}: {total} unique instances)"
        )
    elif new_on_date:
        title_suffix = f"\n(New installs on {new_on_date}: {total} unique instances)"
    elif active_on_date:
        title_suffix = f"\n(Active on {active_on_date}: {total} unique instances)"
    else:
        title_suffix = f"\n({total} unique instances)"
    fig.suptitle(
        f"{CHART_TITLE}{title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    dimension_order = [
        "Cloud Provider",
        "Compute Platform",
        "Storage Backend",
        "Auth Provider",
        "Architecture",
        "Deployment Mode",
    ]
    if include_extended_facets:
        dimension_order += [
            "Registry Version",
            "Embeddings Backend",
            "Cloud Detection Method",
        ]

    for idx, dim_name in enumerate(dimension_order):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx][col_idx]
        _plot_single_facet(ax, distributions[dim_name], dim_name, total)

    for _ax in fig.axes:
        tufte_axes(_ax)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    parser.add_argument(
        "--new-on-date",
        default=None,
        help=(
            "Optional YYYY-MM-DD. When provided, the chart is restricted to "
            "instances first-seen on that date (brand-new installs). Implies "
            "--include-extended-facets unless that flag is explicitly disabled. "
            "Mutually exclusive with --active-on-date."
        ),
    )
    parser.add_argument(
        "--new-range-start",
        default=None,
        help=(
            "Optional YYYY-MM-DD lower bound for --new-on-date. When provided, "
            "the new-installs filter matches first-seen dates in "
            "[--new-range-start, --new-on-date] inclusive. Useful for showing a "
            "7-day or 14-day window of new installs when single-day samples are "
            "too small."
        ),
    )
    parser.add_argument(
        "--include-extended-facets",
        action="store_true",
        help=(
            "When set, adds three additional facets (Registry Version, "
            "Embeddings Backend, Cloud Detection Method) for a 9-panel chart. "
            "Most useful for the new-installs view."
        ),
    )
    args = parser.parse_args()

    if args.active_on_date and args.new_on_date:
        logger.error("--active-on-date and --new-on-date are mutually exclusive")
        raise SystemExit(2)

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

    include_extended = args.include_extended_facets or bool(args.new_on_date)

    if args.new_range_start and not args.new_on_date:
        logger.error("--new-range-start requires --new-on-date")
        raise SystemExit(2)

    if args.new_on_date:
        rows = _filter_rows_new_on_date(
            rows,
            args.new_on_date,
            range_start=args.new_range_start,
        )
        if not rows:
            window = (
                f"{args.new_range_start}..{args.new_on_date}"
                if args.new_range_start
                else args.new_on_date
            )
            logger.error(f"No new instances in {window}")
            raise SystemExit(1)

    _generate_chart(
        rows,
        args.output,
        active_on_date=args.active_on_date,
        new_on_date=args.new_on_date,
        new_range_start=args.new_range_start,
        include_extended_facets=include_extended,
    )


if __name__ == "__main__":
    main()
