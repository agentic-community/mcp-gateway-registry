"""Plot the fleet auth-path mix and prove whether the fields are being collected.

Schema v6 (registry v1.31.0+) adds three fields to the telemetry *heartbeat*
payload: `auth_path_share_24h` (integer percentages per auth path, summing to
100), `auth_path_volume_bucket_24h` (`1-9`, `10-99`, `100-999`, `1k-9k`,
`10k+`), and `auth_path_window_hours` (0..48). All three are absent together
when the window carried no auth traffic.

This chart has two panels:

1. **Fleet auth-path mix** — horizontal bars, one per auth path, share
   percentages, sorted descending. The share is *volume-weighted*: each
   instance contributes its latest non-null share weighted by the midpoint of
   its volume bucket, so a registry serving 10k logins a day does not count the
   same as one serving three. When `--metrics` points at a
   `metrics-YYYY-MM-DD.json` whose `auth_path.fleet_share_pct` is populated,
   those numbers are used verbatim so the chart and the report table can never
   disagree; the CSV is only re-aggregated when that block is missing or empty.

2. **Collection coverage** — the collection funnel (total instances ->
   instances whose latest event reports `schema_version` 6 -> instances
   reporting a non-null auth-path mix) followed by the volume-bucket
   distribution, on one shared "instances" x-axis. This panel is the point of
   the chart: it answers "are the fields actually being emitted and stored?".

The empty state is the state that ships first. The deployed collector Lambda
predates schema v6, and an undeclared key is dropped rather than stored, so
zero rows carry these fields today. When nothing reports a mix, the figure is a
single plain-language note naming the two likely causes, with the
total-instance count so the reader knows the fleet size is known and only the
new fields are missing. The script never raises on missing or malformed data:
it always writes the PNG and exits 0.

CSV-only and column-tolerant: a historical export with none of the four new
columns produces the empty state, not a KeyError.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import sys as _sys

import matplotlib.pyplot as plt

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_style import (  # noqa: E402
    GRID_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    TEXT_COLOR,
    apply_tufte_style,
    tufte_axes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

CHART_TITLE: str = "AI Registry -- Auth Path Mix and Collection Coverage"
FIGURE_WIDTH: int = 14
FIGURE_HEIGHT: int = 8
EMPTY_FIGURE_HEIGHT: float = 5.0

SCHEMA_V6: str = "6"

# The registry clamps auth_path_window_hours to this ceiling before sending.
WINDOW_HOURS_MAX: int = 48

# The 13 auth paths the registry is allowed to report. Anything else is a
# client-side bug or a newer registry than this report knows about; drop it
# rather than plotting a label nobody can interpret.
AUTH_PATH_ALLOWLIST: frozenset[str] = frozenset(
    {
        "session_cookie",
        "self_signed",
        "jwt",
        "boto3",
        "federation-static",
        "network-trusted",
        "keycloak",
        "cognito",
        "entra",
        "okta",
        "auth0",
        "pingfederate",
        "unknown",
    }
)

# Bucket midpoints used to volume-weight each instance's share. The registry
# reports None (not "0") for a window with no traffic, so the collector's "0"
# bucket is unreachable; it is mapped anyway so a stray value weights nothing
# instead of crashing.
BUCKET_MIDPOINTS: dict[str, float] = {
    "0": 0.0,
    "1-9": 3.0,
    "10-99": 30.0,
    "100-999": 300.0,
    "1k-9k": 3000.0,
    "10k+": 10000.0,
}

# Display order for the bucket distribution: smallest deployment to largest.
BUCKET_ORDER: list[str] = ["1-9", "10-99", "100-999", "1k-9k", "10k+"]

# An instance that reports a share but no recognizable bucket still has a real
# mix; weight it as the smallest possible traffic rather than discarding it.
FALLBACK_WEIGHT: float = 1.0

COLOR_MIX_BAR: str = "#4878a8"
COLOR_MIX_UNKNOWN: str = "#9aa5ad"
COLOR_FUNNEL_TOTAL: str = "#cfd6db"
COLOR_FUNNEL_SCHEMA: str = "#8fa9bf"
COLOR_FUNNEL_REPORTING: str = "#2f6690"
COLOR_BUCKET: str = "#7f7f7f"

# Row count a panel is styled for. Fewer rows thin the bars rather than
# fattening them into slabs; more rows just fill the grid.
REFERENCE_ROWS: int = 8


class InstanceState:
    """Per-registry_id rollup of everything this chart needs.

    Tracks the latest event overall (for `schema_version`) separately from the
    latest event that actually carried an auth-path mix, because a registry
    reports the fields only in windows with traffic: the newest heartbeat may
    legitimately have no mix while an older one does.
    """

    __slots__ = ("latest_share_ts", "latest_ts", "schema_version", "share", "volume", "window")

    def __init__(self) -> None:
        self.latest_ts: str = ""
        self.schema_version: str = ""
        self.latest_share_ts: str = ""
        self.share: dict[str, float] | None = None
        self.volume: str = ""
        self.window: int | None = None


def _read_csv(
    path: str,
) -> tuple[list[dict[str, str]], list[str]]:
    """Load all rows plus the header from a single telemetry CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    logger.info(f"Read {len(rows)} rows from {path}")
    return rows, fieldnames


def _parse_share(
    raw: str | None,
) -> dict[str, float] | None:
    """Parse an `auth_path_share_24h` cell into a cleaned share mapping.

    Returns None for an empty cell, malformed JSON, a non-object payload, or an
    object with no usable entries. Keys outside the allowlist and non-positive
    or non-numeric values are dropped, so a partially bad payload still
    contributes the parts that make sense.
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        logger.debug(f"Unparseable auth_path_share_24h cell: {text[:80]!r}")
        return None
    if not isinstance(payload, dict):
        return None

    cleaned: dict[str, float] = {}
    for key, value in payload.items():
        if key not in AUTH_PATH_ALLOWLIST:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        try:
            pct = float(value)
        except (TypeError, ValueError):
            continue
        if pct <= 0:
            continue
        cleaned[key] = pct
    return cleaned or None


def _parse_window_hours(
    raw: str | None,
) -> int | None:
    """Parse `auth_path_window_hours`, tolerating blanks and junk.

    The registry clamps the window to 0..48 before sending it, so anything
    outside that range is corrupt and is treated as not reported rather than
    captioning the chart with a nonsense number of hours.
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        hours = int(float(text))
    except (TypeError, ValueError):
        return None
    if not 0 <= hours <= WINDOW_HOURS_MAX:
        logger.debug(f"Discarding out-of-range auth_path_window_hours: {text!r}")
        return None
    return hours


def _collect_instances(
    rows: list[dict[str, str]],
) -> dict[str, InstanceState]:
    """Roll every event up to one InstanceState per registry_id."""
    instances: dict[str, InstanceState] = {}
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        if not rid:
            continue
        state = instances.get(rid)
        if state is None:
            state = InstanceState()
            instances[rid] = state

        ts = row.get("ts") or ""
        if ts >= state.latest_ts:
            state.latest_ts = ts
            state.schema_version = (row.get("schema_version") or "").strip()

        share = _parse_share(row.get("auth_path_share_24h"))
        if share is not None and ts >= state.latest_share_ts:
            state.latest_share_ts = ts
            state.share = share
            state.volume = (row.get("auth_path_volume_bucket_24h") or "").strip()
            state.window = _parse_window_hours(row.get("auth_path_window_hours"))

    logger.info(f"Rolled {len(rows)} events up to {len(instances)} unique instances")
    return instances


def _instances_reporting_on(
    rows: list[dict[str, str]],
    date: str,
) -> int:
    """Count unique instances that reported a non-null mix on one YYYY-MM-DD."""
    seen: set[str] = set()
    for row in rows:
        if (row.get("ts") or "")[:10] != date:
            continue
        rid = (row.get("registry_id") or "").strip()
        if not rid or rid in seen:
            continue
        if _parse_share(row.get("auth_path_share_24h")) is not None:
            seen.add(rid)
    return len(seen)


def _weighted_fleet_share(
    instances: dict[str, InstanceState],
) -> dict[str, float]:
    """Volume-weighted fleet share per auth path, rounded to 1 dp.

    Each reporting instance contributes its latest non-null share, normalized
    to sum to 1 (so dropped unknown keys do not deflate it), scaled by the
    midpoint of its volume bucket. Sorted by descending share.
    """
    weighted: dict[str, float] = {}
    total_weight = 0.0
    for state in instances.values():
        if not state.share:
            continue
        share_sum = sum(state.share.values())
        if share_sum <= 0:
            continue
        weight = BUCKET_MIDPOINTS.get(state.volume, FALLBACK_WEIGHT)
        if weight <= 0:
            weight = FALLBACK_WEIGHT
        for path, pct in state.share.items():
            weighted[path] = weighted.get(path, 0.0) + weight * (pct / share_sum)
        total_weight += weight

    if total_weight <= 0:
        return {}
    shares = {path: round(w / total_weight * 100, 1) for path, w in weighted.items()}
    return dict(sorted(shares.items(), key=lambda kv: (-kv[1], kv[0])))


def _bucket_counts(
    instances: dict[str, InstanceState],
) -> dict[str, int]:
    """Instance count per volume bucket, among instances reporting a mix."""
    counts: Counter[str] = Counter()
    for state in instances.values():
        if not state.share:
            continue
        counts[state.volume or "(bucket absent)"] += 1
    return dict(counts)


def _window_summary(
    instances: dict[str, InstanceState],
) -> dict[str, float] | None:
    """min / median / max of the reported auth-path window, or None."""
    hours = [s.window for s in instances.values() if s.share and s.window is not None]
    if not hours:
        return None
    return {
        "min": min(hours),
        "median": round(statistics.median(hours), 1),
        "max": max(hours),
    }


def _load_metrics_block(
    metrics_path: str | None,
) -> dict:
    """Read the `auth_path` block out of a metrics-DATE.json, if usable.

    A missing file, unreadable JSON, or a missing/blank `auth_path` key all
    return an empty dict, which means "recompute from the CSV".
    """
    if not metrics_path:
        return {}
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics JSON not found, recomputing from CSV: {metrics_path}")
        return {}
    try:
        with open(metrics_path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning(f"Metrics JSON unreadable ({exc}), recomputing from CSV")
        return {}
    block = payload.get("auth_path")
    if not isinstance(block, dict):
        return {}
    return block


def _coerce_share_map(
    raw: object,
) -> dict[str, float]:
    """Coerce a metrics `fleet_share_pct` object into sorted float shares.

    A path rounded down to 0.0 in the metrics JSON is kept, not dropped: the
    report table lists it, so the chart lists it too.
    """
    if not isinstance(raw, dict):
        return {}
    shares: dict[str, float] = {}
    for key, value in raw.items():
        if key not in AUTH_PATH_ALLOWLIST:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        try:
            pct = float(value)
        except (TypeError, ValueError):
            continue
        if pct < 0:
            continue
        shares[key] = round(pct, 1)
    return dict(sorted(shares.items(), key=lambda kv: (-kv[1], kv[0])))


def _coerce_count_map(
    raw: object,
) -> dict[str, int]:
    """Coerce a metrics `volume_bucket_counts` object into int counts."""
    if not isinstance(raw, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in raw.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[str(key)] = count
    return counts


def _build_view(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    metrics_block: dict,
    yesterday: str | None,
) -> dict:
    """Assemble everything the two panels draw, from the CSV and metrics JSON.

    The metrics block wins whenever it carries a populated `fleet_share_pct`,
    so the chart matches the rendered report table exactly. Otherwise every
    number is re-aggregated from the CSV.
    """
    instances = _collect_instances(rows)
    total_instances = len(instances)
    reporting = {rid: s for rid, s in instances.items() if s.share}
    csv_view = {
        "fleet_share_pct": _weighted_fleet_share(instances),
        "reporting_instances": len(reporting),
        "schema_v6_instances": sum(1 for s in instances.values() if s.schema_version == SCHEMA_V6),
        "total_instances": total_instances,
        "volume_bucket_counts": _bucket_counts(instances),
        "window_hours": _window_summary(instances),
        "reporting_instances_yesterday": (
            _instances_reporting_on(rows, yesterday) if yesterday else 0
        ),
        "columns_present": [
            name
            for name in (
                "schema_version",
                "auth_path_share_24h",
                "auth_path_volume_bucket_24h",
                "auth_path_window_hours",
            )
            if name in fieldnames
        ],
        "yesterday": yesterday,
        "source": "csv",
    }

    metrics_share = _coerce_share_map(metrics_block.get("fleet_share_pct"))
    if not metrics_share:
        if metrics_block:
            logger.info("Metrics auth_path block has no fleet share; using CSV aggregation")
        return csv_view

    logger.info(f"Using fleet share from metrics JSON ({len(metrics_share)} paths)")
    view = dict(csv_view)
    view["source"] = "metrics"
    view["fleet_share_pct"] = metrics_share
    for key in (
        "reporting_instances",
        "schema_v6_instances",
        "total_instances",
        "reporting_instances_yesterday",
    ):
        value = metrics_block.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            view[key] = value
    buckets = _coerce_count_map(metrics_block.get("volume_bucket_counts"))
    if buckets:
        view["volume_bucket_counts"] = buckets
    window = metrics_block.get("window_hours")
    if isinstance(window, dict) and window:
        view["window_hours"] = window
    return view


def _window_caption(
    window: dict | None,
) -> str:
    """One-line description of the reported measurement window."""
    if not window:
        return "window not reported"
    low = window.get("min")
    median = window.get("median")
    high = window.get("max")
    if median is None:
        return "window not reported"
    if low == high:
        return f"window {median}h"
    return f"window {low}-{high}h (median {median}h)"


def _pin_row_pitch(
    ax: plt.Axes,
    lowest: float,
    highest: float,
) -> None:
    """Fix the y-limits so one bar row always occupies the same slice of height.

    Without this, matplotlib fits the y-limits to the bars it was given, so a
    panel with one row draws a slab spanning the whole axes and a panel with
    twelve draws hairlines. Padding short panels out to REFERENCE_ROWS, centred,
    keeps bar weight identical across panels and across snapshots as adoption
    grows.
    """
    rows = highest - lowest + 1.0
    pad = max(0.0, REFERENCE_ROWS - rows) / 2
    ax.set_ylim(lowest - 0.5 - pad, highest + 0.5 + pad)


def _plot_mix_panel(
    ax: plt.Axes,
    view: dict,
) -> None:
    """Panel 1: volume-weighted fleet auth-path mix as horizontal bars."""
    shares: dict[str, float] = view["fleet_share_pct"]
    # barh draws bottom-up, so reverse to put the largest share on top.
    labels = list(reversed(list(shares.keys())))
    values = [shares[label] for label in labels]
    colors = [COLOR_MIX_UNKNOWN if label == "unknown" else COLOR_MIX_BAR for label in labels]

    ax.barh(labels, values, color=colors, height=0.62, edgecolor="none")

    span = max(values) if values else 1.0
    for label, value in zip(labels, values, strict=False):
        # A path can be real yet round to 0.0; say so rather than printing "0.0%"
        # next to an invisible bar.
        annotation = "<0.1%" if value < 0.05 else f"{value:.1f}%"
        ax.text(
            value + span * 0.015,
            label,
            annotation,
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )

    source = "from metrics JSON" if view["source"] == "metrics" else "aggregated from CSV"
    ax.set_title(
        f"Auth path share of authenticated requests\n"
        f"volume-weighted across {view['reporting_instances']} reporting "
        f"instances, {source}",
        fontsize=11,
        loc="left",
    )
    ax.set_xlabel("Share of authenticated requests (%)")
    ax.set_xlim(0, span * 1.14)
    _pin_row_pitch(ax, 0.0, float(len(labels) - 1))
    tufte_axes(ax, grid="x")


def _plot_coverage_panel(
    ax: plt.Axes,
    view: dict,
) -> None:
    """Panel 2: collection funnel plus the volume-bucket distribution.

    Both groups share one x-axis in instances, which is honest: every bucket
    count is a subset of the reporting-instances bar directly above it.
    """
    total = view["total_instances"]
    schema_v6 = view["schema_v6_instances"]
    reporting = view["reporting_instances"]
    buckets: dict[str, int] = view["volume_bucket_counts"]

    def pct_of_total(value: int) -> str:
        if total <= 0:
            return ""
        return f" ({value / total * 100:.1f}% of fleet)"

    # Funnel steps, widest first. Each step is a subset of the one above it, so
    # a reader can see exactly where the fields stop making it through.
    steps: list[tuple[str, int, str, str]] = [
        ("instances in fleet", total, COLOR_FUNNEL_TOTAL, f"{total:,}"),
        (
            "reporting schema_version 6",
            schema_v6,
            COLOR_FUNNEL_SCHEMA,
            f"{schema_v6:,}{pct_of_total(schema_v6)}",
        ),
        (
            "reporting an auth-path mix",
            reporting,
            COLOR_FUNNEL_REPORTING,
            f"{reporting:,}{pct_of_total(reporting)}",
        ),
    ]
    yesterday = view["yesterday"]
    if yesterday:
        fresh = view["reporting_instances_yesterday"]
        steps.append(
            (
                f"...still reporting on {yesterday}",
                fresh,
                COLOR_FUNNEL_REPORTING,
                f"{fresh:,}{pct_of_total(fresh)}",
            )
        )

    # (y position, label, value, color, annotation); barh draws bottom-up, so
    # the first funnel step gets the highest y.
    funnel_top = float(len(steps) - 1)
    entries: list[tuple[float, str, int, str, str]] = [
        (funnel_top - idx, label, value, color, annotation)
        for idx, (label, value, color, annotation) in enumerate(steps)
    ]

    ordered_buckets = [b for b in BUCKET_ORDER if buckets.get(b)]
    ordered_buckets += sorted(b for b in buckets if b not in BUCKET_ORDER and buckets[b])
    # Buckets sit below the funnel, largest bucket highest, after a gap row.
    bucket_base = -1.2
    for idx, bucket in enumerate(reversed(ordered_buckets)):
        count = buckets[bucket]
        entries.append((bucket_base - idx, bucket, count, COLOR_BUCKET, f"{count:,}"))

    positions = [e[0] for e in entries]
    ax.barh(
        positions,
        [e[2] for e in entries],
        color=[e[3] for e in entries],
        height=0.6,
        edgecolor="none",
    )

    span = max(e[2] for e in entries) or 1
    for y, _, value, _color, annotation in entries:
        ax.text(
            value + span * 0.015,
            y,
            annotation,
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([e[1] for e in entries])
    ax.set_xlim(0, span * 1.22)
    ax.set_xlabel("Instances")
    _pin_row_pitch(ax, min(positions), max(positions))

    group_label_x = span * 1.20
    ax.text(
        group_label_x,
        funnel_top + 0.55,
        "collection funnel",
        ha="right",
        va="center",
        fontsize=9,
        color=SECONDARY_COLOR,
        style="italic",
    )
    if ordered_buckets:
        ax.axhline(-0.55, color=GRID_COLOR, linewidth=0.8)
        ax.text(
            group_label_x,
            -0.78,
            f"24h auth volume per reporting instance -- {_window_caption(view['window_hours'])}",
            ha="right",
            va="center",
            fontsize=9,
            color=SECONDARY_COLOR,
            style="italic",
        )

    coverage = (reporting / total * 100) if total > 0 else 0.0
    ax.set_title(
        f"Is the telemetry actually landing?\n"
        f"{reporting:,} of {total:,} instances report a mix ({coverage:.1f}% coverage)",
        fontsize=11,
        loc="left",
    )
    tufte_axes(ax, grid="x")


def _plot_empty_state(
    fig: plt.Figure,
    view: dict,
) -> None:
    """Draw the not-collected-yet figure: one axes, a note, and the fleet size.

    This is what ships until both halves of the pipeline are current, so it has
    to say plainly which half is behind rather than looking like a broken chart.
    """
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    tufte_axes(ax, grid="none")

    missing = [
        name
        for name in (
            "schema_version",
            "auth_path_share_24h",
            "auth_path_volume_bucket_24h",
            "auth_path_window_hours",
        )
        if name not in view["columns_present"]
    ]
    if missing:
        column_note = (
            f"Export is missing {len(missing)} of the 4 auth-path columns, "
            "so the bastion export predates the schema too."
        )
    else:
        column_note = "All 4 auth-path columns are exported, every value is empty."

    # Say which cause the data actually points at. A fleet with zero schema v6
    # instances has not shipped the release; a fleet with some means the
    # emitting side is live and the loss is downstream of it.
    if view["schema_v6_instances"] > 0:
        verdict = (
            f"{view['schema_v6_instances']:,} instances already report schema_version 6, "
            "so cause 2 is the live suspect."
        )
    else:
        verdict = "No instance reports schema_version 6 yet, so cause 1 is sufficient on its own."

    lines = [
        "No auth-path telemetry collected yet.",
        "",
        f"{view['total_instances']:,} instances reported telemetry in this window;"
        f" {view['schema_v6_instances']:,} of them report schema_version 6.",
        "None of them carry an auth_path_share_24h value.",
        "",
        "Two things explain this, and they are not mutually exclusive:",
        "",
        "1. No registry is running the release that emits the fields yet.",
        "    Only schema v6 heartbeats carry them; startup events never do,",
        "    and a window with no auth traffic reports nothing at all.",
        "",
        "2. The deployed collector Lambda predates the schema.",
        "    It does not declare the three fields, so it drops them on ingest",
        "    and the rows in DocumentDB have nothing to export.",
        "",
        verdict,
        column_note,
    ]

    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.6,
        color=PRIMARY_COLOR,
    )
    logger.info("No auth-path data present; drew the not-collected-yet state")


def _plot_chart(
    view: dict,
    output_path: str,
    snapshot_date: str | None,
) -> None:
    """Render the figure (populated or empty state) and save the PNG."""
    apply_tufte_style()

    title = CHART_TITLE
    if snapshot_date:
        title += f"\nSnapshot through {snapshot_date}"

    if not view["fleet_share_pct"] or view["reporting_instances"] <= 0:
        # The note needs a fraction of the height two panels of bars need;
        # a full-height figure would embed a mostly blank image in the report.
        fig = plt.figure(figsize=(FIGURE_WIDTH, EMPTY_FIGURE_HEIGHT))
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
        _plot_empty_state(fig, view)
    else:
        fig, (ax_mix, ax_coverage) = plt.subplots(
            1,
            2,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            gridspec_kw={"width_ratios": [1.15, 1.0]},
        )
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
        _plot_mix_panel(ax_mix, view)
        _plot_coverage_panel(ax_coverage, view)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Auth-path chart saved to {output_path}")


def main() -> None:
    """CLI entry: generate the auth-path mix + collection-coverage chart."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot the volume-weighted fleet auth-path mix and the collection "
            "coverage of the schema v6 auth-path fields."
        ),
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to a single registry_metrics.csv (dated snapshot)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Optional metrics-YYYY-MM-DD.json. When its auth_path.fleet_share_pct "
            "is populated those numbers are plotted verbatim, so the chart and the "
            "report table cannot disagree."
        ),
    )
    parser.add_argument(
        "--yesterday",
        default=None,
        help="YYYY-MM-DD of the previous complete day, used to label the snapshot",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if not os.path.exists(args.csv):
        logger.warning(f"CSV not found, drawing the empty state: {args.csv}")
    else:
        try:
            rows, fieldnames = _read_csv(args.csv)
        except (OSError, csv.Error) as exc:
            logger.warning(f"CSV unreadable ({exc}), drawing the empty state")

    view = _build_view(
        rows=rows,
        fieldnames=fieldnames,
        metrics_block=_load_metrics_block(args.metrics),
        yesterday=args.yesterday,
    )
    logger.info(
        f"total={view['total_instances']} schema_v6={view['schema_v6_instances']} "
        f"reporting={view['reporting_instances']} "
        f"yesterday={view['reporting_instances_yesterday']} "
        f"paths={len(view['fleet_share_pct'])} source={view['source']}"
    )

    _plot_chart(view=view, output_path=args.output, snapshot_date=args.yesterday)


if __name__ == "__main__":
    main()
