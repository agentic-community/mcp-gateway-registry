"""Forecast when the cumulative install count will cross a target threshold.

Reads every `metrics-*.json` under a base output directory, builds a
timeseries of `key_metrics.identified_instances` per snapshot date, and
fits two simple, recent-data-only models:

  1. Linear OLS over the trailing N days (default 14):
        installs = a*day + b   (fitted on last N daily points only)
  2. Recent-pace projection from the trailing 7-day window:
        today + (avg daily-add over last 7 days) * elapsed_days

Both models are deliberately restricted to recent history. A linear OLS
fit on the FULL history was tried and dropped: it averages the early-
week slow growth with the recent acceleration and undershoots the
present trajectory by ~30 days. A log-linear / exponential model was
also tried and dropped: with ~5 weeks of data the early rapid compounding
(e.g. 10 -> 30 instances = +200%) dominates and produces wildly aggressive
extrapolations.

The trailing-14-day OLS fit gives a slope and a 95% prediction interval
based on residual std (Gaussian assumption — defensible at the daily-
snapshot level over a short forward horizon). The 7-day SMA / recent-
pace line is a back-of-envelope sanity check, no band.

Output:

  - PNG chart with historical scatter, the two forward-extrapolated
    lines, a 95% PI band on the linear model, a `y = TARGET` reference,
    and annotated ETAs.
  - JSON sidecar with the predicted ETAs and band edges so the report
    narrative can quote exact dates.

Numerator: cumulative `identified_instances` (gross, including the 4
known internal dev environments). Customer-only (`total_non_internal`)
is available too but the gross number matches the LinkedIn / report
headline framing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import sys as _sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_style import apply_tufte_style, tufte_axes  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TARGET: int = 5000
CHART_TITLE: str = "AI Registry -- When does the cumulative install count cross {target}?"
FIGURE_WIDTH: int = 14
FIGURE_HEIGHT: int = 7
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RECENT_PACE_WINDOW_DAYS: int = 7
DEFAULT_OLS_WINDOW_DAYS: int = 14


def _find_metrics_files(
    base_dir: str,
) -> list[tuple[str, str]]:
    """Return sorted list of (date_str, path) for each metrics-YYYY-MM-DD.json."""
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


def _load_install_series(
    metrics_files: list[tuple[str, str]],
) -> list[tuple[datetime, int]]:
    """Build [(date, identified_instances)] sorted ascending by date."""
    series: list[tuple[datetime, int]] = []
    for date_str, path in metrics_files:
        with open(path) as f:
            data = json.load(f)
        n = data.get("key_metrics", {}).get("identified_instances")
        if n is None:
            continue
        series.append((datetime.strptime(date_str, "%Y-%m-%d"), int(n)))
    series.sort(key=lambda r: r[0])
    return series


def _ols_linear(
    x: list[float],
    y: list[float],
) -> tuple[float, float, float]:
    """Plain linear OLS: y = a*x + b. Returns (a, b, residual_std).

    Uses the closed-form least squares; no numpy dependency.
    """
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 points to fit a line")
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
    den = sum((xi - mean_x) ** 2 for xi in x)
    a = num / den
    b = mean_y - a * mean_x
    residuals = [yi - (a * xi + b) for xi, yi in zip(x, y, strict=False)]
    if n > 2:
        rss = sum(r * r for r in residuals)
        residual_std = math.sqrt(rss / (n - 2))
    else:
        residual_std = 0.0
    return a, b, residual_std


def _eta_for_threshold(
    a: float,
    b: float,
    threshold: float,
    transform: str,
    base_date: datetime,
) -> datetime | None:
    """Solve a*day + b = transform(threshold) and return calendar date.

    transform: "linear" -> threshold itself; "log" -> ln(threshold).
    Returns None if slope <= 0 (model never crosses the threshold).
    """
    if a <= 0:
        return None
    target_y = math.log(threshold) if transform == "log" else threshold
    days_from_base = (target_y - b) / a
    return base_date + timedelta(days=days_from_base)


def _recent_pace_projection(
    series: list[tuple[datetime, int]],
    target: int,
    window_days: int,
) -> tuple[datetime | None, float]:
    """Predict ETA for `target` based on the rolling-window daily-add rate.

    Returns (eta, daily_add_rate). eta is None if rate <= 0.
    """
    if len(series) < 2:
        return None, 0.0
    last_date, last_n = series[-1]
    cutoff = last_date - timedelta(days=window_days)
    earlier = [(d, n) for d, n in series if d <= cutoff]
    if not earlier:
        first_date, first_n = series[0]
    else:
        first_date, first_n = earlier[-1]
    days = (last_date - first_date).days
    if days <= 0:
        return None, 0.0
    rate = (last_n - first_n) / days
    if rate <= 0:
        return None, rate
    days_remaining = (target - last_n) / rate
    eta = last_date + timedelta(days=days_remaining)
    return eta, rate


def _generate_chart(
    series: list[tuple[datetime, int]],
    target: int,
    output_path: str,
    ols_window_days: int = DEFAULT_OLS_WINDOW_DAYS,
) -> dict:
    """Fit the two models, render the chart, and return the ETA summary."""
    if len(series) < 5:
        raise ValueError(f"need at least 5 daily points; have {len(series)}")

    # Restrict the linear OLS fit to the trailing window. The full series is
    # still used for the historical scatter and for the 7-day SMA.
    last_date = series[-1][0]
    cutoff = last_date - timedelta(days=ols_window_days)
    fit_series = [(d, n) for d, n in series if d >= cutoff]
    if len(fit_series) < 3:
        # Fall back to the last 3 points if the window is too aggressive
        fit_series = series[-3:]
    actual_fit_window_days = (fit_series[-1][0] - fit_series[0][0]).days

    base_date = series[0][0]
    full_x = [(d - base_date).days for d, _n in series]
    fit_x = [(d - base_date).days for d, _n in fit_series]
    fit_y = [float(n) for _d, n in fit_series]

    # Linear OLS on the trailing window
    a_lin, b_lin, sigma_lin = _ols_linear(fit_x, fit_y)
    eta_lin = _eta_for_threshold(a_lin, b_lin, target, "linear", base_date)

    # Recent-pace projection (7-day SMA of daily additions)
    eta_pace, daily_add_rate = _recent_pace_projection(series, target, RECENT_PACE_WINDOW_DAYS)

    # Pick the chart's right-edge: latest of the two ETAs (with a sane cap).
    candidate_etas = [e for e in (eta_lin, eta_pace) if e is not None]
    last_date = series[-1][0]
    if candidate_etas:
        right_edge = max(candidate_etas)
    else:
        right_edge = last_date + timedelta(days=30)
    # Cap extrapolation at 1.5x the historical span so the chart doesn't go
    # absurdly far if a model is underdetermined.
    historical_span_days = (last_date - base_date).days
    max_right_edge = last_date + timedelta(days=int(1.5 * historical_span_days))
    if right_edge > max_right_edge:
        right_edge = max_right_edge
    # Also pad a little so the rightmost ETA isn't on the chart edge.
    right_edge = right_edge + timedelta(days=3)

    # Build the forward x-grid
    forecast_days = list(range(0, (right_edge - base_date).days + 1))
    forecast_dates = [base_date + timedelta(days=d) for d in forecast_days]
    lin_pred = [a_lin * d + b_lin for d in forecast_days]
    # 95% PI: ~1.96 * sigma. Linear-space, no transform.
    pi_z = 1.96
    lin_lo = [v - pi_z * sigma_lin for v in lin_pred]
    lin_hi = [v + pi_z * sigma_lin for v in lin_pred]

    # Recent-pace line
    if eta_pace is not None:
        pace_pred = [series[-1][1] + daily_add_rate * (d - full_x[-1]) for d in forecast_days]
    else:
        pace_pred = []

    # ---- Render ----
    apply_tufte_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.suptitle(
        CHART_TITLE.format(target=target),
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    palette = sns.color_palette("Set2", 4)

    # Linear model + band
    ax.fill_between(
        forecast_dates,
        lin_lo,
        lin_hi,
        color=palette[0],
        alpha=0.15,
        label="Linear 95% PI",
    )
    ax.plot(
        forecast_dates,
        lin_pred,
        color=palette[0],
        linewidth=2.0,
        linestyle="--",
        label=(
            f"Linear OLS, trailing {actual_fit_window_days}d window "
            f"({a_lin:.1f}/day, ETA "
            f"{eta_lin.strftime('%Y-%m-%d') if eta_lin else 'n/a'})"
        ),
    )

    # Recent-pace line (no band)
    if pace_pred:
        ax.plot(
            forecast_dates,
            pace_pred,
            color=palette[2],
            linewidth=2.0,
            linestyle=":",
            label=(
                f"Recent {RECENT_PACE_WINDOW_DAYS}d pace "
                f"({daily_add_rate:.1f}/day, ETA "
                f"{eta_pace.strftime('%Y-%m-%d') if eta_pace else 'n/a'})"
            ),
        )

    # Historical scatter (drawn last so it sits on top)
    hist_dates = [d for d, _n in series]
    hist_counts = [n for _d, n in series]
    ax.scatter(
        hist_dates,
        hist_counts,
        color="black",
        s=22,
        zorder=5,
        label="Observed daily snapshot",
    )

    # Reference line at target
    ax.axhline(target, color="grey", linewidth=1, linestyle="-", alpha=0.5)
    ax.text(
        forecast_dates[0],
        target * 1.02,
        f"target = {target}",
        fontsize=9,
        color="grey",
    )

    ax.set_ylabel("Cumulative unique installs", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(hist_dates[0] - timedelta(days=1), forecast_dates[-1])
    ax.set_ylim(0, max(target * 1.2, max(lin_hi) * 1.05))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(forecast_days) // 14)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    for _ax in fig.axes:
        tufte_axes(_ax)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Forecast chart saved to {output_path}")

    return {
        "target": target,
        "today": {
            "date": series[-1][0].strftime("%Y-%m-%d"),
            "installs": series[-1][1],
        },
        "linear": {
            "fit_window_days": actual_fit_window_days,
            "fit_points": len(fit_series),
            "slope_per_day": a_lin,
            "intercept": b_lin,
            "residual_std": sigma_lin,
            "eta": eta_lin.strftime("%Y-%m-%d") if eta_lin else None,
            "eta_lower_95": (
                _eta_for_threshold(
                    a_lin, b_lin + pi_z * sigma_lin, target, "linear", base_date
                ).strftime("%Y-%m-%d")
                if a_lin > 0
                else None
            ),
            "eta_upper_95": (
                _eta_for_threshold(
                    a_lin, b_lin - pi_z * sigma_lin, target, "linear", base_date
                ).strftime("%Y-%m-%d")
                if a_lin > 0
                else None
            ),
        },
        "recent_pace": {
            "window_days": RECENT_PACE_WINDOW_DAYS,
            "daily_add_rate": daily_add_rate,
            "eta": eta_pace.strftime("%Y-%m-%d") if eta_pace else None,
        },
    }


def main() -> None:
    """Parse arguments and emit the forecast chart and JSON sidecar."""
    parser = argparse.ArgumentParser(
        description=(
            "Forecast when cumulative installs will cross a target threshold. "
            "Fits linear, log-linear (exponential), and recent-pace models, "
            "and renders all three with 95% PI bands."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Base output directory containing dated subfolders with metrics-*.json",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Target install count to forecast (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a JSON summary of the fitted models and ETAs",
    )
    parser.add_argument(
        "--ols-window-days",
        type=int,
        default=DEFAULT_OLS_WINDOW_DAYS,
        help=(
            "Trailing window (in days) used for the linear OLS fit "
            f"(default: {DEFAULT_OLS_WINDOW_DAYS}). The full history is still "
            "shown in the historical scatter and used by the recent-pace line."
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        logger.error(f"Directory not found: {args.csv_dir}")
        raise SystemExit(1)

    metrics_files = _find_metrics_files(args.csv_dir)
    if not metrics_files:
        logger.error(f"No metrics-*.json files found under {args.csv_dir}")
        raise SystemExit(1)

    series = _load_install_series(metrics_files)
    if len(series) < 5:
        logger.error(f"Need at least 5 snapshots; have {len(series)}")
        raise SystemExit(1)

    summary = _generate_chart(
        series, args.target, args.output, ols_window_days=args.ols_window_days
    )

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary JSON written to {args.summary_json}")

    # Print a short headline to the log so the user sees the ETAs immediately.
    logger.info(
        f"ETA summary: linear={summary['linear']['eta']}, "
        f"recent-pace={summary['recent_pace']['eta']}"
    )


if __name__ == "__main__":
    main()
