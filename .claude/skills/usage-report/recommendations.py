"""Rule-based Recommendations section for the usage report.

Each rule examines the structured `vars_` dict from render_report._build_template_vars
and emits a numbered bullet when its trigger fires. No LLM in the loop.

Adding a new rule: define a function `rule_<name>(vars_) -> str | None` and
register it in RULES below. Rules return None when their trigger doesn't fire.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# ============================================================================
# Helper utilities
# ============================================================================


def _load_recent_forecast_history(
    search_dir: str,
    current_date: str,
    n: int = 4,
) -> list[dict]:
    """Load the last N install-forecast JSON files (excluding current)."""
    import re
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    candidates = []
    for entry in sorted(os.listdir(search_dir), reverse=True):
        full = os.path.join(search_dir, entry)
        if not os.path.isdir(full) or not date_re.match(entry):
            continue
        if entry >= current_date:
            continue
        path = os.path.join(full, f"install-forecast-{entry}.json")
        if os.path.exists(path):
            candidates.append((entry, path))
        if len(candidates) >= n:
            break
    out = []
    for date_str, path in candidates:
        try:
            with open(path) as f:
                d = json.load(f)
                d["_snapshot_date"] = date_str
                out.append(d)
        except Exception:
            pass
    return out


def _crossed_multiple(
    current: int,
    previous: int,
    multiple: int,
) -> bool:
    """True iff `current` is at or above a multiple of `multiple` and `previous` was below."""
    if previous >= current or multiple <= 0:
        return False
    last_threshold = (previous // multiple + 1) * multiple
    return current >= last_threshold > previous


def _is_new_high(
    current: int | float,
    historical_max: int | float,
) -> bool:
    """True iff current strictly exceeds historical max."""
    return current > historical_max


# ============================================================================
# Rules
# ============================================================================


def rule_milestone_install_count(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Trigger when total instances crosses a 100-multiple."""
    cur = vars_.get("total_instances", 0)
    prev = vars_.get("prev_total_instances", 0)
    if not isinstance(cur, int) or not isinstance(prev, int):
        return None
    for milestone in (1000, 900, 800, 700, 600, 500):
        if cur >= milestone > prev:
            return (
                f"**Crossed {milestone} unique registry installs.** "
                f"The fleet went from {prev} to {cur} since the previous report. "
                f"Worth marking as a milestone in any external comms."
            )
    return None


def rule_confirmed_alive_milestone(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Trigger when Confirmed Alive crosses a 10-multiple or hits a new high."""
    cur = vars_.get("confirmed_alive", 0)
    prev = vars_.get("prev_confirmed_alive", 0)
    if not isinstance(cur, int) or not isinstance(prev, int):
        return None
    if _crossed_multiple(cur, prev, 10):
        return (
            f"**Confirmed Alive crossed {(cur // 10) * 10}.** "
            f"From {prev} to {cur}. This is the leading revenue-countable indicator: "
            f"registries that have heartbeat at least 5 times in the last 7 days."
        )
    if cur > prev and cur >= 50:
        return (
            f"**Confirmed Alive at {cur} (was {prev}).** New high in the durable "
            f"customer tier. The leading revenue-countable indicator continues to climb."
        )
    return None


def rule_stronger_alive_milestone(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Trigger when Stronger Alive crosses a 10-multiple."""
    cur = vars_.get("stronger_alive", 0)
    prev = vars_.get("prev_stronger_alive", 0)
    if not isinstance(cur, int) or not isinstance(prev, int):
        return None
    if _crossed_multiple(cur, prev, 10):
        return (
            f"**Stronger Alive crossed {(cur // 10) * 10}** ({prev} -> {cur}). "
            f"Registries with at least 10 heartbeats in the last 14 days. "
            f"This is the trailing durability signal -- harder to fake than Confirmed Alive."
        )
    return None


def rule_install_pace_acceleration(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Detect sustained acceleration: recent_pace > linear for 3+ consecutive reports."""
    search_dir = args.search_dir or str(Path(args.output_dir).parent)
    history = _load_recent_forecast_history(search_dir, args.date, n=3)

    cur_recent = float(vars_.get("forecast_recent_rate", 0))
    cur_linear = float(vars_.get("forecast_linear_rate", 0))

    consecutive_acceleration = cur_recent > cur_linear
    if consecutive_acceleration:
        for h in history:
            r = h.get("recent_pace", {}).get("daily_add_rate", 0)
            l = h.get("linear", {}).get("slope_per_day", 0)
            if r <= l:
                consecutive_acceleration = False
                break
    if consecutive_acceleration and len(history) >= 2:
        return (
            f"**Sustained install-pace acceleration.** Recent pace ({cur_recent:.1f}/day) "
            f"has been ahead of the 14-day OLS slope ({cur_linear:.1f}/day) for at least "
            f"{len(history) + 1} consecutive reports. This is acceleration, not noise."
        )

    if cur_recent < cur_linear * 0.85:
        return (
            f"**Install-pace deceleration.** Recent pace ({cur_recent:.1f}/day) is "
            f"meaningfully below the 14-day OLS ({cur_linear:.1f}/day). Worth checking "
            f"whether this is a holiday/weekend artifact or a structural slowdown."
        )
    return None


def rule_install_forecast_eta(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Highlight when both forecast models converge tightly or diverge significantly."""
    linear_eta_str = vars_.get("forecast_linear_eta", "")
    recent_eta_str = vars_.get("forecast_recent_eta", "")
    if not (linear_eta_str and recent_eta_str):
        return None
    try:
        from datetime import datetime
        linear_eta = datetime.strptime(linear_eta_str, "%Y-%m-%d")
        recent_eta = datetime.strptime(recent_eta_str, "%Y-%m-%d")
        gap = abs((linear_eta - recent_eta).days)
    except ValueError:
        return None
    if gap <= 2:
        return (
            f"**Forecast models converged within {gap} day{'s' if gap != 1 else ''}** "
            f"(linear: {linear_eta_str}, recent-pace: {recent_eta_str}). "
            f"The 1,000-install milestone is forecastable to within a small window."
        )
    if gap >= 7:
        return (
            f"**Forecast models diverged by {gap} days** "
            f"(linear: {linear_eta_str}, recent-pace: {recent_eta_str}). "
            f"Check the recent daily-active chart to understand whether the spread reflects "
            f"weekend volatility or a structural shift in install velocity."
        )
    return None


def rule_longest_customer_milestone(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Trigger when longest customer crosses a 7-day-multiple (week milestone)."""
    cur_days = vars_.get("longest_non_internal_days", 0)
    if not isinstance(cur_days, int) or cur_days == 0:
        return None
    # Week-level milestones (28, 35, 42, 49, 56, 63, 70 ...)
    if cur_days % 7 == 0 and cur_days >= 28:
        weeks = cur_days // 7
        rid = vars_.get("longest_non_internal_id_short", "")
        profile = vars_.get("longest_non_internal_profile", "")
        return (
            f"**Longest customer crossed {weeks}-week mark** ({cur_days} days). "
            f"`{rid}` ({profile}) is now the longest-running non-internal instance. "
            f"Strong customer-story candidate for an external case study."
        )
    return None


def rule_30day_cohort_milestone(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Trigger when 30-day cohort crosses a 1pp threshold or hits a new count milestone."""
    cur_pct = float(vars_.get("pct_30d", 0))
    cur_count = vars_.get("monthly_count", 0)
    if cur_pct >= 5.0 and cur_count >= 25:
        return (
            f"**30-day cohort at {cur_pct:.1f}% ({cur_count} customers).** "
            f"The retention curve continues to bend up. Customers who survive past the "
            f"first month are now a meaningful share of the install base, not a rounding error."
        )
    return None


def rule_ltv_milestone(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Trigger when cumulative LTV (all-days) crosses a $5K-multiple."""
    ltv_str = vars_.get("ltv_cumulative_all_days", "$0")
    try:
        ltv = float(ltv_str.replace("$", "").replace(",", ""))
    except ValueError:
        return None
    if ltv == 0:
        return None
    # Simple milestone: report whenever LTV crosses each $5K boundary
    milestone = int(ltv // 5000) * 5000
    if milestone < 5000:
        return None
    arr_proven = vars_.get("arr_proven_str", "$?M")
    arr_all_days = vars_.get("arr_all_days_str", "$?M")
    return (
        f"**Cumulative LTV at {ltv_str}** (all-days, hypothetical AWS list-price). "
        f"At the current 7-day daily run rate, the implied ARR is {arr_proven}-{arr_all_days}. "
        f"The deployed customer fleet's economic footprint is growing roughly in line with "
        f"the active-instance count."
    )


def rule_one_day_wonder_trend(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Trigger when one-day-wonder share moves >= 1pp in either direction."""
    cur = float(vars_.get("one_day_wonder_pct_str", 0))
    # We need previous metrics to compare; stash via load
    search_dir = args.search_dir or str(Path(args.output_dir).parent)
    import re
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    prev_pct = None
    for entry in sorted(os.listdir(search_dir), reverse=True):
        if not date_re.match(entry):
            continue
        if entry >= args.date:
            continue
        path = os.path.join(search_dir, entry, f"metrics-{entry}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    prev_pct = json.load(f).get("stickiness", {}).get("one_day_wonder_pct")
                    break
            except Exception:
                pass
    if prev_pct is None:
        return None
    delta = cur - prev_pct
    if abs(delta) < 1.0:
        return None
    if delta < 0:
        return (
            f"**One-day-wonder share down {abs(delta):.1f}pp** ({prev_pct:.1f}% -> {cur:.1f}%). "
            f"The retention curve is bending up, even with new acquisition diluting the denominator."
        )
    return (
        f"**One-day-wonder share up {delta:.1f}pp** ({prev_pct:.1f}% -> {cur:.1f}%). "
        f"Acquisition is outpacing aging-in. Expected during growth surges, but worth "
        f"checking that the absolute multi-day count is still growing."
    )


def rule_github_growth(
    vars_: dict,
    args: argparse.Namespace,
) -> str | None:
    """Trigger when stars OR forks jump notably in one report period."""
    stars_delta_str = vars_.get("github_stars_delta_str", "0")
    forks_delta_str = vars_.get("github_forks_delta_str", "0")
    try:
        stars_delta = int(stars_delta_str.replace("+", "").replace(",", ""))
        forks_delta = int(forks_delta_str.replace("+", "").replace(",", ""))
    except ValueError:
        return None
    if stars_delta >= 5 or forks_delta >= 5:
        return (
            f"**GitHub growth spike.** +{stars_delta} stars, +{forks_delta} forks since the "
            f"previous report. Likely social/blog-post visibility. Worth checking referrer "
            f"data on the repo if available."
        )
    return None


# ============================================================================
# Driver
# ============================================================================


RULES = [
    rule_milestone_install_count,
    rule_confirmed_alive_milestone,
    rule_stronger_alive_milestone,
    rule_install_pace_acceleration,
    rule_install_forecast_eta,
    rule_longest_customer_milestone,
    rule_30day_cohort_milestone,
    rule_ltv_milestone,
    rule_one_day_wonder_trend,
    rule_github_growth,
]


def build_recommendations(
    vars_: dict,
    args: argparse.Namespace,
) -> str:
    """Run every rule and return a numbered markdown list of fired bullets."""
    bullets = []
    for rule in RULES:
        try:
            result = rule(vars_, args)
            if result:
                bullets.append(result)
        except Exception as e:
            bullets.append(f"_(rule {rule.__name__} failed: {type(e).__name__})_")

    if not bullets:
        return "_No rule-based recommendations triggered for this snapshot._"

    return "\n".join(f"{i + 1}. {b}" for i, b in enumerate(bullets))
