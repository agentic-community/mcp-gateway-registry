"""Shared helper that compares the recent install pace with the 14-day trend.

The Install Forecast narrative and the forecast recommendations both describe
whether installs are accelerating. They used two different tests (any
difference in rate versus a 7-day gap between ETAs), so one report could call
the models converged in one section and diverged in another. Both now call
forecast_trend_from_rates.
"""

# The recent 7-day pace counts as ahead of or behind the 14-day OLS slope only
# when it differs by more than this fraction of the slope.
FORECAST_RATE_TOLERANCE: float = 0.15
FORECAST_TREND_AHEAD: str = "ahead"
FORECAST_TREND_BEHIND: str = "behind"
FORECAST_TREND_CONVERGED: str = "converged"


def forecast_trend_from_rates(
    recent_rate: float,
    linear_rate: float,
) -> str:
    """Classify the 7-day pace against the 14-day OLS slope.

    Returns ahead, behind, or converged.
    """
    if linear_rate <= 0:
        return FORECAST_TREND_CONVERGED
    if recent_rate > linear_rate * (1 + FORECAST_RATE_TOLERANCE):
        return FORECAST_TREND_AHEAD
    if recent_rate < linear_rate * (1 - FORECAST_RATE_TOLERANCE):
        return FORECAST_TREND_BEHIND
    return FORECAST_TREND_CONVERGED
