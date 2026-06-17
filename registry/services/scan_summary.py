"""Lightweight security-scan summaries for list endpoints.

List endpoints (servers, agents, skills) need just enough scan information to
colour the shield icon on each card: whether the scan failed and the per-severity
issue counts. Returning the full scan document per card forced the frontend into
an N+1 fetch (one GET /{type}/{path}/security-scan per rendered card). Instead,
the list endpoints bulk-load every scan once and attach this summary inline.

The full scan document is still served by the per-item endpoint, fetched lazily
only when the user opens the scan detail view.

The scan repositories store one document per scan *run* (insert, not upsert), so
the raw collection grows with total scan history. The services feed this builder
from the repo's ``list_latest()``, which collapses history to one (newest)
document per path at the data layer (a ``$group`` aggregation on DocumentDB), so
the bulk read stays O(active paths) rather than O(total scans ever). The dedup in
build_scan_summary_map below is therefore a defensive second line — correct even
if a backend ever returns duplicate paths.
"""

from typing import Any

# Keys read from a stored scan document to build the icon summary. Different scan
# documents key their path field differently: servers use "server_path", agents
# "agent_path", skills "skill_path". Each summary map only ever contains one type,
# so checking all three is harmless.
_PATH_KEYS: tuple[str, ...] = ("server_path", "agent_path", "skill_path")


def build_scan_summary(
    scan: dict[str, Any],
) -> dict[str, Any]:
    """Build the lightweight icon summary for a single scan document.

    Args:
        scan: Stored scan document (dict form of SecurityScanResult).

    Returns:
        Dict with the fields the shield icon needs: scan_failed and the four
        per-severity counts.
    """
    return {
        "scan_failed": scan.get("scan_failed", False),
        "critical_issues": scan.get("critical_issues", 0),
        "high_severity": scan.get("high_severity", 0),
        "medium_severity": scan.get("medium_severity", 0),
        "low_severity": scan.get("low_severity", 0),
    }


def _scan_path(
    scan: dict[str, Any],
) -> str | None:
    """Return the resource path a scan document is keyed by, or None.

    Trailing slashes are stripped so a scan stored as "/foo/" collapses onto the
    same key as a list item whose path is "/foo" (matching get_latest's
    with/without-slash normalisation).
    """
    raw = next((scan[key] for key in _PATH_KEYS if scan.get(key)), None)
    if not raw:
        return None
    normalized = raw.rstrip("/")
    return normalized or raw  # don't collapse a bare "/" to empty


def build_scan_summary_map(
    scans: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a path -> latest-icon-summary map from a list of scan documents.

    The DocumentDB scan repositories store one document per scan *run* (insert,
    not upsert), so list_all() returns full history — a path can appear many
    times. We keep the entry with the newest scan_timestamp per path rather than
    trusting the cursor's sort order, so the badge always reflects the latest
    scan (matching the per-item get_latest endpoint). scan_timestamp is ISO-8601
    with a trailing "Z", which sorts correctly as a plain string.

    Args:
        scans: All scan documents (e.g. from a repository's list_all()).

    Returns:
        Mapping of resource path to its lightweight scan summary. Scans without
        a recognisable path key are skipped.
    """
    latest_ts: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for scan in scans:
        path = _scan_path(scan)
        if not path:
            continue
        # Empty string sorts below any real timestamp, so a doc missing the field
        # only wins when nothing better has been seen for this path.
        ts = scan.get("scan_timestamp") or ""
        if path not in summaries or ts >= latest_ts[path]:
            latest_ts[path] = ts
            summaries[path] = build_scan_summary(scan)
    return summaries
