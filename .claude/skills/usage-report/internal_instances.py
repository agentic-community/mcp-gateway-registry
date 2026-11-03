"""Shared helper that decides which registry instances are internal.

An instance is internal if either:
1. Its registry_id appears in the known-internal-instances.md allowlist, or
2. Its latest non-empty `internal_only_deployment` telemetry value is true.
   Registries send this field from release 1.24.5 (issue #1216).

Every usage-report script that computes customer-only numbers must use this
helper, so the allowlist and the telemetry flag are applied the same way
everywhere (issue #1911).
"""

import logging
import os
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)

INTERNAL_ID_PATTERN = re.compile(
    r"`([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})`"
)
TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes"})


def _load_allowlist_ids(
    path: str | None,
) -> set[str]:
    """Parse the known-internal-instances.md allowlist into a set of registry IDs."""
    if not path or not os.path.exists(path):
        logger.info("No known-internal-instances file found, using the telemetry flag only")
        return set()

    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            ids.update(INTERNAL_ID_PATTERN.findall(line))
    logger.info(f"Loaded {len(ids)} known internal instance IDs from {path}")
    return ids


def _flagged_internal_ids(
    rows: Iterable[dict[str, str]],
) -> set[str]:
    """Return registry IDs whose latest non-empty internal_only_deployment is true."""
    latest: dict[str, tuple[str, str]] = {}
    for row in rows:
        rid = (row.get("registry_id") or "").strip()
        value = (row.get("internal_only_deployment") or "").strip().lower()
        if not rid or rid == "null" or not value:
            continue
        ts = row.get("ts") or ""
        if rid not in latest or ts > latest[rid][0]:
            latest[rid] = (ts, value)

    ids = {rid for rid, (_, value) in latest.items() if value in TRUE_VALUES}
    logger.info(f"Found {len(ids)} instances that report internal_only_deployment=true")
    return ids


def load_internal_instance_ids(
    allowlist_path: str | None,
    rows: Iterable[dict[str, str]],
) -> set[str]:
    """Return every internal registry ID from the allowlist and the telemetry flag.

    Args:
        allowlist_path: Path to known-internal-instances.md. None or a missing
            file means no allowlist entries.
        rows: Telemetry CSV rows (dicts with registry_id, ts and
            internal_only_deployment).

    Returns:
        The union of allowlisted and self-declared internal registry IDs.
    """
    ids = _load_allowlist_ids(allowlist_path) | _flagged_internal_ids(rows)
    logger.info(f"Treating {len(ids)} instances as internal")
    return ids
