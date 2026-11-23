"""Clear the security-pending tag when a rescan resolves the scan that set it.

A registration scan that fails, or that reports findings, tags the server,
agent or skill ``security-pending`` (issue #2032). Before this module, a later
rescan that passed left the tag in place until an admin edited the tags by hand.

The scanner does not record that it set the tag, so the rule is inferred from
scan history: the tag is cleared on a passing rescan unless the scan before it
also passed. If the previous scan passed, the tag was added afterwards, by a
person, and is kept.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

SECURITY_PENDING_TAG: str = "security-pending"


def _previous_scan_passed(
    previous_scan: dict[str, Any] | None,
) -> bool:
    """Return True when the stored scan before this one was a clean pass."""
    if not isinstance(previous_scan, dict):
        return False
    return previous_scan.get("is_safe") is True


def tags_after_passing_scan(
    tags: list[str] | None,
    previous_scan: dict[str, Any] | None,
) -> list[str] | None:
    """Return the tags with security-pending removed, or None to leave them alone.

    Call only after a scan that passed.

    Args:
        tags: The entity's current tags.
        previous_scan: The latest stored scan result from before this rescan,
            or None when there was none (for example, a scan that raised).

    Returns:
        The new tag list when the tag should be cleared, otherwise None.
    """
    if not tags or SECURITY_PENDING_TAG not in tags:
        return None
    if _previous_scan_passed(previous_scan):
        logger.info(
            "Keeping 'security-pending': the scan before this one passed, so the tag "
            "was added after it rather than by the scanner"
        )
        return None
    return [tag for tag in tags if tag != SECURITY_PENDING_TAG]
