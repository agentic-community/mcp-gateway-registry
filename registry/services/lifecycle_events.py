"""Lifecycle event helpers shared across server, agent, and skill routers.

Centralizes the ``scan_complete`` webhook so all three asset types emit the
same event shape (Issue #1330). The webhook itself is sent by
``send_registration_webhook``; these helpers build the event-specific payload
and dispatch it fire-and-forget.
"""

import asyncio
import logging

from ..core.config import settings
from ..schemas.security import SecurityScanResult
from ..utils.credential_encryption import strip_credentials_from_dict
from .webhook_service import send_registration_webhook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Cap the scan_error message so no stack trace or internal detail is sent to an
# external webhook consumer (Issue #1330, security review).
_SCAN_ERROR_MAX_LEN: int = 200

# The tag the registry applies to assets that fail a security scan.
_SECURITY_PENDING_TAG: str = "security-pending"


class EnforcedStatusError(ValueError):
    """Raised when a registration's status does not match the enforced status.

    Routes catch this and return an HTTP 400 with the message.
    """


def _record_status_rejected(
    registration_type: str,
) -> None:
    """Increment the enforced-status rejection counter. Never raises."""
    try:
        from registry.observability.meters import registration_status_rejected_total

        registration_status_rejected_total.labels(registration_type=registration_type).inc()
    except Exception as e:
        logger.debug(f"registration_status_rejected metric inc failed: {type(e).__name__}: {e}")


def user_can_change_lifecycle_status(
    asset_name: str,
    user_context: dict,
) -> bool:
    """Whether the user may change an asset's lifecycle status (Issue #1330).

    Admins always may (existing admin scopes predate this permission). Other
    users need the 'change_lifecycle_status' UI permission for the asset.

    Args:
        asset_name: Display name of the asset (for the per-service permission check).
        user_context: Authenticated user context.

    Returns:
        True if the user may change lifecycle status, else False.
    """
    if user_context.get("is_admin"):
        return True

    from ..auth.dependencies import user_has_ui_permission_for_service

    return user_has_ui_permission_for_service(
        "change_lifecycle_status",
        asset_name,
        user_context.get("ui_permissions", {}),
    )


def enforce_registration_status(
    requested_status: str | None,
    registration_type: str = "server",
) -> str | None:
    """Apply the REGISTRATION_ENFORCED_STATUS policy to a registration.

    When the policy is unset, returns the requested status unchanged (no
    behavior change). When set, a missing status is forced to the enforced
    value, a matching status is accepted, and a mismatched status raises
    EnforcedStatusError (which routes map to HTTP 400).

    Args:
        requested_status: The status supplied on the registration request, if any.
        registration_type: One of "server", "agent", or "skill" (for the metric).

    Returns:
        The status to persist: the enforced value when the policy is set,
        otherwise the requested status unchanged.

    Raises:
        EnforcedStatusError: When the policy is set and a different explicit
            status was supplied.
    """
    enforced = settings.registration_enforced_status
    if not enforced:
        return requested_status

    enforced = enforced.lower().strip()

    if requested_status is None:
        logger.info(
            f"Enforced status '{enforced}' applied to new {registration_type} registration"
        )
        return enforced

    if requested_status.lower().strip() != enforced:
        _record_status_rejected(registration_type)
        raise EnforcedStatusError(
            f"This registry mandates an initial lifecycle status of '{enforced}'. "
            f"Registration with status '{requested_status}' is not allowed. "
            f"Register with status '{enforced}' (or omit status), then have an "
            f"authorized user change the lifecycle status afterward."
        )

    return enforced


def _sanitize_scan_error(
    scan_error: str | None,
) -> str | None:
    """Reduce a scan exception to a short, non-sensitive message.

    Collapses whitespace and truncates so stack traces, internal URLs, or full
    exception detail never leave the registry via the webhook.

    Args:
        scan_error: Raw error string (e.g. f"{type(e).__name__}: {e}").

    Returns:
        A short cleaned message, or None when there was no error.
    """
    if not scan_error:
        return None

    cleaned = " ".join(scan_error.split())
    return cleaned[:_SCAN_ERROR_MAX_LEN]


def _build_scan_fields(
    scan_result: SecurityScanResult | None,
    server_entry: dict,
    scan_error: str | None,
    auto_disabled: bool,
) -> dict:
    """Build the 'scan' block for a scan_complete webhook payload.

    Args:
        scan_result: The completed scan result, or None when the scan raised.
        server_entry: The asset card (used to read applied tags).
        scan_error: Raw error string when the scan raised, else None.
        auto_disabled: Whether the asset was disabled as a result of the scan.

    Returns:
        A JSON-serializable dict describing the scan outcome.
    """
    if scan_result is None:
        return {
            "is_safe": None,
            "severity_counts": {},
            "tags_applied": [],
            "auto_disabled": False,
            "scan_error": _sanitize_scan_error(scan_error),
        }

    tags = server_entry.get("tags", []) or []
    return {
        "is_safe": scan_result.is_safe,
        "severity_counts": {
            "critical": scan_result.critical_issues,
            "high": scan_result.high_severity,
            "medium": scan_result.medium_severity,
            "low": scan_result.low_severity,
        },
        "tags_applied": [t for t in tags if t == _SECURITY_PENDING_TAG],
        "auto_disabled": auto_disabled,
        "scan_error": None,
    }


def fire_scan_complete_event(
    asset_entry: dict,
    scan_result: SecurityScanResult | None,
    scan_error: str | None = None,
    auto_disabled: bool = False,
    registration_type: str = "server",
) -> None:
    """Emit a scan_complete webhook for any asset type. Fire-and-forget.

    Never raises: a failure here must not affect the calling scan workflow.

    Args:
        asset_entry: The asset card dict (server/agent/skill).
        scan_result: Completed scan result, or None if the scan raised.
        scan_error: Raw error string when the scan raised, else None.
        auto_disabled: Whether the asset was disabled due to the scan result.
        registration_type: One of "server", "agent", or "skill".
    """
    try:
        scan_fields = _build_scan_fields(
            scan_result,
            asset_entry,
            scan_error,
            auto_disabled,
        )
        asyncio.create_task(
            send_registration_webhook(
                event_type="scan_complete",
                registration_type=registration_type,
                card_data=strip_credentials_from_dict(dict(asset_entry)),
                performed_by=asset_entry.get("registered_by"),
                extra_fields={"scan": scan_fields},
            )
        )
        logger.info(
            f"scan_complete event dispatched: type={registration_type}, "
            f"is_safe={getattr(scan_result, 'is_safe', None)}"
        )
    except Exception as e:
        logger.warning(f"Failed to dispatch scan_complete event: {type(e).__name__}: {e}")
