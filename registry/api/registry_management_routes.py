"""
Registry management routes for administrative operations.

Provides endpoints for registry operators to manage telemetry,
diagnostics, and other internal registry functions.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from registry.auth.dependencies import nginx_proxied_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/registry-management", tags=["Registry Management"])


def _require_admin(user_context: dict) -> None:
    """
    Verify user has admin permissions.

    Args:
        user_context: User context from authentication

    Raises:
        HTTPException: If user is not an admin
    """
    if not user_context.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permissions are required for this operation",
        )


@router.post("/telemetry/heartbeat")
async def force_heartbeat(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Force an immediate heartbeat telemetry event (admin only).

    Bypasses the 24-hour lock and sends a heartbeat event immediately.
    Useful for verifying telemetry pipeline or after configuration changes.

    Returns:
        Status of the heartbeat send attempt.
    """
    _require_admin(user_context)

    from registry.core.telemetry import send_forced_heartbeat

    result = await send_forced_heartbeat()

    if result["status"] == "disabled":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Telemetry is disabled. Set MCP_TELEMETRY_DISABLED=0 to enable.",
        )

    return result


@router.post("/telemetry/startup")
async def force_startup_ping(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Force an immediate startup telemetry event (admin only).

    Bypasses the 60-second lock and sends a startup ping immediately.
    Useful for verifying telemetry pipeline connectivity.

    Returns:
        Status of the startup send attempt.
    """
    _require_admin(user_context)

    from registry.core.telemetry import send_forced_startup

    result = await send_forced_startup()

    if result["status"] == "disabled":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Telemetry is disabled. Set MCP_TELEMETRY_DISABLED=0 to enable.",
        )

    return result
