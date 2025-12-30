from __future__ import annotations

import logging
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Annotated,
)

import httpx
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from ...auth.dependencies import (
    nginx_proxied_auth,
)
from ...core.config import settings
from ...services.agent_service import agent_service
from .utils import (
    _filter_agents_by_access,
    _normalize_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/agents/{path:path}/health")
async def check_agent_health(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Perform a live /ping health check against an agent endpoint."""
    path = _normalize_path(path)

    agent_card = agent_service.get_agent_info(path)
    if not agent_card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    accessible = _filter_agents_by_access([agent_card], user_context)
    if not accessible:
        logger.warning(
            "User %s attempted to health check agent %s without permission",
            user_context.get("username"),
            path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this agent",
        )

    if not agent_service.is_agent_enabled(path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot perform health check on a disabled agent",
        )

    base_url = str(agent_card.url).rstrip("/")
    ping_url = f"{base_url}/ping"
    timeout_seconds = max(1, settings.health_check_timeout_seconds)

    status_label = "unknown"
    detail = None
    status_code = None
    response_time_ms = None
    start_time = datetime.now(timezone.utc)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(ping_url)
        status_code = response.status_code
        response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        if response.status_code == 200:
            status_label = "healthy"
        else:
            status_label = "unhealthy"
            detail = f"Agent responded with HTTP {response.status_code}"
    except httpx.TimeoutException:
        status_label = "unhealthy"
        detail = "Health check timed out"
    except httpx.HTTPError as exc:
        status_label = "unhealthy"
        detail = f"Health check failed: {exc}"
    except Exception as exc:
        status_label = "unhealthy"
        detail = f"Unexpected health check error: {exc}"

    last_checked_iso = datetime.now(timezone.utc).isoformat()

    logger.info("Agent health check for %s (%s) completed with status %s", path, ping_url, status_label)

    return {
        "agent_path": path,
        "ping_url": ping_url,
        "status": status_label,
        "status_code": status_code,
        "detail": detail,
        "response_time_ms": response_time_ms,
        "last_checked_iso": last_checked_iso,
    }

