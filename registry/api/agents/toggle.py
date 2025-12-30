from __future__ import annotations

import logging
from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import JSONResponse

from ...auth.dependencies import (
    nginx_proxied_auth,
)
from ...services.agent_service import agent_service
from .utils import (
    _check_agent_permission,
    _filter_agents_by_access,
    _normalize_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/agents/{path:path}/toggle")
async def toggle_agent(
    path: str,
    enabled: bool,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Enable or disable an agent."""
    path = _normalize_path(path)

    agent_card = agent_service.get_agent_info(path)
    if not agent_card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    accessible = _filter_agents_by_access([agent_card], user_context)
    if not accessible:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this agent",
        )

    _check_agent_permission("toggle_service", agent_card.name, user_context)

    success = agent_service.toggle_agent(path, enabled)

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Failed to toggle agent state"},
        )

    from ...search.service import faiss_service

    await faiss_service.add_or_update_entity(
        path,
        agent_card.model_dump(),
        "a2a_agent",
        enabled,
    )

    logger.info(
        "Agent '%s' (%s) toggled to %s by user '%s'",
        agent_card.name,
        path,
        enabled,
        user_context.get("username"),
    )

    return {
        "message": f"Agent {'enabled' if enabled else 'disabled'} successfully",
        "path": path,
        "is_enabled": enabled,
    }
