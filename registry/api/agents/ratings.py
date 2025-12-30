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
from pydantic import BaseModel

from ...auth.dependencies import (
    nginx_proxied_auth,
)
from ...services.agent_service import agent_service
from .utils import (
    _filter_agents_by_access,
    _normalize_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class RatingRequest(BaseModel):
    rating: int


@router.post("/agents/{path:path}/rate")
async def rate_agent(
    path: str,
    request: RatingRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Save integer ratings to agent card."""
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
            "User %s attempted to rate agent %s without permission",
            user_context.get("username"),
            path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this agent",
        )

    try:
        avg_rating = agent_service.update_rating(path, user_context["username"], request.rating)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Unexpected error updating rating: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save rating",
        )

    return {
        "message": "Rating added successfully",
        "average_rating": avg_rating,
    }


@router.get("/agents/{path:path}/rating")
async def get_agent_rating(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Get agent rating information."""
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

    return {
        "num_stars": agent_card.num_stars,
        "rating_details": agent_card.rating_details,
    }
