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
from ...schemas.agent_models import (
    AgentCard,
    AgentRegistrationRequest,
)
from ...services.agent_service import agent_service
from .utils import (
    _check_agent_permission,
    _filter_agents_by_access,
    _normalize_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/agents/{path:path}")
async def get_agent(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Get a single agent by path."""
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
            "User %s attempted to access agent %s without permission",
            user_context.get("username"),
            path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this agent",
        )

    return agent_card.model_dump()


@router.put("/agents/{path:path}")
async def update_agent(
    path: str,
    request: AgentRegistrationRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Update an existing agent card."""
    path = _normalize_path(path)

    existing_agent = agent_service.get_agent_info(path)
    if not existing_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    _check_agent_permission("modify_service", existing_agent.name, user_context)

    if not user_context["is_admin"] and existing_agent.registered_by != user_context["username"]:
        logger.warning(
            "User %s attempted to update agent %s owned by %s",
            user_context.get("username"),
            path,
            existing_agent.registered_by,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update agents you registered",
        )

    tag_list = [tag.strip() for tag in request.tags.split(",") if tag.strip()]

    try:
        updated_agent = AgentCard(
            protocol_version=request.protocol_version,
            name=request.name,
            description=request.description,
            url=request.url,
            path=path,
            version=request.version,
            capabilities={"streaming": request.streaming},
            provider=request.provider,
            security_schemes=request.security_schemes or {},
            skills=request.skills or [],
            tags=tag_list,
            license=request.license,
            visibility=request.visibility,
            registered_by=existing_agent.registered_by,
            registered_at=existing_agent.registered_at,
            is_enabled=existing_agent.is_enabled,
            num_stars=existing_agent.num_stars,
        )

        from ...utils.agent_validator import agent_validator

        validation_result = await agent_validator.validate_agent_card(
            updated_agent,
            verify_endpoint=False,
        )

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Agent card validation failed",
                    "errors": validation_result.errors,
                },
            )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid agent card: {str(exc)}",
        )

    success = agent_service.update_agent(path, updated_agent)

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Failed to save updated agent data"},
        )

    from ...search.service import faiss_service

    is_enabled = agent_service.is_agent_enabled(path)
    await faiss_service.add_or_update_entity(
        path,
        updated_agent.model_dump(),
        "a2a_agent",
        is_enabled,
    )

    logger.info(
        "Agent '%s' (%s) updated by user '%s'",
        updated_agent.name,
        path,
        user_context.get("username"),
    )

    return updated_agent.model_dump()


@router.delete("/agents/{path:path}")
async def delete_agent(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Delete an agent from the registry."""
    path = _normalize_path(path)

    existing_agent = agent_service.get_agent_info(path)
    if not existing_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found at path '{path}'",
        )

    if not user_context["is_admin"] and existing_agent.registered_by != user_context["username"]:
        logger.warning(
            "User %s attempted to delete agent %s without permission",
            user_context.get("username"),
            path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins or agent owners can delete agents",
        )

    success = agent_service.remove_agent(path)

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Failed to delete agent"},
        )

    from ...search.service import faiss_service

    await faiss_service.remove_entity(path)

    logger.info(
        "Agent at path '%s' deleted by user '%s'",
        path,
        user_context.get("username"),
    )

    return JSONResponse(
        status_code=status.HTTP_204_NO_CONTENT,
        content=None,
    )

