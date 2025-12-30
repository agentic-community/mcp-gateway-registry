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
    AgentProvider,
    AgentRegistrationRequest,
)
from ...services.agent_service import agent_service
from .utils import (
    _normalize_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/agents/register")
async def register_agent(
    request: AgentRegistrationRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Register a new A2A agent in the registry."""
    ui_permissions = user_context.get("ui_permissions", {})
    publish_permissions = ui_permissions.get("publish_agent", [])

    if not publish_permissions:
        logger.warning(
            "User %s attempted to register agent without permission",
            user_context.get("username"),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to register agents",
        )

    logger.info("Agent registration request from user '%s'", user_context.get("username"))
    logger.info("Name: %s, Path: %s, URL: %s", request.name, request.path, request.url)

    path = _normalize_path(request.path, request.name)

    if agent_service.get_agent_info(path):
        logger.error("Agent registration failed: path '%s' already exists", path)
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": f"Agent with path '{path}' already exists",
                "suggestion": "Use a different path or update the existing agent",
            },
        )

    tag_list = [tag.strip() for tag in request.tags.split(",") if tag.strip()]

    provider_obj = None
    if request.provider:
        provider_obj = AgentProvider(
            organization=request.provider.get("organization", ""),
            url=request.provider.get("url", ""),
        )

    try:
        from ...utils.agent_validator import agent_validator

        agent_card = AgentCard(
            protocol_version=request.protocol_version,
            name=request.name,
            description=request.description,
            url=request.url,
            path=path,
            version=request.version,
            capabilities={"streaming": request.streaming},
            provider=provider_obj,
            security_schemes=request.security_schemes or {},
            skills=request.skills or [],
            tags=tag_list,
            license=request.license,
            visibility=request.visibility,
            registered_by=user_context["username"],
        )

        validation_result = await agent_validator.validate_agent_card(
            agent_card,
            verify_endpoint=True,
        )

        if not validation_result.is_valid:
            logger.error("Agent validation failed: %s", validation_result.errors)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Agent card validation failed",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                },
            )

    except ValueError as exc:
        logger.error("Invalid agent card data: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid agent card: {str(exc)}",
        )

    success = agent_service.register_agent(agent_card)

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Failed to save agent data",
                "suggestion": "Check server logs for details",
            },
        )

    from ...search.service import faiss_service

    is_enabled = agent_service.is_agent_enabled(path)
    await faiss_service.add_or_update_entity(
        path,
        agent_card.model_dump(),
        "a2a_agent",
        is_enabled,
    )

    logger.info(
        "New agent registered: '%s' at path '%s' by user '%s'",
        request.name,
        path,
        user_context.get("username"),
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "Agent registered successfully",
            "agent": {
                "name": agent_card.name,
                "path": agent_card.path,
                "url": str(agent_card.url),
                "num_skills": len(agent_card.skills),
                "registered_at": (
                    agent_card.registered_at.isoformat() if agent_card.registered_at else None
                ),
                "is_enabled": is_enabled,
            },
        },
    )

