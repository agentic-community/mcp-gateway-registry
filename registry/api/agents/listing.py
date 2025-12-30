from __future__ import annotations

import logging
from typing import (
    Annotated,
    Optional,
)

from fastapi import (
    APIRouter,
    Depends,
    Query,
)

from ...auth.dependencies import (
    nginx_proxied_auth,
)
from ...schemas.agent_models import (
    AgentInfo,
)
from ...services.agent_service import agent_service
from .utils import (
    _filter_agents_by_access,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/agents")
async def list_agents(
    query: Optional[str] = Query(None, description="Search query string"),
    enabled_only: bool = Query(False, description="Show only enabled agents"),
    visibility: Optional[str] = Query(None, description="Filter by visibility"),
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """List all agents filtered by user permissions."""
    all_agents = agent_service.get_all_agents()
    accessible_agents = _filter_agents_by_access(all_agents, user_context)

    filtered_agents = []
    search_query = query.lower() if query else ""

    for agent in accessible_agents:
        if enabled_only and not agent_service.is_agent_enabled(agent.path):
            continue

        if visibility and agent.visibility != visibility:
            continue

        searchable_text = (
            f"{agent.name.lower()} {agent.description.lower()} "
            f"{' '.join(agent.tags)} {' '.join([s.name for s in agent.skills])}"
        )

        if not search_query or search_query in searchable_text:
            streaming = agent.capabilities.get("streaming", False) if agent.capabilities else False
            provider_name = agent.provider.organization if agent.provider else None

            agent_info = AgentInfo(
                name=agent.name,
                description=agent.description,
                path=agent.path,
                url=str(agent.url),
                tags=agent.tags,
                skills=[s.name for s in agent.skills],
                num_skills=len(agent.skills),
                num_stars=agent.num_stars,
                is_enabled=agent_service.is_agent_enabled(agent.path),
                provider=provider_name,
                streaming=streaming,
                trust_level=agent.trust_level,
            )
            filtered_agents.append(agent_info)

    logger.info(
        "User %s listed %s agents (out of %s total)",
        user_context.get("username"),
        len(filtered_agents),
        len(all_agents),
    )

    return {
        "agents": [agent.model_dump() for agent in filtered_agents],
        "total_count": len(filtered_agents),
    }

