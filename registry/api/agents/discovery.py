from __future__ import annotations

import logging
from typing import (
    Annotated,
    List,
    Optional,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
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


@router.post("/agents/discover")
async def discover_agents_by_skills(
    skills: List[str],
    tags: Optional[List[str]] = None,
    max_results: int = Query(10, ge=1, le=100),
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Discover agents by required skills."""
    if not skills:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one skill must be specified",
        )

    logger.info(
        "User %s discovering agents with skills: %s",
        user_context.get("username"),
        skills,
    )

    all_agents = agent_service.get_all_agents()
    accessible_agents = _filter_agents_by_access(all_agents, user_context)

    matched_agents = []
    required_skills = set(s.lower() for s in skills)
    required_tags = set(t.lower() for t in tags) if tags else set()

    for agent in accessible_agents:
        if not agent_service.is_agent_enabled(agent.path):
            continue

        agent_skills = set(skill.id.lower() for skill in agent.skills) | set(
            skill.name.lower() for skill in agent.skills
        )

        skill_matches = required_skills & agent_skills
        if not skill_matches:
            continue

        agent_tags = set(t.lower() for t in agent.tags)
        tag_matches = required_tags & agent_tags if required_tags else set()

        skill_match_score = len(skill_matches) / len(required_skills)
        tag_match_score = len(tag_matches) / len(required_tags) if required_tags else 0.0

        trust_boost = {
            "unverified": 0.0,
            "community": 0.2,
            "verified": 0.5,
            "trusted": 1.0,
        }.get(agent.trust_level, 0.0)

        relevance_score = (
            0.6 * skill_match_score + 0.2 * tag_match_score + 0.2 * trust_boost
        )

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
            is_enabled=True,
            provider=provider_name,
            streaming=streaming,
            trust_level=agent.trust_level,
        )

        matched_agents.append(
            {
                **agent_info.model_dump(),
                "relevance_score": round(relevance_score, 2),
                "matched_skills": list(skill_matches),
            }
        )

    matched_agents.sort(key=lambda x: x["relevance_score"], reverse=True)
    matched_agents = matched_agents[:max_results]

    logger.info("Found %s agents matching skills: %s", len(matched_agents), skills)

    return {
        "agents": matched_agents,
        "query": {
            "skills": skills,
            "tags": tags,
        },
    }


@router.post("/agents/discover/semantic")
async def discover_agents_semantic(
    query: str,
    max_results: int = Query(10, ge=1, le=100),
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Discover agents using natural language semantic search."""
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    logger.info(
        "User %s semantic search for agents: %s",
        user_context.get("username"),
        query,
    )

    from ...search.service import faiss_service

    try:
        results = await faiss_service.search_entities(
            query=query,
            entity_types=["a2a_agent"],
            enabled_only=True,
            max_results=max_results,
        )

        all_agents = agent_service.get_all_agents()
        agent_map = {agent.path: agent for agent in all_agents}

        accessible_results = []
        for result in results:
            agent_card = agent_map.get(result.get("path"))
            if not agent_card:
                continue

            if not _filter_agents_by_access([agent_card], user_context):
                continue

            streaming = (
                agent_card.capabilities.get("streaming", False)
                if agent_card.capabilities
                else False
            )
            provider_name = agent_card.provider.organization if agent_card.provider else None

            agent_info = AgentInfo(
                name=agent_card.name,
                description=agent_card.description,
                path=agent_card.path,
                url=str(agent_card.url),
                tags=agent_card.tags,
                skills=[s.name for s in agent_card.skills],
                num_skills=len(agent_card.skills),
                num_stars=agent_card.num_stars,
                is_enabled=True,
                provider=provider_name,
                streaming=streaming,
                trust_level=agent_card.trust_level,
            )

            accessible_results.append(
                {
                    **agent_info.model_dump(),
                    "score": result.get("relevance_score", 0.0),
                }
            )

        logger.info(
            "Semantic search returned %s agents for query: %s",
            len(accessible_results),
            query,
        )

        return {
            "agents": accessible_results,
            "query": query,
        }

    except Exception as exc:
        logger.error("Error in semantic agent search: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Semantic search failed",
        )
