"""Factory for creating A2A tools from discovered agents."""

import logging
from typing import Any, Dict, Optional

from strands.multiagent.a2a import A2AServerTool

from models import DiscoveredAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def create_a2a_tool_from_agent_card(
    agent: DiscoveredAgent,
    auth_token: Optional[str] = None,
) -> A2AServerTool:
    """Create an A2AServerTool from a discovered agent card.

    This allows the Travel Assistant Agent to use discovered agents
    as tools without hardcoding their details.

    Args:
        agent: Discovered agent metadata from registry
        auth_token: Optional JWT token for authentication

    Returns:
        A2AServerTool instance ready to use

    Example:
        >>> discovered = await discover_agents("flight booking")
        >>> booking_tool = create_a2a_tool_from_agent_card(discovered[0])
        >>> result = await booking_tool.invoke(skill_id="reserve_flight", params={...})
    """
    logger.info(f"Creating A2A tool for agent: {agent.name}")
    logger.debug(f"Agent URL: {agent.url}, Skills: {agent.skills}")

    # Create A2AServerTool with agent card data
    tool = A2AServerTool(
        name=agent.path.lstrip("/").replace("/", "_"),  # Convert path to valid tool name
        url=agent.url,
        description=agent.description,
        auth_token=auth_token,
    )

    logger.info(f"A2A tool created: {tool.name}")
    return tool


async def invoke_agent_skill(
    agent: DiscoveredAgent,
    skill_id: str,
    parameters: Dict[str, Any],
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info(f"Invoking skill '{skill_id}' on agent '{agent.name}'")
    logger.debug(f"Parameters: {parameters}")

    # Create tool from agent card
    tool = create_a2a_tool_from_agent_card(agent, auth_token)

    # Invoke the skill
    try:
        result = await tool.invoke(skill_id=skill_id, parameters=parameters)
        logger.info(f"Skill invocation successful")
        return result
    except Exception as e:
        logger.error(f"Skill invocation failed: {e}", exc_info=True)
        raise
