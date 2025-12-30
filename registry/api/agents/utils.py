from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from fastapi import (
    HTTPException,
    status,
)

from ...auth.dependencies import (
    user_has_ui_permission_for_service,
)
from ...schemas.agent_models import (
    AgentCard,
)

logger = logging.getLogger(__name__)


def _normalize_path(
    path: Optional[str],
    agent_name: Optional[str] = None,
) -> str:
    """Normalize agent path format.

    If path is None, derives it from agent_name by converting to lowercase and replacing
    spaces with hyphens.

    Args:
        path: Agent path to normalize, or None to auto-generate.
        agent_name: Agent name used for auto-generating path if needed.

    Returns:
        Normalized path string.

    Raises:
        ValueError: If path is None and agent_name is not provided.
    """
    if path is None:
        if not agent_name:
            raise ValueError(
                "Path is required or agent_name must be provided for auto-generation"
            )
        path = agent_name.lower().replace(" ", "-")

    if not path.startswith("/"):
        path = "/" + path

    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")

    return path


def _check_agent_permission(
    permission: str,
    agent_name: str,
    user_context: Dict[str, Any],
) -> None:
    """Check if user has permission for agent operation."""
    if not user_has_ui_permission_for_service(
        permission,
        agent_name,
        user_context.get("ui_permissions", {}),
    ):
        logger.warning(
            "User %s attempted to perform %s on agent %s without permission",
            user_context.get("username"),
            permission,
            agent_name,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to {permission} for {agent_name}",
        )


def _filter_agents_by_access(
    agents: List[AgentCard],
    user_context: Dict[str, Any],
) -> List[AgentCard]:
    """Filter agents based on user access permissions."""
    accessible: List[AgentCard] = []
    user_groups = set(user_context.get("groups", []))
    username = user_context["username"]
    is_admin = user_context.get("is_admin", False)

    accessible_agent_list = user_context.get("accessible_agents", [])
    logger.debug(
        "User %s accessible agents from UI-Scopes: %s",
        username,
        accessible_agent_list,
    )

    for agent in agents:
        if is_admin:
            accessible.append(agent)
            continue

        if "all" not in accessible_agent_list and agent.path not in accessible_agent_list:
            continue

        if agent.visibility == "public":
            accessible.append(agent)
            continue

        if agent.visibility == "private":
            if agent.registered_by == username:
                accessible.append(agent)
            continue

        if agent.visibility == "group-restricted":
            agent_groups = set(agent.allowed_groups)
            if agent_groups & user_groups:
                accessible.append(agent)
            continue

    return accessible

