"""
DEPRECATED: This module is deprecated. Use registry.services.scope_service instead.

This module is kept for backward compatibility only. All functions are thin
wrappers around the new scope_service module with deprecation warnings.

The old YAML-based implementation has been removed.
"""

import logging
from typing import (
    Any,
)

from ..services.scope_service import (
    add_server_to_groups as _add_server_to_groups,
)
from ..services.scope_service import (
    create_group as _create_group,
)
from ..services.scope_service import (
    delete_group as _delete_group,
)
from ..services.scope_service import (
    group_exists as _group_exists,
)
from ..services.scope_service import (
    list_groups as _list_groups,
)
from ..services.scope_service import (
    remove_server_from_groups as _remove_server_from_groups,
)
from ..services.scope_service import (
    remove_server_scopes as _remove_server_scopes,
)
from ..services.scope_service import (
    trigger_auth_server_reload as _trigger_auth_server_reload,
)
from ..services.scope_service import (
    update_server_scopes as _update_server_scopes,
)

logger = logging.getLogger(__name__)


async def update_server_scopes(
    server_path: str,
    server_name: str,
    tools: list[str],
) -> bool:
    """
    DEPRECATED: Use registry.services.scope_service.update_server_scopes instead.

    Update scopes for a server (add or update) and reload auth server.

    Args:
        server_path: The server's path (e.g., '/example-server')
        server_name: The server's display name
        tools: List of tool names the server provides

    Returns:
        True if successful, False otherwise
    """
    logger.warning(
        "scopes_manager.update_server_scopes is deprecated, "
        "use scope_service.update_server_scopes instead"
    )
    return await _update_server_scopes(server_path, server_name, tools)


async def remove_server_scopes(
    server_path: str,
) -> bool:
    """
    DEPRECATED: Use registry.services.scope_service.remove_server_scopes instead.

    Remove scopes for a server and reload auth server.

    Args:
        server_path: The server's path (e.g., '/example-server')

    Returns:
        True if successful, False otherwise
    """
    logger.warning(
        "scopes_manager.remove_server_scopes is deprecated, "
        "use scope_service.remove_server_scopes instead"
    )
    return await _remove_server_scopes(server_path)


async def add_server_to_groups(
    server_path: str,
    group_names: list[str],
) -> bool:
    """
    DEPRECATED: Use registry.services.scope_service.add_server_to_groups instead.

    Add a server and all its known tools/methods to specific groups in the scopes repository.

    Args:
        server_path: The server's path (e.g., '/example-server')
        group_names: List of group names to add the server to

    Returns:
        True if successful, False otherwise
    """
    logger.warning(
        "scopes_manager.add_server_to_groups is deprecated, "
        "use scope_service.add_server_to_groups instead"
    )
    return await _add_server_to_groups(server_path, group_names)


async def remove_server_from_groups(
    server_path: str,
    group_names: list[str],
) -> bool:
    """
    DEPRECATED: Use registry.services.scope_service.remove_server_from_groups instead.

    Remove a server from specific groups in the scopes repository.

    Args:
        server_path: The server's path (e.g., '/example-server')
        group_names: List of group names to remove the server from

    Returns:
        True if successful, False otherwise
    """
    logger.warning(
        "scopes_manager.remove_server_from_groups is deprecated, "
        "use scope_service.remove_server_from_groups instead"
    )
    return await _remove_server_from_groups(server_path, group_names)


async def create_group_in_scopes(
    group_name: str,
    description: str = "",
) -> bool:
    """
    DEPRECATED: Use registry.services.scope_service.create_group instead.

    Create a new group entry in the scopes repository and add it to group_mappings.

    Args:
        group_name: Name of the group (e.g., 'mcp-servers-custom/read')
        description: Optional description

    Returns:
        True if successful, False otherwise
    """
    logger.warning(
        "scopes_manager.create_group_in_scopes is deprecated, "
        "use scope_service.create_group instead"
    )
    return await _create_group(group_name, description)


async def delete_group_from_scopes(
    group_name: str,
    remove_from_mappings: bool = True,
) -> bool:
    """
    DEPRECATED: Use registry.services.scope_service.delete_group instead.

    Delete a group from the scopes repository and optionally from group_mappings.

    Args:
        group_name: Name of the group to delete
        remove_from_mappings: Whether to remove from group_mappings section

    Returns:
        True if successful, False otherwise
    """
    logger.warning(
        "scopes_manager.delete_group_from_scopes is deprecated, "
        "use scope_service.delete_group instead"
    )
    return await _delete_group(group_name, remove_from_mappings)


async def list_groups_from_scopes() -> dict[str, Any]:
    """
    DEPRECATED: Use registry.services.scope_service.list_groups instead.

    List all groups defined in the scopes repository.

    Returns:
        Dict with group information including server counts and mappings
    """
    logger.warning(
        "scopes_manager.list_groups_from_scopes is deprecated, "
        "use scope_service.list_groups instead"
    )
    return await _list_groups()


async def group_exists_in_scopes(
    group_name: str,
) -> bool:
    """
    DEPRECATED: Use registry.services.scope_service.group_exists instead.

    Check if a group exists in the scopes repository.

    Args:
        group_name: Name of the group to check

    Returns:
        True if group exists, False otherwise
    """
    logger.warning(
        "scopes_manager.group_exists_in_scopes is deprecated, "
        "use scope_service.group_exists instead"
    )
    return await _group_exists(group_name)


async def trigger_auth_server_reload() -> bool:
    """
    DEPRECATED: Use registry.services.scope_service.trigger_auth_server_reload instead.

    Trigger the auth server to reload its scopes configuration.

    Returns:
        True if successful, False otherwise
    """
    logger.warning(
        "scopes_manager.trigger_auth_server_reload is deprecated, "
        "use scope_service.trigger_auth_server_reload instead"
    )
    return await _trigger_auth_server_reload()
