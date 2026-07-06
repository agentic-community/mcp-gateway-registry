"""
Scope service - Business logic layer for scope management.

This service wraps the scope repository and implements high-level business
logic for managing server scopes, groups, and authorization rules.
"""

import logging
from typing import (
    Any,
)

import httpx

from ..auth.internal import generate_internal_token
from ..auth.privileged_constants import PRIVILEGED_SCOPE_NAMES
from ..core.config import settings
from ..repositories.factory import get_scope_repository
from .server_service import server_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
STANDARD_METHODS: list[str] = [
    "initialize",
    "notifications/initialized",
    "ping",
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/templates/list",
]

# PRIVILEGED_SCOPE_NAMES is imported from registry.auth.privileged_constants
# (single source of truth, shared with the repository-layer guard and the
# admin-derivation rule). Re-exported here so existing callers/tests that import
# it from this module keep working.


def _grants_all(granted_resources: object) -> bool:
    """Return True if a ui_permission value grants access to "all" resources.

    Mirrors the ``"all" in resources`` test in
    ``registry.auth.dependencies._user_is_admin`` exactly, including its
    behaviour for BOTH shapes the value can take:

    - ``["all"]`` (list) -> membership check
    - ``"all"`` (bare string) -> equality (the string case ``_user_is_admin``
      accepts via substring matching; we accept it explicitly here)

    Keeping this aligned with ``_user_is_admin`` is the whole point of the guard:
    if the admin-derivation rule treats a value as admin-conferring, the guard
    must treat the same value as privileged, regardless of list-vs-string shape.
    """
    if isinstance(granted_resources, str):
        return granted_resources == "all"
    if isinstance(granted_resources, list | tuple | set | frozenset):
        return "all" in granted_resources
    return False


class PrivilegedScopeWriteError(Exception):
    """Raised when a non-admin actor attempts to write a privileged scope/group."""


def _import_touches_privileged_scope(
    scope_name: str,
    group_mappings: list | None,
    ui_permissions: dict | None,
) -> bool:
    """Return True if an import_group write touches a privileged scope/group.

    A write is considered privileged when the scope itself is privileged, when it
    maps any privileged group, or when its ui_permissions grant a privileged
    permission to "all" servers (an admin-equivalent grant).

    Args:
        scope_name: The scope/group being written.
        group_mappings: IdP group names this scope maps to.
        ui_permissions: UI permission grants for this scope.

    Returns:
        True if the write requires admin privileges.
    """
    if scope_name in PRIVILEGED_SCOPE_NAMES:
        return True

    for mapped in group_mappings or []:
        if mapped in PRIVILEGED_SCOPE_NAMES:
            return True

    # A scope that grants any permission to "all" resources is admin-equivalent
    # (see _user_is_admin, which confers admin on any mutating-prefix action with
    # "all"). _grants_all handles both the list (["all"]) and bare-string ("all")
    # shapes so a string-shaped grant cannot slip past this last line of defense.
    for granted_resources in (ui_permissions or {}).values():
        if _grants_all(granted_resources):
            return True

    return False


async def _trigger_auth_server_reload() -> bool:
    """
    Trigger the auth server to reload its scopes configuration.

    Uses a self-signed JWT (signed with the shared SECRET_KEY) for
    internal service-to-service authentication.

    Returns:
        True if successful, False otherwise
    """
    try:
        token = generate_internal_token(
            subject="registry-service",
            purpose="reload-scopes",
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.auth_server_url}/internal/reload-scopes",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )

            if response.status_code == 200:
                logger.info("Successfully triggered auth server scope reload")
                return True
            else:
                logger.error(
                    f"Failed to reload auth server scopes: HTTP {response.status_code}"
                )
                return False

    except Exception as e:
        logger.error(f"Failed to trigger auth server reload: {e}")
        # Non-fatal - scopes will be picked up on next restart
        return False


async def update_server_scopes(
    server_path: str,
    server_name: str,
    tools: list[str],
) -> bool:
    """
    Update scopes for a server (add or update) and reload auth server.

    This adds the server to unrestricted read and execute scopes.

    Args:
        server_path: The server's path (e.g., '/example-server')
        server_name: The server's display name
        tools: List of tool names the server provides

    Returns:
        True if successful, False otherwise
    """
    try:
        scope_repo = get_scope_repository()

        # Add to unrestricted scopes (both read and execute)
        await scope_repo.add_server_scope(
            server_path=server_path,
            scope_name="mcp-servers-unrestricted/read",
            methods=STANDARD_METHODS,
            tools=tools,
        )

        await scope_repo.add_server_scope(
            server_path=server_path,
            scope_name="mcp-servers-unrestricted/execute",
            methods=STANDARD_METHODS,
            tools=tools,
        )

        logger.info(f"Successfully updated scopes for server {server_path} with {len(tools)} tools")

        # Reload auth server
        await _trigger_auth_server_reload()

        return True

    except Exception as e:
        logger.error(f"Failed to update server scopes for {server_path}: {e}")
        return False


async def remove_server_scopes(
    server_path: str,
) -> bool:
    """
    Remove a server from all scopes and reload auth server.

    Args:
        server_path: The server's path (e.g., '/example-server')

    Returns:
        True if successful, False otherwise
    """
    try:
        scope_repo = get_scope_repository()

        # Remove from all scopes
        await scope_repo.remove_server_from_all_scopes(server_path=server_path)

        logger.info(f"Successfully removed server {server_path} from all scopes")

        # Reload auth server
        await _trigger_auth_server_reload()

        return True

    except Exception as e:
        logger.error(f"Failed to remove server scopes for {server_path}: {e}")
        return False


async def add_server_to_groups(
    server_path: str,
    group_names: list[str],
) -> bool:
    """
    Add a server and all its known tools/methods to specific groups.

    Gets the server's tools from the registry and adds them to the
    specified groups using the standard methods.

    Args:
        server_path: The server's path (e.g., '/example-server')
        group_names: List of group names to add the server to

    Returns:
        True if successful, False otherwise
    """
    try:
        scope_repo = get_scope_repository()

        # Get server info to find its tools
        server_info = await server_service.get_server_info(server_path)
        if not server_info:
            logger.error(f"Server {server_path} not found in registry")
            return False

        # Get the tools from the last health check
        tool_list = server_info.get("tool_list", [])
        tool_names = [
            tool["name"] for tool in tool_list if isinstance(tool, dict) and "name" in tool
        ]

        logger.info(f"Found {len(tool_names)} tools for server {server_path}: {tool_names}")

        # Add server to each group
        for group_name in group_names:
            # Check if group exists
            if not await scope_repo.group_exists(group_name):
                logger.warning(f"Group {group_name} not found in scopes")
                continue

            # Add server to this group
            await scope_repo.add_server_scope(
                server_path=server_path,
                scope_name=group_name,
                methods=STANDARD_METHODS,
                tools=tool_names,
            )

            # Add to UI-Scopes for web interface visibility
            server_name = server_info.get("server_name", server_path.lstrip("/").rstrip("/"))
            await scope_repo.add_server_to_ui_scopes(
                group_name=group_name,
                server_name=server_name,
            )

            logger.info(f"Added server {server_path} to group {group_name}")

        # Reload auth server
        await _trigger_auth_server_reload()

        return True

    except Exception as e:
        logger.error(f"Failed to add server {server_path} to groups {group_names}: {e}")
        return False


async def remove_server_from_groups(
    server_path: str,
    group_names: list[str],
) -> bool:
    """
    Remove a server from specific groups.

    Args:
        server_path: The server's path (e.g., '/example-server')
        group_names: List of group names to remove the server from

    Returns:
        True if successful, False otherwise
    """
    try:
        scope_repo = get_scope_repository()

        # Get server info for UI-Scopes updates
        server_info = await server_service.get_server_info(server_path)
        if server_info:
            server_name = server_info.get("server_name", server_path.lstrip("/").rstrip("/"))
        else:
            # If server not found, derive name from path
            server_name = server_path.lstrip("/").rstrip("/")

        # Remove server from each group
        for group_name in group_names:
            # Check if group exists
            if not await scope_repo.group_exists(group_name):
                logger.warning(f"Group {group_name} not found in scopes")
                continue

            # Remove server from this group
            await scope_repo.remove_server_scope(
                server_path=server_path,
                scope_name=group_name,
            )

            # Remove from UI-Scopes
            await scope_repo.remove_server_from_ui_scopes(
                group_name=group_name,
                server_name=server_name,
            )

            logger.info(f"Removed server {server_path} from group {group_name}")

        # Reload auth server
        await _trigger_auth_server_reload()

        return True

    except Exception as e:
        logger.error(f"Failed to remove server {server_path} from groups {group_names}: {e}")
        return False


async def create_group(
    group_name: str,
    description: str = "",
) -> bool:
    """
    Create a new group in scopes and add it to group_mappings.

    Args:
        group_name: Name of the group (e.g., 'mcp-servers-custom/read')
        description: Optional description

    Returns:
        True if successful, False otherwise
    """
    try:
        scope_repo = get_scope_repository()

        # Check if group already exists
        if await scope_repo.group_exists(group_name):
            logger.warning(f"Group {group_name} already exists in scopes")
            return False

        # Create the group
        await scope_repo.create_group(
            group_name=group_name,
            description=description,
        )

        logger.info(
            f"Successfully created group {group_name} in scopes, group_mappings, and UI-Scopes"
        )

        # Reload auth server
        await _trigger_auth_server_reload()

        return True

    except Exception as e:
        logger.error(f"Failed to create group {group_name} in scopes: {e}")
        return False


async def delete_group(
    group_name: str,
    remove_from_mappings: bool = True,
) -> bool:
    """
    Delete a group from scopes and optionally from group_mappings.

    Args:
        group_name: Name of the group to delete
        remove_from_mappings: Whether to remove from group_mappings section

    Returns:
        True if successful, False otherwise
    """
    try:
        scope_repo = get_scope_repository()

        # Check if group exists
        if not await scope_repo.group_exists(group_name):
            logger.warning(f"Group {group_name} not found in scopes")
            return False

        # Delete the group
        await scope_repo.delete_group(
            group_name=group_name,
            remove_from_mappings=remove_from_mappings,
        )

        logger.info(f"Successfully deleted group {group_name} from scopes")

        # Reload auth server
        await _trigger_auth_server_reload()

        return True

    except Exception as e:
        logger.error(f"Failed to delete group {group_name} from scopes: {e}")
        return False


async def import_group(
    scope_name: str,
    scope_type: str = "server_scope",
    description: str = "",
    server_access: list = None,
    group_mappings: list = None,
    ui_permissions: dict = None,
    agent_access: list = None,
    is_idp_managed: bool = True,
    allow_privileged: bool = False,
) -> bool:
    """
    Import a complete group definition with all document types.

    This creates/updates all group-related data structures based on the provided
    definition. The group_name is derived from scope_name.

    Args:
        scope_name: Name of the scope/group
        scope_type: Type of scope (default: server_scope)
        description: Description of the group
        server_access: Optional list of server access definitions
        group_mappings: Optional list of group names this group maps to
        ui_permissions: Optional dictionary of UI permissions
        agent_access: Optional list of agent paths this group can access
        is_idp_managed: Whether PATCH/DELETE should call the upstream IdP.
            See issue #946. Defaults to True to preserve pre-#946 behavior.
        allow_privileged: Whether the caller is admin-authorized to write a
            privileged (admin-conferring) scope. Defaults to False (fail-closed);
            only admin-gated callers pass True. Enforced as defense-in-depth at
            BOTH this service layer (privileged scope_name / group_mappings /
            ui_permissions) and the repository layer (admin-conferring
            ui_permissions), so a caller reaching either layer without a
            route-level admin check cannot self-assign admin.

    Returns:
        True if successful, False otherwise

    Raises:
        PrivilegedScopeWriteError: If a non-admin actor attempts to write a
            privileged scope/group.
    """
    # Defense-in-depth: a privileged scope write must come from an admin-authorized
    # caller, regardless of any route-level check. This service-layer guard is
    # broader than the repository's _grants_admin (it also catches a privileged
    # scope_name and privileged group_mappings, e.g. mapping a group to
    # mcp-registry-admin -- the original privesc vector).
    if not allow_privileged and _import_touches_privileged_scope(
        scope_name, group_mappings, ui_permissions
    ):
        logger.warning(
            "Rejected non-admin privileged scope write for '%s' (group_mappings=%s)",
            scope_name,
            group_mappings,
        )
        raise PrivilegedScopeWriteError(
            f"Writing privileged scope '{scope_name}' requires administrator privileges"
        )

    try:
        scope_repo = get_scope_repository()

        # Use scope_name as the group_name
        group_name = scope_name

        # Call repository import_group method
        success = await scope_repo.import_group(
            group_name=group_name,
            description=description,
            server_access=server_access,
            group_mappings=group_mappings,
            ui_permissions=ui_permissions,
            agent_access=agent_access,
            is_idp_managed=is_idp_managed,
            allow_privileged=allow_privileged,
        )

        if success:
            logger.info(f"Successfully imported group definition for {group_name}")
        else:
            logger.error(f"Failed to import group definition for {group_name}")

        return success

    except Exception as e:
        logger.error(f"Failed to import group {scope_name}: {e}")
        return False


async def get_group(group_name: str) -> dict[str, Any]:
    """
    Get full details of a specific group from scopes storage.

    Args:
        group_name: Name of the group

    Returns:
        Dict with complete group information including server_access, group_mappings, and ui_permissions
    """
    try:
        scope_repo = get_scope_repository()

        # Get group details
        group_data = await scope_repo.get_group(group_name)

        if not group_data:
            logger.warning(f"Group {group_name} not found in scopes")
            return None

        logger.info(f"Retrieved group {group_name} from scopes")
        return group_data

    except Exception as e:
        logger.error(f"Failed to get group {group_name} from scopes: {e}")
        return None


async def list_groups() -> dict[str, Any]:
    """
    List all groups defined in scopes.

    Returns:
        Dict with group information including server counts and mappings
    """
    try:
        scope_repo = get_scope_repository()

        # Get all groups
        groups_data = await scope_repo.list_groups()

        logger.info(f"Found {groups_data.get('total_count', 0)} groups in scopes")

        return groups_data

    except Exception as e:
        logger.error(f"Failed to list groups from scopes: {e}")
        return {
            "total_count": 0,
            "groups": {},
            "error": str(e),
        }


async def group_exists(
    group_name: str,
) -> bool:
    """
    Check if a group exists in scopes.

    Args:
        group_name: Name of the group to check

    Returns:
        True if group exists, False otherwise
    """
    try:
        scope_repo = get_scope_repository()
        return await scope_repo.group_exists(group_name)
    except Exception as e:
        logger.error(f"Error checking if group exists in scopes: {e}")
        return False


async def trigger_auth_server_reload() -> bool:
    """
    Trigger the auth server to reload its scopes configuration.

    Public wrapper around the private function.

    Returns:
        True if successful, False otherwise
    """
    return await _trigger_auth_server_reload()


async def add_group_mapping_to_scope(
    scope_name: str,
    group_id: str,
) -> bool:
    """
    Add a group mapping (IdP group ID) to an existing scope's group_mappings.

    This is used when creating a group in an IdP (like Entra ID) that returns
    group IDs (GUIDs) in tokens. We need to map both the group name and ID
    so that token validation works correctly.

    Args:
        scope_name: Name of the scope to update
        group_id: IdP group ID to add to group_mappings

    Returns:
        True if successful, False otherwise
    """
    try:
        scope_repo = get_scope_repository()

        # Use the existing add_group_mapping method which adds
        # an entry to the scope's group_mappings array
        success = await scope_repo.add_group_mapping(scope_name, group_id)

        if success:
            logger.info(f"Added group ID {group_id} to scope {scope_name} group_mappings")
        else:
            logger.error(f"Failed to add group ID {group_id} to scope {scope_name} group_mappings")
        return success

    except Exception as e:
        logger.error(f"Error adding group mapping to scope {scope_name}: {e}")
        return False
