from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
)

import yaml

logger = logging.getLogger(__name__)


def load_scopes_config() -> dict[str, Any]:
    """Load the scopes configuration from `auth_server/scopes.yml`."""
    try:
        import os

        scopes_path = os.getenv("SCOPES_CONFIG_PATH")
        logger.debug("SCOPES_CONFIG_PATH env var: %s", scopes_path)

        if not scopes_path:
            scopes_file = Path(__file__).parent.parent.parent / "auth_server" / "scopes.yml"
        else:
            scopes_file = Path(scopes_path)

        if not scopes_file.exists():
            alt_scopes_file = (
                Path(__file__).parent.parent.parent / "auth_server" / "auth_config" / "scopes.yml"
            )
            if alt_scopes_file.exists():
                scopes_file = alt_scopes_file
                logger.debug(
                    "Scopes config file not found at primary location, using EFS mount location: %s",
                    scopes_file,
                )

        logger.debug("Looking for scopes config at: %s (exists=%s)", scopes_file, scopes_file.exists())

        if not scopes_file.exists():
            logger.warning("Scopes config file not found at %s", scopes_file)
            return {}

        with open(scopes_file, "r") as handle:
            config = yaml.safe_load(handle) or {}

        if not isinstance(config, dict):
            logger.warning("Scopes config root must be a mapping; got %s", type(config).__name__)
            return {}

        logger.info(
            "Loaded scopes configuration with %s group mappings",
            len(config.get("group_mappings", {})),
        )
        return config
    except Exception as exc:
        logger.error("Failed to load scopes configuration: %s", exc, exc_info=True)
        return {}


SCOPES_CONFIG: dict[str, Any] = load_scopes_config()


def map_cognito_groups_to_scopes(
    groups: list[str],
) -> list[str]:
    """Map Cognito groups to MCP scopes using the scopes.yml configuration."""
    scopes: list[str] = []
    group_mappings = SCOPES_CONFIG.get("group_mappings", {})

    for group in groups:
        if group in group_mappings:
            group_scopes = group_mappings[group]
            scopes.extend(group_scopes)
            logger.debug("Mapped group '%s' to scopes: %s", group, group_scopes)
        else:
            logger.debug("No scope mapping found for group: %s", group)

    seen: set[str] = set()
    unique_scopes: list[str] = []
    for scope in scopes:
        if scope not in seen:
            seen.add(scope)
            unique_scopes.append(scope)

    logger.info("Final mapped scopes: %s", unique_scopes)
    return unique_scopes


def get_ui_permissions_for_user(
    user_scopes: list[str],
) -> dict[str, list[str]]:
    """Get UI permissions for a user based on their scopes."""
    ui_permissions: dict[str, set[str]] = {}
    ui_scopes = SCOPES_CONFIG.get("UI-Scopes", {})

    for scope in user_scopes:
        if scope in ui_scopes:
            scope_config = ui_scopes[scope]
            logger.debug("Processing UI scope '%s' with config: %s", scope, scope_config)

            for permission, services in scope_config.items():
                if permission not in ui_permissions:
                    ui_permissions[permission] = set()

                if services == ["all"] or (isinstance(services, list) and "all" in services):
                    ui_permissions[permission].add("all")
                    logger.debug("UI permission '%s' granted for all services", permission)
                else:
                    if isinstance(services, list):
                        ui_permissions[permission].update(services)
                        logger.debug(
                            "UI permission '%s' granted for services: %s",
                            permission,
                            services,
                        )

    result = {key: list(value) for key, value in ui_permissions.items()}
    logger.info("Final UI permissions for user: %s", result)
    return result


def user_has_ui_permission_for_service(
    permission: str,
    service_name: str,
    user_ui_permissions: dict[str, list[str]],
) -> bool:
    """Check if user has a specific UI permission for a specific service."""
    if permission not in user_ui_permissions:
        return False

    allowed_services = user_ui_permissions[permission]
    has_permission = "all" in allowed_services or service_name in allowed_services

    logger.debug(
        "Permission check: %s for %s = %s (allowed: %s)",
        permission,
        service_name,
        has_permission,
        allowed_services,
    )
    return has_permission


def get_accessible_services_for_user(
    user_ui_permissions: dict[str, list[str]],
) -> list[str]:
    """Get list of services the user can see based on their list_service permission."""
    list_permissions = user_ui_permissions.get("list_service", [])

    if "all" in list_permissions:
        return ["all"]

    return list_permissions


def get_accessible_agents_for_user(
    user_ui_permissions: dict[str, list[str]],
) -> list[str]:
    """Get list of agents the user can see based on their list_agents permission."""
    list_permissions = user_ui_permissions.get("list_agents", [])

    if "all" in list_permissions:
        return ["all"]

    return list_permissions


def get_servers_for_scope(
    scope: str,
) -> list[str]:
    """Get list of server names that a scope provides access to."""
    scope_config = SCOPES_CONFIG.get(scope, [])
    server_names: list[str] = []

    for server_config in scope_config:
        if isinstance(server_config, dict) and "server" in server_config:
            server_names.append(server_config["server"])

    return list(set(server_names))


def user_has_wildcard_access(
    user_scopes: list[str],
) -> bool:
    """Check if user has wildcard access to all servers via their scopes."""
    for scope in user_scopes:
        servers = get_servers_for_scope(scope)
        if "*" in servers:
            logger.debug("User scope '%s' grants wildcard access to all servers", scope)
            return True

    return False


def get_user_accessible_servers(
    user_scopes: list[str],
) -> list[str]:
    """Get list of all servers the user has access to based on their scopes."""
    accessible_servers: set[str] = set()

    logger.debug("get_user_accessible_servers called with scopes: %s", user_scopes)
    logger.debug("Available scope configs: %s", list(SCOPES_CONFIG.keys()))

    for scope in user_scopes:
        logger.debug("Processing scope: %s", scope)
        server_names = get_servers_for_scope(scope)
        logger.debug("Scope %s maps to servers: %s", scope, server_names)
        accessible_servers.update(server_names)

    logger.debug(
        "User with scopes %s has access to servers: %s",
        user_scopes,
        list(accessible_servers),
    )
    return list(accessible_servers)


def user_can_modify_servers(
    user_groups: list[str],
    user_scopes: list[str],
) -> bool:
    """Check if user can modify servers (toggle, edit)."""
    if "mcp-registry-admin" in user_groups:
        return True

    if "mcp-servers-unrestricted/execute" in user_scopes:
        return True

    if "mcp-registry-user" in user_groups and "mcp-registry-admin" not in user_groups:
        return False

    execute_scopes = [scope for scope in user_scopes if "/execute" in scope]
    return len(execute_scopes) > 0


def user_can_access_server(
    server_name: str,
    user_scopes: list[str],
) -> bool:
    """Check if user can access a specific server."""
    accessible_servers = get_user_accessible_servers(user_scopes)
    return server_name in accessible_servers

