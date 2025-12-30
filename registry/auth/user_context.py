from __future__ import annotations

from typing import (
    Any,
    Optional,
)

from .scopes import (
    get_accessible_agents_for_user,
    get_accessible_services_for_user,
    get_ui_permissions_for_user,
    get_user_accessible_servers,
    user_can_modify_servers,
    user_has_wildcard_access,
)


def build_registry_user_context(
    *,
    username: str,
    groups: list[str],
    scopes: list[str],
    auth_method: str,
    provider: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a normalized user context payload used by registry routes."""
    ui_permissions = get_ui_permissions_for_user(scopes)
    accessible_servers = get_user_accessible_servers(scopes)
    accessible_services = get_accessible_services_for_user(ui_permissions)
    accessible_agents = get_accessible_agents_for_user(ui_permissions)
    can_modify = user_can_modify_servers(groups, scopes)

    context: dict[str, Any] = {
        "username": username,
        "groups": groups,
        "scopes": scopes,
        "auth_method": auth_method,
        "provider": provider,
        "accessible_servers": accessible_servers,
        "accessible_services": accessible_services,
        "accessible_agents": accessible_agents,
        "ui_permissions": ui_permissions,
        "can_modify_servers": can_modify,
        "is_admin": user_has_wildcard_access(scopes),
    }

    if extra:
        context.update(extra)

    return context

