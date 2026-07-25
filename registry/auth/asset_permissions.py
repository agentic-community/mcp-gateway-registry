"""Canonical asset-permission scope naming and checks.

Single source of truth for the per-asset UI-Scope permission model shared by
every first-class asset family: servers, agents, skills, and custom entities.
Prior to this module each family open-coded its own scope-name strings and its
own ``"all"``-or-named check, so the four families drifted (different verbs,
different pluralization, subtly different fail-closed behavior). Centralizing
here means:

- The irregular on-disk scope names live in ONE place (``_SCOPE_NAMES``). New
  code refers to a ``(family, action)`` tuple and never has to remember whether
  create is spelled ``register``/``publish``/``create`` or whether the resource
  is singular/plural.
- Every family gets the same fail-closed check semantics via
  ``user_has_asset_permission`` and the same discovery projection via
  ``accessible_resources_for``.

IMPORTANT — the on-disk names are load-bearing legacy. They are persisted as
``ui_permissions`` keys in every deployment's scope store, in seed data, and in
the frontend permission editor. This module deliberately preserves the EXACT
existing names (``list_service`` singular, ``list_agents``/``list_skills``
plural, ``register_service``/``publish_agent``/``publish_skill``); it does NOT
normalize them. Renaming a persisted scope key is a separate, breaking data
migration and must never be folded into a code refactor.

This is a dependency-free leaf module (only the standard library) so any layer
can import it without a cycle. The dynamic custom-entity family delegates its
naming to ``registry.services.custom_entity_scopes.entity_scope`` via a lazy
import inside the one function that needs it, keeping this module import-clean.
"""

from typing import Literal

# Asset families with a per-asset UI-Scope permission layer.
AssetFamily = Literal["server", "agent", "skill", "custom_entity"]

# Logical actions. Not every family supports every action; ``_SCOPE_NAMES``
# defines exactly which (family, action) pairs are valid. ``list`` is the
# discovery action every family gates on.
AssetAction = Literal["list", "get", "create", "modify", "delete", "toggle", "health_check"]

# (family, action) -> the EXACT on-disk ui_permission key. These strings match
# what is persisted in the scope store today; do not "fix" them here (see the
# module docstring). custom_entity is intentionally ABSENT: its keys are
# per-type dynamic (``list_<type>_entity``) and are produced by
# ``asset_scope_name`` via ``entity_scope`` instead of a static lookup.
_SCOPE_NAMES: dict[tuple[str, str], str] = {
    # Servers (issue #663/#717 naming: singular "service").
    ("server", "list"): "list_service",
    ("server", "create"): "register_service",
    ("server", "modify"): "modify_service",
    ("server", "delete"): "delete_service",
    ("server", "toggle"): "toggle_service",
    ("server", "health_check"): "health_check_service",
    # Agents (plural "agents"; "get_agent" singular is a real read scope).
    ("agent", "list"): "list_agents",
    ("agent", "get"): "get_agent",
    ("agent", "create"): "publish_agent",
    ("agent", "modify"): "modify_agent",
    ("agent", "delete"): "delete_agent",
    ("agent", "toggle"): "toggle_agent",
    # Skills (plural "skills"; create is "publish", matching agents).
    ("skill", "list"): "list_skills",
    ("skill", "create"): "publish_skill",
    ("skill", "modify"): "modify_skill",
    ("skill", "delete"): "delete_skill",
    ("skill", "toggle"): "toggle_skill",
}


def asset_scope_name(
    family: str,
    action: str,
    type_name: str | None = None,
) -> str:
    """Return the on-disk ui_permission key for a (family, action).

    For the static families (server/agent/skill) this is a table lookup. For the
    dynamic ``custom_entity`` family the key is per-type
    (``<action>_<type_name>_entity``), so this delegates to the custom-entity
    naming SSOT ``entity_scope`` — the single place that owns that convention.

    Args:
        family: The asset family (server/agent/skill/custom_entity).
        action: The logical action (list/create/modify/delete/...).
        type_name: The custom type name; REQUIRED when family == custom_entity,
            ignored otherwise.

    Returns:
        The ui_permission key string as persisted in the scope store.

    Raises:
        ValueError: If family == custom_entity without a type_name, or if the
            (family, action) pair is not a defined permission.
    """
    if family == "custom_entity":
        if not type_name:
            raise ValueError("custom_entity scope name requires a type_name")
        # Lazy import: custom_entity_scopes lives in the service layer; importing
        # it at module load would make this leaf module depend on services.
        from ..services.custom_entity_scopes import entity_scope

        return entity_scope(action, type_name)

    try:
        return _SCOPE_NAMES[(family, action)]
    except KeyError:
        raise ValueError(
            f"No scope defined for asset family '{family}' action '{action}'"
        ) from None


def user_has_asset_permission(
    family: str,
    action: str,
    resource_name: str,
    user_context: dict,
    *,
    type_name: str | None = None,
) -> bool:
    """Return True if the caller may perform ``action`` on ``resource_name``.

    Canonical per-asset authorization check, uniform across all families:

    - Admin (``user_context["is_admin"]``) is the catch-all bypass.
    - Otherwise the caller must hold the family/action scope for this specific
      resource (or ``"all"``) in their ``ui_permissions``.
    - Fails closed: a missing/empty ``ui_permissions`` denies.

    Args:
        family: The asset family (server/agent/skill/custom_entity).
        action: The logical action (list/create/modify/delete/...).
        resource_name: The specific resource being acted on (server/agent/skill
            name, custom-type name, etc.). Compared against the scope's granted
            resource list.
        user_context: The authenticated request context.
        type_name: The custom type name; REQUIRED when family == custom_entity.

    Returns:
        True if access is permitted, False otherwise.
    """
    if user_context.get("is_admin", False):
        return True
    scope = asset_scope_name(family, action, type_name)
    granted = (user_context.get("ui_permissions") or {}).get(scope) or []
    return "all" in granted or resource_name in granted


def accessible_resources_for(
    family: str,
    ui_permissions: dict[str, list[str]],
) -> list[str]:
    """Project the resources a family's list scope grants for discovery.

    Faithful drop-in for the historical ``get_accessible_services_for_user`` /
    ``get_accessible_agents_for_user``: a PURE ``ui_permissions`` projection with
    NO ``is_admin`` handling. The admin bypass is applied at the enforcement site
    (``user_can_access_*`` / ``user_has_asset_permission``), not baked into the
    stored ``accessible_*`` value — keeping the derived context byte-identical to
    the pre-refactor behavior.

    - ``list_<family>: ["all"]`` -> ``["all"]``.
    - Otherwise the explicitly named resources (may be empty -> discovers
      nothing, so lacking the list scope hides even public resources; fail
      closed).

    Args:
        family: The asset family (server/agent/skill). custom_entity is per-type
            and has no single global list projection; use
            ``user_has_asset_permission(..., action="list", type_name=...)``.
        ui_permissions: The user's ui_permissions dict (from
            ``get_ui_permissions_for_user``).

    Returns:
        ``["all"]`` or the list of named resources the caller may discover.
    """
    scope = asset_scope_name(family, "list")
    granted = ui_permissions.get(scope) or []
    if "all" in granted:
        return ["all"]
    return list(granted)
