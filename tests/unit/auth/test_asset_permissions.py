"""Unit tests for the canonical asset-permission module.

Guards the single source of truth for per-asset UI-Scope naming and checks
shared by servers, agents, skills, and custom entities:

- The (family, action) -> on-disk scope-name map returns the EXACT persisted
  names (no pluralization normalization) and delegates the dynamic
  custom-entity family to entity_scope.
- user_has_asset_permission is uniform and fails closed across families.
- accessible_resources_for is a pure ui_permissions projection (no is_admin).
"""

import logging

import pytest

from registry.auth.asset_permissions import (
    accessible_resources_for,
    asset_scope_name,
    user_has_asset_permission,
)

logger = logging.getLogger(__name__)


@pytest.mark.unit
class TestAssetScopeName:
    """The (family, action) -> on-disk name map preserves legacy names."""

    def test_server_names_are_singular_service(self):
        assert asset_scope_name("server", "list") == "list_service"
        assert asset_scope_name("server", "create") == "register_service"
        assert asset_scope_name("server", "modify") == "modify_service"
        assert asset_scope_name("server", "delete") == "delete_service"
        assert asset_scope_name("server", "toggle") == "toggle_service"
        assert asset_scope_name("server", "health_check") == "health_check_service"

    def test_agent_names_are_plural_list_with_publish_create(self):
        assert asset_scope_name("agent", "list") == "list_agents"
        assert asset_scope_name("agent", "get") == "get_agent"
        assert asset_scope_name("agent", "create") == "publish_agent"
        assert asset_scope_name("agent", "modify") == "modify_agent"
        assert asset_scope_name("agent", "delete") == "delete_agent"

    def test_skill_names_are_plural_list_with_publish_create(self):
        assert asset_scope_name("skill", "list") == "list_skills"
        assert asset_scope_name("skill", "create") == "publish_skill"
        assert asset_scope_name("skill", "modify") == "modify_skill"
        assert asset_scope_name("skill", "delete") == "delete_skill"
        assert asset_scope_name("skill", "toggle") == "toggle_skill"

    def test_custom_entity_delegates_to_entity_scope(self):
        assert asset_scope_name("custom_entity", "list", "policy") == "list_policy_entity"
        assert asset_scope_name("custom_entity", "create", "dataset") == "create_dataset_entity"
        assert asset_scope_name("custom_entity", "modify", "n8n") == "modify_n8n_entity"
        assert asset_scope_name("custom_entity", "delete", "policy") == "delete_policy_entity"

    def test_custom_entity_requires_type_name(self):
        with pytest.raises(ValueError, match="requires a type_name"):
            asset_scope_name("custom_entity", "list")

    def test_unknown_pair_raises(self):
        with pytest.raises(ValueError, match="No scope defined"):
            asset_scope_name("server", "get")  # servers have no per-item get scope
        with pytest.raises(ValueError, match="No scope defined"):
            asset_scope_name("bogus", "list")


@pytest.mark.unit
class TestUserHasAssetPermission:
    """Uniform, fail-closed per-asset check."""

    def test_admin_bypasses_all_families(self):
        ctx = {"is_admin": True, "ui_permissions": {}}
        assert user_has_asset_permission("server", "modify", "anything", ctx)
        assert user_has_asset_permission("skill", "delete", "anything", ctx)
        assert user_has_asset_permission("custom_entity", "list", "policy", ctx, type_name="policy")

    def test_named_grant_allows_only_that_resource(self):
        ctx = {"is_admin": False, "ui_permissions": {"modify_skill": ["cool-skill"]}}
        assert user_has_asset_permission("skill", "modify", "cool-skill", ctx)
        assert not user_has_asset_permission("skill", "modify", "other-skill", ctx)

    def test_all_grant_allows_any_resource(self):
        ctx = {"is_admin": False, "ui_permissions": {"list_skills": ["all"]}}
        assert user_has_asset_permission("skill", "list", "whatever", ctx)

    def test_missing_ui_permissions_denies(self):
        assert not user_has_asset_permission("skill", "list", "x", {"is_admin": False})
        assert not user_has_asset_permission("skill", "list", "x", {})

    def test_empty_grant_denies(self):
        ctx = {"is_admin": False, "ui_permissions": {"list_skills": []}}
        assert not user_has_asset_permission("skill", "list", "x", ctx)

    def test_custom_entity_scope_keyed_by_type(self):
        ctx = {"is_admin": False, "ui_permissions": {"list_policy_entity": ["all"]}}
        assert user_has_asset_permission("custom_entity", "list", "policy", ctx, type_name="policy")
        # A different type's scope must not grant this one.
        assert not user_has_asset_permission(
            "custom_entity", "list", "dataset", ctx, type_name="dataset"
        )


@pytest.mark.unit
class TestAccessibleResourcesFor:
    """Pure ui_permissions projection; no is_admin short-circuit."""

    def test_all_grant_returns_all(self):
        assert accessible_resources_for("skill", {"list_skills": ["all"]}) == ["all"]
        assert accessible_resources_for("server", {"list_service": ["all"]}) == ["all"]
        assert accessible_resources_for("agent", {"list_agents": ["all"]}) == ["all"]

    def test_named_grant_returns_names(self):
        assert accessible_resources_for("skill", {"list_skills": ["a", "b"]}) == ["a", "b"]

    def test_no_grant_returns_empty(self):
        assert accessible_resources_for("skill", {}) == []
        assert accessible_resources_for("skill", {"list_skills": []}) == []

    def test_does_not_consult_is_admin(self):
        # Even if an is_admin-looking key is present, the projection ignores it:
        # the admin bypass belongs at the enforcement site, not the projection.
        assert accessible_resources_for("skill", {"list_skills": []}) == []
