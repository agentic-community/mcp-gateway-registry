"""The read-only ``view_all`` permission (issue #2099): parsing, admin, discovery."""

import pytest

from registry.auth.asset_permissions import (
    VIEW_ALL_PERMISSION,
    accessible_resources_for,
    can_view_all,
    ui_permissions_grant_view_all,
    user_can_discover_asset,
    user_has_asset_permission,
)
from registry.auth.dependencies import _user_is_admin
from registry.auth.privileged_constants import is_admin_conferring_action
from registry.auth.tool_filter import filter_tools_for_user

HOLDER_PERMISSIONS: dict = {VIEW_ALL_PERMISSION: ["all"]}
ALL_ACTIONS: tuple[str, ...] = ("list", "get", "create", "modify", "delete", "toggle")
TOOLS: list[dict] = [{"name": "read_note"}, {"name": "delete_note"}]


def _holder(
    **extra,
) -> dict:
    """A non-admin user context whose only grant is view_all."""
    context = {
        "username": "ro-admin",
        "is_admin": False,
        "groups": ["registry-read-only-admin"],
        "accessible_servers": [],
        "accessible_services": [],
        "accessible_agents": [],
        "accessible_skills": [],
        "accessible_tools": {},
        "ui_permissions": dict(HOLDER_PERMISSIONS),
    }
    context.update(extra)
    return context


@pytest.mark.unit
class TestViewAllIsNotAdmin:
    """view_all can never confer admin."""

    def test_view_all_is_not_admin_conferring(self):
        assert is_admin_conferring_action(VIEW_ALL_PERMISSION) is False

    def test_user_is_admin_false_for_holder(self):
        assert _user_is_admin(HOLDER_PERMISSIONS) is False


@pytest.mark.unit
class TestViewAllGrantShape:
    """Only an exact "all" in a list grants view_all."""

    def test_all_grants(self):
        assert ui_permissions_grant_view_all({"view_all": ["all"]}) is True

    @pytest.mark.parametrize(
        "value",
        [["some-agent"], "all", [], None, {"all": True}, ["ALL"]],
    )
    def test_other_shapes_grant_nothing(self, value):
        assert ui_permissions_grant_view_all({"view_all": value}) is False

    def test_missing_key_grants_nothing(self):
        assert ui_permissions_grant_view_all({"list_agents": ["all"]}) is False

    def test_can_view_all_handles_missing_context(self):
        assert can_view_all(None) is False
        assert can_view_all({}) is False


@pytest.mark.unit
class TestDiscovery:
    """view_all opens discovery for every family, through the read helper only."""

    @pytest.mark.parametrize("family", ["server", "agent", "skill"])
    def test_discover_asset_honors_view_all(self, family):
        assert user_can_discover_asset(family, "anything", _holder()) is True

    def test_discover_custom_entity_honors_view_all(self):
        assert user_can_discover_asset("custom_entity", "dataset", _holder(), type_name="dataset")

    def test_discover_without_view_all_uses_list_grant(self):
        context = _holder(ui_permissions={"list_agents": ["agent-a"]})
        assert user_can_discover_asset("agent", "agent-a", context) is True
        assert user_can_discover_asset("agent", "agent-b", context) is False

    @pytest.mark.parametrize("family", ["server", "agent", "skill"])
    @pytest.mark.parametrize("action", ALL_ACTIONS)
    def test_user_has_asset_permission_ignores_view_all(self, family, action):
        """Writes and variable-action routes call this; view_all must not reach it."""
        try:
            result = user_has_asset_permission(family, action, "anything", _holder())
        except ValueError:
            pytest.skip(f"{family} has no {action} permission")
        assert result is False

    @pytest.mark.parametrize("family", ["server", "agent", "skill"])
    def test_accessible_lists_unchanged(self, family):
        """Token binding reads these lists, so view_all must not change them."""
        assert accessible_resources_for(family, HOLDER_PERMISSIONS) == []


@pytest.mark.unit
class TestToolLists:
    """Holders see full tool definitions in registry REST responses."""

    def test_holder_sees_every_tool(self):
        result = filter_tools_for_user("srv", TOOLS, _holder(), server_path="/srv")
        assert [tool["name"] for tool in result] == ["read_note", "delete_note"]

    def test_non_holder_without_grants_sees_none(self):
        context = _holder(ui_permissions={})
        assert filter_tools_for_user("srv", TOOLS, context, server_path="/srv") == []
