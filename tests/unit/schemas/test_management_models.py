"""Unit tests for the ScopeConfig models in registry/schemas/management.py.

Added for issue #1494 (scope creation from the UI): scope_config on the IAM
group create/update endpoints is now a validated Pydantic model instead of a
free-form dict.
"""

import pytest
from pydantic import ValidationError

from registry.schemas.management import (
    AgentAccessRule,
    GroupCreateRequest,
    GroupUpdateRequest,
    ScopeConfig,
    ServerAccessRule,
)


@pytest.mark.unit
class TestServerAccessRule:
    def test_minimal_rule(self):
        rule = ServerAccessRule(server="currenttime")
        assert rule.server == "currenttime"
        assert rule.methods == []
        assert rule.tools == []

    def test_tools_accepts_star_string(self):
        rule = ServerAccessRule(server="api", tools="*")
        assert rule.tools == "*"

    def test_empty_server_rejected(self):
        with pytest.raises(ValidationError):
            ServerAccessRule(server="")

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError):
            ServerAccessRule(server="api", methodz=["tools/call"])


@pytest.mark.unit
class TestAgentAccessRule:
    def test_minimal_rule(self):
        rule = AgentAccessRule(agent="/my-planner")
        assert rule.agent == "/my-planner"
        assert rule.actions == []

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError):
            AgentAccessRule(agent="/a", grant="invoke")


@pytest.mark.unit
class TestScopeConfig:
    def test_defaults_are_none_except_create_in_idp(self):
        """Omitted access fields stay None so PATCH can preserve existing values."""
        cfg = ScopeConfig()
        assert cfg.server_access is None
        assert cfg.ui_permissions is None
        assert cfg.agent_access is None
        assert cfg.group_mappings is None
        assert cfg.create_in_idp is False

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ScopeConfig(sever_access=[])  # typo'd key
        assert "sever_access" in str(exc_info.value)

    def test_server_access_union_discrimination(self):
        """Dicts with a server key become ServerAccessRule; agent key -> AgentAccessRule."""
        cfg = ScopeConfig(
            server_access=[
                {"server": "api", "methods": ["tools/call"], "tools": ["t1"]},
                {"agent": "/my-planner", "actions": ["invoke_agent"]},
            ]
        )
        assert isinstance(cfg.server_access[0], ServerAccessRule)
        assert isinstance(cfg.server_access[1], AgentAccessRule)

    def test_rule_missing_server_and_agent_rejected(self):
        with pytest.raises(ValidationError):
            ScopeConfig(server_access=[{"methods": ["tools/call"]}])

    def test_ui_permissions_shape_enforced(self):
        with pytest.raises(ValidationError):
            ScopeConfig(ui_permissions={"list_service": "not-a-list"})

    def test_frontend_shaped_payload_validates(self):
        """The exact JSON _buildScopeJson produces (minus name/description) validates."""
        cfg = ScopeConfig(
            server_access=[
                {
                    "server": "currenttime",
                    "methods": ["initialize", "tools/list", "tools/call"],
                    "tools": ["current_time_by_timezone"],
                }
            ],
            ui_permissions={
                "list_service": ["currenttime"],
                "health_check_service": ["currenttime"],
            },
            agent_access=[],
            group_mappings=["currenttime-users"],
            create_in_idp=False,
        )
        assert cfg.group_mappings == ["currenttime-users"]

    def test_model_dump_exclude_unset_preserves_wire_shape(self):
        """Rules dumped with exclude_unset match what the caller sent (no added keys)."""
        cfg = ScopeConfig(server_access=[{"server": "api"}])
        dumped = cfg.server_access[0].model_dump(exclude_unset=True)
        assert dumped == {"server": "api"}


@pytest.mark.unit
class TestRequestModels:
    def test_create_request_accepts_typed_scope_config(self):
        req = GroupCreateRequest(
            name="g",
            scope_config={"create_in_idp": True, "server_access": [{"server": "api"}]},
        )
        assert isinstance(req.scope_config, ScopeConfig)
        assert req.scope_config.create_in_idp is True

    def test_update_request_accepts_typed_scope_config(self):
        req = GroupUpdateRequest(scope_config={"ui_permissions": {"list_service": ["api"]}})
        assert isinstance(req.scope_config, ScopeConfig)

    def test_create_request_rejects_malformed_scope_config(self):
        with pytest.raises(ValidationError):
            GroupCreateRequest(name="g", scope_config={"unknown_key": 1})
