"""Tests for AgentRegistrationRequest allowed_groups field and validators."""

import pytest

from registry.schemas.agent_models import AgentRegistrationRequest


MINIMAL_AGENT_KWARGS = {
    "name": "test-agent",
    "url": "https://example.com",
    "supported_protocol": "a2a",
}


@pytest.mark.unit
class TestAgentRegistrationRequestAllowedGroups:
    """Tests for allowed_groups on AgentRegistrationRequest."""

    def test_allowed_groups_defaults_to_empty_list(self):
        """allowed_groups should default to empty list."""
        req = AgentRegistrationRequest(**MINIMAL_AGENT_KWARGS)
        assert req.allowed_groups == []

    def test_allowed_groups_accepted_via_camel_case_alias(self):
        """allowedGroups alias should work."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="group-restricted",
            allowedGroups=["team-a", "team-b"],
        )
        assert req.allowed_groups == ["team-a", "team-b"]

    def test_allowed_groups_accepted_via_snake_case(self):
        """allowed_groups should work directly."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="group-restricted",
            allowed_groups=["team-a"],
        )
        assert req.allowed_groups == ["team-a"]

    def test_allowed_groups_from_comma_separated_string(self):
        """Comma-separated string should be normalized to list."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="group-restricted",
            allowed_groups="finance-team, dev-team, ops-team",
        )
        assert req.allowed_groups == ["finance-team", "dev-team", "ops-team"]

    def test_allowed_groups_string_strips_whitespace(self):
        """Whitespace around group names should be stripped."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="group-restricted",
            allowed_groups="  team-a ,  team-b  , team-c ",
        )
        assert req.allowed_groups == ["team-a", "team-b", "team-c"]

    def test_allowed_groups_list_strips_whitespace(self):
        """Whitespace in list elements should be stripped."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="group-restricted",
            allowed_groups=["  team-a  ", "team-b "],
        )
        assert req.allowed_groups == ["team-a", "team-b"]

    def test_allowed_groups_string_filters_empty_segments(self):
        """Empty segments from trailing commas should be filtered out."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="group-restricted",
            allowed_groups="team-a,,team-b,",
        )
        assert req.allowed_groups == ["team-a", "team-b"]

    def test_allowed_groups_none_normalizes_to_empty_list(self):
        """None should be normalized to empty list."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="public",
            allowed_groups=None,
        )
        assert req.allowed_groups == []

    def test_group_restricted_without_groups_raises_error(self):
        """group-restricted without allowed_groups should raise ValueError."""
        with pytest.raises(ValueError, match="requires at least one allowed_group"):
            AgentRegistrationRequest(
                **MINIMAL_AGENT_KWARGS,
                visibility="group-restricted",
                allowed_groups=[],
            )

    def test_group_restricted_with_empty_string_raises_error(self):
        """group-restricted with empty string should raise ValueError."""
        with pytest.raises(ValueError, match="requires at least one allowed_group"):
            AgentRegistrationRequest(
                **MINIMAL_AGENT_KWARGS,
                visibility="group-restricted",
                allowed_groups="",
            )

    def test_public_visibility_with_empty_groups_is_valid(self):
        """Public visibility should not require allowed_groups."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="public",
        )
        assert req.allowed_groups == []

    def test_private_visibility_with_empty_groups_is_valid(self):
        """Private visibility should not require allowed_groups."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="private",
        )
        assert req.allowed_groups == []

    def test_invalid_group_name_format_raises_error(self):
        """Group names with special characters should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid group name"):
            AgentRegistrationRequest(
                **MINIMAL_AGENT_KWARGS,
                visibility="group-restricted",
                allowed_groups=["valid-team", "invalid team with spaces"],
            )

    def test_group_name_with_allowed_special_chars(self):
        """Group names with hyphens, underscores, dots should be accepted."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="group-restricted",
            allowed_groups=["finance-team", "dev_ops", "org.engineering"],
        )
        assert req.allowed_groups == ["finance-team", "dev_ops", "org.engineering"]

    def test_max_items_exceeded_raises_error(self):
        """More than 50 groups should raise a validation error."""
        with pytest.raises(ValueError):
            AgentRegistrationRequest(
                **MINIMAL_AGENT_KWARGS,
                visibility="group-restricted",
                allowed_groups=[f"group-{i}" for i in range(51)],
            )

    def test_exactly_50_groups_is_valid(self):
        """Exactly 50 groups should be accepted."""
        req = AgentRegistrationRequest(
            **MINIMAL_AGENT_KWARGS,
            visibility="group-restricted",
            allowed_groups=[f"group-{i}" for i in range(50)],
        )
        assert len(req.allowed_groups) == 50
