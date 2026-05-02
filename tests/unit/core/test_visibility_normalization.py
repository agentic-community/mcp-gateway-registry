"""Unit tests for visibility normalization in Pydantic models.

Tests verify that AgentCard, AgentRegistrationRequest, and ServerInfo
all normalize 'internal' -> 'private' and 'group' -> 'group-restricted'.
"""

import pytest
from pydantic import ValidationError

from registry.schemas.agent_models import (
    AgentCard,
    AgentRegistrationRequest,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_minimal_agent_card(**overrides) -> AgentCard:
    """Build an AgentCard with minimal required fields plus overrides."""
    defaults = {
        "name": "test-agent",
        "path": "/test/agent",
        "url": "https://test.example.com",
        "version": "1.0.0",
        "protocol_version": "1.0",
        "description": "Test agent",
    }
    defaults.update(overrides)
    return AgentCard(**defaults)


def _build_minimal_registration(**overrides) -> AgentRegistrationRequest:
    """Build an AgentRegistrationRequest with minimal required fields."""
    defaults = {
        "name": "test-agent",
        "url": "https://test.example.com",
        "supported_protocol": "a2a",
    }
    defaults.update(overrides)
    return AgentRegistrationRequest(**defaults)


# ---------------------------------------------------------------------------
# AgentCard visibility normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentCardVisibilityNormalization:
    """Tests for visibility normalization in AgentCard."""

    def test_internal_normalized_to_private(self):
        """AgentCard with visibility='internal' should normalize to 'private'."""
        agent = _build_minimal_agent_card(visibility="internal")
        assert agent.visibility == "private"

    def test_private_accepted(self):
        """AgentCard with visibility='private' should stay 'private'."""
        agent = _build_minimal_agent_card(visibility="private")
        assert agent.visibility == "private"

    def test_public_accepted(self):
        """AgentCard with visibility='public' should stay 'public'."""
        agent = _build_minimal_agent_card(visibility="public")
        assert agent.visibility == "public"

    def test_group_normalized_to_group_restricted(self):
        """AgentCard with visibility='group' should normalize to 'group-restricted'."""
        agent = _build_minimal_agent_card(
            visibility="group",
            allowed_groups=["developers"],
        )
        assert agent.visibility == "group-restricted"

    def test_group_restricted_accepted(self):
        """AgentCard with visibility='group-restricted' should stay."""
        agent = _build_minimal_agent_card(
            visibility="group-restricted",
            allowed_groups=["developers"],
        )
        assert agent.visibility == "group-restricted"

    def test_case_insensitive(self):
        """AgentCard should accept visibility in any case."""
        agent = _build_minimal_agent_card(visibility="Internal")
        assert agent.visibility == "private"

    def test_invalid_visibility_rejected(self):
        """AgentCard should reject invalid visibility values."""
        with pytest.raises(ValidationError, match="Visibility must be one of"):
            _build_minimal_agent_card(visibility="secret")

    def test_backward_compat_old_data_with_internal(self):
        """Old agent data with 'internal' should load as 'private'."""
        old_data = {
            "name": "old-agent",
            "path": "/old/agent",
            "url": "https://old.example.com",
            "version": "1.0.0",
            "protocol_version": "1.0",
            "description": "Old agent with internal visibility",
            "visibility": "internal",
            "trust_level": "community",
        }
        agent = AgentCard(**old_data)
        assert agent.visibility == "private"


# ---------------------------------------------------------------------------
# AgentRegistrationRequest visibility normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistrationVisibilityNormalization:
    """Tests for visibility normalization in AgentRegistrationRequest."""

    def test_internal_normalized_to_private(self):
        """Registration with visibility='internal' should normalize to 'private'."""
        req = _build_minimal_registration(visibility="internal")
        assert req.visibility == "private"

    def test_private_accepted(self):
        """Registration with visibility='private' should stay 'private'."""
        req = _build_minimal_registration(visibility="private")
        assert req.visibility == "private"

    def test_group_normalized_to_group_restricted(self):
        """Registration with visibility='group' should normalize to 'group-restricted'."""
        req = _build_minimal_registration(visibility="group", allowed_groups=["test-group"])
        assert req.visibility == "group-restricted"

    def test_default_is_public(self):
        """Registration visibility defaults to 'public'."""
        req = _build_minimal_registration()
        assert req.visibility == "public"

    def test_invalid_visibility_rejected(self):
        """Registration should reject invalid visibility values."""
        with pytest.raises(ValidationError, match="Visibility must be one of"):
            _build_minimal_registration(visibility="hidden")
