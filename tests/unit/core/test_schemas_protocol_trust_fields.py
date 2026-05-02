"""Unit tests for supported_protocol, trust_level, and visibility field changes.

Tests cover:
- AgentCard default values for trust_level and visibility
- AgentCard supported_protocol field (optional, None default)
- AgentInfo new fields (visibility, supported_protocol)
- AgentRegistrationRequest validators (supported_protocol, trust_level)
- Backward compatibility for old agents without supported_protocol
"""

import pytest
from pydantic import ValidationError

from registry.schemas.agent_models import (
    AgentCard,
    AgentInfo,
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
# AgentCard defaults and supported_protocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentCardDefaults:
    """Tests for AgentCard default field values."""

    def test_trust_level_defaults_to_community(self):
        """AgentCard trust_level should default to 'community'."""
        agent = _build_minimal_agent_card()
        assert agent.trust_level == "community"

    def test_visibility_defaults_to_public(self):
        """AgentCard visibility should default to 'public'."""
        agent = _build_minimal_agent_card()
        assert agent.visibility == "public"

    def test_supported_protocol_defaults_to_none(self):
        """AgentCard supported_protocol should default to None."""
        agent = _build_minimal_agent_card()
        assert agent.supported_protocol is None

    def test_supported_protocol_a2a(self):
        """AgentCard accepts 'a2a' as supported_protocol."""
        agent = _build_minimal_agent_card(supported_protocol="a2a")
        assert agent.supported_protocol == "a2a"

    def test_supported_protocol_other(self):
        """AgentCard accepts 'other' as supported_protocol."""
        agent = _build_minimal_agent_card(supported_protocol="other")
        assert agent.supported_protocol == "other"

    def test_supported_protocol_camel_case_alias(self):
        """AgentCard accepts camelCase alias 'supportedProtocol'."""
        agent = _build_minimal_agent_card(supportedProtocol="a2a")
        assert agent.supported_protocol == "a2a"

    def test_supported_protocol_serializes_with_alias(self):
        """supported_protocol serializes as 'supportedProtocol' in camelCase output."""
        agent = _build_minimal_agent_card(supported_protocol="a2a")
        data = agent.model_dump(by_alias=True)
        assert "supportedProtocol" in data
        assert data["supportedProtocol"] == "a2a"

    def test_trust_level_camel_case_alias(self):
        """AgentCard accepts camelCase alias 'trustLevel'."""
        agent = _build_minimal_agent_card(trustLevel="verified")
        assert agent.trust_level == "verified"

    def test_trust_level_all_valid_values(self):
        """AgentCard accepts all valid trust_level values."""
        for level in ["unverified", "community", "verified", "trusted"]:
            agent = _build_minimal_agent_card(trust_level=level)
            assert agent.trust_level == level

    def test_trust_level_invalid_value_rejected(self):
        """AgentCard rejects invalid trust_level values."""
        with pytest.raises(ValidationError, match="Trust level must be one of"):
            _build_minimal_agent_card(trust_level="invalid")


# ---------------------------------------------------------------------------
# AgentCard backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentCardBackwardCompat:
    """Tests for backward compatibility with old agents."""

    def test_old_agent_data_without_supported_protocol(self):
        """Old agent data without supported_protocol loads with None default."""
        old_data = {
            "name": "old-agent",
            "path": "/old/agent",
            "url": "https://old.example.com",
            "version": "1.0.0",
            "protocol_version": "1.0",
            "description": "Old agent without supported_protocol",
            "visibility": "public",
            "trust_level": "unverified",
        }
        agent = AgentCard(**old_data)
        assert agent.supported_protocol is None

    def test_old_agent_with_unverified_trust_still_valid(self):
        """Old agents with 'unverified' trust_level still load correctly."""
        agent = _build_minimal_agent_card(trust_level="unverified")
        assert agent.trust_level == "unverified"

    def test_old_agent_with_internal_visibility_still_valid(self):
        """Old agents with 'internal' visibility load correctly as 'private'."""
        agent = _build_minimal_agent_card(visibility="internal")
        assert agent.visibility == "private"


# ---------------------------------------------------------------------------
# AgentInfo new fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentInfoFields:
    """Tests for AgentInfo visibility and supported_protocol fields."""

    def test_trust_level_defaults_to_community(self):
        """AgentInfo trust_level should default to 'community'."""
        info = AgentInfo(
            name="test",
            path="/test",
            url="https://test.example.com",
        )
        assert info.trust_level == "community"

    def test_visibility_defaults_to_public(self):
        """AgentInfo visibility should default to 'public'."""
        info = AgentInfo(
            name="test",
            path="/test",
            url="https://test.example.com",
        )
        assert info.visibility == "public"

    def test_supported_protocol_defaults_to_none(self):
        """AgentInfo supported_protocol should default to None."""
        info = AgentInfo(
            name="test",
            path="/test",
            url="https://test.example.com",
        )
        assert info.supported_protocol is None

    def test_supported_protocol_a2a(self):
        """AgentInfo accepts 'a2a' as supported_protocol."""
        info = AgentInfo(
            name="test",
            path="/test",
            url="https://test.example.com",
            supported_protocol="a2a",
        )
        assert info.supported_protocol == "a2a"

    def test_supported_protocol_camel_case_alias(self):
        """AgentInfo accepts camelCase alias 'supportedProtocol'."""
        info = AgentInfo(
            name="test",
            path="/test",
            url="https://test.example.com",
            supportedProtocol="other",
        )
        assert info.supported_protocol == "other"

    def test_all_fields_serialized(self):
        """AgentInfo serializes visibility and supported_protocol."""
        info = AgentInfo(
            name="test",
            path="/test",
            url="https://test.example.com",
            visibility="public",
            trust_level="community",
            supported_protocol="a2a",
        )
        data = info.model_dump(by_alias=True)
        assert data["trustLevel"] == "community"
        assert data["visibility"] == "public"
        assert data["supportedProtocol"] == "a2a"


# ---------------------------------------------------------------------------
# AgentRegistrationRequest validators
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentRegistrationRequest:
    """Tests for AgentRegistrationRequest model and validators."""

    def test_supported_protocol_required(self):
        """supported_protocol is required on registration."""
        with pytest.raises(ValidationError, match="supportedProtocol"):
            AgentRegistrationRequest(
                name="test",
                url="https://test.example.com",
            )

    def test_supported_protocol_a2a(self):
        """Registration accepts 'a2a' protocol."""
        req = _build_minimal_registration(supported_protocol="a2a")
        assert req.supported_protocol == "a2a"

    def test_supported_protocol_other(self):
        """Registration accepts 'other' protocol."""
        req = _build_minimal_registration(supported_protocol="other")
        assert req.supported_protocol == "other"

    def test_supported_protocol_normalized_to_lowercase(self):
        """supported_protocol is normalized to lowercase."""
        req = _build_minimal_registration(supported_protocol="A2A")
        assert req.supported_protocol == "a2a"

    def test_supported_protocol_invalid_rejected(self):
        """Invalid supported_protocol values are rejected."""
        with pytest.raises(ValidationError, match="supported_protocol must be one of"):
            _build_minimal_registration(supported_protocol="mcp")

    def test_supported_protocol_camel_case_alias(self):
        """Registration accepts camelCase alias 'supportedProtocol'."""
        req = AgentRegistrationRequest(
            name="test",
            url="https://test.example.com",
            supportedProtocol="a2a",
        )
        assert req.supported_protocol == "a2a"

    def test_trust_level_defaults_to_community(self):
        """Registration trust_level defaults to 'community'."""
        req = _build_minimal_registration()
        assert req.trust_level == "community"

    def test_trust_level_all_valid_values(self):
        """Registration accepts all valid trust_level values."""
        for level in ["unverified", "community", "verified", "trusted"]:
            req = _build_minimal_registration(trust_level=level)
            assert req.trust_level == level

    def test_trust_level_invalid_rejected(self):
        """Invalid trust_level values are rejected."""
        with pytest.raises(ValidationError, match="trust_level must be one of"):
            _build_minimal_registration(trust_level="unknown")

    def test_trust_level_camel_case_alias(self):
        """Registration accepts camelCase alias 'trustLevel'."""
        req = AgentRegistrationRequest(
            name="test",
            url="https://test.example.com",
            supportedProtocol="a2a",
            trustLevel="verified",
        )
        assert req.trust_level == "verified"

    def test_visibility_defaults_to_public(self):
        """Registration visibility defaults to 'public'."""
        req = _build_minimal_registration()
        assert req.visibility == "public"

    def test_full_registration_with_all_new_fields(self):
        """Full registration with all new fields set."""
        req = _build_minimal_registration(
            supported_protocol="a2a",
            trust_level="verified",
            visibility="group-restricted",
            allowed_groups=["test-group"],
        )
        assert req.supported_protocol == "a2a"
        assert req.trust_level == "verified"
        assert req.visibility == "group-restricted"
