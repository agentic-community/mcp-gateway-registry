"""Unit tests for UUID fields in all card models."""

from uuid import UUID

import pytest
from pydantic import HttpUrl

from registry.core.schemas import ServerInfo
from registry.schemas.agent_models import AgentCard
from registry.schemas.registry_card import RegistryCard
from registry.schemas.skill_models import SkillCard


@pytest.mark.unit
class TestRegistryCardUUID:
    """Tests for UUID field in RegistryCard."""

    def test_uuid_auto_generated(self):
        """Test that UUID is auto-generated on creation."""
        card = RegistryCard(
            registry_id="test-registry",
            name="Test Registry",
            federation_endpoint=HttpUrl("https://example.com/api/v1/federation"),
        )

        assert isinstance(card.id, UUID)
        assert card.id is not None

    def test_uuid_unique_per_instance(self):
        """Test that each instance gets a unique UUID."""
        card1 = RegistryCard(
            registry_id="test-registry",
            name="Test Registry",
            federation_endpoint=HttpUrl("https://example.com/api/v1/federation"),
        )

        card2 = RegistryCard(
            registry_id="test-registry",
            name="Test Registry",
            federation_endpoint=HttpUrl("https://example.com/api/v1/federation"),
        )

        assert card1.id != card2.id

    def test_uuid_serialization(self):
        """Test that UUID serializes to string in JSON."""
        card = RegistryCard(
            registry_id="test-registry",
            name="Test Registry",
            federation_endpoint=HttpUrl("https://example.com/api/v1/federation"),
        )

        json_data = card.model_dump(mode="json")

        assert "id" in json_data
        assert isinstance(json_data["id"], str)
        # Should be a valid UUID string
        UUID(json_data["id"])

    def test_uuid_deserialization(self):
        """Test that UUID deserializes from string."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"

        card = RegistryCard(
            id=uuid_str,
            registry_id="test-registry",
            name="Test Registry",
            federation_endpoint=HttpUrl("https://example.com/api/v1/federation"),
        )

        assert isinstance(card.id, UUID)
        assert str(card.id) == uuid_str


@pytest.mark.unit
class TestServerInfoUUID:
    """Tests for UUID field in ServerInfo."""

    def test_uuid_auto_generated(self):
        """Test that UUID is auto-generated on creation."""
        server = ServerInfo(
            server_name="test-server",
            path="/test/server",
        )

        assert isinstance(server.id, UUID)
        assert server.id is not None

    def test_uuid_unique_per_instance(self):
        """Test that each instance gets a unique UUID."""
        server1 = ServerInfo(
            server_name="test-server",
            path="/test/server",
        )

        server2 = ServerInfo(
            server_name="test-server",
            path="/test/server",
        )

        assert server1.id != server2.id

    def test_uuid_serialization(self):
        """Test that UUID serializes correctly."""
        server = ServerInfo(
            server_name="test-server",
            path="/test/server",
        )

        json_data = server.model_dump(mode="json")

        assert "id" in json_data
        assert isinstance(json_data["id"], str)
        UUID(json_data["id"])


@pytest.mark.unit
class TestAgentCardUUID:
    """Tests for UUID field in AgentCard."""

    def test_uuid_auto_generated(self):
        """Test that UUID is auto-generated on creation."""
        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
        )

        assert isinstance(agent.id, UUID)
        assert agent.id is not None

    def test_uuid_unique_per_instance(self):
        """Test that each instance gets a unique UUID."""
        agent1 = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
        )

        agent2 = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
        )

        assert agent1.id != agent2.id

    def test_uuid_serialization(self):
        """Test that UUID serializes correctly."""
        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
        )

        json_data = agent.model_dump(mode="json")

        assert "id" in json_data
        assert isinstance(json_data["id"], str)
        UUID(json_data["id"])


@pytest.mark.unit
class TestSkillCardUUID:
    """Tests for UUID field in SkillCard."""

    def test_uuid_auto_generated(self):
        """Test that UUID is auto-generated on creation."""
        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
        )

        assert isinstance(skill.id, UUID)
        assert skill.id is not None

    def test_uuid_unique_per_instance(self):
        """Test that each instance gets a unique UUID."""
        skill1 = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
        )

        skill2 = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
        )

        assert skill1.id != skill2.id

    def test_uuid_serialization(self):
        """Test that UUID serializes correctly."""
        skill = SkillCard(
            path="/skills/test-skill",
            name="test-skill",
            description="Test skill",
            skill_md_url=HttpUrl("https://example.com/SKILL.md"),
        )

        json_data = skill.model_dump(mode="json")

        assert "id" in json_data
        assert isinstance(json_data["id"], str)
        UUID(json_data["id"])


@pytest.mark.unit
class TestUUIDBackwardsCompatibility:
    """Tests for backwards compatibility with existing data without UUID."""

    def test_serverinfo_without_uuid(self):
        """Test loading ServerInfo data without UUID field."""
        old_data = {
            "server_name": "old-server",
            "path": "/old/server",
            "description": "Old server without UUID",
        }

        # Should auto-generate UUID
        server = ServerInfo(**old_data)

        assert isinstance(server.id, UUID)
        assert server.id is not None

    def test_agentcard_without_uuid(self):
        """Test loading AgentCard data without UUID field."""
        old_data = {
            "name": "old-agent",
            "path": "/old/agent",
            "url": "https://old.example.com",
            "version": "1.0.0",
            "protocol_version": "1.0.0",
            "description": "Old agent without UUID",
        }

        # Should auto-generate UUID
        agent = AgentCard(**old_data)

        assert isinstance(agent.id, UUID)
        assert agent.id is not None

    def test_skillcard_without_uuid(self):
        """Test loading SkillCard data without UUID field."""
        old_data = {
            "path": "/skills/old-skill",
            "name": "old-skill",
            "description": "Old skill without UUID",
            "skill_md_url": "https://example.com/SKILL.md",
        }

        # Should auto-generate UUID
        skill = SkillCard(**old_data)

        assert isinstance(skill.id, UUID)
        assert skill.id is not None
