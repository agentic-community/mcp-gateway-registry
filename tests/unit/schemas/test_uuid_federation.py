"""Unit tests for UUID field preservation from federated registries."""

from uuid import UUID

import pytest

from registry.core.schemas import ServerInfo
from registry.schemas.agent_models import AgentCard
from registry.schemas.skill_models import SkillCard


@pytest.mark.unit
class TestFederatedUUIDPreservation:
    """Tests that UUIDs from federated registries are preserved."""

    def test_serverinfo_preserves_federated_uuid(self):
        """Test that UUID from federated registry is preserved."""
        # Simulate data from a peer registry with existing UUID
        federated_uuid = "550e8400-e29b-41d4-a716-446655440000"

        federated_data = {
            "id": federated_uuid,
            "server_name": "federated-server",
            "path": "/federated/server",
            "description": "Server from peer registry",
            "external_tags": ["federated"],
        }

        # Load the data
        server = ServerInfo(**federated_data)

        # UUID should be preserved, not regenerated
        assert isinstance(server.id, UUID)
        assert str(server.id) == federated_uuid

    def test_serverinfo_generates_uuid_when_missing(self):
        """Test that UUID is generated when not present in federated data."""
        federated_data = {
            "server_name": "federated-server",
            "path": "/federated/server",
            "description": "Server from old peer registry",
            "external_tags": ["federated"],
        }

        # Load the data
        server = ServerInfo(**federated_data)

        # UUID should be auto-generated
        assert isinstance(server.id, UUID)
        assert server.id is not None

    def test_agentcard_preserves_federated_uuid(self):
        """Test that Agent UUID from federated registry is preserved."""
        federated_uuid = "660e8400-e29b-41d4-a716-446655440000"

        federated_data = {
            "id": federated_uuid,
            "name": "federated-agent",
            "path": "/federated/agent",
            "url": "https://federated.example.com",
            "version": "1.0.0",
            "protocol_version": "1.0.0",
            "description": "Agent from peer registry",
            "external_tags": ["federated"],
        }

        agent = AgentCard(**federated_data)

        # UUID should be preserved
        assert isinstance(agent.id, UUID)
        assert str(agent.id) == federated_uuid

    def test_agentcard_generates_uuid_when_missing(self):
        """Test that Agent UUID is generated when not present."""
        federated_data = {
            "name": "federated-agent",
            "path": "/federated/agent",
            "url": "https://federated.example.com",
            "version": "1.0.0",
            "protocol_version": "1.0.0",
            "description": "Agent from old peer registry",
            "external_tags": ["federated"],
        }

        agent = AgentCard(**federated_data)

        # UUID should be auto-generated
        assert isinstance(agent.id, UUID)
        assert agent.id is not None

    def test_skillcard_preserves_federated_uuid(self):
        """Test that Skill UUID from federated registry is preserved."""
        federated_uuid = "770e8400-e29b-41d4-a716-446655440000"

        federated_data = {
            "id": federated_uuid,
            "path": "/skills/federated-skill",
            "name": "federated-skill",
            "description": "Skill from peer registry",
            "skill_md_url": "https://federated.example.com/SKILL.md",
            "external_tags": ["federated"],
        }

        skill = SkillCard(**federated_data)

        # UUID should be preserved
        assert isinstance(skill.id, UUID)
        assert str(skill.id) == federated_uuid

    def test_skillcard_generates_uuid_when_missing(self):
        """Test that Skill UUID is generated when not present."""
        federated_data = {
            "path": "/skills/federated-skill",
            "name": "federated-skill",
            "description": "Skill from old peer registry",
            "skill_md_url": "https://federated.example.com/SKILL.md",
            "external_tags": ["federated"],
        }

        skill = SkillCard(**federated_data)

        # UUID should be auto-generated
        assert isinstance(skill.id, UUID)
        assert skill.id is not None

    def test_multiple_servers_same_data_different_uuids(self):
        """Test that creating multiple servers from same data generates different UUIDs."""
        # Simulate syncing same server from peer registry at different times
        # without UUID in the data (old peer registry)
        federated_data = {
            "server_name": "federated-server",
            "path": "/federated/server",
            "description": "Server from old peer",
        }

        # First sync
        server1 = ServerInfo(**federated_data)

        # Second sync (data without UUID)
        server2 = ServerInfo(**federated_data)

        # Each instance gets a unique UUID
        assert server1.id != server2.id

    def test_uuid_in_json_roundtrip(self):
        """Test UUID preservation through JSON serialization/deserialization."""
        original_uuid = "880e8400-e29b-41d4-a716-446655440000"

        server = ServerInfo(
            id=original_uuid,
            server_name="test-server",
            path="/test/server",
            external_tags=["federated"],
        )

        # Serialize to JSON
        json_data = server.model_dump(mode="json")

        # UUID should be in JSON as string
        assert json_data["id"] == original_uuid

        # Deserialize back
        restored = ServerInfo(**json_data)

        # UUID should be preserved
        assert str(restored.id) == original_uuid
