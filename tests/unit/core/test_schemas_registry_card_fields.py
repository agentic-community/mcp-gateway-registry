"""Unit tests for Registry Card fields added to ServerInfo and AgentCard."""

from datetime import UTC, datetime

import pytest

from registry.core.schemas import AgentProvider, ServerInfo
from registry.schemas.agent_models import AgentCard
from registry.schemas.registry_card import LifecycleStatus


@pytest.mark.unit
class TestServerInfoRegistryCardFields:
    """Tests for Registry Card fields in ServerInfo model."""

    def test_default_lifecycle_status(self):
        """Test that default lifecycle status is ACTIVE."""
        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
        )
        assert server.status == LifecycleStatus.ACTIVE

    def test_custom_lifecycle_status(self):
        """Test setting custom lifecycle status."""
        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
            status=LifecycleStatus.DEPRECATED,
        )
        assert server.status == LifecycleStatus.DEPRECATED

    def test_all_lifecycle_statuses(self):
        """Test all lifecycle status values."""
        statuses = [
            LifecycleStatus.ACTIVE,
            LifecycleStatus.DEPRECATED,
            LifecycleStatus.DRAFT,
            LifecycleStatus.BETA,
        ]

        for status in statuses:
            server = ServerInfo(
                server_name="test-server",
                path="test/server",
                description="Test server",
                version="1.0.0",
                status=status,
            )
            assert server.status == status

    def test_provider_default_population(self):
        """Test that provider is populated with default values when None."""
        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
        )

        # Provider should be auto-populated from settings
        assert server.provider is not None
        assert isinstance(server.provider, AgentProvider)
        assert server.provider.organization is not None
        assert server.provider.url is not None

    def test_custom_provider(self):
        """Test setting custom provider."""
        custom_provider = AgentProvider(
            organization="Custom Org",
            url="https://custom.example.com",
        )

        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
            provider=custom_provider,
        )

        assert server.provider == custom_provider
        assert server.provider.organization == "Custom Org"
        assert server.provider.url == "https://custom.example.com"

    def test_source_timestamps_default_none(self):
        """Test that source timestamps default to None."""
        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
        )

        assert server.source_created_at is None
        assert server.source_updated_at is None

    def test_source_timestamps_with_values(self):
        """Test setting source timestamps."""
        created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)

        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
            source_created_at=created,
            source_updated_at=updated,
        )

        assert server.source_created_at == created
        assert server.source_updated_at == updated

    def test_external_tags_default_empty(self):
        """Test that external_tags defaults to empty list."""
        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
        )

        assert server.external_tags == []

    def test_external_tags_with_values(self):
        """Test setting external tags."""
        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
            external_tags=["federated", "external", "verified"],
        )

        assert server.external_tags == ["federated", "external", "verified"]
        assert len(server.external_tags) == 3

    def test_all_registry_card_fields_together(self):
        """Test setting all registry card fields together."""
        created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        provider = AgentProvider(
            organization="Test Org",
            url="https://test.example.com",
        )

        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
            status=LifecycleStatus.BETA,
            provider=provider,
            source_created_at=created,
            source_updated_at=updated,
            external_tags=["tag1", "tag2"],
        )

        assert server.status == LifecycleStatus.BETA
        assert server.provider == provider
        assert server.source_created_at == created
        assert server.source_updated_at == updated
        assert server.external_tags == ["tag1", "tag2"]

    def test_json_serialization_with_registry_card_fields(self):
        """Test JSON serialization of registry card fields."""
        created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        provider = AgentProvider(
            organization="Test Org",
            url="https://test.example.com",
        )

        server = ServerInfo(
            server_name="test-server",
            path="test/server",
            description="Test server",
            version="1.0.0",
            status=LifecycleStatus.DEPRECATED,
            provider=provider,
            source_created_at=created,
            source_updated_at=updated,
            external_tags=["federated"],
        )

        json_data = server.model_dump(mode="json")

        assert json_data["status"] == "deprecated"
        assert "provider" in json_data
        assert json_data["provider"]["organization"] == "Test Org"
        assert "source_created_at" in json_data
        assert "source_updated_at" in json_data
        assert json_data["external_tags"] == ["federated"]

        # Round-trip
        restored = ServerInfo(**json_data)
        assert restored.status == LifecycleStatus.DEPRECATED
        assert restored.provider.organization == "Test Org"
        assert restored.external_tags == ["federated"]

    def test_backwards_compatibility_without_new_fields(self):
        """Test that old data without new fields loads successfully."""
        old_data = {
            "server_name": "old-server",
            "path": "old/server",
            "description": "Old server without registry card fields",
            "version": "1.0.0",
            "tags": ["old"],
        }

        # Should load successfully with defaults
        server = ServerInfo(**old_data)

        assert server.status == LifecycleStatus.ACTIVE
        assert server.provider is not None  # Auto-populated
        assert server.source_created_at is None
        assert server.source_updated_at is None
        assert server.external_tags == []


@pytest.mark.unit
class TestAgentCardRegistryCardFields:
    """Tests for Registry Card fields in AgentCard model."""

    def test_default_lifecycle_status(self):
        """Test that default lifecycle status is ACTIVE."""
        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
        )
        assert agent.status == LifecycleStatus.ACTIVE

    def test_custom_lifecycle_status(self):
        """Test setting custom lifecycle status."""
        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
            status=LifecycleStatus.DRAFT,
        )
        assert agent.status == LifecycleStatus.DRAFT

    def test_source_timestamps_default_none(self):
        """Test that source timestamps default to None."""
        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
        )

        assert agent.source_created_at is None
        assert agent.source_updated_at is None

    def test_source_timestamps_with_values(self):
        """Test setting source timestamps."""
        created = datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 2, 15, 0, 0, 0, tzinfo=UTC)

        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
            sourceCreatedAt=created,
            sourceUpdatedAt=updated,
        )

        assert agent.source_created_at == created
        assert agent.source_updated_at == updated

    def test_external_tags_default_empty(self):
        """Test that external_tags defaults to empty list."""
        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
        )

        assert agent.external_tags == []

    def test_external_tags_with_values(self):
        """Test setting external tags."""
        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
            externalTags=["federated", "verified"],
        )

        assert agent.external_tags == ["federated", "verified"]

    def test_all_registry_card_fields_together(self):
        """Test setting all registry card fields together."""
        created = datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 2, 15, 0, 0, 0, tzinfo=UTC)

        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
            status=LifecycleStatus.BETA,
            sourceCreatedAt=created,
            sourceUpdatedAt=updated,
            externalTags=["tag1", "tag2"],
        )

        assert agent.status == LifecycleStatus.BETA
        assert agent.source_created_at == created
        assert agent.source_updated_at == updated
        assert agent.external_tags == ["tag1", "tag2"]

    def test_json_serialization_with_camel_case_aliases(self):
        """Test JSON serialization uses camelCase aliases."""
        created = datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2024, 2, 15, 0, 0, 0, tzinfo=UTC)

        agent = AgentCard(
            name="test-agent",
            path="/test/agent",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test agent",
            status=LifecycleStatus.ACTIVE,
            sourceCreatedAt=created,
            sourceUpdatedAt=updated,
            externalTags=["federated"],
        )

        json_data = agent.model_dump(by_alias=True, mode="json")

        assert json_data["status"] == "active"
        assert "sourceCreatedAt" in json_data
        assert "sourceUpdatedAt" in json_data
        assert "externalTags" in json_data
        assert json_data["externalTags"] == ["federated"]

    def test_backwards_compatibility_without_new_fields(self):
        """Test that old data without new fields loads successfully."""
        old_data = {
            "name": "old-agent",
            "path": "/old/agent",
            "url": "https://old.example.com",
            "version": "1.0.0",
            "protocol_version": "1.0.0",
            "description": "Old agent without registry card fields",
            "enabled": True,
            "visibility": "public",
            "trust_level": "verified",
            "tags": ["old"],
        }

        # Should load successfully with defaults
        agent = AgentCard(**old_data)

        assert agent.status == LifecycleStatus.ACTIVE
        assert agent.source_created_at is None
        assert agent.source_updated_at is None
        assert agent.external_tags == []

    def test_snake_case_and_camel_case_both_work(self):
        """Test that both snake_case and camelCase field names work."""
        created = datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)

        # Test with camelCase (aliases)
        agent1 = AgentCard(
            name="test-agent-1",
            path="/test/agent1",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test",
            sourceCreatedAt=created,
            externalTags=["tag1"],
        )

        # Test with snake_case (actual field names)
        agent2 = AgentCard(
            name="test-agent-2",
            path="/test/agent2",
            url="https://test.example.com",
            version="1.0.0",
            protocol_version="1.0.0",
            description="Test",
            source_created_at=created,
            external_tags=["tag2"],
        )

        assert agent1.source_created_at == created
        assert agent1.external_tags == ["tag1"]
        assert agent2.source_created_at == created
        assert agent2.external_tags == ["tag2"]
