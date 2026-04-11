"""
Unit tests for AWS Registry federation schema models.

This module provides tests for the AWS Registry federation Pydantic models:
- AwsRegistryConfig (aliased as AgentCoreRegistryConfig): Configuration for a single AWS Agent Registry
- AwsRegistryFederationConfig (aliased as AgentCoreFederationConfig): AWS Agent Registry federation configuration
- FederationConfig: Root federation config with aws_registry support

Tests cover:
- Default values for all fields
- Custom value assignment
- FederationConfig.aws_registry integration
- is_any_federation_enabled() with aws_registry
- get_enabled_federations() with aws_registry
- Backward compatibility: old 'agentcore' key in dict input
"""

import pytest

from registry.schemas.federation_schema import (
    AgentCoreFederationConfig,
    AgentCoreRegistryConfig,
    AnthropicFederationConfig,
    AsorFederationConfig,
    FederationConfig,
)


# =============================================================================
# AgentCoreRegistryConfig Tests
# =============================================================================


@pytest.mark.unit
class TestAgentCoreRegistryConfig:
    """Tests for AgentCoreRegistryConfig model."""

    def test_required_registry_id(self):
        """Registry ID is required and must be provided."""
        config = AgentCoreRegistryConfig(registry_id="my-registry-123")
        assert config.registry_id == "my-registry-123"

    def test_default_descriptor_types(self):
        """Default descriptor types should include MCP, A2A, CUSTOM, AGENT_SKILLS."""
        config = AgentCoreRegistryConfig(registry_id="test-reg")
        assert config.descriptor_types == ["MCP", "A2A", "CUSTOM", "AGENT_SKILLS"]

    def test_custom_descriptor_types(self):
        """Custom descriptor types should override the defaults."""
        config = AgentCoreRegistryConfig(
            registry_id="test-reg",
            descriptor_types=["MCP", "A2A"],
        )
        assert config.descriptor_types == ["MCP", "A2A"]

    def test_empty_descriptor_types(self):
        """Empty descriptor types list should be allowed."""
        config = AgentCoreRegistryConfig(
            registry_id="test-reg",
            descriptor_types=[],
        )
        assert config.descriptor_types == []

    def test_default_sync_status_filter(self):
        """Default sync status filter should be APPROVED."""
        config = AgentCoreRegistryConfig(registry_id="test-reg")
        assert config.sync_status_filter == "APPROVED"

    def test_custom_sync_status_filter(self):
        """Custom sync status filter should override the default."""
        config = AgentCoreRegistryConfig(
            registry_id="test-reg",
            sync_status_filter="PENDING",
        )
        assert config.sync_status_filter == "PENDING"

    def test_missing_registry_id_raises_error(self):
        """Creating without registry_id should raise a validation error."""
        with pytest.raises(Exception):
            AgentCoreRegistryConfig()

    def test_default_aws_account_id_is_none(self):
        """Default aws_account_id should be None (same-account)."""
        config = AgentCoreRegistryConfig(registry_id="test-reg")
        assert config.aws_account_id is None

    def test_custom_aws_account_id(self):
        """aws_account_id should accept a custom value."""
        config = AgentCoreRegistryConfig(
            registry_id="test-reg",
            aws_account_id="123456789012",
        )
        assert config.aws_account_id == "123456789012"

    def test_default_registry_aws_region_is_none(self):
        """Default aws_region should be None (inherits from parent config)."""
        config = AgentCoreRegistryConfig(registry_id="test-reg")
        assert config.aws_region is None

    def test_custom_registry_aws_region(self):
        """Per-registry aws_region should override parent."""
        config = AgentCoreRegistryConfig(
            registry_id="test-reg",
            aws_region="eu-west-1",
        )
        assert config.aws_region == "eu-west-1"

    def test_default_assume_role_arn_is_none(self):
        """Default assume_role_arn should be None."""
        config = AgentCoreRegistryConfig(registry_id="test-reg")
        assert config.assume_role_arn is None

    def test_custom_assume_role_arn(self):
        """assume_role_arn should accept a custom IAM role ARN."""
        config = AgentCoreRegistryConfig(
            registry_id="test-reg",
            aws_account_id="123456789012",
            assume_role_arn="arn:aws:iam::123456789012:role/AgentCoreReadOnly",
        )
        assert config.assume_role_arn == "arn:aws:iam::123456789012:role/AgentCoreReadOnly"

    def test_cross_account_config_all_fields(self):
        """Cross-account config should set account, region, role, and registry."""
        config = AgentCoreRegistryConfig(
            registry_id="reg-cross-001",
            aws_account_id="987654321098",
            aws_region="eu-west-1",
            assume_role_arn="arn:aws:iam::987654321098:role/FederationRole",
            descriptor_types=["MCP"],
            sync_status_filter="APPROVED",
        )
        assert config.registry_id == "reg-cross-001"
        assert config.aws_account_id == "987654321098"
        assert config.aws_region == "eu-west-1"
        assert config.assume_role_arn == "arn:aws:iam::987654321098:role/FederationRole"
        assert config.descriptor_types == ["MCP"]


# =============================================================================
# AgentCoreFederationConfig Tests
# =============================================================================


@pytest.mark.unit
class TestAgentCoreFederationConfig:
    """Tests for AgentCoreFederationConfig model."""

    def test_default_enabled_is_false(self):
        """Default enabled should be False."""
        config = AgentCoreFederationConfig()
        assert config.enabled is False

    def test_default_aws_region(self):
        """Default AWS region should be us-east-1."""
        config = AgentCoreFederationConfig()
        assert config.aws_region == "us-east-1"

    def test_custom_aws_region(self):
        """Custom AWS region should override the default."""
        config = AgentCoreFederationConfig(aws_region="eu-west-1")
        assert config.aws_region == "eu-west-1"

    def test_default_sync_on_startup_is_false(self):
        """Default sync_on_startup should be False."""
        config = AgentCoreFederationConfig()
        assert config.sync_on_startup is False

    def test_default_sync_interval_minutes(self):
        """Default sync interval should be 60 minutes."""
        config = AgentCoreFederationConfig()
        assert config.sync_interval_minutes == 60

    def test_custom_sync_interval_minutes(self):
        """Custom sync interval should override the default."""
        config = AgentCoreFederationConfig(sync_interval_minutes=30)
        assert config.sync_interval_minutes == 30

    def test_default_sync_timeout_seconds(self):
        """Default sync timeout should be 300 seconds."""
        config = AgentCoreFederationConfig()
        assert config.sync_timeout_seconds == 300

    def test_custom_sync_timeout_seconds(self):
        """Custom sync timeout should override the default."""
        config = AgentCoreFederationConfig(sync_timeout_seconds=120)
        assert config.sync_timeout_seconds == 120

    def test_default_max_concurrent_fetches(self):
        """Default max concurrent fetches should be 5."""
        config = AgentCoreFederationConfig()
        assert config.max_concurrent_fetches == 5

    def test_custom_max_concurrent_fetches(self):
        """Custom max concurrent fetches should override the default."""
        config = AgentCoreFederationConfig(max_concurrent_fetches=10)
        assert config.max_concurrent_fetches == 10

    def test_default_registries_is_empty(self):
        """Default registries should be an empty list."""
        config = AgentCoreFederationConfig()
        assert config.registries == []

    def test_registries_with_entries(self):
        """Registries should accept a list of AgentCoreRegistryConfig objects."""
        registry = AgentCoreRegistryConfig(registry_id="reg-001")
        config = AgentCoreFederationConfig(registries=[registry])
        assert len(config.registries) == 1
        assert config.registries[0].registry_id == "reg-001"

    def test_multiple_registries(self):
        """Multiple registries should be supported."""
        registries = [
            AgentCoreRegistryConfig(registry_id="reg-001"),
            AgentCoreRegistryConfig(registry_id="reg-002"),
            AgentCoreRegistryConfig(
                registry_id="reg-003",
                descriptor_types=["MCP"],
                sync_status_filter="PENDING",
            ),
        ]
        config = AgentCoreFederationConfig(registries=registries)
        assert len(config.registries) == 3
        assert config.registries[2].descriptor_types == ["MCP"]
        assert config.registries[2].sync_status_filter == "PENDING"

    def test_fully_custom_config(self):
        """All fields should be overridable at once."""
        config = AgentCoreFederationConfig(
            enabled=True,
            aws_region="ap-southeast-1",
            sync_on_startup=True,
            sync_interval_minutes=15,
            sync_timeout_seconds=60,
            max_concurrent_fetches=2,
            registries=[
                AgentCoreRegistryConfig(registry_id="prod-reg"),
            ],
        )
        assert config.enabled is True
        assert config.aws_region == "ap-southeast-1"
        assert config.sync_on_startup is True
        assert config.sync_interval_minutes == 15
        assert config.sync_timeout_seconds == 60
        assert config.max_concurrent_fetches == 2
        assert len(config.registries) == 1


# =============================================================================
# FederationConfig AgentCore Integration Tests
# =============================================================================


@pytest.mark.unit
class TestFederationConfigAwsRegistry:
    """Tests for FederationConfig with aws_registry field."""

    def test_default_aws_registry_field_exists(self):
        """FederationConfig should have an aws_registry field with defaults."""
        config = FederationConfig()
        assert isinstance(config.aws_registry, AgentCoreFederationConfig)
        assert config.aws_registry.enabled is False

    def test_aws_registry_custom_config(self):
        """FederationConfig should accept custom aws_registry configuration."""
        aws_config = AgentCoreFederationConfig(
            enabled=True,
            aws_region="us-west-2",
            registries=[
                AgentCoreRegistryConfig(registry_id="my-reg"),
            ],
        )
        config = FederationConfig(aws_registry=aws_config)
        assert config.aws_registry.enabled is True
        assert config.aws_registry.aws_region == "us-west-2"
        assert len(config.aws_registry.registries) == 1

    def test_backward_compat_agentcore_key(self):
        """FederationConfig should accept old 'agentcore' key from MongoDB."""
        config = FederationConfig(**{
            "agentcore": {"enabled": True, "aws_region": "eu-west-1"},
        })
        assert config.aws_registry.enabled is True
        assert config.aws_registry.aws_region == "eu-west-1"

    def test_is_any_federation_enabled_all_disabled(self):
        """is_any_federation_enabled should return False when all are disabled."""
        config = FederationConfig()
        assert config.is_any_federation_enabled() is False

    def test_is_any_federation_enabled_only_aws_registry(self):
        """is_any_federation_enabled should return True when only aws_registry is enabled."""
        config = FederationConfig(
            aws_registry=AgentCoreFederationConfig(enabled=True),
        )
        assert config.is_any_federation_enabled() is True

    def test_is_any_federation_enabled_aws_registry_and_anthropic(self):
        """is_any_federation_enabled should return True when multiple are enabled."""
        config = FederationConfig(
            aws_registry=AgentCoreFederationConfig(enabled=True),
        )
        assert config.anthropic.enabled is False
        assert config.is_any_federation_enabled() is True

    def test_get_enabled_federations_none_enabled(self):
        """get_enabled_federations should return empty list when none are enabled."""
        config = FederationConfig()
        assert config.get_enabled_federations() == []

    def test_get_enabled_federations_only_aws_registry(self):
        """get_enabled_federations should include 'aws_registry' when enabled."""
        config = FederationConfig(
            aws_registry=AgentCoreFederationConfig(enabled=True),
        )
        enabled = config.get_enabled_federations()
        assert "aws_registry" in enabled
        assert len(enabled) == 1

    def test_get_enabled_federations_excludes_disabled(self):
        """get_enabled_federations should not include disabled federations."""
        config = FederationConfig(
            aws_registry=AgentCoreFederationConfig(enabled=False),
        )
        enabled = config.get_enabled_federations()
        assert "aws_registry" not in enabled

    def test_get_enabled_federations_multiple_enabled(self):
        """get_enabled_federations should list all enabled federation names."""
        config = FederationConfig(
            anthropic=AnthropicFederationConfig(enabled=True),
            asor=AsorFederationConfig(enabled=True),
            aws_registry=AgentCoreFederationConfig(enabled=True),
        )
        enabled = config.get_enabled_federations()
        assert "anthropic" in enabled
        assert "asor" in enabled
        assert "aws_registry" in enabled
        assert len(enabled) == 3
