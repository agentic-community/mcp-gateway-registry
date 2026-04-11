"""Simplified federation configuration schemas."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class AnthropicServerConfig(BaseModel):
    """Anthropic server configuration."""

    name: str


class AnthropicFederationConfig(BaseModel):
    """Anthropic federation configuration."""

    enabled: bool = False
    endpoint: str = "https://registry.modelcontextprotocol.io"
    sync_on_startup: bool = False
    servers: list[AnthropicServerConfig] = Field(default_factory=list)


class AsorAgentConfig(BaseModel):
    """ASOR agent configuration."""

    id: str


class AsorFederationConfig(BaseModel):
    """ASOR federation configuration."""

    enabled: bool = False
    endpoint: str = ""
    auth_env_var: str | None = None
    sync_on_startup: bool = False
    agents: list[AsorAgentConfig] = Field(default_factory=list)


class AwsRegistryConfig(BaseModel):
    """Configuration for a single AWS Agent Registry to sync from.

    For cross-account or cross-region access, provide aws_account_id,
    assume_role_arn, and/or aws_region per registry. The gateway assumes
    the IAM role via STS to read from the remote registry.
    """

    registry_id: str
    aws_account_id: str | None = None
    aws_region: str | None = None
    assume_role_arn: str | None = None
    descriptor_types: list[str] = Field(
        default_factory=lambda: ["MCP", "A2A", "CUSTOM", "AGENT_SKILLS"]
    )
    sync_status_filter: str = "APPROVED"


class AwsRegistryFederationConfig(BaseModel):
    """AWS Agent Registry federation configuration."""

    enabled: bool = False
    aws_region: str = "us-east-1"
    sync_on_startup: bool = False
    sync_interval_minutes: int = 60
    sync_timeout_seconds: int = 300
    max_concurrent_fetches: int = 5
    registries: list[AwsRegistryConfig] = Field(default_factory=list)


class FederationConfig(BaseModel):
    """Root federation configuration."""

    anthropic: AnthropicFederationConfig = Field(default_factory=AnthropicFederationConfig)
    asor: AsorFederationConfig = Field(default_factory=AsorFederationConfig)
    aws_registry: AwsRegistryFederationConfig = Field(default_factory=AwsRegistryFederationConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_agentcore_key(cls, data: Any) -> Any:
        """Accept old 'agentcore' key as alias for 'aws_registry'.

        MongoDB documents created before the rename use 'agentcore'.
        This validator transparently maps the old key so existing
        documents deserialize without a migration script.
        """
        if isinstance(data, dict) and "agentcore" in data and "aws_registry" not in data:
            data["aws_registry"] = data.pop("agentcore")
        return data

    def is_any_federation_enabled(self) -> bool:
        """Check if any federation is enabled."""
        return self.anthropic.enabled or self.asor.enabled or self.aws_registry.enabled

    def get_enabled_federations(self) -> list[str]:
        """Get list of enabled federation names."""
        enabled = []
        if self.anthropic.enabled:
            enabled.append("anthropic")
        if self.asor.enabled:
            enabled.append("asor")
        if self.aws_registry.enabled:
            enabled.append("aws_registry")
        return enabled


# Backward-compatible aliases for the renamed classes
AgentCoreRegistryConfig = AwsRegistryConfig
AgentCoreFederationConfig = AwsRegistryFederationConfig


# Add missing FederatedServer class for compatibility
class FederatedServer(BaseModel):
    """Federated server configuration."""

    name: str
    endpoint: str
    enabled: bool = True
