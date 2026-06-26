"""Simplified federation configuration schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class AiCatalogSourceConfig(BaseModel):
    """A single ARD ai-catalog.json ingestion source (issue #1296, Phase 3).

    Provide either ``uri`` (a direct ai-catalog.json URL) or ``domain`` (resolved
    via ``https://<domain>/.well-known/ai-catalog.json``). ``source_id`` is a
    stable, path-safe identifier reused as the ingested items' ``registry_name``
    and ``/{source_id}/...`` path prefix (so search/origin attribution is a pure
    function of the record). ``expected_identity`` optionally pins the required
    ``trustManifest.identity`` for this source (domain-anchored trust).
    """

    source_id: str = Field(..., min_length=1, max_length=64)
    uri: str | None = None
    domain: str | None = None
    expected_identity: str | None = None

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: str | None) -> str | None:
        """Reject anything that isn't a bare hostname.

        ``domain`` is interpolated into ``https://{domain}/.well-known/...``, so a
        value containing a scheme, path, userinfo, port, or whitespace could shape
        an unexpected URL. Restrict it to a plain hostname.
        """
        if v is None:
            return v
        v = v.strip()
        if not v or any(c in v for c in ("/", "@", ":", " ", "\\", "?", "#")) or "://" in v:
            raise ValueError("domain must be a bare hostname (no scheme/path/port/userinfo)")
        return v

    def resolve_uri(self) -> str:
        """Return the effective catalog URL (uri wins; else domain .well-known)."""
        if self.uri:
            return self.uri
        return f"https://{self.domain}/.well-known/ai-catalog.json"


class AiCatalogFederationConfig(BaseModel):
    """ARD ai-catalog.json ingestion configuration (Phase 3).

    Sits alongside the Anthropic/ASOR/AWS adapters. ``sources`` is managed through
    the federation-config API and the External Registries UI, mirroring the other
    upstream registry types.
    """

    enabled: bool = False
    sync_on_startup: bool = False
    sync_interval_minutes: int = Field(default=60, ge=5, le=1440)
    max_depth: int = Field(default=3, ge=0, le=10)
    fetch_timeout_seconds: int = Field(default=15, ge=1, le=120)
    polite_interval_ms: int = Field(default=200, ge=0, le=10_000)
    same_domain_only: bool = True
    trust_enforcement: Literal["reject", "flag", "off"] = Field(
        default="reject",
        description=(
            "Domain-anchored trust policy when an entry's URN publisher FQDN does not "
            "match the catalog host trustManifest.identity: reject | flag | off."
        ),
    )
    sources: list[AiCatalogSourceConfig] = Field(default_factory=list)


class FederationConfig(BaseModel):
    """Root federation configuration."""

    anthropic: AnthropicFederationConfig = Field(default_factory=AnthropicFederationConfig)
    asor: AsorFederationConfig = Field(default_factory=AsorFederationConfig)
    aws_registry: AwsRegistryFederationConfig = Field(default_factory=AwsRegistryFederationConfig)
    ai_catalog: AiCatalogFederationConfig = Field(default_factory=AiCatalogFederationConfig)

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
        return (
            self.anthropic.enabled
            or self.asor.enabled
            or self.aws_registry.enabled
            or self.ai_catalog.enabled
        )

    def get_enabled_federations(self) -> list[str]:
        """Get list of enabled federation names."""
        enabled = []
        if self.anthropic.enabled:
            enabled.append("anthropic")
        if self.asor.enabled:
            enabled.append("asor")
        if self.aws_registry.enabled:
            enabled.append("aws_registry")
        if self.ai_catalog.enabled:
            enabled.append("ai_catalog")
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
