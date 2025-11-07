"""
Pydantic models for registry federation configuration.

Supports federation with external registries like Anthropic MCP Registry
and Workday ASOR for enterprise agent discovery.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class DisplayOptions(BaseModel):
    """Display options for federated servers/agents."""

    mark_as_federated: bool = Field(
        default=True,
        description="Whether to mark federated items with a visual indicator"
    )
    attribution_label: str = Field(
        default="External Registry",
        description="Label to show source registry"
    )
    separate_section: bool = Field(
        default=True,
        description="Display in separate section from local servers"
    )
    read_only: bool = Field(
        default=True,
        description="Federated items are read-only"
    )


class AnthropicServerConfig(BaseModel):
    """Configuration for a single Anthropic MCP Registry server to federate."""

    name: str = Field(
        ...,
        description="Server name in Anthropic format (e.g., ai.smithery/github)"
    )
    endpoint: Optional[str] = Field(
        None,
        description="Full API endpoint URL (if not using default pattern)"
    )
    requires_auth: bool = Field(
        default=False,
        description="Whether this server requires authentication"
    )
    auth_type: Optional[str] = Field(
        None,
        description="Authentication type (api-key, oauth, bearer)"
    )
    auth_env_var: Optional[str] = Field(
        None,
        description="Environment variable name containing auth credentials"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this server should be synced"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for this server"
    )

    @field_validator("auth_env_var")
    @classmethod
    def validate_auth_env_var(cls, v: Optional[str], info) -> Optional[str]:
        """Validate that auth_env_var is provided when requires_auth is true."""
        requires_auth = info.data.get("requires_auth", False)
        if requires_auth and not v:
            logger.warning(
                f"Server {info.data.get('name')} requires auth but no auth_env_var provided"
            )
        return v


class AnthropicFederationConfig(BaseModel):
    """Configuration for Anthropic MCP Registry federation."""

    enabled: bool = Field(
        default=False,
        description="Enable federation with Anthropic MCP Registry"
    )
    endpoint: str = Field(
        default="https://registry.modelcontextprotocol.io",
        description="Anthropic MCP Registry API base URL"
    )
    api_version: str = Field(
        default="v0.1",
        description="API version to use"
    )
    servers: List[AnthropicServerConfig] = Field(
        default_factory=list,
        description="List of servers to federate from Anthropic Registry"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds (default: 1 hour)",
        ge=60,
        le=86400
    )
    sync_interval_seconds: int = Field(
        default=300,
        description="Sync interval in seconds (default: 5 minutes)",
        ge=60,
        le=3600
    )
    sync_on_startup: bool = Field(
        default=True,
        description="Whether to sync on registry startup"
    )
    display_options: DisplayOptions = Field(
        default_factory=lambda: DisplayOptions(
            attribution_label="Anthropic MCP Registry"
        ),
        description="Display options for Anthropic servers"
    )
    timeout_seconds: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
        ge=5,
        le=120
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed requests",
        ge=1,
        le=10
    )


class AsorAgentConfig(BaseModel):
    """Configuration for a single ASOR agent to federate."""

    id: str = Field(
        ...,
        description="Agent ID in ASOR"
    )
    endpoint: str = Field(
        ...,
        description="Full API endpoint URL for this agent"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this agent should be synced"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for this agent"
    )


class AsorFederationConfig(BaseModel):
    """Configuration for Workday ASOR (Agent Service Operating Registry) federation."""

    enabled: bool = Field(
        default=False,
        description="Enable federation with ASOR"
    )
    endpoint: str = Field(
        default="https://api.asor.workday.com/v1",
        description="ASOR API base URL"
    )
    agents: List[AsorAgentConfig] = Field(
        default_factory=list,
        description="List of agents to federate from ASOR"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds (default: 1 hour)",
        ge=60,
        le=86400
    )
    sync_interval_seconds: int = Field(
        default=300,
        description="Sync interval in seconds (default: 5 minutes)",
        ge=60,
        le=3600
    )
    sync_on_startup: bool = Field(
        default=True,
        description="Whether to sync on registry startup"
    )
    display_options: DisplayOptions = Field(
        default_factory=lambda: DisplayOptions(
            attribution_label="ASOR"
        ),
        description="Display options for ASOR agents"
    )
    auth_type: Optional[str] = Field(
        default="oauth2",
        description="Authentication type (oauth2, api-key)"
    )
    auth_env_var: Optional[str] = Field(
        default="ASOR_API_KEY",
        description="Environment variable containing auth credentials"
    )
    timeout_seconds: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
        ge=5,
        le=120
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed requests",
        ge=1,
        le=10
    )


class FederationConfig(BaseModel):
    """Root configuration for registry federation."""

    anthropic: AnthropicFederationConfig = Field(
        default_factory=AnthropicFederationConfig,
        description="Anthropic MCP Registry federation settings"
    )
    asor: AsorFederationConfig = Field(
        default_factory=AsorFederationConfig,
        description="ASOR federation settings"
    )

    def is_any_federation_enabled(self) -> bool:
        """Check if any federation is enabled."""
        return self.anthropic.enabled or self.asor.enabled

    def get_enabled_federations(self) -> List[str]:
        """Get list of enabled federation sources."""
        enabled = []
        if self.anthropic.enabled:
            enabled.append("anthropic")
        if self.asor.enabled:
            enabled.append("asor")
        return enabled


class FederatedServer(BaseModel):
    """Represents a server from a federated registry."""

    source: str = Field(
        ...,
        description="Source registry (anthropic, asor, etc.)"
    )
    server_name: str = Field(
        ...,
        description="Server name from source registry"
    )
    description: Optional[str] = Field(
        None,
        description="Server description"
    )
    version: str = Field(
        default="1.0.0",
        description="Server version"
    )
    proxy_pass_url: Optional[str] = Field(
        None,
        description="Proxy URL to reach the server"
    )
    transport_type: str = Field(
        default="streamable-http",
        description="Transport type"
    )
    requires_auth: bool = Field(
        default=False,
        description="Whether server requires authentication"
    )
    auth_headers: Optional[List[Dict[str, str]]] = Field(
        default_factory=list,
        description="Authentication headers"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Server tags"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from source registry"
    )
    cached_at: Optional[str] = Field(
        None,
        description="When this server data was cached (ISO 8601)"
    )
    is_read_only: bool = Field(
        default=True,
        description="Whether this is a read-only federated server"
    )
    attribution_label: str = Field(
        default="External Registry",
        description="Label showing source registry"
    )
