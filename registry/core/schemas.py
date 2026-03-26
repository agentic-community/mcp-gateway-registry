from typing import Any
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from registry.schemas.agent_models import AgentProvider
from registry.schemas.registry_card import LifecycleStatus


class ServerVersion(BaseModel):
    """Represents a single version of an MCP server.

    Used for multi-version server support where different versions
    can run simultaneously behind a single endpoint.
    """

    version: str = Field(..., description="Version identifier (e.g., 'v2.0.0', 'v1.5.0')")
    proxy_pass_url: str = Field(..., description="Backend URL for this version")
    status: str = Field(default="stable", description="Version status: stable, deprecated, beta")
    is_default: bool = Field(
        default=False, description="Whether this is the default (latest) version"
    )
    released: str | None = Field(default=None, description="Release date (ISO format)")
    sunset_date: str | None = Field(
        default=None, description="Deprecation sunset date (ISO format)"
    )
    description: str | None = Field(
        default=None, description="Version-specific description (if different from main)"
    )


class ServerInfo(BaseModel):
    """Server information model."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier (UUID) for this server",
    )
    server_name: str
    description: str = ""
    path: str
    proxy_pass_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    num_tools: int = 0
    license: str = "N/A"
    tool_list: list[dict[str, Any]] = Field(default_factory=list)
    is_enabled: bool = False
    transport: str | None = Field(
        default="auto", description="Preferred transport: sse, streamable-http, or auto"
    )
    supported_transports: list[str] = Field(
        default_factory=lambda: ["streamable-http"], description="List of supported transports"
    )
    mcp_endpoint: str | None = Field(
        default=None,
        description="Full URL for the MCP streamable-http endpoint. If set, used directly for health checks and client connections instead of appending /mcp to proxy_pass_url. Example: 'https://server.com/custom-path'",
    )
    sse_endpoint: str | None = Field(
        default=None,
        description="Full URL for the SSE endpoint. If set, used directly for health checks and client connections instead of appending /sse to proxy_pass_url. Example: 'https://server.com/events'",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata for organization, compliance, or integration purposes",
    )
    # Version routing fields
    version: str | None = Field(
        default=None,
        description="Current version identifier (e.g., 'v1.0.0'). None for legacy single-version servers.",
    )
    versions: list[ServerVersion] | None = Field(
        default=None,
        description="List of available versions. None = single-version server (backward compatible).",
    )
    default_version: str | None = Field(
        default=None, description="Default version identifier for routing (e.g., 'v2.0.0')"
    )
    is_active: bool = Field(
        default=True,
        description="Whether this is the active version. False for inactive versions in multi-version setup.",
    )
    version_group: str | None = Field(
        default=None, description="Groups related versions together (derived from path)"
    )
    other_version_ids: list[str] = Field(
        default_factory=list, description="IDs of other versions in this group (for quick lookup)"
    )

    def get_default_proxy_url(self) -> str:
        """Get the proxy URL for the default version."""
        if not self.versions:
            return self.proxy_pass_url or ""

        for v in self.versions:
            if v.is_default or v.version == self.default_version:
                return v.proxy_pass_url

        # Fallback to first version or original proxy_pass_url
        if self.versions:
            return self.versions[0].proxy_pass_url
        return self.proxy_pass_url or ""

    def has_multiple_versions(self) -> bool:
        """Check if server has multiple versions configured."""
        return self.versions is not None and len(self.versions) > 1

    # Federation and access control fields
    visibility: str = Field(
        default="public",
        description="Federation visibility: public (shared with all peers), group-restricted (shared with allowed_groups only), or internal (never shared)",
    )
    allowed_groups: list[str] = Field(
        default_factory=list, description="Groups with access when visibility is group-restricted"
    )
    sync_metadata: dict[str, Any] | None = Field(
        default=None, description="Metadata for items synced from peer registries"
    )

    # ANS Integration
    ans_metadata: dict[str, Any] | None = Field(
        default=None,
        alias="ansMetadata",
        description="ANS (Agent Name Service) verification metadata",
    )

    # Backend authentication (replaces legacy auth_type)
    auth_scheme: str = Field(
        default="none",
        description="Authentication scheme for backend server: none, bearer, api_key",
    )
    auth_credential_encrypted: str | None = Field(
        default=None,
        description="Encrypted auth credential (Fernet). Never returned in API responses.",
    )
    auth_header_name: str | None = Field(
        default=None,
        description="Custom header name. Default: 'Authorization' for bearer, 'X-API-Key' for api_key.",
    )
    credential_updated_at: str | None = Field(
        default=None, description="ISO timestamp of last credential update."
    )

    # Lifecycle and federation metadata fields
    status: LifecycleStatus = Field(
        default=LifecycleStatus.ACTIVE,
        description="Lifecycle status",
    )
    provider: AgentProvider | None = Field(
        default=None,
        description="Provider organization and URL",
    )
    source_created_at: datetime | None = Field(
        default=None,
        description="Original creation timestamp from source system",
    )
    source_updated_at: datetime | None = Field(
        default=None,
        description="Last update timestamp from source system",
    )
    external_tags: list[str] = Field(
        default_factory=list,
        description="Tags from external/source system (separate from local tags)",
    )

    @field_validator("visibility")
    @classmethod
    def _validate_visibility(
        cls,
        v: str,
    ) -> str:
        """Validate visibility value."""
        valid_values = ["public", "group-restricted", "internal"]
        if v not in valid_values:
            raise ValueError(f"Visibility must be one of: {', '.join(valid_values)}")
        return v

    @model_validator(mode="after")
    def _populate_provider_default(self) -> "ServerInfo":
        """Populate default provider from config if not set."""
        if self.provider is None:
            from registry.core.config import settings

            self.provider = AgentProvider(
                organization=settings.registry_organization_name,
                url=settings.registry_url,
            )
        return self


class ToolDescription(BaseModel):
    """Parsed tool description sections."""

    main: str = "No description available."
    args: str | None = None
    returns: str | None = None
    raises: str | None = None


class ToolInfo(BaseModel):
    """Tool information model."""

    name: str
    parsed_description: ToolDescription
    tool_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    server_path: str | None = None
    server_name: str | None = None

    class Config:
        populate_by_name = True


class HealthStatus(BaseModel):
    """Health check status model."""

    status: str
    last_checked_iso: str | None = None
    num_tools: int = 0


class SessionData(BaseModel):
    """Session data model."""

    username: str
    auth_method: str = "oauth2"
    provider: str = "local"


class ServiceRegistrationRequest(BaseModel):
    """Service registration request model."""

    name: str = Field(..., min_length=1)
    description: str = ""
    path: str = Field(..., min_length=1)
    proxy_pass_url: str = Field(..., min_length=1)
    tags: str = ""
    num_tools: int = Field(0, ge=0)
    license: str = "N/A"
    transport: str | None = Field(
        default="auto", description="Preferred transport: sse, streamable-http, or auto"
    )
    supported_transports: str = Field(
        default="streamable-http", description="Comma-separated list of supported transports"
    )
    mcp_endpoint: str | None = Field(
        default=None,
        description="Full URL for the MCP streamable-http endpoint. If set, used directly for health checks and client connections instead of appending /mcp to proxy_pass_url. Example: 'https://server.com/custom-path'",
    )
    sse_endpoint: str | None = Field(
        default=None,
        description="Full URL for the SSE endpoint. If set, used directly for health checks and client connections instead of appending /sse to proxy_pass_url. Example: 'https://server.com/events'",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata for organization, compliance, or integration purposes",
    )
    visibility: str = Field(
        default="public",
        description="Federation visibility: public (shared with all peers), group-restricted (shared with allowed_groups only), or internal (never shared)",
    )
    allowed_groups: list[str] = Field(
        default_factory=list, description="Groups with access when visibility is group-restricted"
    )
    auth_scheme: str = Field(
        default="none", description="Authentication scheme: none, bearer, api_key"
    )
    auth_credential: str | None = Field(
        default=None,
        description="Plaintext credential (encrypted before storage, never stored as-is)",
    )
    auth_header_name: str | None = Field(
        default=None, description="Custom header name for API key auth. Default: X-API-Key"
    )
    status: LifecycleStatus = Field(
        default=LifecycleStatus.ACTIVE,
        description="Lifecycle status: active, deprecated, draft, or beta",
    )


class AuthCredentialUpdateRequest(BaseModel):
    """Request model for updating server auth credentials via PATCH."""

    auth_scheme: str = Field(..., description="Authentication scheme: none, bearer, api_key")
    auth_credential: str | None = Field(
        default=None, description="New credential (required if auth_scheme is not 'none')"
    )
    auth_header_name: str | None = Field(
        default=None, description="Custom header name. Default: X-API-Key for api_key"
    )


class OAuth2Provider(BaseModel):
    """OAuth2 provider information."""

    name: str
    display_name: str
    icon: str | None = None


class FaissMetadata(BaseModel):
    """FAISS metadata model."""

    id: int
    text_for_embedding: str
    full_server_info: ServerInfo
