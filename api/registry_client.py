#!/usr/bin/env python3
"""
MCP Gateway Registry Client - Standalone Pydantic-based client for the Registry API.

This client provides a type-safe interface to the MCP Gateway Registry API endpoints
documented in:
- /home/ubuntu/repos/mcp-gateway-registry/docs/api-specs/server-management.yaml (Server Management)
- /home/ubuntu/repos/mcp-gateway-registry/docs/api-specs/a2a-agent-management.yaml (Agent Management)

Authentication is handled via JWT tokens retrieved from AWS SSM Parameter Store using
the get-m2m-token.sh script.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import quote
from uuid import UUID

import requests
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration for servers."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class ServiceRegistration(BaseModel):
    """Service registration request model (UI-based registration)."""

    name: str = Field(..., description="Service name")
    description: str = Field(..., description="Service description")
    path: str = Field(..., description="Service path")
    proxy_pass_url: str = Field(..., description="Proxy pass URL")
    tags: str | None = Field(None, description="Comma-separated tags")
    num_tools: int | None = Field(None, description="Number of tools")
    license: str | None = Field(None, description="License type")


class InternalServiceRegistration(BaseModel):
    """Internal service registration model (Admin/M2M registration)."""

    service_path: str = Field(
        ..., alias="path", description="Service path (e.g., /cloudflare-docs)"
    )
    name: str | None = Field(None, description="Service name")
    description: str | None = Field(None, description="Service description")
    proxy_pass_url: str | None = Field(None, description="Proxy pass URL")
    version: str | None = Field(None, description="Server version (e.g., v1.0.0, v2.0.0)")
    status: str | None = Field(None, description="Version status (stable, beta, deprecated)")
    auth_provider: str | None = Field(None, description="Authentication provider")
    auth_scheme: str | None = Field(
        None, description="Authentication scheme (e.g., 'bearer', 'api_key', 'none')"
    )
    supported_transports: list[str] | None = Field(None, description="Supported transports")
    headers: dict[str, str] | None = Field(None, description="Custom headers")
    tool_list_json: str | None = Field(None, description="Tool list as JSON string")
    tags: list[str] | None = Field(None, description="Categorization tags")
    overwrite: bool | None = Field(False, description="Overwrite if exists")
    mcp_endpoint: str | None = Field(
        None,
        description="Full URL for the MCP streamable-http endpoint (overrides proxy_pass_url + /mcp)",
    )
    sse_endpoint: str | None = Field(
        None, description="Full URL for the SSE endpoint (overrides proxy_pass_url + /sse)"
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Additional custom metadata for organization, compliance, or integration purposes",
    )
    provider_organization: str | None = Field(None, description="Provider organization name")
    provider_url: str | None = Field(None, description="Provider URL")
    source_created_at: str | None = Field(
        None, description="Original creation timestamp (ISO format)"
    )
    source_updated_at: str | None = Field(None, description="Last update timestamp (ISO format)")
    external_tags: list[str] | None = Field(None, description="Tags from external/source system")
    auth_credential: str | None = Field(
        None,
        description="Plaintext auth credential (Bearer token or API key). Encrypted before storage.",
    )

    model_config = ConfigDict(populate_by_name=True)


class Server(BaseModel):
    """Server information model."""

    path: str = Field(..., description="Service path")
    display_name: str = Field(..., description="Service display name")
    description: str = Field(..., description="Service description")
    is_enabled: bool = Field(..., description="Whether service is enabled")
    health_status: HealthStatus = Field(..., description="Health status")
    status: str = Field(
        default="active",
        description="Lifecycle status (active, deprecated, draft, beta)",
    )


class ServerDetail(BaseModel):
    """Detailed server information model."""

    path: str = Field(..., description="Service path")
    name: str = Field(..., description="Service name")
    description: str = Field(..., description="Service description")
    url: str = Field(..., description="Service URL")
    is_enabled: bool = Field(..., description="Whether service is enabled")
    num_tools: int = Field(..., description="Number of tools")
    health_status: str = Field(..., description="Health status")
    last_health_check: datetime | None = Field(None, description="Last health check timestamp")
    status: str = Field(
        default="active", description="Server status (active, deprecated, draft, beta)"
    )
    provider: dict[str, str] | None = Field(
        None, description="Provider information (organization, url)"
    )
    source_created_at: str | None = Field(None, description="Creation timestamp in source system")
    source_updated_at: str | None = Field(
        None, description="Last update timestamp in source system"
    )
    external_tags: list[str] = Field(default_factory=list, description="Tags from external source")


class ServerDetailResponse(BaseModel):
    """Response model for single server retrieval via GET /api/servers/{path}."""

    server_name: str = Field(default="", description="Server display name")
    description: str = Field(default="", description="Server description")
    path: str = Field(..., description="Server path (e.g., /my-server)")
    proxy_pass_url: str | None = Field(None, description="Backend URL")
    tags: list[str] = Field(default_factory=list, description="Server tags")
    num_tools: int = Field(default=0, description="Number of tools")
    tool_list: list[dict[str, Any]] = Field(default_factory=list, description="Tool definitions")
    is_enabled: bool = Field(default=False, description="Whether server is enabled")
    health_status: str | None = Field(None, description="Health status")
    transport: str | None = Field(None, description="Transport type")
    version: str | None = Field(None, description="Server version")
    versions: list[dict[str, Any]] | None = Field(None, description="Version list")
    license: str = Field(default="N/A", description="License")
    registered_by: str | None = Field(None, description="Who registered")

    model_config = ConfigDict(extra="allow")


class ServerListResponse(BaseModel):
    """Server list response model."""

    servers: list[Server] = Field(..., description="List of servers")
    total_count: int = Field(..., description="Total count of matching servers (all pages)")
    limit: int = Field(..., description="Page size applied")
    offset: int = Field(..., description="Offset applied")
    has_next: bool = Field(..., description="Whether more pages exist")


class ServiceResponse(BaseModel):
    """Service operation response model."""

    path: str = Field(..., description="Service path")
    name: str = Field(..., description="Service name")
    message: str = Field(..., description="Response message")


class ToggleResponse(BaseModel):
    """Toggle service response model."""

    path: str = Field(..., description="Service path")
    is_enabled: bool = Field(..., description="Current enabled status")
    message: str = Field(..., description="Response message")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error detail message")
    error_code: str | None = Field(None, description="Error code")
    request_id: str | None = Field(None, description="Request ID")


class SecurityScanResult(BaseModel):
    """Security scan result model."""

    analysis_results: dict[str, Any] = Field(..., description="Analysis results by analyzer")
    tool_results: list[dict[str, Any]] = Field(..., description="Detailed tool scan results")


class RescanResponse(BaseModel):
    """Server rescan response model."""

    server_url: str = Field(..., description="Server URL that was scanned")
    server_path: str = Field(..., description="Server path")
    scan_timestamp: str = Field(..., description="Scan timestamp")
    is_safe: bool = Field(..., description="Whether server is safe")
    critical_issues: int = Field(..., description="Number of critical issues")
    high_severity: int = Field(..., description="Number of high severity issues")
    medium_severity: int = Field(..., description="Number of medium severity issues")
    low_severity: int = Field(..., description="Number of low severity issues")
    analyzers_used: list[str] = Field(..., description="Analyzers used in scan")
    scan_failed: bool = Field(..., description="Whether scan failed")
    error_message: str | None = Field(None, description="Error message if scan failed")
    raw_output: dict[str, Any] | None = Field(None, description="Raw scan output")


class AgentSecurityScanResponse(BaseModel):
    """Agent security scan results response model."""

    analysis_results: dict[str, Any] = Field(
        default_factory=dict, description="Analysis results by analyzer"
    )
    scan_results: dict[str, Any] = Field(
        default_factory=dict, description="Scan results and metadata"
    )


class AgentRescanResponse(BaseModel):
    """Agent rescan response model."""

    agent_path: str = Field(..., description="Agent path")
    agent_url: str = Field(..., description="Agent URL that was scanned")
    scan_timestamp: str = Field(..., description="Scan timestamp")
    is_safe: bool = Field(..., description="Whether agent is safe")
    critical_issues: int = Field(..., description="Number of critical issues")
    high_severity: int = Field(..., description="Number of high severity issues")
    medium_severity: int = Field(..., description="Number of medium severity issues")
    low_severity: int = Field(..., description="Number of low severity issues")
    analyzers_used: list[str] = Field(..., description="Analyzers used in scan")
    scan_failed: bool = Field(..., description="Whether scan failed")
    error_message: str | None = Field(None, description="Error message if scan failed")
    output_file: str | None = Field(None, description="Path to scan output file")


class SkillSecurityScanResponse(BaseModel):
    """Skill security scan results response model."""

    skill_path: str = Field(..., description="Skill path")
    skill_md_url: str | None = Field(None, description="Skill SKILL.md URL")
    scan_timestamp: str = Field(..., description="Scan timestamp")
    is_safe: bool = Field(..., description="Whether skill is safe")
    critical_issues: int = Field(default=0, description="Number of critical issues")
    high_severity: int = Field(default=0, description="Number of high severity issues")
    medium_severity: int = Field(default=0, description="Number of medium severity issues")
    low_severity: int = Field(default=0, description="Number of low severity issues")
    analyzers_used: list[str] = Field(default_factory=list, description="Analyzers used in scan")
    raw_output: dict[str, Any] = Field(default_factory=dict, description="Raw scanner output")
    scan_failed: bool = Field(default=False, description="Whether scan failed")
    error_message: str | None = Field(None, description="Error message if scan failed")


class SkillRescanResponse(BaseModel):
    """Skill rescan response model."""

    skill_path: str = Field(..., description="Skill path")
    skill_md_url: str | None = Field(None, description="Skill SKILL.md URL")
    scan_timestamp: str = Field(..., description="Scan timestamp")
    is_safe: bool = Field(..., description="Whether skill is safe")
    critical_issues: int = Field(default=0, description="Number of critical issues")
    high_severity: int = Field(default=0, description="Number of high severity issues")
    medium_severity: int = Field(default=0, description="Number of medium severity issues")
    low_severity: int = Field(default=0, description="Number of low severity issues")
    analyzers_used: list[str] = Field(default_factory=list, description="Analyzers used in scan")
    raw_output: dict[str, Any] = Field(default_factory=dict, description="Raw scanner output")
    scan_failed: bool = Field(default=False, description="Whether scan failed")
    error_message: str | None = Field(None, description="Error message if scan failed")


class GroupListResponse(BaseModel):
    """Group list response model."""

    groups: list[dict[str, Any]] = Field(..., description="List of groups")
    total: int = Field(..., description="Total number of groups")


# Agent Management Models


class AgentProvider(str, Enum):
    """Agent provider enumeration."""

    ANTHROPIC = "anthropic"
    CUSTOM = "custom"
    OTHER = "other"


class AgentVisibility(str, Enum):
    """Agent visibility enumeration."""

    PUBLIC = "public"
    PRIVATE = "private"
    GROUP_RESTRICTED = "group-restricted"


class Provider(BaseModel):
    """
    A2A Agent Provider information.

    Represents the service provider of an agent with organization name and website URL.
    Per A2A specification, if provider is present, both organization and url are required.
    """

    organization: str = Field(..., description="Provider organization name")
    url: str = Field(..., description="Provider website or documentation URL")


class SecuritySchemeType(str, Enum):
    """Security scheme type enumeration (A2A spec values)."""

    API_KEY = "apiKey"
    HTTP = "http"
    OAUTH2 = "oauth2"
    OPENID_CONNECT = "openIdConnect"


class SecurityScheme(BaseModel):
    """
    Security scheme model.
    Note: Uses snake_case internally but serializes to camelCase for A2A compliance.
    """

    type: SecuritySchemeType = Field(..., description="Security scheme type")
    scheme: str | None = Field(
        None,
        description="HTTP auth scheme: basic, bearer, digest",
    )
    in_: str | None = Field(
        None,
        alias="in",
        description="API key location: header, query, cookie",
    )
    name: str | None = Field(
        None,
        description="Name of header/query/cookie for API key",
    )
    bearer_format: str | None = Field(
        None,
        alias="bearerFormat",
        description="Bearer token format hint (e.g., JWT)",
    )
    flows: dict[str, Any] | None = Field(
        None,
        description="OAuth2 flows configuration",
    )
    openid_connect_url: str | None = Field(
        None,
        alias="openIdConnectUrl",
        description="OpenID Connect discovery URL",
    )
    description: str | None = Field(None, description="Security scheme description")

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase on input


class Skill(BaseModel):
    """
    Agent skill definition per A2A protocol specification.
    Note: Uses snake_case internally but serializes to camelCase for A2A compliance.
    """

    id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(..., description="Detailed skill description")
    tags: list[str] = Field(default_factory=list, description="Skill categorization tags")
    examples: list[str] | None = Field(None, description="Usage scenarios and examples")
    input_modes: list[str] | None = Field(
        None, alias="inputModes", description="Skill-specific input MIME types"
    )
    output_modes: list[str] | None = Field(
        None, alias="outputModes", description="Skill-specific output MIME types"
    )
    security: list[dict[str, list[str]]] | None = Field(
        None, description="Skill-level security requirements"
    )

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase on input


class AgentRegistration(BaseModel):
    """
    Agent registration request model matching server AgentCard schema.
    This model represents a complete agent card following the A2A protocol
    specification (v0.3.0), with extensions for MCP Gateway Registry integration.
    Note: Uses snake_case internally but serializes to camelCase for A2A compliance.
    """

    # Required A2A fields
    protocol_version: str = Field(
        "1.0", alias="protocolVersion", description="A2A protocol version (e.g., '1.0')"
    )
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    url: str = Field(..., description="Agent endpoint URL (HTTP or HTTPS)")
    version: str = Field(..., description="Agent version")
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Feature declarations (e.g., {'streaming': true})"
    )
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"],
        alias="defaultInputModes",
        description="Supported input MIME types",
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"],
        alias="defaultOutputModes",
        description="Supported output MIME types",
    )
    skills: list[Skill] = Field(default_factory=list, description="Agent capabilities (skills)")

    # Optional A2A fields
    preferred_transport: str | None = Field(
        "JSONRPC",
        alias="preferredTransport",
        description="Preferred transport protocol: JSONRPC, GRPC, HTTP+JSON",
    )
    provider: Provider | None = Field(None, description="Agent provider information per A2A spec")
    icon_url: str | None = Field(None, alias="iconUrl", description="Agent icon URL")
    documentation_url: str | None = Field(
        None, alias="documentationUrl", description="Documentation URL"
    )
    security_schemes: dict[str, SecurityScheme | dict[str, Any]] = Field(
        default_factory=dict,
        alias="securitySchemes",
        description="Supported authentication methods",
    )
    security: list[dict[str, list[str]]] | None = Field(
        None, description="Security requirements array"
    )
    supports_authenticated_extended_card: bool | None = Field(
        None,
        alias="supportsAuthenticatedExtendedCard",
        description="Supports extended card with auth",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # MCP Gateway Registry extensions (optional - not part of A2A spec)
    path: str | None = Field(
        None,
        description="Registry path (e.g., /agents/my-agent). Optional - auto-generated if not provided.",
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    is_enabled: bool = Field(
        False, alias="isEnabled", description="Whether agent is enabled in registry"
    )
    num_stars: int = Field(0, ge=0, alias="numStars", description="Community rating")
    license: str = Field("N/A", description="License information")
    registered_at: datetime | None = Field(
        None, alias="registeredAt", description="Registration timestamp"
    )
    updated_at: datetime | None = Field(
        None, alias="updatedAt", description="Last update timestamp"
    )
    registered_by: str | None = Field(
        None, alias="registeredBy", description="Username who registered agent"
    )
    visibility: str = Field("public", description="public, private, or group-restricted")
    allowed_groups: list[str] = Field(
        default_factory=list, alias="allowedGroups", description="Groups with access"
    )
    signature: str | None = Field(None, description="JWS signature for card integrity")
    trust_level: str = Field(
        "unverified", alias="trustLevel", description="unverified, community, verified, trusted"
    )
    supported_protocol: str | None = Field(
        None, alias="supportedProtocol", description="Agent protocol: a2a or other"
    )

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase on input


class AgentCard(BaseModel):
    """Agent card model (summary view)."""

    name: str = Field(..., description="Agent name")
    path: str = Field(..., description="Agent path")
    url: str = Field(..., description="Agent URL")
    num_skills: int = Field(..., description="Number of skills")
    registered_at: datetime | None = Field(None, description="Registration timestamp")
    is_enabled: bool = Field(..., description="Whether agent is enabled")
    status: str = Field(
        default="active", description="Agent status (active, deprecated, draft, beta)"
    )
    source_created_at: str | None = Field(
        None, alias="sourceCreatedAt", description="Creation timestamp in source system"
    )
    source_updated_at: str | None = Field(
        None, alias="sourceUpdatedAt", description="Last update timestamp in source system"
    )
    external_tags: list[str] = Field(
        default_factory=list, alias="externalTags", description="Tags from external source"
    )
    supported_protocol: str | None = Field(
        None, alias="supportedProtocol", description="Agent protocol: 'a2a' or 'other'"
    )

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase on input


class AgentRegistrationResponse(BaseModel):
    """Agent registration response model."""

    message: str = Field(..., description="Response message")
    agent: AgentCard = Field(..., description="Registered agent card")


class SkillDetail(BaseModel):
    """
    Detailed skill model - same as Skill.
    Note: Uses snake_case internally but serializes to camelCase for A2A compliance.
    """

    id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(..., description="Detailed skill description")
    tags: list[str] = Field(default_factory=list, description="Skill categorization tags")
    examples: list[str] | None = Field(None, description="Usage scenarios and examples")
    input_modes: list[str] | None = Field(
        None, alias="inputModes", description="Skill-specific input MIME types"
    )
    output_modes: list[str] | None = Field(
        None, alias="outputModes", description="Skill-specific output MIME types"
    )
    security: list[dict[str, list[str]]] | None = Field(
        None, description="Skill-level security requirements"
    )

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase on input


class AgentDetail(BaseModel):
    """
    Detailed agent model matching server AgentCard schema.
    This model represents a complete agent card following the A2A protocol
    specification (v0.3.0), with extensions for MCP Gateway Registry integration.
    Note: Uses snake_case internally but serializes to camelCase for A2A compliance.
    """

    # Required A2A fields
    protocol_version: str = Field(..., alias="protocolVersion", description="A2A protocol version")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    url: str = Field(..., description="Agent endpoint URL")
    version: str = Field(..., description="Agent version")
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Feature declarations (e.g., {'streaming': true})"
    )
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"],
        alias="defaultInputModes",
        description="Supported input MIME types",
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"],
        alias="defaultOutputModes",
        description="Supported output MIME types",
    )
    skills: list[SkillDetail] = Field(
        default_factory=list, description="Agent capabilities (skills)"
    )

    # Optional A2A fields
    preferred_transport: str | None = Field(
        "JSONRPC",
        alias="preferredTransport",
        description="Preferred transport protocol: JSONRPC, GRPC, HTTP+JSON",
    )
    provider: Provider | None = Field(None, description="Agent provider information per A2A spec")
    icon_url: str | None = Field(None, alias="iconUrl", description="Agent icon URL")
    documentation_url: str | None = Field(
        None, alias="documentationUrl", description="Documentation URL"
    )
    security_schemes: dict[str, SecurityScheme | dict[str, Any]] = Field(
        default_factory=dict,
        alias="securitySchemes",
        description="Supported authentication methods",
    )
    security: list[dict[str, list[str]]] | None = Field(
        None, description="Security requirements array"
    )
    supports_authenticated_extended_card: bool | None = Field(
        None,
        alias="supportsAuthenticatedExtendedCard",
        description="Supports extended card with auth",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # MCP Gateway Registry extensions (optional - not part of A2A spec)
    path: str | None = Field(None, description="Registry path")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    is_enabled: bool = Field(False, alias="isEnabled", description="Whether agent is enabled")
    num_stars: int = Field(0, ge=0, alias="numStars", description="Community rating")
    license: str = Field("N/A", description="License information")
    registered_at: datetime | None = Field(
        None, alias="registeredAt", description="Registration timestamp"
    )
    updated_at: datetime | None = Field(
        None, alias="updatedAt", description="Last update timestamp"
    )
    registered_by: str | None = Field(
        None, alias="registeredBy", description="Username who registered agent"
    )
    visibility: str = Field("public", description="Visibility level")
    allowed_groups: list[str] = Field(
        default_factory=list, alias="allowedGroups", description="Groups with access"
    )
    trust_level: str = Field("community", alias="trustLevel", description="Trust level")
    ans_metadata: dict[str, Any] | None = Field(
        default=None,
        alias="ansMetadata",
        description="ANS (Agent Name Service) verification metadata",
    )
    signature: str | None = Field(None, description="JWS signature for card integrity")
    status: str = Field(
        default="active", description="Agent status (active, deprecated, draft, beta)"
    )
    source_created_at: str | None = Field(
        None, alias="sourceCreatedAt", description="Creation timestamp in source system"
    )
    source_updated_at: str | None = Field(
        None, alias="sourceUpdatedAt", description="Last update timestamp in source system"
    )
    external_tags: list[str] = Field(
        default_factory=list, alias="externalTags", description="Tags from external source"
    )
    supported_protocol: str | None = Field(
        None, alias="supportedProtocol", description="Agent protocol: 'a2a' or 'other'"
    )

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase on input


class AgentListItem(BaseModel):
    """
    Agent list item model (AgentInfo from server).
    Note: Uses snake_case internally but serializes to camelCase for A2A compliance.
    """

    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    path: str = Field(..., description="Agent path")
    url: str = Field(..., description="Agent URL")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    skills: list[str] = Field(default_factory=list, description="Skill names")
    num_skills: int = Field(default=0, alias="numSkills", description="Number of skills")
    num_stars: float = Field(
        default=0.0, alias="numStars", description="Average community rating (0.0-5.0)"
    )
    is_enabled: bool = Field(
        default=False, alias="isEnabled", description="Whether agent is enabled"
    )
    provider: str | None = Field(None, description="Agent provider")
    streaming: bool = Field(default=False, description="Supports streaming")
    trust_level: str = Field(default="unverified", alias="trustLevel", description="Trust level")
    ans_metadata: dict[str, Any] | None = Field(
        default=None,
        alias="ansMetadata",
        description="ANS (Agent Name Service) verification metadata",
    )
    sync_metadata: dict[str, Any] | None = Field(
        default=None,
        alias="syncMetadata",
        description="Federation sync metadata for items from peer registries",
    )
    status: str = Field(
        default="active",
        description="Lifecycle status (active, deprecated, draft, beta)",
    )

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase on input


class AgentListResponse(BaseModel):
    """Agent list response model."""

    agents: list[AgentListItem] = Field(..., description="List of agents")
    total_count: int = Field(..., description="Total count of matching agents (all pages)")
    limit: int = Field(..., description="Page size applied")
    offset: int = Field(..., description="Offset applied")
    has_next: bool = Field(..., description="Whether more pages exist")


class AgentToggleResponse(BaseModel):
    """Agent toggle response model."""

    path: str = Field(..., description="Agent path")
    is_enabled: bool = Field(..., description="Current enabled status")
    message: str = Field(..., description="Response message")


class SkillDiscoveryRequest(BaseModel):
    """Skill-based discovery request model."""

    skills: list[str] = Field(..., description="List of required skills")
    tags: list[str] | None = Field(None, description="Optional tag filters")


class DiscoveredAgent(BaseModel):
    """Discovered agent model (skill-based)."""

    path: str = Field(..., description="Agent path")
    name: str = Field(..., description="Agent name")
    relevance_score: float = Field(..., description="Matching score (0.0 to 1.0)")
    matching_skills: list[str] = Field(..., description="Matching skills")


class AgentDiscoveryResponse(BaseModel):
    """Agent discovery response model (skill-based)."""

    agents: list[DiscoveredAgent] = Field(..., description="Discovered agents")


class SemanticDiscoveredAgent(BaseModel):
    """Semantically discovered agent model with full AgentCard fields."""

    # Core identification
    path: str = Field(..., description="Agent path")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    url: str = Field(..., description="Agent endpoint URL")

    # Semantic search relevance
    relevance_score: float = Field(..., description="Semantic similarity score")

    # Agent metadata
    tags: list[str] = Field(default_factory=list, description="Agent tags")
    skills: list[dict[str, Any]] = Field(default_factory=list, description="Agent skills")
    provider: dict[str, str] | None = Field(None, description="Provider information")
    capabilities: dict[str, Any] = Field(default_factory=dict, description="Agent capabilities")
    trust_level: str = Field("unverified", description="Trust level")
    trust_verified: str | None = Field(None, description="ANS trust verification status")
    ans_metadata: dict[str, Any] | None = Field(None, description="ANS verification metadata")
    num_stars: float = Field(0.0, description="Average rating")
    version: str | None = Field(None, description="Agent version")

    # Security and authentication
    security_schemes: dict[str, Any] = Field(default_factory=dict, description="Security schemes")

    # Timestamps
    created_at: str | None = Field(None, description="Creation timestamp")
    updated_at: str | None = Field(None, description="Last update timestamp")

    class Config:
        extra = "allow"  # Allow additional fields from API


class AgentSemanticDiscoveryResponse(BaseModel):
    """Agent semantic discovery response model."""

    agents: list[SemanticDiscoveredAgent] = Field(..., description="Semantically discovered agents")


class MatchingToolResult(BaseModel):
    """Tool matching result with optional schema for display."""

    tool_name: str = Field(..., description="Tool name")
    description: str | None = Field(None, description="Tool description")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    match_context: str | None = Field(None, description="Why this tool matched")
    inputSchema: dict[str, Any] | None = Field(
        None, description="JSON Schema for tool input parameters"
    )


class SyncMetadata(BaseModel):
    """Metadata for items synced from peer registries."""

    is_federated: bool = Field(False, description="Whether this is from a federated registry")
    source_peer_id: str | None = Field(None, description="Source peer registry ID")
    synced_at: str | None = Field(None, description="When item was synced")
    original_path: str | None = Field(None, description="Original path on source registry")
    is_orphaned: bool = Field(False, description="Whether item is orphaned")
    orphaned_at: str | None = Field(None, description="When item became orphaned")
    is_read_only: bool = Field(True, description="Whether item is read-only")


class SemanticDiscoveredServer(BaseModel):
    """Semantically discovered server model."""

    path: str = Field(..., description="Server path")
    server_name: str = Field(..., description="Server name")
    relevance_score: float = Field(..., description="Semantic similarity score")
    description: str | None = Field(None, description="Server description")
    tags: list[str] = Field(default_factory=list, description="Server tags")
    num_tools: int = Field(0, description="Number of tools")
    is_enabled: bool = Field(False, description="Whether server is enabled")
    match_context: str | None = Field(None, description="Why this matched")
    matching_tools: list[MatchingToolResult] = Field(
        default_factory=list, description="Matching tools"
    )
    sync_metadata: SyncMetadata | None = Field(
        None, description="Sync metadata for federated items"
    )
    # Endpoint URL for agent connectivity (computed based on deployment mode)
    endpoint_url: str | None = Field(
        None, description="URL for agents to connect to this MCP server"
    )
    # Raw endpoint fields (for advanced use cases)
    proxy_pass_url: str | None = Field(
        None, description="Base URL for the MCP server backend (internal)"
    )
    mcp_endpoint: str | None = Field(None, description="Explicit streamable-http endpoint URL")
    sse_endpoint: str | None = Field(None, description="Explicit SSE endpoint URL")
    supported_transports: list[str] = Field(
        default_factory=list, description="Supported transport types"
    )


class ToolSearchResult(BaseModel):
    """Tool search result model."""

    server_path: str = Field(..., description="Parent server path")
    server_name: str = Field(..., description="Parent server name")
    tool_name: str = Field(..., description="Tool name")
    description: str | None = Field(None, description="Tool description")
    inputSchema: dict[str, Any] | None = Field(None, description="JSON Schema for tool input")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    match_context: str | None = Field(None, description="Why this tool matched")
    # Endpoint URL for the parent MCP server
    endpoint_url: str | None = Field(
        None, description="URL for agents to connect to the parent MCP server"
    )


class AgentSearchResult(BaseModel):
    """Agent search result with minimal top-level fields.

    Only search-specific fields are at the top level. All agent details
    (name, description, url, skills, etc.) are in the agent_card.
    """

    path: str = Field(..., description="Agent path for identification")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    match_context: str | None = Field(None, description="Why this agent matched")
    agent_card: dict[str, Any] = Field(..., description="Full agent card with all details")


class SkillSearchResult(BaseModel):
    """Skill search result model."""

    path: str = Field(..., description="Skill path")
    skill_name: str = Field(..., description="Skill name")
    description: str | None = Field(None, description="Skill description")
    tags: list[str] = Field(default_factory=list, description="Skill tags")
    skill_md_url: str | None = Field(None, description="Skill markdown URL")
    skill_md_raw_url: str | None = Field(None, description="Skill markdown raw URL")
    version: str | None = Field(None, description="Skill version")
    author: str | None = Field(None, description="Skill author")
    visibility: str | None = Field(None, description="Visibility setting")
    owner: str | None = Field(None, description="Skill owner")
    is_enabled: bool = Field(False, description="Whether skill is enabled")
    health_status: str = Field("unknown", description="Health status")
    last_checked_time: str | None = Field(None, description="Last health check time")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    match_context: str | None = Field(None, description="Why this skill matched")


class VirtualServerSearchResult(BaseModel):
    """Virtual server search result model."""

    path: str = Field(..., description="Virtual server path")
    server_name: str = Field(..., description="Virtual server name")
    description: str | None = Field(None, description="Virtual server description")
    tags: list[str] = Field(default_factory=list, description="Virtual server tags")
    num_tools: int = Field(0, description="Number of tools")
    backend_count: int = Field(0, description="Number of backend servers")
    backend_paths: list[str] = Field(default_factory=list, description="Backend server paths")
    is_enabled: bool = Field(False, description="Whether virtual server is enabled")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    match_context: str | None = Field(None, description="Why this matched")
    matching_tools: list[MatchingToolResult] = Field(
        default_factory=list, description="Matching tools"
    )
    # Endpoint URL for agent connectivity
    endpoint_url: str | None = Field(
        None, description="URL for agents to connect to this virtual MCP server"
    )


class ToolMapping(BaseModel):
    """Tool mapping for virtual MCP servers."""

    tool_name: str = Field(..., description="Original tool name on backend server")
    alias: str | None = Field(None, description="Renamed tool name in virtual server")
    backend_server_path: str = Field(..., description="Backend server path (e.g., /github)")
    backend_version: str | None = Field(None, description="Pin to specific backend version")
    description_override: str | None = Field(None, description="Override tool description")


class ToolScopeOverride(BaseModel):
    """Per-tool scope override for access control."""

    tool_alias: str = Field(..., description="Tool alias or name")
    required_scopes: list[str] = Field(
        default_factory=list, description="Required scopes for this tool"
    )


class VirtualServerCreateRequest(BaseModel):
    """Request to create a virtual MCP server."""

    path: str = Field(..., description="Virtual server path (e.g., /virtual/dev-tools)")
    server_name: str = Field(..., description="Display name for the virtual server")
    description: str | None = Field(None, description="Virtual server description")
    tool_mappings: list[ToolMapping] = Field(
        ..., min_length=1, description="Tool mappings (at least one)"
    )
    required_scopes: list[str] = Field(
        default_factory=list, description="Server-level required scopes"
    )
    tool_scope_overrides: list[ToolScopeOverride] = Field(
        default_factory=list, description="Per-tool scope overrides"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    supported_transports: list[str] = Field(
        default_factory=lambda: ["streamable-http"], description="Supported transports"
    )
    is_enabled: bool = Field(True, description="Whether to enable on creation")


class VirtualServerConfig(BaseModel):
    """Full virtual MCP server configuration."""

    path: str = Field(..., description="Virtual server path")
    server_name: str = Field(..., description="Display name")
    description: str | None = Field(None, description="Description")
    tool_mappings: list[ToolMapping] = Field(default_factory=list, description="Tool mappings")
    required_scopes: list[str] = Field(default_factory=list, description="Server-level scopes")
    tool_scope_overrides: list[ToolScopeOverride] = Field(
        default_factory=list, description="Per-tool scope overrides"
    )
    tags: list[str] = Field(default_factory=list, description="Tags")
    supported_transports: list[str] = Field(
        default_factory=list, description="Supported transports"
    )
    is_enabled: bool = Field(False, description="Whether enabled")
    num_stars: float = Field(0.0, description="Average rating")
    rating_details: list[dict[str, Any]] = Field(
        default_factory=list, description="Individual ratings"
    )
    created_by: str | None = Field(None, description="Creator username")
    created_at: str | None = Field(None, description="Creation timestamp")
    updated_at: str | None = Field(None, description="Last update timestamp")


class VirtualServerListResponse(BaseModel):
    """Response for listing virtual servers."""

    virtual_servers: list[VirtualServerConfig] = Field(
        default_factory=list, description="Virtual servers"
    )
    total: int = Field(0, description="Total count")


class VirtualServerToggleResponse(BaseModel):
    """Response from toggling a virtual server."""

    path: str = Field(..., description="Virtual server path")
    is_enabled: bool = Field(..., description="New enabled state")
    message: str = Field(..., description="Status message")


class VirtualServerDeleteResponse(BaseModel):
    """Response from deleting a virtual server."""

    path: str = Field(..., description="Deleted virtual server path")
    message: str = Field(..., description="Status message")


class SemanticSearchResponse(BaseModel):
    """Comprehensive semantic search response with all entity types."""

    query: str = Field(..., description="Search query")
    search_mode: str = Field("hybrid", description="Search mode: hybrid or lexical-only")
    servers: list[SemanticDiscoveredServer] = Field(
        default_factory=list, description="Matching servers"
    )
    tools: list[ToolSearchResult] = Field(default_factory=list, description="Matching tools")
    agents: list[AgentSearchResult] = Field(default_factory=list, description="Matching agents")
    skills: list[SkillSearchResult] = Field(default_factory=list, description="Matching skills")
    virtual_servers: list[VirtualServerSearchResult] = Field(
        default_factory=list, description="Matching virtual servers"
    )
    total_servers: int = Field(0, description="Total server count")
    total_tools: int = Field(0, description="Total tool count")
    total_agents: int = Field(0, description="Total agent count")
    total_skills: int = Field(0, description="Total skill count")
    total_virtual_servers: int = Field(0, description="Total virtual server count")


class ServerSemanticSearchResponse(BaseModel):
    """Server semantic search response model (legacy, use SemanticSearchResponse)."""

    query: str = Field(..., description="Search query")
    servers: list[SemanticDiscoveredServer] = Field(
        default_factory=list, description="Matching servers"
    )


class RatingDetail(BaseModel):
    """Individual rating detail."""

    user: str = Field(..., description="Username who submitted the rating")
    rating: int = Field(..., ge=1, le=5, description="Rating value (1-5 stars)")


class RatingRequest(BaseModel):
    """Rating submission request."""

    rating: int = Field(..., ge=1, le=5, description="Rating value (1-5 stars)")


class RatingResponse(BaseModel):
    """Rating submission response."""

    message: str = Field(..., description="Success message")
    average_rating: float = Field(..., ge=1.0, le=5.0, description="Updated average rating")


class RatingInfoResponse(BaseModel):
    """Rating information response."""

    num_stars: float = Field(..., ge=0.0, le=5.0, description="Average rating (0.0 if no ratings)")
    rating_details: list[RatingDetail] = Field(..., description="Individual ratings (max 100)")


# Anthropic Registry API Models (v0.1)


class AnthropicRepository(BaseModel):
    """Repository metadata for MCP server source code (Anthropic Registry API)."""

    url: str = Field(..., description="Repository URL for browsing source code")
    source: str = Field(..., description="Repository hosting service identifier (e.g., 'github')")
    id: str | None = Field(None, description="Repository ID from hosting service")
    subfolder: str | None = Field(None, description="Path within monorepo")


class AnthropicStdioTransport(BaseModel):
    """Standard I/O transport configuration (Anthropic Registry API)."""

    type: str = Field(default="stdio")
    command: str | None = Field(None, description="Command to execute")
    args: list[str] | None = Field(None, description="Command arguments")
    env: dict[str, str] | None = Field(None, description="Environment variables")


class AnthropicStreamableHttpTransport(BaseModel):
    """HTTP-based transport configuration (Anthropic Registry API)."""

    type: str = Field(default="streamable-http")
    url: str = Field(..., description="HTTP endpoint URL")
    headers: dict[str, str] | None = Field(None, description="HTTP headers")


class AnthropicSseTransport(BaseModel):
    """Server-Sent Events transport configuration (Anthropic Registry API)."""

    type: str = Field(default="sse")
    url: str = Field(..., description="SSE endpoint URL")


class AnthropicPackage(BaseModel):
    """Package information for MCP server distribution (Anthropic Registry API)."""

    registryType: str = Field(..., description="Registry type (npm, pypi, oci, etc.)")
    identifier: str = Field(..., description="Package identifier or URL")
    version: str = Field(..., description="Specific package version")
    registryBaseUrl: str | None = Field(None, description="Base URL of package registry")
    transport: dict[str, Any] = Field(..., description="Transport configuration")
    runtimeHint: str | None = Field(None, description="Runtime hint (npx, uvx, docker, etc.)")


class AnthropicServerDetail(BaseModel):
    """Detailed MCP server information (Anthropic Registry API)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="Server name in reverse-DNS format")
    description: str = Field(..., description="Server description")
    version: str = Field(..., description="Server version")
    title: str | None = Field(None, description="Human-readable server name")
    repository: AnthropicRepository | None = Field(None, description="Repository information")
    websiteUrl: str | None = Field(None, description="Server website URL")
    packages: list[AnthropicPackage] | None = Field(None, description="Package distributions")
    meta: dict[str, Any] | None = Field(
        None, alias="_meta", serialization_alias="_meta", description="Extensible metadata"
    )


class AnthropicServerResponse(BaseModel):
    """Response for single server query (Anthropic Registry API)."""

    model_config = ConfigDict(populate_by_name=True)

    server: AnthropicServerDetail = Field(..., description="Server details")
    meta: dict[str, Any] | None = Field(
        None, alias="_meta", serialization_alias="_meta", description="Registry-managed metadata"
    )


class AnthropicPaginationMetadata(BaseModel):
    """Pagination information for server lists (Anthropic Registry API)."""

    nextCursor: str | None = Field(None, description="Cursor for next page")
    count: int | None = Field(None, description="Number of items in current page")


class AnthropicServerList(BaseModel):
    """Response for server list queries (Anthropic Registry API)."""

    servers: list[AnthropicServerResponse] = Field(..., description="List of servers")
    metadata: AnthropicPaginationMetadata | None = Field(None, description="Pagination info")


class AnthropicErrorResponse(BaseModel):
    """Standard error response (Anthropic Registry API)."""

    error: str = Field(..., description="Error message")


# Registry Card Models


class RegistryCapabilitiesResponse(BaseModel):
    """Registry capabilities response model."""

    servers: bool = Field(..., description="Supports MCP server registry")
    agents: bool = Field(..., description="Supports A2A agent registry")
    skills: bool = Field(..., description="Supports skill registry")
    prompts: bool = Field(False, description="Supports prompt registry")
    security_scans: bool = Field(True, description="Supports security scanning")
    incremental_sync: bool = Field(False, description="Supports incremental federation sync")
    webhooks: bool = Field(False, description="Supports webhook notifications")


class RegistryAuthConfigResponse(BaseModel):
    """Registry authentication configuration response model."""

    schemes: list[str] = Field(..., description="Supported auth schemes (bearer, oauth2, etc.)")
    oauth2_issuer: str | None = Field(None, description="OAuth2 issuer URL")
    oauth2_token_endpoint: str | None = Field(None, description="OAuth2 token endpoint URL")
    scopes_supported: list[str] = Field(default_factory=list, description="Supported OAuth2 scopes")


class RegistryContactResponse(BaseModel):
    """Registry contact information response model."""

    email: str | None = Field(None, description="Contact email address")
    url: str | None = Field(None, description="Contact URL")


class RegistryCardResponse(BaseModel):
    """Registry Card response model."""

    schema_version: str = Field(..., description="Registry card schema version")
    id: str = Field(..., description="Unique registry identifier (UUID)")
    name: str = Field(..., description="Registry name")
    description: str | None = Field(None, description="Registry description")
    registry_url: str | None = Field(None, description="Base URL of this registry")
    organization_name: str | None = Field(None, description="Organization operating this registry")
    federation_api_version: str = Field(..., description="Federation API version")
    federation_endpoint: str = Field(..., description="Federation endpoint URL")
    capabilities: RegistryCapabilitiesResponse = Field(..., description="Registry capabilities")
    authentication: RegistryAuthConfigResponse = Field(
        ..., description="Authentication configuration"
    )
    visibility_policy: str = Field(..., description="Default visibility policy")
    contact: RegistryContactResponse | None = Field(None, description="Contact information")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str | None = Field(None, description="Creation timestamp")
    updated_at: str | None = Field(None, description="Last update timestamp")


# Management API Models (IAM/User Management)


class M2MAccountRequest(BaseModel):
    """Request model for creating M2M service account."""

    name: str = Field(..., min_length=1, description="Service account name/client ID")
    groups: list[str] = Field(..., min_length=1, description="List of group names")
    description: str | None = Field(None, description="Account description")


class HumanUserRequest(BaseModel):
    """Request model for creating human user account."""

    username: str = Field(..., min_length=1, description="Username")
    email: str = Field(..., description="Email address")
    first_name: str = Field(..., min_length=1, description="First name")
    last_name: str = Field(..., min_length=1, description="Last name")
    groups: list[str] = Field(..., min_length=1, description="List of group names")
    password: str | None = Field(None, description="Initial password")


class UserSummary(BaseModel):
    """User summary model."""

    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str | None = Field(None, description="Email address")
    firstName: str | None = Field(None, description="First name")
    lastName: str | None = Field(None, description="Last name")
    enabled: bool = Field(True, description="Whether user is enabled")
    groups: list[str] = Field(default_factory=list, description="User groups")


class UserListResponse(BaseModel):
    """Response model for list users endpoint."""

    users: list[UserSummary] = Field(default_factory=list, description="List of users")
    total: int = Field(..., description="Total number of users")


class UserDeleteResponse(BaseModel):
    """Response model for delete user endpoint."""

    username: str = Field(..., description="Deleted username")
    deleted: bool = Field(True, description="Deletion status")


class M2MAccountResponse(BaseModel):
    """Response model for M2M account creation."""

    client_id: str = Field(..., description="Client ID (app ID in Entra)")
    client_secret: str = Field(..., description="Client secret")
    groups: list[str] = Field(default_factory=list, description="Assigned groups")
    client_uuid: str | None = Field(None, description="Client UUID (Entra app object ID)")
    service_principal_id: str | None = Field(None, description="Service principal ID (Entra)")


class GroupCreateRequest(BaseModel):
    """Request model for creating a Keycloak group."""

    name: str = Field(..., min_length=1, description="Group name")
    description: str | None = Field(None, description="Group description")


class GroupSummary(BaseModel):
    """Group summary model."""

    id: str = Field(..., description="Group ID")
    name: str = Field(..., description="Group name")
    path: str = Field(..., description="Group path")
    attributes: dict[str, Any] | None = Field(None, description="Group attributes")


class IdPM2MClient(BaseModel):
    """M2M client record as stored in idp_m2m_clients.

    Models the response shape from the /api/iam/m2m-clients direct
    registration API (issue #851).
    """

    client_id: str = Field(..., description="IdP application client ID")
    name: str = Field(..., description="Application name")
    description: str | None = Field(None, description="Application description")
    groups: list[str] = Field(default_factory=list, description="Groups this client belongs to")
    enabled: bool = Field(True, description="Whether the client is active")
    provider: str = Field(..., description="Identity provider (okta, keycloak, entra, manual)")
    idp_app_id: str | None = Field(None, description="IdP internal app ID")
    created_by: str | None = Field(
        None, description="Operator who registered this client (manual records)"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class M2MClientListResponse(BaseModel):
    """Paginated response for GET /api/iam/m2m-clients."""

    total: int = Field(..., description="Total number of matching records")
    limit: int = Field(..., description="Limit applied to this page")
    skip: int = Field(..., description="Offset applied to this page")
    items: list[IdPM2MClient] = Field(default_factory=list, description="Records on this page")


class GroupSyncStatusResponse(BaseModel):
    """Response model for list groups endpoint with sync status."""

    keycloak_groups: list[dict[str, Any]] = Field(
        default_factory=list, description="Groups from Keycloak"
    )
    scopes_groups: dict[str, Any] = Field(
        default_factory=dict, description="Groups from scopes storage"
    )
    synchronized: list[str] = Field(
        default_factory=list, description="Groups in both Keycloak and scopes"
    )
    keycloak_only: list[str] = Field(default_factory=list, description="Groups only in Keycloak")
    scopes_only: list[str] = Field(default_factory=list, description="Groups only in scopes")


class GroupDeleteResponse(BaseModel):
    """Response model for delete group endpoint."""

    name: str = Field(..., description="Deleted group name")
    deleted: bool = Field(True, description="Deletion status")


# ==========================================
# Agent Skills Models
# ==========================================


class SkillRegistrationRequest(BaseModel):
    """Request model for registering a skill."""

    name: str = Field(..., description="Skill name (lowercase alphanumeric with hyphens)")
    skill_md_url: str = Field(..., description="URL to SKILL.md file")
    description: str | None = Field(None, description="Skill description")
    repository_url: str | None = Field(None, description="Repository URL")
    version: str | None = Field(None, description="Skill version (e.g., 1.0.0)")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    target_agents: list[str] = Field(
        default_factory=list, description="Target coding assistants (e.g., claude-code, cursor)"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Custom metadata key-value pairs for search and organization"
    )
    visibility: str = Field(default="public", description="Visibility: public, private, group")
    allowed_groups: list[str] = Field(
        default_factory=list, description="Groups for group visibility"
    )


class SkillCard(BaseModel):
    """Response model for a skill."""

    id: UUID = Field(..., description="Unique identifier (UUID) for this skill")
    name: str = Field(..., description="Skill name")
    path: str = Field(..., description="Skill path (e.g., /skills/pdf-processing)")
    description: str | None = Field(None, description="Skill description")
    skill_md_url: str = Field(..., description="URL to SKILL.md file")
    skill_md_raw_url: str | None = Field(None, description="Raw content URL")
    version: str | None = Field(None, description="Skill version")
    author: str | None = Field(None, description="Skill author")
    visibility: str = Field(default="public", description="Visibility level")
    is_enabled: bool = Field(default=True, description="Whether skill is enabled")
    tags: list[str] = Field(default_factory=list, description="Tags")
    target_agents: list[str] = Field(default_factory=list, description="Target coding assistants")
    metadata: dict[str, Any] | None = Field(
        None, description="Skill metadata (author, version, extra)"
    )
    owner: str | None = Field(None, description="Skill owner")
    registry_name: str | None = Field(None, description="Source registry")
    num_stars: float = Field(default=0, description="Average rating")
    health_status: str = Field(default="unknown", description="Health status")
    status: str = Field(
        default="active",
        description="Lifecycle status (active, deprecated, draft, beta)",
    )
    created_at: str | None = Field(None, description="Creation timestamp")
    updated_at: str | None = Field(None, description="Last update timestamp")


class SkillListResponse(BaseModel):
    """Response model for listing skills."""

    skills: list[SkillCard] = Field(default_factory=list, description="List of skills")
    total_count: int = Field(0, description="Total number of skills")
    limit: int = Field(..., description="Page size applied")
    offset: int = Field(..., description="Offset applied")
    has_next: bool = Field(..., description="Whether more pages exist")


class SkillHealthResponse(BaseModel):
    """Response model for skill health check."""

    path: str = Field(..., description="Skill path")
    healthy: bool = Field(..., description="Whether SKILL.md is accessible")
    status_code: int | None = Field(None, description="HTTP status code")
    error: str | None = Field(None, description="Error message if unhealthy")
    response_time_ms: float | None = Field(None, description="Response time in ms")


class SkillContentResponse(BaseModel):
    """Response model for skill content."""

    content: str = Field(..., description="SKILL.md content")
    url: str = Field(..., description="URL content was fetched from")


class SkillSearchResponse(BaseModel):
    """Response model for skill search."""

    query: str = Field(..., description="Search query")
    skills: list[dict[str, Any]] = Field(default_factory=list, description="Matching skills")
    total_count: int = Field(0, description="Total matches")


class SkillToggleResponse(BaseModel):
    """Response model for skill toggle."""

    path: str = Field(..., description="Skill path")
    is_enabled: bool = Field(..., description="New enabled state")


class SkillRatingResponse(BaseModel):
    """Response model for skill rating."""

    num_stars: float = Field(..., description="Average rating")
    rating_details: list[dict[str, Any]] = Field(
        default_factory=list, description="Individual ratings"
    )


class AppLogEntry(BaseModel):
    """Single application log entry."""

    timestamp: str = Field(..., description="Log timestamp (ISO-8601)")
    hostname: str = Field(..., description="Pod/hostname that emitted the log")
    service: str = Field(..., description="Service name (registry, auth-server)")
    level: str = Field(..., description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    level_no: int = Field(..., description="Numeric log level")
    logger: str = Field(..., description="Python logger name")
    filename: str = Field(..., description="Source filename")
    lineno: int = Field(..., description="Source line number")
    process: int = Field(..., description="Process ID")
    message: str = Field(..., description="Log message")


class AppLogResponse(BaseModel):
    """Response model for application log query."""

    entries: list[AppLogEntry] = Field(default_factory=list, description="Log entries")
    total_count: int = Field(0, description="Total matching entries")
    limit: int = Field(100, description="Applied page size")
    offset: int = Field(0, description="Applied offset")
    has_next: bool = Field(False, description="Whether more entries exist")


class AppLogMetadataResponse(BaseModel):
    """Response model for application log metadata."""

    services: list[str] = Field(default_factory=list, description="Available service names")
    hostnames: list[str] = Field(default_factory=list, description="Available hostnames")
    levels: list[str] = Field(default_factory=list, description="Available log levels")


class RegistryClient:
    """
    MCP Gateway Registry API client.

    Provides methods for interacting with the Registry API endpoints including:
    - Server Management: registration, removal, toggling, health checks
    - Group Management: create, delete, list groups
    - Agent Management: register, update, delete, discover agents (A2A)
    - Management API: IAM/user management, M2M accounts, user CRUD operations

    Authentication is handled via JWT tokens passed to the constructor.
    """

    def __init__(self, registry_url: str, token: str):
        """
        Initialize the Registry Client.

        Args:
            registry_url: Base URL of the registry (e.g., https://registry.mycorp.click)
            token: JWT access token for authentication
        """
        self.registry_url = registry_url.rstrip("/")
        self._token = token

        # Redact token in logs - show only first 8 characters
        redacted_token = f"{token[:8]}..." if len(token) > 8 else "***"
        logger.info(f"Initialized RegistryClient for {self.registry_url} (token: {redacted_token})")

    def _get_headers(self) -> dict[str, str]:
        """
        Get request headers with JWT token.

        Returns:
            Dictionary of HTTP headers
        """
        return {"Authorization": f"Bearer {self._token}"}

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> requests.Response:
        """
        Make HTTP request to the Registry API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data (sent as form-encoded for POST)
            params: Query parameters

        Returns:
            Response object

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.registry_url}{endpoint}"
        headers = self._get_headers()

        logger.debug(f"{method} {url}")

        # Determine content type based on endpoint
        # Agent, Management, Search, Federation, Skills, Virtual Servers, Registry Card, version, and group import endpoints use JSON
        # Server registration uses form data
        if (
            endpoint.startswith("/api/agents")
            or endpoint.startswith("/api/management")
            or endpoint.startswith("/api/iam")
            or endpoint.startswith("/api/search")
            or endpoint.startswith("/api/federation")
            or endpoint.startswith("/api/peers")
            or endpoint.startswith("/api/skills")
            or endpoint.startswith("/api/virtual-servers")
            or endpoint.startswith("/api/v1/registry")
            or endpoint.startswith("/api/v1/health")
            or endpoint == "/api/servers/groups/import"
            or "/auth-credential" in endpoint
            or "/versions" in endpoint
        ):
            # Send as JSON for agent, management, search, federation, and import endpoints
            response = requests.request(
                method=method, url=url, headers=headers, json=data, params=params, timeout=120
            )
        else:
            # Send as form-encoded for server registration
            response = requests.request(
                method=method, url=url, headers=headers, data=data, params=params, timeout=120
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            # For 422 errors, try to extract validation details
            if response.status_code == 422:
                try:
                    error_detail = response.json()
                    logger.error(f"Validation error details: {json.dumps(error_detail, indent=2)}")
                except Exception as e:
                    logger.warning(f"Could not parse 422 error response as JSON: {e}")
            raise
        return response

    def register_service(self, registration: InternalServiceRegistration) -> ServiceResponse:
        """
        Register a new service in the registry.

        Args:
            registration: Service registration data

        Returns:
            Service response with registration details

        Raises:
            requests.HTTPError: If registration fails
        """
        logger.info(f"Registering service: {registration.service_path}")

        # Convert model to dict
        data = registration.model_dump(exclude_none=True, by_alias=True)

        # Convert tags list to comma-separated string for form encoding
        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = ",".join(data["tags"])

        # Convert external_tags list to comma-separated string for form encoding
        if "external_tags" in data and isinstance(data["external_tags"], list):
            data["external_tags"] = ",".join(data["external_tags"])

        # Convert metadata dict to JSON string for form encoding
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = json.dumps(data["metadata"])

        response = self._make_request(method="POST", endpoint="/api/servers/register", data=data)

        logger.info(f"Service registered successfully: {registration.service_path}")
        return ServiceResponse(**response.json())

    def remove_service(self, service_path: str) -> dict[str, Any]:
        """
        Remove a service from the registry.

        Args:
            service_path: Path of service to remove

        Returns:
            Response data

        Raises:
            requests.HTTPError: If removal fails
        """
        logger.info(f"Removing service: {service_path}")

        response = self._make_request(
            method="POST", endpoint="/api/servers/remove", data={"path": service_path}
        )

        logger.info(f"Service removed successfully: {service_path}")
        return response.json()

    def toggle_service(self, service_path: str) -> ToggleResponse:
        """
        Toggle service enabled/disabled status.

        Args:
            service_path: Path of service to toggle

        Returns:
            Toggle response with current status

        Raises:
            requests.HTTPError: If toggle fails
        """
        logger.info(f"Toggling service: {service_path}")

        response = self._make_request(
            method="POST", endpoint="/api/servers/toggle", data={"service_path": service_path}
        )

        result = ToggleResponse(**response.json())
        logger.info(f"Service toggled: {service_path} -> enabled={result.is_enabled}")
        return result

    def update_server_credential(
        self,
        service_path: str,
        auth_scheme: str,
        auth_credential: str = None,
        auth_header_name: str = None,
    ) -> dict[str, Any]:
        """
        Update authentication credentials for a server.

        Args:
            service_path: Path of server to update (e.g., /my-server)
            auth_scheme: Authentication scheme (none, bearer, api_key)
            auth_credential: New credential (required if auth_scheme is not 'none')
            auth_header_name: Custom header name (optional, for api_key)

        Returns:
            Response dict with message and updated auth details

        Raises:
            requests.HTTPError: If update fails
        """
        logger.info(f"Updating auth credential for: {service_path}")

        # Build payload
        payload = {"auth_scheme": auth_scheme}
        if auth_credential:
            payload["auth_credential"] = auth_credential
        if auth_header_name:
            payload["auth_header_name"] = auth_header_name

        response = self._make_request(
            method="PATCH", endpoint=f"/api/servers{service_path}/auth-credential", data=payload
        )

        result = response.json()
        logger.info(f"Credential updated for {service_path}: scheme={result.get('auth_scheme')}")
        return result

    def list_services(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> ServerListResponse:
        """
        List all services in the registry.

        Args:
            limit: Maximum number of services to return per page
            offset: Number of services to skip for pagination

        Returns:
            Server list response

        Raises:
            requests.HTTPError: If list operation fails
        """
        logger.info("Listing all services")

        params = {
            "limit": limit,
            "offset": offset,
        }

        response = self._make_request(method="GET", endpoint="/api/servers", params=params)

        response_data = response.json()
        logger.debug(f"Raw API response: {json.dumps(response_data, indent=2, default=str)}")

        try:
            result = ServerListResponse(**response_data)
            logger.info(
                f"Retrieved {len(result.servers)} services"
                f" (total={result.total_count}, offset={result.offset},"
                f" limit={result.limit}, has_next={result.has_next})"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to parse server list response: {e}")
            logger.error(f"Raw response data: {json.dumps(response_data, indent=2, default=str)}")
            raise

    def healthcheck(self) -> dict[str, Any]:
        """
        Perform health check on all services.

        Returns:
            Health check response with service statuses

        Raises:
            requests.HTTPError: If health check fails
        """
        logger.info("Performing health check on all services")

        response = self._make_request(method="GET", endpoint="/api/servers/health")

        result = response.json()
        logger.info(f"Health check completed: {result.get('status', 'unknown')}")
        return result

    def get_config(self) -> dict[str, Any]:
        """
        Get registry configuration including deployment mode and features.

        Returns:
            Configuration response with deployment_mode, registry_mode,
            nginx_updates_enabled, and features dict

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info("Fetching registry configuration")

        response = self._make_request(method="GET", endpoint="/api/config")

        result = response.json()
        logger.info(
            f"Registry config: deployment_mode={result.get('deployment_mode')}, "
            f"registry_mode={result.get('registry_mode')}"
        )
        return result

    def get_well_known_registry_card(self) -> RegistryCardResponse:
        """
        Get the Registry Card via .well-known discovery endpoint.

        This is the standard discovery endpoint for registry federation, following
        the .well-known convention used for service discovery (similar to
        .well-known/openid-configuration).

        Returns:
            Registry Card response with registry metadata

        Raises:
            requests.HTTPError: If request fails or card not initialized
        """
        logger.info("Fetching registry card via .well-known endpoint")

        response = self._make_request(
            method="GET", endpoint="/api/v1/registry/.well-known/registry-card"
        )

        result = RegistryCardResponse(**response.json())
        logger.info(f"Retrieved registry card: {result.id} (name: {result.name})")
        return result

    def get_registry_card(self) -> RegistryCardResponse:
        """
        Get the Registry Card for this registry instance.

        The Registry Card provides metadata about the registry including:
        - Capabilities (servers, agents, skills, security scans, etc.)
        - Authentication configuration
        - Federation API version and endpoint
        - Contact information

        Returns:
            Registry Card response with registry metadata

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info("Fetching registry card")

        response = self._make_request(method="GET", endpoint="/api/v1/registry/card")

        result = RegistryCardResponse(**response.json())
        logger.info(f"Retrieved registry card: {result.id} (name: {result.name})")
        return result

    def update_registry_card(self, card_data: dict[str, Any]) -> dict[str, Any]:
        """
        Update the Registry Card (admin only).

        This replaces the entire registry card with the provided data.
        For partial updates, use patch_registry_card() instead.

        Args:
            card_data: Complete registry card data

        Returns:
            Response with update confirmation

        Raises:
            requests.HTTPError: If update fails (e.g., insufficient permissions)
        """
        logger.info("Updating registry card")

        response = self._make_request(
            method="POST", endpoint="/api/v1/registry/card", data=card_data
        )

        result = response.json()
        logger.info("Registry card updated successfully")
        return result

    def patch_registry_card(self, updates: dict[str, Any]) -> dict[str, Any]:
        """
        Partially update the Registry Card (admin only).

        Only the fields provided in updates will be modified.
        Other fields will remain unchanged.

        Args:
            updates: Partial registry card updates

        Returns:
            Response with update confirmation

        Raises:
            requests.HTTPError: If update fails (e.g., insufficient permissions)
        """
        logger.info(f"Patching registry card with updates: {list(updates.keys())}")

        response = self._make_request(
            method="PATCH", endpoint="/api/v1/registry/card", data=updates
        )

        result = response.json()
        logger.info("Registry card patched successfully")
        return result

    def add_server_to_groups(self, server_name: str, group_names: list[str]) -> dict[str, Any]:
        """
        Add a server to user groups.

        Args:
            server_name: Name of server
            group_names: List of group names

        Returns:
            Response data

        Raises:
            requests.HTTPError: If operation fails
        """
        logger.info(f"Adding server {server_name} to groups: {group_names}")

        response = self._make_request(
            method="POST",
            endpoint="/api/servers/groups/add",
            data={"server_name": server_name, "group_names": ",".join(group_names)},
        )

        logger.info("Server added to groups successfully")
        return response.json()

    def remove_server_from_groups(self, server_name: str, group_names: list[str]) -> dict[str, Any]:
        """
        Remove a server from user groups.

        Args:
            server_name: Name of server
            group_names: List of group names

        Returns:
            Response data

        Raises:
            requests.HTTPError: If operation fails
        """
        logger.info(f"Removing server {server_name} from groups: {group_names}")

        response = self._make_request(
            method="POST",
            endpoint="/api/servers/groups/remove",
            data={"server_name": server_name, "group_names": ",".join(group_names)},
        )

        logger.info("Server removed from groups successfully")
        return response.json()

    def create_group(
        self, group_name: str, description: str | None = None, create_in_idp: bool = False
    ) -> dict[str, Any]:
        """
        Create a new user group.

        Args:
            group_name: Name of group
            description: Group description
            create_in_idp: Whether to create in IdP (Keycloak/Entra)

        Returns:
            Response data

        Raises:
            requests.HTTPError: If creation fails
        """
        logger.info(f"Creating group: {group_name}")

        data = {"group_name": group_name}
        if description:
            data["description"] = description
        data["create_in_idp"] = str(create_in_idp).lower()

        response = self._make_request(
            method="POST", endpoint="/api/servers/groups/create", data=data
        )

        logger.info(f"Group created successfully: {group_name}")
        return response.json()

    def delete_group(
        self, group_name: str, delete_from_idp: bool = False, force: bool = False
    ) -> dict[str, Any]:
        """
        Delete a user group.

        Args:
            group_name: Name of group
            delete_from_idp: Whether to delete from IdP (Keycloak/Entra)
            force: Force deletion of system groups

        Returns:
            Response data

        Raises:
            requests.HTTPError: If deletion fails
        """
        logger.info(f"Deleting group: {group_name}")

        data = {"group_name": group_name}
        if delete_from_idp:
            data["delete_from_idp"] = True
        if force:
            data["force"] = True

        response = self._make_request(
            method="POST", endpoint="/api/servers/groups/delete", data=data
        )

        logger.info(f"Group deleted successfully: {group_name}")
        return response.json()

    def import_group(self, group_definition: dict[str, Any]) -> dict[str, Any]:
        """
        Import a complete group definition.

        Args:
            group_definition: Complete group definition including:
                - scope_name (required): Name of the scope/group
                - scope_type (optional): Type of scope (default: "server_scope")
                - description (optional): Description of the group
                - server_access (optional): List of server access definitions
                - group_mappings (optional): List of group mappings
                - ui_permissions (optional): Dictionary of UI permissions
                - create_in_idp (optional): Whether to create in IdP (default: false)

        Returns:
            Response data

        Raises:
            requests.HTTPError: If import fails
        """
        scope_name = group_definition.get("scope_name")
        if not scope_name:
            raise ValueError("scope_name is required in group_definition")

        logger.info(f"Importing group definition: {scope_name}")

        response = self._make_request(
            method="POST", endpoint="/api/servers/groups/import", data=group_definition
        )

        logger.info(f"Group imported successfully: {scope_name}")
        return response.json()

    def list_groups(
        self, include_keycloak: bool = True, include_scopes: bool = True
    ) -> GroupSyncStatusResponse:
        """
        List all user groups.

        Args:
            include_keycloak: Include Keycloak information
            include_scopes: Include scope information

        Returns:
            Group list response with sync status

        Raises:
            requests.HTTPError: If list operation fails
        """
        logger.info("Listing all groups")

        params = {
            "include_keycloak": str(include_keycloak).lower(),
            "include_scopes": str(include_scopes).lower(),
        }

        response = self._make_request(method="GET", endpoint="/api/servers/groups", params=params)

        result = GroupSyncStatusResponse(**response.json())
        total_groups = len(result.scopes_groups) + len(result.keycloak_groups)
        logger.info(
            f"Retrieved {total_groups} groups ({len(result.keycloak_groups)} from Keycloak, {len(result.scopes_groups)} from scopes)"
        )
        return result

    def get_group(self, group_name: str) -> dict[str, Any]:
        """
        Get full details of a specific group.

        Args:
            group_name: Name of the group

        Returns:
            Complete group definition with server_access, group_mappings, and ui_permissions

        Raises:
            requests.HTTPError: If get operation fails (404 if group not found)
        """
        logger.info(f"Getting group details: {group_name}")

        response = self._make_request(method="GET", endpoint=f"/api/servers/groups/{group_name}")

        logger.info(f"Retrieved group details for {group_name}")
        return response.json()

    # Agent Management Methods

    def register_agent(self, agent: AgentRegistration) -> AgentRegistrationResponse:
        """
        Register a new A2A agent.

        Args:
            agent: Agent registration data

        Returns:
            Agent registration response

        Raises:
            requests.HTTPError: If registration fails (409 for conflict, 422 for validation error, 403 for permission denied)
        """
        logger.info(f"Registering agent: {agent.path}")

        agent_data = agent.model_dump(exclude_none=True, by_alias=True)
        logger.debug(f"Agent data being sent: {json.dumps(agent_data, indent=2, default=str)}")

        response = self._make_request(
            method="POST", endpoint="/api/agents/register", data=agent_data
        )

        result = AgentRegistrationResponse(**response.json())
        logger.info(f"Agent registered successfully: {agent.path}")
        return result

    def list_agents(
        self,
        query: str | None = None,
        enabled_only: bool = False,
        visibility: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> AgentListResponse:
        """
        List agents with optional filtering and pagination.

        Args:
            query: Search query string
            enabled_only: Show only enabled agents
            visibility: Filter by visibility level (public, private, internal)
            limit: Number of agents to return (1-100, default 20)
            offset: Number of agents to skip (default 0)

        Returns:
            Agent list response with pagination metadata

        Raises:
            requests.HTTPError: If list operation fails
        """
        logger.info(f"Listing agents (limit={limit}, offset={offset})")

        params: dict[str, str | int | bool] = {
            "limit": limit,
            "offset": offset,
        }
        if query:
            params["query"] = query
        if enabled_only:
            params["enabled_only"] = "true"
        if visibility:
            params["visibility"] = visibility

        response = self._make_request(method="GET", endpoint="/api/agents", params=params)

        result = AgentListResponse(**response.json())
        logger.info(
            f"Retrieved {len(result.agents)} agents "
            f"(total: {result.total_count}, offset: {result.offset}, limit: {result.limit})"
        )
        return result

    def get_agent(self, path: str) -> AgentDetail:
        """
        Get detailed information about a specific agent.

        Args:
            path: Agent path (e.g., /code-reviewer)

        Returns:
            Agent detail

        Raises:
            requests.HTTPError: If agent not found (404) or unauthorized (403)
        """
        logger.info(f"Getting agent details: {path}")

        response = self._make_request(method="GET", endpoint=f"/api/agents{path}")

        result = AgentDetail(**response.json())
        logger.info(f"Retrieved agent details: {path}")
        return result

    def update_agent(self, path: str, agent: AgentRegistration) -> AgentDetail:
        """
        Update an existing agent.

        Args:
            path: Agent path
            agent: Updated agent data

        Returns:
            Updated agent detail

        Raises:
            requests.HTTPError: If update fails (404 for not found, 403 for permission denied, 422 for validation error)
        """
        logger.info(f"Updating agent: {path}")

        response = self._make_request(
            method="PUT",
            endpoint=f"/api/agents{path}",
            data=agent.model_dump(exclude_none=True, by_alias=True),
        )

        result = AgentDetail(**response.json())
        logger.info(f"Agent updated successfully: {path}")
        return result

    def delete_agent(self, path: str) -> None:
        """
        Delete an agent from the registry.

        Args:
            path: Agent path

        Raises:
            requests.HTTPError: If deletion fails (404 for not found, 403 for permission denied)
        """
        logger.info(f"Deleting agent: {path}")

        self._make_request(method="DELETE", endpoint=f"/api/agents{path}")

        logger.info(f"Agent deleted successfully: {path}")

    def toggle_agent(self, path: str, enabled: bool) -> AgentToggleResponse:
        """
        Toggle agent enabled/disabled status.

        Args:
            path: Agent path
            enabled: True to enable, False to disable

        Returns:
            Agent toggle response

        Raises:
            requests.HTTPError: If toggle fails (404 for not found, 403 for permission denied)
        """
        logger.info(f"Toggling agent {path} to {'enabled' if enabled else 'disabled'}")

        params = {"enabled": str(enabled).lower()}

        response = self._make_request(
            method="POST", endpoint=f"/api/agents{path}/toggle", params=params
        )

        result = AgentToggleResponse(**response.json())
        logger.info(
            f"Agent toggled: {path} is now {'enabled' if result.is_enabled else 'disabled'}"
        )
        return result

    def discover_agents_by_skills(
        self, skills: list[str], tags: list[str] | None = None, max_results: int = 10
    ) -> AgentDiscoveryResponse:
        """
        Discover agents by required skills.

        Args:
            skills: List of required skills
            tags: Optional tag filters
            max_results: Maximum number of results (default: 10, max: 100)

        Returns:
            Agent discovery response

        Raises:
            requests.HTTPError: If discovery fails (400 for bad request)
        """
        logger.info(f"Discovering agents by skills: {skills}")

        request_data = SkillDiscoveryRequest(skills=skills, tags=tags)
        params = {"max_results": max_results}

        response = self._make_request(
            method="POST",
            endpoint="/api/agents/discover",
            data=request_data.model_dump(exclude_none=True),
            params=params,
        )

        result = AgentDiscoveryResponse(**response.json())
        logger.info(f"Discovered {len(result.agents)} agents matching skills")
        return result

    def discover_agents_semantic(
        self, query: str, max_results: int = 10
    ) -> AgentSemanticDiscoveryResponse:
        """
        Discover agents using semantic search (FAISS vector search).

        Args:
            query: Natural language query (e.g., "Find agents that can analyze code")
            max_results: Maximum number of results (default: 10, max: 100)

        Returns:
            Agent semantic discovery response

        Raises:
            requests.HTTPError: If discovery fails (400 for bad request, 500 for search error)
        """
        logger.info(f"Discovering agents semantically: {query}")

        params = {"query": query, "max_results": max_results}

        response = self._make_request(
            method="POST", endpoint="/api/agents/discover/semantic", params=params
        )

        result = AgentSemanticDiscoveryResponse(**response.json())
        logger.info(f"Discovered {len(result.agents)} agents via semantic search")
        return result

    def semantic_search_servers(
        self,
        query: str,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> ServerSemanticSearchResponse:
        """
        Search for servers using semantic search (vector search).

        Args:
            query: Natural language query (e.g., "time and date services")
            max_results: Maximum number of results (default: 10, max: 100)
            include_draft: Include draft assets in results (default: False)
            include_deprecated: Include deprecated assets in results (default: False)
            include_disabled: Include disabled assets in results (default: False)

        Returns:
            Server semantic search response

        Raises:
            requests.HTTPError: If search fails (400 for bad request, 500 for search error)
        """
        logger.info(f"Searching servers semantically: {query}")

        request_data: dict[str, Any] = {
            "query": query,
            "entity_types": ["mcp_server"],
            "max_results": max_results,
            "include_draft": include_draft,
            "include_deprecated": include_deprecated,
            "include_disabled": include_disabled,
        }

        response = self._make_request(
            method="POST", endpoint="/api/search/semantic", data=request_data
        )

        result = ServerSemanticSearchResponse(**response.json())
        logger.info(f"Found {len(result.servers)} servers via semantic search")
        return result

    def semantic_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        max_results: int = 10,
        include_draft: bool = False,
        include_deprecated: bool = False,
        include_disabled: bool = False,
    ) -> SemanticSearchResponse:
        """
        Comprehensive semantic search across all entity types.

        Args:
            query: Natural language query (e.g., "coding assistants")
            entity_types: Optional list of entity types to search.
                         Valid values: "mcp_server", "tool", "a2a_agent", "skill", "virtual_server"
                         If None, searches all entity types.
            max_results: Maximum number of results per entity type (default: 10, max: 50)
            include_draft: Include draft assets in results (default: False)
            include_deprecated: Include deprecated assets in results (default: False)
            include_disabled: Include disabled assets in results (default: False)

        Returns:
            SemanticSearchResponse with servers, tools, agents, skills, and virtual_servers

        Raises:
            requests.HTTPError: If search fails (400 for bad request, 500 for search error)
        """
        logger.info(f"Semantic search: {query} (entity_types={entity_types})")

        request_data: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "include_draft": include_draft,
            "include_deprecated": include_deprecated,
            "include_disabled": include_disabled,
        }
        if entity_types:
            request_data["entity_types"] = entity_types

        response = self._make_request(
            method="POST", endpoint="/api/search/semantic", data=request_data
        )

        result = SemanticSearchResponse(**response.json())
        logger.info(
            f"Found: {len(result.servers)} servers, {len(result.tools)} tools, "
            f"{len(result.agents)} agents, {len(result.skills)} skills, "
            f"{len(result.virtual_servers)} virtual servers"
        )
        return result

    def rate_agent(self, path: str, rating: int) -> RatingResponse:
        """
        Submit a rating for an agent (1-5 stars).

        Each user can only have one active rating. If user has already rated,
        this updates their existing rating. System maintains a rotating buffer
        of the last 100 ratings.

        Args:
            path: Agent path (e.g., /code-reviewer)
            rating: Rating value (1-5 stars)

        Returns:
            Rating response with success message and updated average rating

        Raises:
            requests.HTTPError: If rating fails (400 for invalid rating, 403 for unauthorized, 404 for not found)
        """
        logger.info(f"Rating agent '{path}' with {rating} stars")

        request_data = RatingRequest(rating=rating)

        response = self._make_request(
            method="POST", endpoint=f"/api/agents{path}/rate", data=request_data.model_dump()
        )

        result = RatingResponse(**response.json())
        logger.info(f"Agent '{path}' rated successfully. New average: {result.average_rating:.2f}")
        return result

    def get_agent_rating(self, path: str) -> RatingInfoResponse:
        """
        Get rating information for an agent.

        Returns average rating and up to 100 most recent individual ratings
        (maintained as rotating buffer).

        Args:
            path: Agent path (e.g., /code-reviewer)

        Returns:
            Rating information with average and individual ratings

        Raises:
            requests.HTTPError: If retrieval fails (403 for unauthorized, 404 for not found)
        """
        logger.info(f"Getting ratings for agent: {path}")

        response = self._make_request(method="GET", endpoint=f"/api/agents{path}/rating")

        result = RatingInfoResponse(**response.json())
        logger.info(
            f"Retrieved ratings for '{path}': {result.num_stars:.2f} stars ({len(result.rating_details)} ratings)"
        )
        return result

    def rescan_agent(self, path: str) -> AgentRescanResponse:
        """
        Trigger a manual security scan for an agent.

        Initiates a new security scan for the specified agent and returns
        the results. This endpoint is useful for re-scanning agents after
        updates or for on-demand security assessments.

        Args:
            path: Agent path (e.g., /code-reviewer)

        Returns:
            Newly generated security scan results

        Raises:
            requests.HTTPError: If scan fails (403 for unauthorized, 404 for not found)
        """
        logger.info(f"Triggering security scan for agent: {path}")

        response = self._make_request(method="POST", endpoint=f"/api/agents{path}/rescan")

        result = AgentRescanResponse(**response.json())
        logger.info(
            f"Security scan completed for '{path}': "
            f"Safe={result.is_safe}, Critical={result.critical_issues}, "
            f"High={result.high_severity}, Medium={result.medium_severity}, "
            f"Low={result.low_severity}"
        )
        return result

    def get_agent_security_scan(self, path: str) -> AgentSecurityScanResponse:
        """
        Get security scan results for an agent.

        Returns the latest security scan results including threat analysis,
        severity levels, and detailed findings from YARA, specification
        validation, and heuristic analyzers.

        Args:
            path: Agent path (e.g., /code-reviewer)

        Returns:
            Security scan results with analysis_results and scan_results

        Raises:
            requests.HTTPError: If retrieval fails (403 for unauthorized, 404 for not found)
        """
        logger.info(f"Getting security scan results for agent: {path}")

        response = self._make_request(method="GET", endpoint=f"/api/agents{path}/security-scan")

        result = AgentSecurityScanResponse(**response.json())
        logger.info(f"Retrieved security scan results for '{path}'")
        return result

    def agent_ans_link(
        self,
        path: str,
        ans_agent_id: str,
    ) -> dict[str, Any]:
        """
        Link an ANS Agent ID to an agent.

        Args:
            path: Agent path (e.g., /code-reviewer)
            ans_agent_id: ANS Agent ID (e.g., ans://v1.example.com)

        Returns:
            Link result with success status, message, and ans_metadata

        Raises:
            requests.HTTPError: If linking fails
        """
        logger.info(f"Linking ANS ID '{ans_agent_id}' to agent: {path}")

        response = self._make_request(
            method="POST",
            endpoint=f"/api/agents{path}/ans/link",
            data={"ans_agent_id": ans_agent_id},
        )

        result = response.json()
        logger.info(f"ANS link result for '{path}': {result.get('message', '')}")
        return result

    def agent_ans_status(
        self,
        path: str,
    ) -> dict[str, Any]:
        """
        Get ANS verification status for an agent.

        Args:
            path: Agent path (e.g., /code-reviewer)

        Returns:
            ANS metadata dict with status, domain, ans_agent_id, etc.

        Raises:
            requests.HTTPError: If retrieval fails (404 if no ANS link)
        """
        logger.info(f"Getting ANS status for agent: {path}")

        response = self._make_request(
            method="GET",
            endpoint=f"/api/agents{path}/ans/status",
        )

        result = response.json()
        logger.info(f"ANS status for '{path}': {result.get('status', 'unknown')}")
        return result

    def agent_ans_unlink(
        self,
        path: str,
    ) -> dict[str, Any]:
        """
        Remove ANS link from an agent.

        Args:
            path: Agent path (e.g., /code-reviewer)

        Returns:
            Unlink result with success status and message

        Raises:
            requests.HTTPError: If unlinking fails
        """
        logger.info(f"Unlinking ANS from agent: {path}")

        response = self._make_request(
            method="DELETE",
            endpoint=f"/api/agents{path}/ans/link",
        )

        result = response.json()
        logger.info(f"ANS unlink result for '{path}': {result.get('message', '')}")
        return result

    def rate_server(self, path: str, rating: int) -> RatingResponse:
        """
        Submit a rating for a server (1-5 stars).

        Each user can only have one active rating. If user has already rated,
        this updates their existing rating. System maintains a rotating buffer
        of the last 100 ratings.

        Args:
            path: Server path (e.g., /cloudflare-docs)
            rating: Rating value (1-5 stars)

        Returns:
            Rating response with success message and updated average rating

        Raises:
            requests.HTTPError: If rating fails (400 for invalid rating, 403 for unauthorized, 404 for not found)
        """
        logger.info(f"Rating server '{path}' with {rating} stars")

        request_data = RatingRequest(rating=rating)

        response = self._make_request(
            method="POST", endpoint=f"/api/servers{path}/rate", data=request_data.model_dump()
        )

        result = RatingResponse(**response.json())
        logger.info(f"Server '{path}' rated successfully. New average: {result.average_rating:.2f}")
        return result

    def get_server(
        self,
        path: str,
    ) -> ServerDetailResponse:
        """
        Get detailed information about a specific server.

        Args:
            path: Server path (e.g., /my-server)

        Returns:
            Server detail response

        Raises:
            requests.HTTPError: If server not found (404) or unauthorized (403)
        """
        logger.info(f"Getting server details: {path}")

        response = self._make_request(method="GET", endpoint=f"/api/servers{path}")

        result = ServerDetailResponse(**response.json())
        logger.info(f"Retrieved server details: {path}")
        return result

    def get_server_rating(self, path: str) -> RatingInfoResponse:
        """
        Get rating information for a server.

        Returns average rating and up to 100 most recent individual ratings
        (maintained as rotating buffer).

        Args:
            path: Server path (e.g., /cloudflare-docs)

        Returns:
            Rating information with average and individual ratings

        Raises:
            requests.HTTPError: If retrieval fails (403 for unauthorized, 404 for not found)
        """
        logger.info(f"Getting ratings for server: {path}")

        response = self._make_request(method="GET", endpoint=f"/api/servers{path}/rating")

        result = RatingInfoResponse(**response.json())
        logger.info(
            f"Retrieved ratings for '{path}': {result.num_stars:.2f} stars ({len(result.rating_details)} ratings)"
        )
        return result

    def get_security_scan(self, path: str) -> SecurityScanResult:
        """
        Get security scan results for a server.

        Returns the latest security scan results including threat analysis,
        severity levels, and detailed findings for each tool.

        Args:
            path: Server path (e.g., /cloudflare-docs)

        Returns:
            Security scan results with analysis_results and tool_results

        Raises:
            requests.HTTPError: If retrieval fails (403 for unauthorized, 404 for not found)
        """
        logger.info(f"Getting security scan results for server: {path}")

        response = self._make_request(method="GET", endpoint=f"/api/servers{path}/security-scan")

        result = SecurityScanResult(**response.json())
        logger.info(f"Retrieved security scan results for '{path}'")
        return result

    def rescan_server(self, path: str) -> RescanResponse:
        """
        Trigger a manual security scan for a server.

        Initiates a new security scan for the specified server and returns
        the results. This operation is admin-only.

        Args:
            path: Server path (e.g., /cloudflare-docs)

        Returns:
            Newly generated security scan results

        Raises:
            requests.HTTPError: If scan fails (403 for non-admin, 404 for not found, 500 for scan error)
        """
        logger.info(f"Triggering security scan for server: {path}")

        response = self._make_request(method="POST", endpoint=f"/api/servers{path}/rescan")

        result = RescanResponse(**response.json())
        safety_status = "SAFE" if result.is_safe else "UNSAFE"
        logger.info(
            f"Security scan completed for '{path}': {safety_status} "
            f"(Critical: {result.critical_issues}, High: {result.high_severity}, "
            f"Medium: {result.medium_severity}, Low: {result.low_severity})"
        )
        return result

    # Anthropic Registry API Methods (v0.1)

    def anthropic_list_servers(
        self, cursor: str | None = None, limit: int | None = None
    ) -> AnthropicServerList:
        """
        List all MCP servers using the Anthropic Registry API format (v0.1).

        This endpoint provides pagination support and returns servers in the
        Anthropic Registry API standard format with reverse-DNS naming.

        Args:
            cursor: Pagination cursor (opaque string from previous response)
            limit: Maximum number of results per page (default: 100, max: 1000)

        Returns:
            Anthropic ServerList with servers and pagination metadata

        Raises:
            requests.HTTPError: If list operation fails
        """
        logger.info("Listing servers via Anthropic Registry API (v0.1)")

        params = {}
        if cursor:
            params["cursor"] = cursor
        if limit:
            params["limit"] = limit

        response = self._make_request(method="GET", endpoint="/v0.1/servers", params=params)

        result = AnthropicServerList(**response.json())
        logger.info(f"Retrieved {len(result.servers)} servers via Anthropic API")
        return result

    def anthropic_list_server_versions(self, server_name: str) -> AnthropicServerList:
        """
        List all versions of a specific server using Anthropic Registry API (v0.1).

        Currently, the registry maintains only one version per server, so this
        returns a single-item list.

        Args:
            server_name: Server name in reverse-DNS format (e.g., "io.mcpgateway/example-server")
                        Will be URL-encoded automatically.

        Returns:
            Anthropic ServerList with single server version

        Raises:
            requests.HTTPError: If server not found (404) or user lacks access (403/404)
        """
        logger.info(f"Listing versions for server: {server_name}")

        # URL-encode the server name
        encoded_name = quote(server_name, safe="")

        response = self._make_request(
            method="GET", endpoint=f"/v0.1/servers/{encoded_name}/versions"
        )

        result = AnthropicServerList(**response.json())
        logger.info(f"Retrieved {len(result.servers)} version(s) for {server_name}")
        return result

    def anthropic_get_server_version(
        self, server_name: str, version: str = "latest"
    ) -> AnthropicServerResponse:
        """
        Get detailed information about a specific server version using Anthropic Registry API (v0.1).

        Args:
            server_name: Server name in reverse-DNS format (e.g., "io.mcpgateway/example-server")
                        Will be URL-encoded automatically.
            version: Version string (e.g., "1.0.0" or "latest"). Default: "latest"
                    Currently only "latest" and "1.0.0" are supported.

        Returns:
            Anthropic ServerResponse with full server details

        Raises:
            requests.HTTPError: If server not found (404), version not found (404),
                              or user lacks access (403/404)
        """
        logger.info(f"Getting server {server_name} version {version}")

        # URL-encode both server name and version
        encoded_name = quote(server_name, safe="")
        encoded_version = quote(version, safe="")

        response = self._make_request(
            method="GET", endpoint=f"/v0.1/servers/{encoded_name}/versions/{encoded_version}"
        )

        result = AnthropicServerResponse(**response.json())
        logger.info(f"Retrieved server details for {server_name} v{version}")
        return result

    # Local Server Version Management Methods

    def remove_server_version(self, path: str, version: str) -> dict:
        """
        Remove a version from a server.

        Args:
            path: Server path (e.g., "/context7")
            version: Version to remove

        Returns:
            Response dict with status and message

        Raises:
            requests.HTTPError: If server not found or cannot remove default
        """
        logger.info(f"Removing version {version} from server {path}")

        encoded_path = quote(path.lstrip("/"), safe="")
        encoded_version = quote(version, safe="")

        response = self._make_request(
            method="DELETE", endpoint=f"/api/servers/{encoded_path}/versions/{encoded_version}"
        )

        return response.json()

    def set_default_version(self, path: str, version: str) -> dict:
        """
        Set the default (latest) version for a server.

        Args:
            path: Server path (e.g., "/context7")
            version: Version to set as default

        Returns:
            Response dict with status and message

        Raises:
            requests.HTTPError: If server or version not found
        """
        logger.info(f"Setting default version to {version} for server {path}")

        encoded_path = quote(path.lstrip("/"), safe="")

        response = self._make_request(
            method="PUT",
            endpoint=f"/api/servers/{encoded_path}/versions/default",
            data={"version": version},
        )

        return response.json()

    def get_server_versions(self, path: str) -> dict:
        """
        Get all versions for a server.

        Args:
            path: Server path (e.g., "/context7")

        Returns:
            Dict with path, default_version, and versions list

        Raises:
            requests.HTTPError: If server not found
        """
        logger.info(f"Getting versions for server {path}")

        encoded_path = quote(path.lstrip("/"), safe="")

        response = self._make_request(
            method="GET", endpoint=f"/api/servers/{encoded_path}/versions"
        )

        return response.json()

    # Management API Methods (IAM/User Management)

    def list_users(self, search: str | None = None, limit: int = 500) -> UserListResponse:
        """
        List Keycloak users (admin only).

        Args:
            search: Optional search string to filter users
            limit: Maximum number of results (default: 500)

        Returns:
            UserListResponse with list of users

        Raises:
            requests.HTTPError: If not authorized (403) or request fails
        """
        logger.info("Listing Keycloak users")

        params = {}
        if search:
            params["search"] = search
        if limit != 500:
            params["limit"] = limit

        response = self._make_request(
            method="GET", endpoint="/api/management/iam/users", params=params
        )

        try:
            response_data = response.json()
            logger.debug(f"Raw API response: {json.dumps(response_data, indent=2, default=str)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            logger.error(f"Raw response text: {response.text}")
            logger.error(f"Response status code: {response.status_code}")
            logger.error(f"Response headers: {dict(response.headers)}")
            raise

        try:
            result = UserListResponse(**response_data)
            logger.info(f"Retrieved {result.total} users")
            return result
        except Exception as e:
            logger.error(f"Failed to parse user list response: {e}")
            logger.error(f"Raw response data: {json.dumps(response_data, indent=2, default=str)}")
            raise

    def create_m2m_account(
        self, name: str, groups: list[str], description: str | None = None
    ) -> M2MAccountResponse:
        """
        Create a machine-to-machine service account.

        Args:
            name: Service account name/client ID
            groups: List of group names for access control
            description: Optional account description

        Returns:
            M2MAccountResponse with client credentials

        Raises:
            requests.HTTPError: If not authorized (403), already exists (400), or request fails
        """
        logger.info(f"Creating M2M service account: {name}")

        data = {"name": name, "groups": groups}
        if description:
            data["description"] = description

        response = self._make_request(
            method="POST", endpoint="/api/management/iam/users/m2m", data=data
        )

        result = M2MAccountResponse(**response.json())
        logger.info(f"M2M account created successfully: {name}")
        return result

    def create_human_user(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        groups: list[str],
        password: str | None = None,
    ) -> UserSummary:
        """
        Create a human user account in Keycloak.

        Args:
            username: Username
            email: Email address
            first_name: First name
            last_name: Last name
            groups: List of group names
            password: Optional initial password

        Returns:
            UserSummary with created user details

        Raises:
            requests.HTTPError: If not authorized (403), already exists (400), or request fails
        """
        logger.info(f"Creating human user: {username}")

        data = {
            "username": username,
            "email": email,
            "firstname": first_name,
            "lastname": last_name,
            "groups": groups,
        }
        if password:
            data["password"] = password

        response = self._make_request(
            method="POST", endpoint="/api/management/iam/users/human", data=data
        )

        result = UserSummary(**response.json())
        logger.info(f"User created successfully: {username}")
        return result

    def delete_user(self, username: str) -> UserDeleteResponse:
        """
        Delete a user by username.

        Args:
            username: Username to delete

        Returns:
            UserDeleteResponse confirming deletion

        Raises:
            requests.HTTPError: If not authorized (403), not found (400/404), or request fails
        """
        logger.info(f"Deleting user: {username}")

        response = self._make_request(
            method="DELETE", endpoint=f"/api/management/iam/users/{username}"
        )

        result = UserDeleteResponse(**response.json())
        logger.info(f"User deleted successfully: {username}")
        return result

    def list_keycloak_iam_groups(self) -> GroupListResponse:
        """
        List Keycloak IAM groups (admin only).

        This is different from list_groups() which returns groups with server associations.
        This method returns raw Keycloak group data without scopes.

        Returns:
            GroupListResponse with list of groups

        Raises:
            requests.HTTPError: If not authorized (403) or request fails
        """
        logger.info("Listing Keycloak IAM groups")

        response = self._make_request(method="GET", endpoint="/api/management/iam/groups")

        result = GroupListResponse(**response.json())
        logger.info(f"Retrieved {result.total} Keycloak groups")
        return result

    def create_keycloak_group(self, name: str, description: str | None = None) -> GroupSummary:
        """
        Create a new Keycloak group (admin only).

        Args:
            name: Group name
            description: Optional group description

        Returns:
            GroupSummary with created group details

        Raises:
            requests.HTTPError: If not authorized (403), already exists (400), or request fails
        """
        logger.info(f"Creating Keycloak group: {name}")

        data = {"name": name}
        if description:
            data["description"] = description

        response = self._make_request(
            method="POST", endpoint="/api/management/iam/groups", data=data
        )

        result = GroupSummary(**response.json())
        logger.info(f"Group created successfully: {name}")
        return result

    def delete_keycloak_group(self, name: str) -> GroupDeleteResponse:
        """
        Delete a Keycloak group by name (admin only).

        Args:
            name: Group name to delete

        Returns:
            GroupDeleteResponse confirming deletion

        Raises:
            requests.HTTPError: If not authorized (403), not found (404), or request fails
        """
        logger.info(f"Deleting Keycloak group: {name}")

        response = self._make_request(
            method="DELETE", endpoint=f"/api/management/iam/groups/{name}"
        )

        result = GroupDeleteResponse(**response.json())
        logger.info(f"Group deleted successfully: {name}")
        return result

    def get_federation_config(self, config_id: str = "default") -> dict[str, Any]:
        """
        Get federation configuration by ID.

        Args:
            config_id: Configuration ID (default: "default")

        Returns:
            Federation configuration dictionary

        Raises:
            requests.HTTPError: If not found (404) or request fails
        """
        logger.info(f"Getting federation config: {config_id}")

        response = self._make_request(
            method="GET", endpoint="/api/federation/config", params={"config_id": config_id}
        )

        result = response.json()
        logger.info(f"Retrieved federation config: {config_id}")
        return result

    def save_federation_config(
        self, config: dict[str, Any], config_id: str = "default"
    ) -> dict[str, Any]:
        """
        Create or update federation configuration.

        Args:
            config: Federation configuration dictionary
            config_id: Configuration ID (default: "default")

        Returns:
            Saved configuration response

        Raises:
            requests.HTTPError: If validation fails (422) or request fails
        """
        logger.info(f"Saving federation config: {config_id}")

        response = self._make_request(
            method="POST",
            endpoint="/api/federation/config",
            params={"config_id": config_id},
            data=config,
        )

        result = response.json()
        logger.info(f"Federation config saved successfully: {config_id}")
        return result

    def delete_federation_config(self, config_id: str = "default") -> dict[str, str]:
        """
        Delete federation configuration.

        Args:
            config_id: Configuration ID to delete

        Returns:
            Deletion confirmation message

        Raises:
            requests.HTTPError: If not found (404) or request fails
        """
        logger.info(f"Deleting federation config: {config_id}")

        response = self._make_request(
            method="DELETE", endpoint=f"/api/federation/config/{config_id}"
        )

        result = response.json()
        logger.info(f"Federation config deleted successfully: {config_id}")
        return result

    def list_federation_configs(self) -> dict[str, Any]:
        """
        List all federation configurations.

        Returns:
            Dictionary with configs list and total count

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info("Listing federation configs")

        response = self._make_request(method="GET", endpoint="/api/federation/configs")

        result = response.json()
        logger.info(f"Retrieved {result.get('total', 0)} federation configs")
        return result

    def add_anthropic_server(self, server_name: str, config_id: str = "default") -> dict[str, Any]:
        """
        Add Anthropic server to federation configuration.

        Args:
            server_name: Server name (e.g., "io.github.jgador/websharp")
            config_id: Configuration ID (default: "default")

        Returns:
            Updated configuration

        Raises:
            requests.HTTPError: If config not found (404), already exists (400), or request fails
        """
        logger.info(f"Adding Anthropic server '{server_name}' to config: {config_id}")

        response = self._make_request(
            method="POST",
            endpoint=f"/api/federation/config/{config_id}/anthropic/servers",
            params={"server_name": server_name},
        )

        result = response.json()
        logger.info(f"Anthropic server added successfully: {server_name}")
        return result

    def remove_anthropic_server(
        self, server_name: str, config_id: str = "default"
    ) -> dict[str, Any]:
        """
        Remove Anthropic server from federation configuration.

        Args:
            server_name: Server name to remove
            config_id: Configuration ID (default: "default")

        Returns:
            Updated configuration

        Raises:
            requests.HTTPError: If config or server not found (404) or request fails
        """
        logger.info(f"Removing Anthropic server '{server_name}' from config: {config_id}")

        response = self._make_request(
            method="DELETE",
            endpoint=f"/api/federation/config/{config_id}/anthropic/servers/{server_name}",
        )

        result = response.json()
        logger.info(f"Anthropic server removed successfully: {server_name}")
        return result

    def add_asor_agent(self, agent_id: str, config_id: str = "default") -> dict[str, Any]:
        """
        Add ASOR agent to federation configuration.

        Args:
            agent_id: Agent ID (e.g., "aws_assistant")
            config_id: Configuration ID (default: "default")

        Returns:
            Updated configuration

        Raises:
            requests.HTTPError: If config not found (404), already exists (400), or request fails
        """
        logger.info(f"Adding ASOR agent '{agent_id}' to config: {config_id}")

        response = self._make_request(
            method="POST",
            endpoint=f"/api/federation/config/{config_id}/asor/agents",
            params={"agent_id": agent_id},
        )

        result = response.json()
        logger.info(f"ASOR agent added successfully: {agent_id}")
        return result

    def remove_asor_agent(self, agent_id: str, config_id: str = "default") -> dict[str, Any]:
        """
        Remove ASOR agent from federation configuration.

        Args:
            agent_id: Agent ID to remove
            config_id: Configuration ID (default: "default")

        Returns:
            Updated configuration

        Raises:
            requests.HTTPError: If config or agent not found (404) or request fails
        """
        logger.info(f"Removing ASOR agent '{agent_id}' from config: {config_id}")

        response = self._make_request(
            method="DELETE", endpoint=f"/api/federation/config/{config_id}/asor/agents/{agent_id}"
        )

        result = response.json()
        logger.info(f"ASOR agent removed successfully: {agent_id}")
        return result

    def sync_federation(
        self, config_id: str = "default", source: str | None = None
    ) -> dict[str, Any]:
        """
        Trigger manual federation sync to import servers/agents.

        Args:
            config_id: Configuration ID (default: "default")
            source: Optional source filter ("anthropic" or "asor"). None syncs all enabled sources.

        Returns:
            Sync results with counts of synced items

        Raises:
            requests.HTTPError: If config not found (404) or request fails
        """
        logger.info(f"Triggering federation sync for config: {config_id}")

        params = {}
        if source:
            params["source"] = source

        response = self._make_request(
            method="POST",
            endpoint="/api/federation/sync",
            params={"config_id": config_id, **params},
        )

        result = response.json()
        logger.info(f"Federation sync completed: {result.get('total_synced', 0)} items synced")
        return result

    # ==========================================
    # Peer Federation Management Methods
    # ==========================================

    def list_peers(self, enabled: bool | None = None) -> dict[str, Any]:
        """
        List all configured peer registries.

        Args:
            enabled: Optional filter by enabled status

        Returns:
            Dictionary with peers list

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info("Listing peer registries")

        params = {}
        if enabled is not None:
            params["enabled"] = str(enabled).lower()

        response = self._make_request(
            method="GET", endpoint="/api/peers", params=params if params else None
        )

        result = response.json()
        logger.info(f"Retrieved {len(result) if isinstance(result, list) else 0} peers")
        return result

    def add_peer(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Add a new peer registry.

        Args:
            config: Peer configuration dictionary with peer_id, name, endpoint, etc.

        Returns:
            Created peer configuration

        Raises:
            requests.HTTPError: If peer already exists (409) or request fails
        """
        peer_id = config.get("peer_id", "unknown")
        logger.info(f"Adding peer registry: {peer_id}")

        response = self._make_request(method="POST", endpoint="/api/peers", data=config)

        result = response.json()
        logger.info(f"Peer registry added successfully: {peer_id}")
        return result

    def get_peer(self, peer_id: str) -> dict[str, Any]:
        """
        Get details of a specific peer registry.

        Args:
            peer_id: Peer registry identifier

        Returns:
            Peer configuration details

        Raises:
            requests.HTTPError: If peer not found (404) or request fails
        """
        logger.info(f"Getting peer registry: {peer_id}")

        response = self._make_request(method="GET", endpoint=f"/api/peers/{peer_id}")

        result = response.json()
        logger.info(f"Retrieved peer registry: {peer_id}")
        return result

    def update_peer(self, peer_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Update an existing peer registry configuration.

        Args:
            peer_id: Peer registry identifier
            config: Updated peer configuration

        Returns:
            Updated peer configuration

        Raises:
            requests.HTTPError: If peer not found (404) or request fails
        """
        logger.info(f"Updating peer registry: {peer_id}")

        response = self._make_request(method="PUT", endpoint=f"/api/peers/{peer_id}", data=config)

        result = response.json()
        logger.info(f"Peer registry updated successfully: {peer_id}")
        return result

    def update_peer_token(self, peer_id: str, federation_token: str) -> dict[str, Any]:
        """
        Update only the federation token for a peer registry.

        This is useful for recovering from token loss (issue #561) or
        rotating tokens without triggering a full peer update.

        Args:
            peer_id: Peer registry identifier
            federation_token: New federation token value

        Returns:
            Success message with peer ID

        Raises:
            requests.HTTPError: If peer not found (404) or request fails
        """
        logger.info(f"Updating federation token for peer: {peer_id}")

        response = self._make_request(
            method="PATCH",
            endpoint=f"/api/peers/{peer_id}/token",
            data={"federation_token": federation_token},
        )

        result = response.json()
        logger.info(f"Federation token updated successfully for peer: {peer_id}")
        return result

    def remove_peer(self, peer_id: str) -> dict[str, Any]:
        """
        Remove a peer registry.

        Args:
            peer_id: Peer registry identifier

        Returns:
            Deletion confirmation

        Raises:
            requests.HTTPError: If peer not found (404) or request fails
        """
        logger.info(f"Removing peer registry: {peer_id}")

        response = self._make_request(method="DELETE", endpoint=f"/api/peers/{peer_id}")

        # Handle 204 No Content response
        if response.status_code == 204:
            logger.info(f"Peer registry removed successfully: {peer_id}")
            return {"status": "deleted", "peer_id": peer_id}

        result = response.json()
        logger.info(f"Peer registry removed successfully: {peer_id}")
        return result

    def sync_peer(self, peer_id: str) -> dict[str, Any]:
        """
        Trigger sync from a specific peer registry.

        Args:
            peer_id: Peer registry identifier

        Returns:
            Sync result with statistics

        Raises:
            requests.HTTPError: If peer not found (404) or request fails
        """
        logger.info(f"Syncing from peer registry: {peer_id}")

        response = self._make_request(method="POST", endpoint=f"/api/peers/{peer_id}/sync")

        result = response.json()
        logger.info(f"Peer sync completed: {peer_id}")
        return result

    def sync_all_peers(self) -> dict[str, Any]:
        """
        Trigger sync from all enabled peer registries.

        Returns:
            Sync results for all peers

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info("Syncing from all peer registries")

        response = self._make_request(method="POST", endpoint="/api/peers/sync")

        result = response.json()
        logger.info("All peer sync completed")
        return result

    def get_peer_status(self, peer_id: str) -> dict[str, Any]:
        """
        Get sync status for a specific peer registry.

        Args:
            peer_id: Peer registry identifier

        Returns:
            Sync status with history

        Raises:
            requests.HTTPError: If peer not found (404) or request fails
        """
        logger.info(f"Getting sync status for peer: {peer_id}")

        response = self._make_request(method="GET", endpoint=f"/api/peers/{peer_id}/status")

        result = response.json()
        logger.info(f"Retrieved sync status for peer: {peer_id}")
        return result

    def enable_peer(self, peer_id: str) -> dict[str, Any]:
        """
        Enable a peer registry.

        Args:
            peer_id: Peer registry identifier

        Returns:
            Updated peer configuration

        Raises:
            requests.HTTPError: If peer not found (404) or request fails
        """
        logger.info(f"Enabling peer registry: {peer_id}")

        response = self._make_request(method="POST", endpoint=f"/api/peers/{peer_id}/enable")

        result = response.json()
        logger.info(f"Peer registry enabled: {peer_id}")
        return result

    def disable_peer(self, peer_id: str) -> dict[str, Any]:
        """
        Disable a peer registry.

        Args:
            peer_id: Peer registry identifier

        Returns:
            Updated peer configuration

        Raises:
            requests.HTTPError: If peer not found (404) or request fails
        """
        logger.info(f"Disabling peer registry: {peer_id}")

        response = self._make_request(method="POST", endpoint=f"/api/peers/{peer_id}/disable")

        result = response.json()
        logger.info(f"Peer registry disabled: {peer_id}")
        return result

    def get_peer_connections(self) -> dict[str, Any]:
        """
        Get all federation connections across all peers.

        Returns:
            Dictionary with connection details

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info("Getting all peer connections")

        response = self._make_request(method="GET", endpoint="/api/peers/connections/all")

        result = response.json()
        logger.info("Retrieved peer connections")
        return result

    def get_shared_resources(self) -> dict[str, Any]:
        """
        Get resource sharing summary across all peers.

        Returns:
            Dictionary with shared resource details

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info("Getting shared resources summary")

        response = self._make_request(method="GET", endpoint="/api/peers/shared-resources")

        result = response.json()
        logger.info("Retrieved shared resources summary")
        return result

    # ==========================================
    # Agent Skills Management Methods
    # ==========================================

    def register_skill(self, request: SkillRegistrationRequest) -> SkillCard:
        """
        Register a new Agent Skill.

        Args:
            request: Skill registration request

        Returns:
            SkillCard with registered skill details

        Raises:
            requests.HTTPError: If skill already exists (409) or validation fails (400/422)
        """
        logger.info(f"Registering skill: {request.name}")

        response = self._make_request(
            method="POST", endpoint="/api/skills", data=request.model_dump(exclude_none=True)
        )

        result = response.json()
        logger.info(f"Skill registered successfully: {result.get('name')} at {result.get('path')}")
        return SkillCard(**result)

    def list_skills(
        self,
        include_disabled: bool = False,
        tag: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SkillListResponse:
        """
        List all Agent Skills.

        Args:
            include_disabled: Include disabled skills
            tag: Filter by tag
            limit: Maximum number of skills to return per page
            offset: Number of skills to skip for pagination

        Returns:
            SkillListResponse with list of skills

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info("Listing skills")

        params: dict[str, str | int] = {
            "limit": limit,
            "offset": offset,
        }
        if include_disabled:
            params["include_disabled"] = "true"
        if tag:
            params["tag"] = tag

        response = self._make_request(method="GET", endpoint="/api/skills", params=params)

        result = response.json()
        skills = [SkillCard(**s) for s in result.get("skills", [])]
        total_count = result.get("total_count", len(skills))
        resp_limit = result.get("limit", limit)
        resp_offset = result.get("offset", offset)
        has_next = result.get("has_next", False)
        logger.info(
            f"Retrieved {len(skills)} skills"
            f" (total={total_count}, offset={resp_offset},"
            f" limit={resp_limit}, has_next={has_next})"
        )
        return SkillListResponse(
            skills=skills,
            total_count=total_count,
            limit=resp_limit,
            offset=resp_offset,
            has_next=has_next,
        )

    def get_skill(self, path: str) -> SkillCard:
        """
        Get details for a specific skill.

        Args:
            path: Skill path or name

        Returns:
            SkillCard with skill details

        Raises:
            requests.HTTPError: If skill not found (404)
        """
        # Normalize path - remove /skills/ prefix if present
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Getting skill: {api_path}")

        response = self._make_request(method="GET", endpoint=f"/api/skills{api_path}")

        result = response.json()
        logger.info(f"Retrieved skill: {result.get('name')}")
        return SkillCard(**result)

    def update_skill(self, path: str, request: SkillRegistrationRequest) -> SkillCard:
        """
        Update an existing skill.

        Args:
            path: Skill path or name
            request: Updated skill data

        Returns:
            Updated SkillCard

        Raises:
            requests.HTTPError: If skill not found (404) or validation fails
        """
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Updating skill: {api_path}")

        response = self._make_request(
            method="PUT",
            endpoint=f"/api/skills{api_path}",
            data=request.model_dump(exclude_none=True),
        )

        result = response.json()
        logger.info(f"Skill updated: {result.get('name')}")
        return SkillCard(**result)

    def delete_skill(self, path: str) -> bool:
        """
        Delete a skill.

        Args:
            path: Skill path or name

        Returns:
            True if deleted successfully

        Raises:
            requests.HTTPError: If skill not found (404) or permission denied (403)
        """
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Deleting skill: {api_path}")

        self._make_request(method="DELETE", endpoint=f"/api/skills{api_path}")

        logger.info(f"Skill deleted: {api_path}")
        return True

    def toggle_skill(self, path: str, enabled: bool) -> SkillToggleResponse:
        """
        Toggle skill enabled/disabled state.

        Args:
            path: Skill path or name
            enabled: New enabled state

        Returns:
            SkillToggleResponse with new state

        Raises:
            requests.HTTPError: If skill not found (404)
        """
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Toggling skill {api_path} to enabled={enabled}")

        response = self._make_request(
            method="POST", endpoint=f"/api/skills{api_path}/toggle", data={"enabled": enabled}
        )

        result = response.json()
        logger.info(f"Skill toggled: {result.get('path')} -> enabled={result.get('is_enabled')}")
        return SkillToggleResponse(**result)

    def check_skill_health(self, path: str) -> SkillHealthResponse:
        """
        Check skill health (SKILL.md accessibility).

        Args:
            path: Skill path or name

        Returns:
            SkillHealthResponse with health status

        Raises:
            requests.HTTPError: If skill not found (404)
        """
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Checking health for skill: {api_path}")

        response = self._make_request(method="GET", endpoint=f"/api/skills{api_path}/health")

        result = response.json()
        logger.info(f"Skill health: {result.get('path')} -> healthy={result.get('healthy')}")
        return SkillHealthResponse(**result)

    def get_skill_content(self, path: str) -> SkillContentResponse:
        """
        Get SKILL.md content for a skill.

        Args:
            path: Skill path or name

        Returns:
            SkillContentResponse with content

        Raises:
            requests.HTTPError: If skill not found (404) or content unavailable
        """
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Getting content for skill: {api_path}")

        response = self._make_request(method="GET", endpoint=f"/api/skills{api_path}/content")

        result = response.json()
        content_len = len(result.get("content", ""))
        logger.info(f"Retrieved skill content: {content_len} characters")
        return SkillContentResponse(**result)

    def search_skills(self, query: str, tags: str | None = None) -> SkillSearchResponse:
        """
        Search for skills by query.

        Args:
            query: Search query
            tags: Optional comma-separated tags filter

        Returns:
            SkillSearchResponse with matching skills

        Raises:
            requests.HTTPError: If request fails
        """
        logger.info(f"Searching skills: query='{query}', tags={tags}")

        params = {"q": query}
        if tags:
            params["tags"] = tags

        response = self._make_request(method="GET", endpoint="/api/skills/search", params=params)

        result = response.json()
        logger.info(f"Found {result.get('total_count', 0)} skills matching '{query}'")
        return SkillSearchResponse(**result)

    def rate_skill(self, path: str, rating: int) -> dict[str, Any]:
        """
        Rate a skill (1-5 stars).

        Args:
            path: Skill path or name
            rating: Rating value (1-5)

        Returns:
            Rating response with average rating

        Raises:
            requests.HTTPError: If skill not found (404) or invalid rating (400)
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Rating skill {api_path}: {rating} stars")

        response = self._make_request(
            method="POST", endpoint=f"/api/skills{api_path}/rate", data={"rating": rating}
        )

        result = response.json()
        logger.info(f"Skill rated: avg={result.get('average_rating')}")
        return result

    def get_skill_rating(self, path: str) -> SkillRatingResponse:
        """
        Get rating information for a skill.

        Args:
            path: Skill path or name

        Returns:
            SkillRatingResponse with rating details

        Raises:
            requests.HTTPError: If skill not found (404)
        """
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Getting rating for skill: {api_path}")

        response = self._make_request(method="GET", endpoint=f"/api/skills{api_path}/rating")

        result = response.json()
        logger.info(f"Skill rating: {result.get('num_stars')} stars")
        return SkillRatingResponse(**result)

    def get_skill_security_scan(self, path: str) -> SkillSecurityScanResponse:
        """
        Get security scan results for a skill.

        Returns the latest security scan results including threat analysis,
        findings by analyzer, and overall safety status.

        Args:
            path: Skill path or name

        Returns:
            Security scan results with analysis_results and scan_results
        """
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Getting security scan results for skill: {api_path}")

        response = self._make_request(method="GET", endpoint=f"/api/skills{api_path}/security-scan")

        result = SkillSecurityScanResponse(**response.json())
        logger.info(f"Retrieved security scan results for skill '{api_path}'")
        return result

    def rescan_skill(self, path: str) -> SkillRescanResponse:
        """
        Trigger a manual security scan for a skill.

        Initiates a new security scan for the specified skill and returns
        the scan results. Requires admin privileges.

        Args:
            path: Skill path or name

        Returns:
            Newly generated security scan results
        """
        api_path = path.replace("/skills/", "/") if path.startswith("/skills/") else f"/{path}"
        logger.info(f"Triggering security scan for skill: {api_path}")

        response = self._make_request(method="POST", endpoint=f"/api/skills{api_path}/rescan")

        result = SkillRescanResponse(**response.json())
        safety_status = "SAFE" if result.is_safe else "UNSAFE"
        logger.info(
            f"Security scan completed for skill '{api_path}': {safety_status} "
            f"(C:{result.critical_issues} H:{result.high_severity} "
            f"M:{result.medium_severity} L:{result.low_severity})"
        )
        return result

    # =========================================================================
    # Virtual MCP Server Operations
    # =========================================================================

    def create_virtual_server(self, request: VirtualServerCreateRequest) -> VirtualServerConfig:
        """
        Create a new virtual MCP server.

        Args:
            request: Virtual server creation request with tool mappings

        Returns:
            VirtualServerConfig with created server details

        Raises:
            requests.HTTPError: If creation fails (400 invalid, 409 conflict)
        """
        logger.info(f"Creating virtual server: {request.path}")
        logger.debug(f"Virtual server config:\n{json.dumps(request.model_dump(), indent=2)}")

        response = self._make_request(
            method="POST", endpoint="/api/virtual-servers", data=request.model_dump()
        )

        result = response.json()
        logger.info(f"Virtual server created: {result.get('path')}")
        return VirtualServerConfig(**result)

    def list_virtual_servers(
        self,
        enabled_only: bool = False,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> VirtualServerListResponse:
        """
        List virtual MCP servers.

        Args:
            enabled_only: If True, only return enabled servers
            tag: Filter by tag
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            VirtualServerListResponse with list of servers
        """
        params = {"limit": limit, "offset": offset}
        if enabled_only:
            params["enabled_only"] = "true"
        if tag:
            params["tag"] = tag

        logger.info(f"Listing virtual servers (enabled_only={enabled_only}, tag={tag})")

        response = self._make_request(method="GET", endpoint="/api/virtual-servers", params=params)

        result = response.json()
        logger.info(f"Found {result.get('total', 0)} virtual servers")
        return VirtualServerListResponse(**result)

    def get_virtual_server(self, path: str) -> VirtualServerConfig:
        """
        Get details of a virtual MCP server.

        Args:
            path: Virtual server path (e.g., /virtual/dev-tools)

        Returns:
            VirtualServerConfig with server details

        Raises:
            requests.HTTPError: If server not found (404)
        """
        api_path = path if path.startswith("/") else f"/{path}"
        logger.info(f"Getting virtual server: {api_path}")

        response = self._make_request(method="GET", endpoint=f"/api/virtual-servers{api_path}")

        result = response.json()
        logger.info(f"Virtual server: {result.get('server_name')}")
        return VirtualServerConfig(**result)

    def update_virtual_server(
        self, path: str, request: VirtualServerCreateRequest
    ) -> VirtualServerConfig:
        """
        Update an existing virtual MCP server.

        Args:
            path: Virtual server path
            request: Updated configuration

        Returns:
            VirtualServerConfig with updated server details

        Raises:
            requests.HTTPError: If server not found (404) or invalid (400)
        """
        api_path = path if path.startswith("/") else f"/{path}"
        logger.info(f"Updating virtual server: {api_path}")
        logger.debug(f"Updated config:\n{json.dumps(request.model_dump(), indent=2)}")

        response = self._make_request(
            method="PUT", endpoint=f"/api/virtual-servers{api_path}", data=request.model_dump()
        )

        result = response.json()
        logger.info(f"Virtual server updated: {result.get('path')}")
        return VirtualServerConfig(**result)

    def delete_virtual_server(self, path: str) -> VirtualServerDeleteResponse:
        """
        Delete a virtual MCP server.

        Args:
            path: Virtual server path

        Returns:
            VirtualServerDeleteResponse with confirmation

        Raises:
            requests.HTTPError: If server not found (404)
        """
        api_path = path if path.startswith("/") else f"/{path}"
        logger.info(f"Deleting virtual server: {api_path}")

        response = self._make_request(method="DELETE", endpoint=f"/api/virtual-servers{api_path}")

        result = response.json()
        logger.info(f"Virtual server deleted: {api_path}")
        return VirtualServerDeleteResponse(**result)

    def toggle_virtual_server(self, path: str, enable: bool) -> VirtualServerToggleResponse:
        """
        Enable or disable a virtual MCP server.

        Args:
            path: Virtual server path
            enable: True to enable, False to disable

        Returns:
            VirtualServerToggleResponse with new state

        Raises:
            requests.HTTPError: If server not found (404)
        """
        api_path = path if path.startswith("/") else f"/{path}"
        action = "enable" if enable else "disable"
        logger.info(f"Toggling virtual server {api_path}: {action}")

        response = self._make_request(
            method="POST", endpoint=f"/api/virtual-servers{api_path}/{action}"
        )

        result = response.json()
        logger.info(f"Virtual server {action}d: {result.get('is_enabled')}")
        return VirtualServerToggleResponse(**result)

    def rate_virtual_server(self, path: str, rating: int) -> dict[str, Any]:
        """
        Rate a virtual MCP server (1-5 stars).

        Args:
            path: Virtual server path
            rating: Rating value (1-5)

        Returns:
            Rating response with average rating

        Raises:
            requests.HTTPError: If server not found (404) or invalid rating (400)
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        api_path = path if path.startswith("/") else f"/{path}"
        logger.info(f"Rating virtual server {api_path}: {rating} stars")

        response = self._make_request(
            method="POST", endpoint=f"/api/virtual-servers{api_path}/rate", data={"rating": rating}
        )

        result = response.json()
        logger.info(f"Virtual server rated: avg={result.get('average_rating')}")
        return result

    def get_virtual_server_rating(self, path: str) -> dict[str, Any]:
        """
        Get rating information for a virtual MCP server.

        Args:
            path: Virtual server path

        Returns:
            Dict with rating details (num_stars, rating_count, etc.)

        Raises:
            requests.HTTPError: If server not found (404)
        """
        api_path = path if path.startswith("/") else f"/{path}"
        logger.info(f"Getting rating for virtual server: {api_path}")

        response = self._make_request(
            method="GET", endpoint=f"/api/virtual-servers{api_path}/rating"
        )

        result = response.json()
        logger.info(f"Virtual server rating: {result.get('num_stars')} stars")
        return result

    def force_heartbeat(self) -> dict[str, Any]:
        """Force an immediate heartbeat telemetry event (admin only).

        Bypasses the 24-hour lock and sends a heartbeat event immediately.

        Returns:
            Dict with status and payload summary.

        Raises:
            requests.HTTPError: If not authorized (403) or telemetry disabled (409)
        """
        logger.info("Forcing heartbeat telemetry event")

        response = self._make_request(
            method="POST",
            endpoint="/api/registry-management/telemetry/heartbeat",
        )

        result = response.json()
        logger.info(f"Heartbeat result: {result.get('status')}")
        return result

    def force_startup_ping(self) -> dict[str, Any]:
        """Force an immediate startup telemetry event (admin only).

        Bypasses the 60-second lock and sends a startup ping immediately.

        Returns:
            Dict with status and payload summary.

        Raises:
            requests.HTTPError: If not authorized (403) or telemetry disabled (409)
        """
        logger.info("Forcing startup telemetry event")

        response = self._make_request(
            method="POST",
            endpoint="/api/registry-management/telemetry/startup",
        )

        result = response.json()
        logger.info(f"Startup ping result: {result.get('status')}")
        return result

    # -------------------------------------------------------------------------
    # Direct M2M client registration (issue #851, /api/iam/m2m-clients)
    #
    # These endpoints write directly to idp_m2m_clients without calling any
    # IdP Admin API. Useful when OKTA_API_TOKEN / equivalent is unavailable.
    # -------------------------------------------------------------------------

    def create_m2m_client(
        self,
        client_id: str,
        client_name: str,
        groups: list[str] | None = None,
        description: str | None = None,
    ) -> IdPM2MClient:
        """Register an M2M client directly (admin only).

        Args:
            client_id: IdP application client ID to register.
            client_name: Human-readable name for the client.
            groups: Group mappings for authorization.
            description: Optional description.

        Returns:
            The persisted M2M client record.

        Raises:
            requests.HTTPError: 401/403 on auth, 409 if client_id already exists,
                422 for invalid payload.
        """
        logger.info(f"Registering M2M client: {client_id}")

        payload: dict[str, Any] = {
            "client_id": client_id,
            "client_name": client_name,
            "groups": list(groups) if groups else [],
        }
        if description is not None:
            payload["description"] = description

        response = self._make_request(
            method="POST",
            endpoint="/api/iam/m2m-clients",
            data=payload,
        )
        return IdPM2MClient(**response.json())

    def list_m2m_clients(
        self,
        provider: str | None = None,
        limit: int = 500,
        skip: int = 0,
    ) -> M2MClientListResponse:
        """List M2M clients with pagination.

        Args:
            provider: Optional provider filter (e.g. "manual", "okta").
            limit: Max records to return (1-1000).
            skip: Offset for pagination.

        Returns:
            Paginated envelope with total, limit, skip, items.

        Raises:
            requests.HTTPError: 401 if unauthenticated.
        """
        logger.info(f"Listing M2M clients (provider={provider}, limit={limit}, skip={skip})")

        params: dict[str, Any] = {"limit": limit, "skip": skip}
        if provider is not None:
            params["provider"] = provider

        response = self._make_request(
            method="GET",
            endpoint="/api/iam/m2m-clients",
            params=params,
        )
        return M2MClientListResponse(**response.json())

    def get_m2m_client(self, client_id: str) -> IdPM2MClient:
        """Get a single M2M client by client_id.

        Args:
            client_id: IdP application client ID.

        Returns:
            The M2M client record.

        Raises:
            requests.HTTPError: 401 if unauthenticated, 404 if not found.
        """
        logger.info(f"Getting M2M client: {client_id}")

        response = self._make_request(
            method="GET",
            endpoint=f"/api/iam/m2m-clients/{quote(client_id, safe='')}",
        )
        return IdPM2MClient(**response.json())

    def patch_m2m_client(
        self,
        client_id: str,
        client_name: str | None = None,
        groups: list[str] | None = None,
        description: str | None = None,
        enabled: bool | None = None,
    ) -> IdPM2MClient:
        """Partially update an M2M client (admin only).

        Only manual records (provider == "manual") can be updated. IdP-synced
        records return 403.

        Fields left as None are NOT sent to the server (unchanged). To clear
        groups, pass an empty list explicitly.

        Args:
            client_id: IdP application client ID to update.
            client_name: New name, or None to leave unchanged.
            groups: New groups list (empty list clears), or None to leave unchanged.
            description: New description, or None to leave unchanged.
            enabled: New enabled flag, or None to leave unchanged.

        Returns:
            The updated M2M client record.

        Raises:
            requests.HTTPError: 401/403 on auth, 404 if not found, 403 if record
                was IdP-synced.
        """
        logger.info(f"Updating M2M client: {client_id}")

        payload: dict[str, Any] = {}
        if client_name is not None:
            payload["client_name"] = client_name
        if groups is not None:
            payload["groups"] = list(groups)
        if description is not None:
            payload["description"] = description
        if enabled is not None:
            payload["enabled"] = enabled

        response = self._make_request(
            method="PATCH",
            endpoint=f"/api/iam/m2m-clients/{quote(client_id, safe='')}",
            data=payload,
        )
        return IdPM2MClient(**response.json())

    def delete_m2m_client(self, client_id: str) -> None:
        """Delete a manual M2M client (admin only).

        Only manual records (provider == "manual") can be deleted.

        Args:
            client_id: IdP application client ID to delete.

        Raises:
            requests.HTTPError: 401/403 on auth, 404 if not found, 403 if record
                was IdP-synced.
        """
        logger.info(f"Deleting M2M client: {client_id}")

        self._make_request(
            method="DELETE",
            endpoint=f"/api/iam/m2m-clients/{quote(client_id, safe='')}",
        )


    # -------------------------------------------------------------------------
    # Application Logs (admin-only, issue #886)
    # -------------------------------------------------------------------------

    def get_logs(
        self,
        service: str | None = None,
        level: str | None = None,
        hostname: str | None = None,
        search: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AppLogResponse:
        """Query application logs (admin only).

        Args:
            service: Filter by service name.
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            hostname: Filter by hostname/pod.
            search: Substring search in log messages.
            start: Start timestamp (ISO-8601).
            end: End timestamp (ISO-8601).
            limit: Page size (1-10000).
            offset: Offset for pagination.

        Returns:
            AppLogResponse with matching log entries.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if service:
            params["service"] = service
        if level:
            params["level"] = level
        if hostname:
            params["hostname"] = hostname
        if search:
            params["search"] = search
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._make_request(
            method="GET",
            endpoint="/api/admin/logs",
            params=params,
        )
        return AppLogResponse(**response.json())

    def get_log_metadata(self) -> AppLogMetadataResponse:
        """Get available filter values for application logs (admin only).

        Returns:
            AppLogMetadataResponse with services, hostnames, and levels.
        """
        response = self._make_request(
            method="GET",
            endpoint="/api/admin/logs/metadata",
        )
        return AppLogMetadataResponse(**response.json())

    def get_log_services(self) -> list[str]:
        """Get list of distinct service names from application logs (admin only).

        Returns:
            List of service name strings.
        """
        metadata = self.get_log_metadata()
        return metadata.services


def _format_tool_result(
    tool: ToolSearchResult,
) -> dict[str, Any]:
    """
    Format a tool search result for display to the agent.

    The search API returns inputSchema directly, so no additional server lookup is needed.

    Args:
        tool: Tool search result

    Returns:
        Formatted tool information dict
    """
    result = {
        "tool_name": tool.tool_name,
        "server_path": tool.server_path,
        "server_name": tool.server_name,
        "description": tool.description or "No description available",
        "relevance_score": tool.relevance_score,
        "supported_transports": ["streamable_http"],
    }

    # Use inputSchema from search result if available
    if tool.inputSchema:
        result["tool_schema"] = tool.inputSchema

    return result
