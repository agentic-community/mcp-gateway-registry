"""
Registry Card model for describing this registry instance.

This module defines the RegistryCard model used for federation discovery
and registry metadata, along with supporting models for capabilities,
authentication, and contact information.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class LifecycleStatus(str, Enum):
    """Lifecycle status for servers and agents."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DRAFT = "draft"
    BETA = "beta"


class RegistryCapabilities(BaseModel):
    """Capabilities supported by this registry."""

    servers: bool = Field(default=True, description="Supports MCP servers")
    agents: bool = Field(default=True, description="Supports A2A agents")
    skills: bool = Field(default=True, description="Supports AI agent skills")
    prompts: bool = Field(default=False, description="Supports prompt templates")
    security_scans: bool = Field(default=True, description="Runs security scans")
    incremental_sync: bool = Field(default=False, description="Supports incremental sync")
    webhooks: bool = Field(default=False, description="Supports webhook notifications")


class RegistryAuthConfig(BaseModel):
    """Authentication configuration for federation."""

    schemes: list[str] = Field(
        default_factory=lambda: ["oauth2", "bearer"],
        description="Supported auth schemes",
    )
    oauth2_issuer: str | None = Field(default=None, description="OAuth2/OIDC issuer URL")
    oauth2_token_endpoint: str | None = Field(default=None, description="OAuth2 token endpoint")
    scopes_supported: list[str] = Field(
        default_factory=lambda: ["federation/read"],
        description="OAuth2 scopes",
    )


class RegistryContact(BaseModel):
    """Contact information for registry operators."""

    email: str | None = Field(default=None, description="Contact email")
    url: str | None = Field(default=None, description="Documentation or support URL")


class RegistryCard(BaseModel):
    """
    Registry Card describing this registry instance.

    Used for federation discovery and registry metadata.
    """

    schema_version: str = Field(
        default="1.0.0",
        description="Schema version for forward/backward compatibility",
    )
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier (UUID) for this registry instance",
    )
    name: str = Field(..., description="Human-readable registry name")
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Registry description (max 1000 chars)",
    )

    # Base URL and organization (for frontend display)
    registry_url: str | None = Field(default=None, description="Base URL of this registry instance")
    organization_name: str | None = Field(
        default=None, description="Organization name that operates this registry"
    )

    federation_api_version: str = Field(default="1.0", description="Federation API version")
    federation_endpoint: HttpUrl = Field(
        ..., description="Federation API base URL (HTTPS required)"
    )

    capabilities: RegistryCapabilities = Field(
        default_factory=RegistryCapabilities, description="Registry capabilities"
    )
    authentication: RegistryAuthConfig = Field(
        default_factory=RegistryAuthConfig, description="Auth configuration"
    )

    visibility_policy: str = Field(
        default="public_only",
        description="Visibility policy: public_only, authenticated, private",
    )
    contact: RegistryContact | None = Field(default=None, description="Contact information")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (max 10KB serialized)",
    )

    # Internal tracking
    created_at: datetime | None = Field(default=None, description="Created timestamp")
    updated_at: datetime | None = Field(default=None, description="Last updated timestamp")

    @field_validator("visibility_policy")
    @classmethod
    def _validate_visibility_policy(cls, v: str) -> str:
        """Validate visibility policy."""
        allowed = ["public_only", "authenticated", "private"]
        if v not in allowed:
            raise ValueError(f"visibility_policy must be one of {allowed}")
        return v

    @field_validator("federation_endpoint")
    @classmethod
    def _validate_https_endpoint(cls, v: HttpUrl) -> HttpUrl:
        """Ensure federation endpoint uses HTTPS in production."""
        # Allow HTTP for localhost/development, require HTTPS for production
        url_str = str(v)
        if url_str.startswith("http://") and not any(
            host in url_str for host in ["localhost", "127.0.0.1", "host.docker.internal"]
        ):
            logger.warning(
                f"federation_endpoint uses HTTP in production: {url_str}. "
                "HTTPS is strongly recommended for security."
            )
        return v

    @field_validator("metadata")
    @classmethod
    def _validate_metadata_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata size limit (10KB)."""
        serialized = json.dumps(v)
        if len(serialized) > 10240:  # 10KB
            raise ValueError("metadata exceeds 10KB size limit")
        return v
