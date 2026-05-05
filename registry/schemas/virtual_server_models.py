"""
Virtual MCP Server data models.

Defines the schema for virtual servers that aggregate tools from multiple
backend MCP servers into a single unified endpoint with fine-grained
access control, tool aliasing, and version pinning.
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class ToolMapping(BaseModel):
    """Maps a tool from a backend server into a virtual server.

    Each mapping selects a specific tool from a backend MCP server,
    optionally renaming it (alias) and pinning it to a specific version.
    """

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(
        ...,
        min_length=1,
        description="Original tool name on the backend server",
    )
    alias: str | None = Field(
        None,
        description="Renamed tool name in virtual server (for conflict resolution)",
    )
    backend_server_path: str = Field(
        ...,
        min_length=1,
        description="Backend server path (e.g., '/github')",
    )
    backend_version: str | None = Field(
        None,
        description="Pinned version (None = active version, e.g., 'v1.5.0' = pinned)",
    )
    description_override: str | None = Field(
        None,
        max_length=1024,
        description="Override the tool's description in this virtual server",
    )

    @field_validator("backend_server_path")
    @classmethod
    def validate_backend_path(
        cls,
        v: str,
    ) -> str:
        """Validate backend server path starts with /."""
        if not v.startswith("/"):
            raise ValueError("Backend server path must start with '/'")
        return v


class ToolScopeOverride(BaseModel):
    """Per-tool scope override for fine-grained access control.

    Allows requiring additional scopes to see or call specific tools
    beyond the virtual server's base required_scopes.
    """

    model_config = ConfigDict(populate_by_name=True)

    tool_alias: str = Field(
        ...,
        min_length=1,
        description="Tool alias or original tool_name",
    )
    required_scopes: list[str] = Field(
        ...,
        min_length=1,
        description="Scopes needed to see/call this tool",
    )


class VirtualServerConfig(BaseModel):
    """Full virtual MCP server configuration.

    A virtual server aggregates tools from multiple backend MCP servers
    into a single endpoint. It supports tool aliasing, version pinning,
    and scope-based access control.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Identity
    path: str = Field(
        ...,
        description="Unique path and MongoDB _id (e.g., '/virtual/dev-essentials')",
    )
    server_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable name for the virtual server",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Description of the virtual server's purpose",
    )

    @field_validator("description", mode="before")
    @classmethod
    def coerce_null_description(cls, v: Any) -> str:
        """Coerce None to empty string for MongoDB documents with null description."""
        if v is None:
            return ""
        return v

    # Tool configuration
    tool_mappings: list[ToolMapping] = Field(
        default_factory=list,
        max_length=500,
        description="List of tools mapped from backend servers (max 500)",
    )

    # Access control
    required_scopes: list[str] = Field(
        default_factory=list,
        description="Scopes required to access any tool on this virtual server",
    )
    tool_scope_overrides: list[ToolScopeOverride] = Field(
        default_factory=list,
        description="Per-tool scope overrides for fine-grained access",
    )

    # State
    is_enabled: bool = Field(
        default=False,
        description="Whether the virtual server is enabled and routable",
    )

    # Categorization
    tags: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Tags for categorization and filtering (max 50)",
    )
    supported_transports: list[str] = Field(
        default_factory=lambda: ["streamable-http"],
        description="Supported MCP transport types",
    )

    # Rating
    num_stars: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Average star rating (0-5)",
    )
    rating_details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of individual ratings with user and rating",
    )

    # Audit
    created_by: str | None = Field(
        None,
        description="Username of the creator",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="Last update timestamp",
    )

    @field_validator("path")
    @classmethod
    def validate_path(
        cls,
        v: str,
    ) -> str:
        """Validate path starts with /virtual/ to avoid collision with real servers."""
        if not v.startswith("/virtual/"):
            raise ValueError("Virtual server path must start with '/virtual/'")
        # Validate path segment after /virtual/
        segment = v[len("/virtual/") :]
        if not segment:
            raise ValueError("Virtual server path must have a name after '/virtual/'")
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", segment):
            raise ValueError(
                "Virtual server path segment must be lowercase alphanumeric "
                "with single hyphens, not starting or ending with hyphen"
            )
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(
        cls,
        v: list[str],
    ) -> list[str]:
        """Validate each tag is within max length."""
        for tag in v:
            if len(tag) > 64:
                raise ValueError(f"Tag '{tag[:20]}...' exceeds max length of 64 characters")
        return v

    @field_validator("server_name")
    @classmethod
    def validate_server_name(
        cls,
        v: str,
    ) -> str:
        """Validate server name is not empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Server name cannot be empty or whitespace-only")
        return stripped


class VirtualServerInfo(BaseModel):
    """Lightweight virtual server summary for listings."""

    model_config = ConfigDict(populate_by_name=True)

    path: str = Field(..., description="Virtual server path")
    server_name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Server description")

    @field_validator("description", mode="before")
    @classmethod
    def coerce_null_description(cls, v: Any) -> str:
        """Coerce None to empty string for MongoDB documents with null description."""
        if v is None:
            return ""
        return v

    tool_count: int = Field(default=0, description="Number of mapped tools")
    backend_count: int = Field(default=0, description="Number of unique backend servers")
    backend_paths: list[str] = Field(
        default_factory=list,
        description="List of unique backend server paths",
    )
    is_enabled: bool = Field(default=False, description="Whether the server is enabled")
    tags: list[str] = Field(default_factory=list, description="Tags")
    num_stars: float = Field(default=0.0, description="Average star rating")
    rating_details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of individual ratings",
    )
    created_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CreateVirtualServerRequest(BaseModel):
    """Request model for creating a virtual server."""

    model_config = ConfigDict(populate_by_name=True)

    server_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable name",
    )
    path: str | None = Field(
        None,
        description="Custom path (auto-generated from name if not provided)",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Description of the virtual server",
    )
    tool_mappings: list[ToolMapping] = Field(
        default_factory=list,
        max_length=500,
        description="Tools to map from backend servers (max 500)",
    )
    required_scopes: list[str] = Field(
        default_factory=list,
        description="Scopes required for access",
    )
    tool_scope_overrides: list[ToolScopeOverride] = Field(
        default_factory=list,
        description="Per-tool scope overrides",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Tags for categorization (max 50)",
    )
    supported_transports: list[str] = Field(
        default_factory=lambda: ["streamable-http"],
        description="Supported transport types",
    )


class UpdateVirtualServerRequest(BaseModel):
    """Request model for updating a virtual server."""

    model_config = ConfigDict(populate_by_name=True)

    server_name: str | None = Field(
        None,
        min_length=1,
        max_length=128,
        description="Updated name",
    )
    description: str | None = Field(
        None,
        max_length=2048,
        description="Updated description",
    )
    tool_mappings: list[ToolMapping] | None = Field(
        None,
        description="Updated tool mappings",
    )
    required_scopes: list[str] | None = Field(
        None,
        description="Updated required scopes",
    )
    tool_scope_overrides: list[ToolScopeOverride] | None = Field(
        None,
        description="Updated per-tool scope overrides",
    )
    tags: list[str] | None = Field(
        None,
        description="Updated tags",
    )
    supported_transports: list[str] | None = Field(
        None,
        description="Updated transport types",
    )


class ToggleVirtualServerRequest(BaseModel):
    """Request model for toggling virtual server enabled state."""

    enabled: bool = Field(..., description="New enabled state")


class ToolCatalogEntry(BaseModel):
    """A tool available in the registry, from the global tool catalog.

    Aggregates tool information across all enabled backend servers.
    """

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str = Field(..., description="Tool name")
    server_path: str = Field(..., description="Backend server path")
    server_name: str = Field(default="", description="Backend server display name")
    description: str = Field(default="", description="Tool description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool input parameters",
    )
    available_versions: list[str] = Field(
        default_factory=list,
        description="Available versions of the backend server",
    )


class ResolvedTool(BaseModel):
    """A tool resolved from a virtual server's tool mappings.

    Contains the final tool name (alias or original), its source backend,
    and the full tool metadata for serving in tools/list responses.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="Tool name (alias if set, otherwise original)")
    original_name: str = Field(..., description="Original tool name on backend")
    backend_server_path: str = Field(..., description="Backend server path")
    backend_version: str | None = Field(None, description="Pinned version if set")
    description: str = Field(default="", description="Tool description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool input",
    )
    required_scopes: list[str] = Field(
        default_factory=list,
        description="Scopes required for this specific tool",
    )
