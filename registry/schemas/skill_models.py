"""
Agent Skills data models following agentskills.io specification.

All recommendations incorporated:
- VisibilityEnum for type-safe visibility
- Explicit path field in SkillCard
- HttpUrl validation for URLs
- ToolReference for allowed_tools linking
- CompatibilityRequirement for machine-readable requirements
- Progressive disclosure tier models
- Owner field for access control
- Content versioning fields
"""

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
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


class VisibilityEnum(str, Enum):
    """Visibility options for skills."""

    PUBLIC = "public"
    PRIVATE = "private"
    GROUP = "group"


class SkillMetadata(BaseModel):
    """Optional metadata for skills."""

    author: str | None = None
    version: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class CompatibilityRequirement(BaseModel):
    """Machine-readable compatibility constraint."""

    type: Literal["product", "tool", "api", "environment"] = Field(
        ..., description="Type of requirement"
    )
    target: str = Field(..., description="Target identifier (e.g., 'claude-code', 'python>=3.10')")
    min_version: str | None = None
    max_version: str | None = None
    required: bool = Field(default=True, description="False = optional enhancement")


class ToolReference(BaseModel):
    """Reference to a tool with optional filtering."""

    tool_name: str = Field(..., description="Tool name (e.g., 'Read', 'Bash')")
    server_path: str | None = Field(
        None, description="MCP server path (e.g., '/servers/claude-tools')"
    )
    version: str | None = None
    capabilities: list[str] = Field(
        default_factory=list, description="Capability filters (e.g., ['git:*'])"
    )


class SkillResource(BaseModel):
    """Reference to a skill resource file."""

    path: str = Field(..., description="Relative path from skill root")
    type: Literal["script", "reference", "asset"] = Field(...)
    size_bytes: int = Field(default=0)
    description: str | None = None
    language: str | None = Field(None, description="Programming language for scripts")


class SkillCard(BaseModel):
    """Full skill profile following Agent Skills specification."""

    model_config = ConfigDict(populate_by_name=True)

    # Explicit path - immutable after creation
    path: str = Field(..., description="Unique skill path (e.g., /skills/pdf-processing)")
    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Skill name: lowercase alphanumeric and hyphens only",
    )
    description: str = Field(
        ..., min_length=1, max_length=1024, description="What the skill does and when to use it"
    )

    # URLs with validation
    skill_md_url: HttpUrl = Field(
        ..., description="URL to the SKILL.md file as provided by the user"
    )
    skill_md_raw_url: HttpUrl | None = Field(
        None,
        description="Raw URL for fetching SKILL.md content (auto-translated from skill_md_url)",
    )
    repository_url: HttpUrl | None = Field(
        None, description="URL to the git repository containing the skill"
    )

    # Skill metadata
    license: str | None = Field(
        None, description="License name or reference to bundled license file"
    )
    compatibility: str | None = Field(
        None, max_length=500, description="Human-readable environment requirements"
    )
    requirements: list[CompatibilityRequirement] = Field(
        default_factory=list, description="Machine-readable compatibility requirements"
    )
    target_agents: list[str] = Field(
        default_factory=list,
        description="Target coding assistants (e.g., ['claude-code', 'cursor'])",
    )
    metadata: SkillMetadata | None = Field(
        None, description="Additional metadata (author, version, etc.)"
    )

    # Tool references
    allowed_tools: list[ToolReference] = Field(
        default_factory=list, description="Tools the skill may use with capabilities"
    )

    # Categorization
    tags: list[str] = Field(default_factory=list, description="Tags for categorization and search")

    # Access control
    visibility: VisibilityEnum = Field(
        default=VisibilityEnum.PUBLIC, description="Visibility scope"
    )
    allowed_groups: list[str] = Field(
        default_factory=list, description="Groups allowed to view (when visibility=group)"
    )
    owner: str | None = Field(None, description="Owner email/username for private visibility")

    # State
    is_enabled: bool = Field(default=True, description="Whether the skill is enabled")
    registry_name: str = Field(default="local", description="Registry this skill belongs to")
    health_status: Literal["healthy", "unhealthy", "unknown"] = Field(
        default="unknown", description="Health status from last SKILL.md accessibility check"
    )
    last_checked_time: datetime | None = Field(None, description="When health was last checked")

    # Rating
    num_stars: float = Field(default=0.0, ge=0.0, le=5.0, description="Average rating (1-5 stars)")
    rating_details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of individual user ratings with user and rating fields",
    )

    # Content versioning
    content_version: str | None = Field(None, description="Hash of SKILL.md for cache validation")
    content_updated_at: datetime | None = Field(
        None, description="When SKILL.md content was last updated"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("name")
    @classmethod
    def validate_name(
        cls,
        v: str,
    ) -> str:
        """Validate name follows Agent Skills spec."""
        import re

        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError(
                "Name must be lowercase alphanumeric with single hyphens, "
                "not starting or ending with hyphen"
            )
        return v

    @field_validator("path")
    @classmethod
    def validate_path(
        cls,
        v: str,
    ) -> str:
        """Validate path format."""
        if not v.startswith("/skills/"):
            raise ValueError("Path must start with /skills/")
        return v


class SkillInfo(BaseModel):
    """Lightweight skill summary for listings."""

    model_config = ConfigDict(populate_by_name=True)

    path: str = Field(..., description="Unique skill path")
    name: str
    description: str
    skill_md_url: str
    skill_md_raw_url: str | None = Field(None, description="Raw URL for fetching SKILL.md content")
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    version: str | None = None
    compatibility: str | None = None
    target_agents: list[str] = Field(default_factory=list)
    is_enabled: bool = True
    visibility: VisibilityEnum = VisibilityEnum.PUBLIC
    allowed_groups: list[str] = Field(default_factory=list)
    registry_name: str = "local"
    owner: str | None = Field(
        None, description="Owner email/username for private visibility access control"
    )
    num_stars: float = Field(default=0.0, ge=0.0, le=5.0, description="Average rating (1-5 stars)")
    health_status: Literal["healthy", "unhealthy", "unknown"] = Field(
        default="unknown", description="Health status from last SKILL.md accessibility check"
    )
    last_checked_time: datetime | None = Field(None, description="When health was last checked")


class SkillRegistrationRequest(BaseModel):
    """Request model for skill registration."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=1024)
    skill_md_url: HttpUrl = Field(..., description="URL to SKILL.md file")
    repository_url: HttpUrl | None = None
    version: str | None = Field(None, max_length=32, description="Skill version (e.g., 1.0.0)")
    license: str | None = None
    compatibility: str | None = Field(None, max_length=500)
    requirements: list[CompatibilityRequirement] = Field(default_factory=list)
    target_agents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    allowed_tools: list[ToolReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    visibility: VisibilityEnum = Field(default=VisibilityEnum.PUBLIC)
    allowed_groups: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(
        cls,
        v: str,
    ) -> str:
        """Validate name follows Agent Skills spec."""
        import re

        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError(
                "Name must be lowercase alphanumeric with single hyphens, "
                "not starting or ending with hyphen"
            )
        return v


class SkillSearchResult(BaseModel):
    """Skill search result with relevance score."""

    skill: SkillInfo
    score: float = Field(description="Relevance score 0-1")
    match_context: str | None = Field(None, description="Snippet showing where query matched")
    required_mcp_servers: list[str] = Field(
        default_factory=list, description="MCP servers providing required tools"
    )
    missing_tools: list[str] = Field(
        default_factory=list, description="Tools not available in registry"
    )


class ToggleStateRequest(BaseModel):
    """Request model for toggling skill state."""

    enabled: bool = Field(..., description="New enabled state")


# Progressive Disclosure Models


class SkillTier1_Metadata(BaseModel):
    """Tier 1: Always available, ~100 tokens."""

    path: str
    name: str
    description: str
    skill_md_url: str
    skill_md_raw_url: str | None = Field(None, description="Raw URL for fetching SKILL.md content")
    tags: list[str] = Field(default_factory=list)
    compatibility: str | None = None
    target_agents: list[str] = Field(default_factory=list)


class SkillTier2_Instructions(BaseModel):
    """Tier 2: Loaded when activated, <5000 tokens."""

    skill_md_body: str = Field(..., description="Full SKILL.md content")
    metadata: SkillMetadata | None = None
    allowed_tools: list[ToolReference] = Field(default_factory=list)
    requirements: list[CompatibilityRequirement] = Field(default_factory=list)


class SkillTier3_Resources(BaseModel):
    """Tier 3: Loaded on-demand."""

    available_resources: list[SkillResource] = Field(default_factory=list)


class SkillResourceManifest(BaseModel):
    """Manifest of available resources for a skill."""

    scripts: list[SkillResource] = Field(default_factory=list)
    references: list[SkillResource] = Field(default_factory=list)
    assets: list[SkillResource] = Field(default_factory=list)


class ToolValidationResult(BaseModel):
    """Result of tool availability validation."""

    all_available: bool
    missing_tools: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    mcp_servers_required: list[str] = Field(default_factory=list)


class DiscoveryResponse(BaseModel):
    """Response for coding assistant discovery endpoint."""

    skills: list[SkillTier1_Metadata]
    total_count: int
    page: int = 0
    page_size: int = 100
