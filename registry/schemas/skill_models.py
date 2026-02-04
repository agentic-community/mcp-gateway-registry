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
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
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
    return datetime.now(timezone.utc)


class VisibilityEnum(str, Enum):
    """Visibility options for skills."""

    PUBLIC = "public"
    PRIVATE = "private"
    GROUP = "group"


class SkillMetadata(BaseModel):
    """Optional metadata for skills."""

    author: Optional[str] = None
    version: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class CompatibilityRequirement(BaseModel):
    """Machine-readable compatibility constraint."""

    type: Literal["product", "tool", "api", "environment"] = Field(
        ...,
        description="Type of requirement"
    )
    target: str = Field(
        ...,
        description="Target identifier (e.g., 'claude-code', 'python>=3.10')"
    )
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    required: bool = Field(
        default=True,
        description="False = optional enhancement"
    )


class ToolReference(BaseModel):
    """Reference to a tool with optional filtering."""

    tool_name: str = Field(
        ...,
        description="Tool name (e.g., 'Read', 'Bash')"
    )
    server_path: Optional[str] = Field(
        None,
        description="MCP server path (e.g., '/servers/claude-tools')"
    )
    version: Optional[str] = None
    capabilities: List[str] = Field(
        default_factory=list,
        description="Capability filters (e.g., ['git:*'])"
    )


class SkillResource(BaseModel):
    """Reference to a skill resource file."""

    path: str = Field(..., description="Relative path from skill root")
    type: Literal["script", "reference", "asset"] = Field(...)
    size_bytes: int = Field(default=0)
    description: Optional[str] = None
    language: Optional[str] = Field(
        None,
        description="Programming language for scripts"
    )


class SkillCard(BaseModel):
    """Full skill profile following Agent Skills specification."""

    model_config = ConfigDict(
        populate_by_name=True
    )

    # Explicit path - immutable after creation
    path: str = Field(
        ...,
        description="Unique skill path (e.g., /skills/pdf-processing)"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Skill name: lowercase alphanumeric and hyphens only"
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="What the skill does and when to use it"
    )

    # URLs with validation
    skill_md_url: HttpUrl = Field(
        ...,
        description="URL to the SKILL.md file in a git repository"
    )
    repository_url: Optional[HttpUrl] = Field(
        None,
        description="URL to the git repository containing the skill"
    )

    # Skill metadata
    license: Optional[str] = Field(
        None,
        description="License name or reference to bundled license file"
    )
    compatibility: Optional[str] = Field(
        None,
        max_length=500,
        description="Human-readable environment requirements"
    )
    requirements: List[CompatibilityRequirement] = Field(
        default_factory=list,
        description="Machine-readable compatibility requirements"
    )
    target_agents: List[str] = Field(
        default_factory=list,
        description="Target coding assistants (e.g., ['claude-code', 'cursor'])"
    )
    metadata: Optional[SkillMetadata] = Field(
        None,
        description="Additional metadata (author, version, etc.)"
    )

    # Tool references
    allowed_tools: List[ToolReference] = Field(
        default_factory=list,
        description="Tools the skill may use with capabilities"
    )

    # Categorization
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and search"
    )

    # Access control
    visibility: VisibilityEnum = Field(
        default=VisibilityEnum.PUBLIC,
        description="Visibility scope"
    )
    allowed_groups: List[str] = Field(
        default_factory=list,
        description="Groups allowed to view (when visibility=group)"
    )
    owner: Optional[str] = Field(
        None,
        description="Owner email/username for private visibility"
    )

    # State
    is_enabled: bool = Field(
        default=True,
        description="Whether the skill is enabled"
    )
    registry_name: str = Field(
        default="local",
        description="Registry this skill belongs to"
    )

    # Content versioning
    content_version: Optional[str] = Field(
        None,
        description="Hash of SKILL.md for cache validation"
    )
    content_updated_at: Optional[datetime] = Field(
        None,
        description="When SKILL.md content was last updated"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=_utc_now
    )
    updated_at: datetime = Field(
        default_factory=_utc_now
    )

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

    model_config = ConfigDict(
        populate_by_name=True
    )

    path: str = Field(..., description="Unique skill path")
    name: str
    description: str
    skill_md_url: str
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    version: Optional[str] = None
    compatibility: Optional[str] = None
    target_agents: List[str] = Field(default_factory=list)
    is_enabled: bool = True
    visibility: VisibilityEnum = VisibilityEnum.PUBLIC
    allowed_groups: List[str] = Field(default_factory=list)
    registry_name: str = "local"


class SkillRegistrationRequest(BaseModel):
    """Request model for skill registration."""

    model_config = ConfigDict(
        populate_by_name=True
    )

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=1024)
    skill_md_url: HttpUrl = Field(..., description="URL to SKILL.md file")
    repository_url: Optional[HttpUrl] = None
    license: Optional[str] = None
    compatibility: Optional[str] = Field(None, max_length=500)
    requirements: List[CompatibilityRequirement] = Field(default_factory=list)
    target_agents: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    allowed_tools: List[ToolReference] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    visibility: VisibilityEnum = Field(default=VisibilityEnum.PUBLIC)
    allowed_groups: List[str] = Field(default_factory=list)

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
    match_context: Optional[str] = Field(
        None,
        description="Snippet showing where query matched"
    )
    required_mcp_servers: List[str] = Field(
        default_factory=list,
        description="MCP servers providing required tools"
    )
    missing_tools: List[str] = Field(
        default_factory=list,
        description="Tools not available in registry"
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
    tags: List[str] = Field(default_factory=list)
    compatibility: Optional[str] = None
    target_agents: List[str] = Field(default_factory=list)


class SkillTier2_Instructions(BaseModel):
    """Tier 2: Loaded when activated, <5000 tokens."""

    skill_md_body: str = Field(..., description="Full SKILL.md content")
    metadata: Optional[SkillMetadata] = None
    allowed_tools: List[ToolReference] = Field(default_factory=list)
    requirements: List[CompatibilityRequirement] = Field(default_factory=list)


class SkillTier3_Resources(BaseModel):
    """Tier 3: Loaded on-demand."""

    available_resources: List[SkillResource] = Field(default_factory=list)


class SkillResourceManifest(BaseModel):
    """Manifest of available resources for a skill."""

    scripts: List[SkillResource] = Field(default_factory=list)
    references: List[SkillResource] = Field(default_factory=list)
    assets: List[SkillResource] = Field(default_factory=list)


class ToolValidationResult(BaseModel):
    """Result of tool availability validation."""

    all_available: bool
    missing_tools: List[str] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)
    mcp_servers_required: List[str] = Field(default_factory=list)


class DiscoveryResponse(BaseModel):
    """Response for coding assistant discovery endpoint."""

    skills: List[SkillTier1_Metadata]
    total_count: int
    page: int = 0
    page_size: int = 100
