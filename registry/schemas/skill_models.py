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
from uuid import UUID, uuid4

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
    type: Literal["script", "reference", "asset", "agent"] = Field(...)
    size_bytes: int = Field(default=0)
    description: str | None = None
    language: str | None = Field(None, description="Programming language for scripts")


class SkillResourceManifest(BaseModel):
    """Manifest of available resources for a skill."""

    scripts: list[SkillResource] = Field(default_factory=list)
    references: list[SkillResource] = Field(default_factory=list)
    assets: list[SkillResource] = Field(default_factory=list)
    agents: list[SkillResource] = Field(default_factory=list)


class FileHash(BaseModel):
    """SHA-256 hash for a single file in the skill directory."""

    path: str = Field(..., description="Relative path (e.g. 'SKILL.md' or 'references/arch.md')")
    sha256: str = Field(..., description="Full SHA-256 hex digest of the file content")
    size_bytes: int = Field(default=0, description="File size at hash time")


class ContentIntegrity(BaseModel):
    """Content integrity record computed at registration or refresh.

    Stores per-file SHA-256 hashes and a composite hash derived from all
    individual hashes, enabling drift detection without re-fetching content.
    """

    composite_hash: str = Field(
        ..., description="SHA-256 of the sorted, concatenated per-file hashes"
    )
    file_hashes: list[FileHash] = Field(default_factory=list)
    computed_at: datetime = Field(default_factory=_utc_now)
    drift_detected: bool = Field(
        default=False, description="True when a drift check found content differs from this baseline"
    )
    last_drift_check: datetime | None = Field(
        None, description="When drift was last checked"
    )
    drifted_files: list[str] = Field(
        default_factory=list, description="Paths of files that changed since baseline"
    )


class SkillCard(BaseModel):
    """Full skill profile following Agent Skills specification."""

    model_config = ConfigDict(populate_by_name=True)

    # Unique identifier
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier (UUID) for this skill",
    )

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
    skill_md_content: str | None = Field(
        None,
        description="Inline SKILL.md content for federated skills (stored in DB instead of URL fetch)",
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

    # Source authentication (for private Git repos)
    # Literal keeps the wire-format strings compatible with existing clients
    # while rejecting unsupported schemes at validation time.  Adding a new
    # scheme requires updating both this list and SkillRegistrationRequest.
    auth_scheme: Literal["none", "global_credentials", "bearer", "api_key"] = Field(
        default="none",
        description="Auth scheme for fetching SKILL.md: none, global_credentials, bearer, api_key",
    )
    auth_credential_encrypted: str | None = Field(
        None,
        description="Encrypted credential for SKILL.md fetching",
    )
    auth_header_name: str | None = Field(
        None,
        description="Custom header name for credential (default: Authorization for bearer, PRIVATE-TOKEN for api_key)",
    )
    credential_updated_at: datetime | None = Field(
        None, description="When the credential was last updated"
    )

    # Resource manifest (companion files: references, scripts, agents, assets)
    resource_manifest: SkillResourceManifest | None = Field(
        None, description="Manifest of companion resource files discovered in the skill directory"
    )

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

    # Content integrity (full hash of SKILL.md + all resources)
    content_integrity: ContentIntegrity | None = Field(
        None, description="Per-file hashes and composite hash for drift detection"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    # Registry Card fields for federation
    status: str = Field(
        default="active",
        description="Lifecycle status (default: active for existing assets)",
    )
    source_created_at: datetime | None = Field(
        None, description="Creation timestamp from federated source"
    )
    source_updated_at: datetime | None = Field(
        None, description="Last update timestamp from federated source"
    )
    external_tags: list[str] = Field(
        default_factory=list, description="Tags from external/federated registries"
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

    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(..., description="Unique identifier (UUID) for this skill")
    path: str = Field(..., description="Unique skill path")
    name: str
    description: str
    skill_md_url: str
    skill_md_raw_url: str | None = Field(None, description="Raw URL for fetching SKILL.md content")
    repository_url: HttpUrl | None = Field(
        None, description="URL to the git repository containing the skill"
    )
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    version: str | None = None
    metadata: SkillMetadata | None = None
    compatibility: str | None = None
    target_agents: list[str] = Field(default_factory=list)
    is_enabled: bool = True
    visibility: VisibilityEnum = VisibilityEnum.PUBLIC
    allowed_groups: list[str] = Field(default_factory=list)
    registry_name: str = "local"
    owner: str | None = Field(
        None, description="Owner email/username for private visibility access control"
    )
    auth_scheme: Literal["none", "global_credentials", "bearer", "api_key"] = Field(
        default="none",
        description="Auth scheme for fetching SKILL.md: none, global_credentials, bearer, api_key",
    )
    auth_header_name: str | None = Field(
        None,
        description="Custom header name for credential (default: Authorization for bearer, PRIVATE-TOKEN for api_key)",
    )
    num_stars: float = Field(default=0.0, ge=0.0, le=5.0, description="Average rating (1-5 stars)")
    health_status: Literal["healthy", "unhealthy", "unknown"] = Field(
        default="unknown", description="Health status from last SKILL.md accessibility check"
    )
    last_checked_time: datetime | None = Field(None, description="When health was last checked")

    # Registry Card fields for federation
    status: str = Field(
        default="active",
        description="Lifecycle status (default: active for existing assets)",
    )
    source_created_at: datetime | None = Field(
        None, description="Creation timestamp from federated source"
    )
    source_updated_at: datetime | None = Field(
        None, description="Last update timestamp from federated source"
    )
    external_tags: list[str] = Field(
        default_factory=list, description="Tags from external/federated registries"
    )


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
    status: str = Field(
        default="draft",
        description="Lifecycle status (default: draft). Allowed: active, deprecated, draft, beta",
    )
    auth_scheme: Literal["none", "global_credentials", "bearer", "api_key"] = Field(
        default="none",
        description="Auth scheme for fetching SKILL.md from private repos: none, global_credentials, bearer, api_key",
    )
    auth_credential: str | None = Field(
        None,
        description="Credential (token/key) for fetching SKILL.md; encrypted before storage, never persisted in plaintext",
    )
    auth_header_name: str | None = Field(
        None,
        description="Custom header name (default: Authorization for bearer, PRIVATE-TOKEN for api_key)",
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
    status: str = Field(
        default="active",
        description="Lifecycle status (default: active for existing assets)",
    )


class SkillTier2_Instructions(BaseModel):
    """Tier 2: Loaded when activated, <5000 tokens."""

    skill_md_body: str = Field(..., description="Full SKILL.md content")
    metadata: SkillMetadata | None = None
    allowed_tools: list[ToolReference] = Field(default_factory=list)
    requirements: list[CompatibilityRequirement] = Field(default_factory=list)


class SkillTier3_Resources(BaseModel):
    """Tier 3: Loaded on-demand."""

    available_resources: list[SkillResource] = Field(default_factory=list)


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
