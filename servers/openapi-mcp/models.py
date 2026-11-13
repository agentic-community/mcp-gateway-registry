"""Models for what the registry sends the container and what it returns."""

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
)

SLUG_PATTERN: str = r"^[a-z0-9]+(-[a-z0-9]+)*$"
TARGET_NAME_PATTERN: str = r"^[a-z][a-z0-9]{1,23}$"

AuthScheme = Literal["none", "api_key_header", "api_key_query", "bearer", "basic"]
TargetStatusValue = Literal["ok", "error"]


class TargetOptions(BaseModel):
    """How one target's operations become tools. Shared by preview and push."""

    name: str = Field(..., pattern=TARGET_NAME_PATTERN)
    tool_name_overrides: dict[str, str] = Field(default_factory=dict)
    description_overrides: dict[str, str] = Field(default_factory=dict)
    include_tags: list[str] = Field(default_factory=list)
    include_operations: list[str] = Field(default_factory=list)
    keep_output_schemas: bool = False


class HubTargetState(TargetOptions):
    """One REST target as the registry pushes it. The credential is plaintext in memory."""

    description: str = ""
    base_url: str
    spec: dict[str, Any]
    auth_scheme: AuthScheme = "none"
    auth_header_name: str | None = None
    auth_query_param: str | None = None
    auth_username: str | None = None
    auth_credential: SecretStr | None = None


class HubServerState(BaseModel):
    """The full state of one OpenAPI MCP server."""

    slug: str = Field(..., pattern=SLUG_PATTERN)
    name: str
    description: str = ""
    generation: int = Field(..., ge=0)
    targets: list[HubTargetState] = Field(default_factory=list)


class PreviewRequest(TargetOptions):
    """Ask the container what tools a spec would produce, with no network I/O."""

    target_name: str = Field(..., pattern=TARGET_NAME_PATTERN)
    name: str = Field(default="preview", pattern=TARGET_NAME_PATTERN)
    spec: dict[str, Any]


class TargetStatus(BaseModel):
    """How one target fared when its hub was built."""

    status: TargetStatusValue
    tools: list[str] = Field(default_factory=list)
    error: str | None = None


class ApplyResponse(BaseModel):
    """Response to PUT /internal/servers/{slug}."""

    slug: str
    generation: int
    targets: dict[str, TargetStatus]


class PreviewResponse(BaseModel):
    """Response to POST /internal/preview."""

    tools: list[str]
    tags: dict[str, int]
    warnings: list[str]


class ServerStatus(BaseModel):
    """One running hub in GET /internal/status."""

    generation: int
    targets: dict[str, TargetStatus]


class StatusResponse(BaseModel):
    """Response to GET /internal/status."""

    servers: dict[str, ServerStatus]
    started_at: float
