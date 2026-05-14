import re
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from registry.constants import DeploymentType, LocalRuntimeType, TransportType
from registry.schemas.agent_models import AgentProvider
from registry.schemas.registry_card import LifecycleStatus

_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


_RFC7230_TOKEN_RE = re.compile(r"[A-Za-z0-9!#$%&'*+\-.^_`|~]+")


class CustomHeader(BaseModel):
    """A single user-defined HTTP header attached to an MCP server."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="HTTP header name. Must be a valid RFC 7230 token.",
    )
    value: str = Field(
        ...,
        max_length=4096,
        description="HTTP header value. Stored encrypted; never returned in list responses.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _RFC7230_TOKEN_RE.fullmatch(v):
            raise ValueError("Header name contains invalid characters (must be RFC 7230 token)")
        return v

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: str) -> str:
        if "\r" in v or "\n" in v:
            raise ValueError("Header value cannot contain CR or LF")
        return v


class CustomHeaderEncrypted(BaseModel):
    """Stored form: header name + Fernet-encrypted value."""

    name: str
    value_encrypted: str


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


def _validate_deployment_invariants(obj: Any) -> None:
    """Enforce remote-vs-local field invariants on a server-like object.

    Used by ServerInfo's @model_validator. The object must expose:
    deployment, local_runtime, proxy_pass_url, mcp_endpoint, sse_endpoint,
    auth_scheme. `versions` (multi-version routing) is checked via getattr so
    callers that don't have such a field don't need to define one.

    For deployment='local' the helper also forces transport='stdio' and
    supported_transports=['stdio'] on the object.
    """
    if obj.deployment == DeploymentType.LOCAL:
        if obj.local_runtime is None:
            raise ValueError("deployment='local' requires local_runtime")
        if obj.proxy_pass_url is not None:
            raise ValueError("deployment='local' must not set proxy_pass_url")
        if obj.mcp_endpoint is not None:
            raise ValueError("deployment='local' must not set mcp_endpoint")
        if obj.sse_endpoint is not None:
            raise ValueError("deployment='local' must not set sse_endpoint")
        if obj.auth_scheme not in ("none", ""):
            raise ValueError(
                "deployment='local' must use auth_scheme='none' "
                "(local servers handle auth via env vars on the user's machine)"
            )
        if getattr(obj, "versions", None) is not None:
            raise ValueError("deployment='local' does not support multi-version routing")
        obj.transport = TransportType.STDIO
        obj.supported_transports = [TransportType.STDIO]
    else:
        # deployment == "remote"
        if obj.local_runtime is not None:
            raise ValueError("deployment='remote' must not set local_runtime")
        if not obj.proxy_pass_url:
            raise ValueError("deployment='remote' requires proxy_pass_url")


class LocalRuntime(BaseModel):
    """How to launch a local (stdio) MCP server on a developer's machine.

    The registry stores the recipe; it does NOT run the server. Health checks
    do not apply. The recipe is emitted as IDE config (Claude Code, Cursor, etc.)
    via the Connect modal.
    """

    type: Literal["npx", "docker", "uvx", "command"] = Field(
        ...,
        description=(
            "Launcher type. npx/uvx: package name. docker: image ref. "
            "command: raw executable path (admin-only, highest trust)."
        ),
    )
    package: str = Field(
        ...,
        min_length=1,
        description="Package name, image reference, or command path depending on `type`.",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Argv-style arguments passed to the launcher (no shell interpolation).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Environment variables. Values may be literal or ${VAR} placeholders. "
            "Literal-looking secrets are rejected at registration time."
        ),
    )
    required_env: list[str] = Field(
        default_factory=list,
        description=(
            "Env var names the user MUST provide at connect time. MUST NOT overlap with `env` keys."
        ),
    )

    # docker-only
    image_digest: str | None = Field(
        default=None,
        description="Pinned image digest, e.g. 'sha256:abc...'. Encouraged for supply-chain hardening.",
    )
    platforms: list[str] | None = Field(
        default=None,
        description="Supported platforms, e.g. ['linux/amd64', 'linux/arm64'].",
    )

    # npx/uvx-only
    version: str | None = Field(
        default=None,
        description="Package version pin, e.g. '1.2.0'. Encouraged.",
    )

    @model_validator(mode="after")
    def _validate_runtime_consistency(self) -> "LocalRuntime":
        """Validate runtime fields and required_env disjointness from env."""
        # required_env keys must not overlap with env keys (kiro round-1 feedback)
        overlap = set(self.required_env) & set(self.env.keys())
        if overlap:
            raise ValueError(f"required_env keys must not also appear in env: {sorted(overlap)}")

        # platforms only meaningful for docker
        if self.platforms is not None and self.type != LocalRuntimeType.DOCKER:
            raise ValueError("platforms is only valid for docker runtime")

        # image_digest only meaningful for docker
        if self.image_digest is not None and self.type != LocalRuntimeType.DOCKER:
            raise ValueError("image_digest is only valid for docker runtime")

        # image_digest format check (only when provided): require the full
        # 'sha256:<64 hex>' shape so malformed digests fail at registration
        # rather than silently propagating to clients.
        if self.image_digest is not None and not _IMAGE_DIGEST_RE.fullmatch(self.image_digest):
            raise ValueError(
                f"image_digest must match 'sha256:<64 hex chars>', got: {self.image_digest!r}"
            )

        return self


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
        description="Federation visibility: public (shared with all peers), group-restricted (shared with allowed_groups only), or private (never shared). 'internal' is accepted as an alias for 'private'.",
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

    # Custom HTTP headers (encrypted values, names public)
    custom_header_names: list[str] = Field(
        default_factory=list,
        description="Names of custom HTTP headers defined for this server.",
    )
    custom_headers_encrypted: list[CustomHeaderEncrypted] | None = Field(
        default=None,
        description="List of {name, value_encrypted} pairs. Never serialized to API consumers.",
    )
    custom_headers_updated_at: str | None = Field(
        default=None,
        description="ISO timestamp of last custom-headers update.",
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
    deployment: Literal["remote", "local"] = Field(
        default=DeploymentType.REMOTE,
        description=(
            "Deployment model: 'remote' (HTTP-reachable, registry proxies) or "
            "'local' (stdio, runs on developer's machine via launch recipe)."
        ),
    )
    local_runtime: LocalRuntime | None = Field(
        default=None,
        description="Launch recipe. Required when deployment='local', forbidden otherwise.",
    )
    registered_by: str | None = Field(
        default=None,
        description=(
            "Username of the user who registered this server. Audit trail; "
            "load-bearing for local servers (executable recipe approval). "
            "Records the ORIGINAL registrant only — edits do not update this "
            "field. The general audit log captures who last touched the entry."
        ),
    )

    @field_validator("visibility")
    @classmethod
    def _validate_visibility(
        cls,
        v: str,
    ) -> str:
        """Validate and normalize visibility value.

        Accepts "internal" as alias for "private" and "group" as alias
        for "group-restricted" for backward compatibility.
        """
        from registry.utils.visibility import validate_visibility

        return validate_visibility(v)

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

    @model_validator(mode="after")
    def _validate_deployment_consistency(self) -> "ServerInfo":
        """Enforce remote/local field invariants. See _validate_deployment_invariants."""
        _validate_deployment_invariants(self)
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
        description="Federation visibility: public (shared with all peers), group-restricted (shared with allowed_groups only), or private (never shared). 'internal' is accepted as an alias for 'private'.",
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
