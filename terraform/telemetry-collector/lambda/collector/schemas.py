"""
Pydantic validation schemas for telemetry events.

Matches schemas from registry/core/telemetry.py (issue #558 client implementation).

NOTE on embeddings_backend_kind: keep the regex allowlist here in sync with
_BACKEND_KIND_PATTERNS in registry/core/telemetry.py. The return values of
_derive_embeddings_backend_kind() must be a subset of the values this regex
accepts.

NOTE on cloud_detection_method: keep _CLOUD_DETECTION_METHOD_PATTERN in sync
with the _DETECTION_METHOD_* constants in registry/core/telemetry.py.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Allowlist of values for embeddings_backend_kind across both StartupEvent
# and HeartbeatEvent. Kept as a module-level constant so both models can
# reference the same regex.
_EMBEDDINGS_BACKEND_KIND_PATTERN = (
    r"^(sentence-transformers|bedrock|openai|azure-openai|voyage|cohere|other|unknown)$"
)

# Allowlist of values for cloud_detection_method. Added in schema v3
# (issue #986). Extended in issue #1120 with explicit, operator_declared,
# operator_declined.
_CLOUD_DETECTION_METHOD_PATTERN = (
    r"^(env|dmi|ecs_meta|k8s_heuristic|imds|"
    r"explicit|operator_declared|operator_declined|unknown)$"
)

# Allowlist of values for internal_deployment_type. Added in schema v5
# (issue #1216). Keep in sync with InternalDeploymentType in registry/core/config.py.
_INTERNAL_DEPLOYMENT_TYPE_PATTERN = r"^(none|dev|workshop|other)$"


def _check_cloud_detection_consistency(cloud: str, method: str | None) -> None:
    """Reject payloads where cloud and cloud_detection_method are inconsistent.

    Absent method (None) is always acceptable for backwards compatibility with
    pre-v3 clients.
    """
    if method is None:
        return
    if method == "ecs_meta" and cloud != "aws":
        raise ValueError(
            f"cloud_detection_method=ecs_meta requires cloud=aws, got cloud={cloud!r}"
        )
    if method == "unknown" and cloud != "unknown":
        raise ValueError(
            f"cloud_detection_method=unknown requires cloud=unknown, got cloud={cloud!r}"
        )
    if cloud == "unknown" and method not in ("unknown", "operator_declined", None):
        raise ValueError(
            f"cloud=unknown requires cloud_detection_method in (unknown, operator_declined, None), "
            f"got method={method!r}"
        )
    if method == "operator_declared" and cloud not in ("on_premises", "other"):
        raise ValueError(
            f"cloud_detection_method=operator_declared requires cloud in "
            f"{{on_premises, other}}, got cloud={cloud!r}"
        )
    if method == "operator_declined" and cloud != "unknown":
        raise ValueError(
            f"cloud_detection_method=operator_declined requires cloud=unknown, "
            f"got cloud={cloud!r}"
        )
    if cloud in ("on_premises", "other") and method not in ("operator_declared", "explicit"):
        raise ValueError(
            f"cloud={cloud!r} requires cloud_detection_method in "
            f"{{operator_declared, explicit}}, got method={method!r}"
        )


class StartupEvent(BaseModel):
    """
    Startup telemetry event (Tier 1 - opt-out, default ON).

    Sent once at registry startup to track:
    - Version distribution
    - Python version compatibility
    - OS and architecture
    - Deployment configurations
    - Auth provider usage
    """

    event: str = Field(..., pattern="^startup$")
    registry_id: str | None = Field(default=None, max_length=36, description="Registry card UUID")
    v: str = Field(..., min_length=1, max_length=200, description="Registry version")
    py: str = Field(..., pattern=r"^\d+\.\d+$", description="Python version (major.minor)")
    os: str = Field(..., pattern="^(linux|darwin|windows)$", description="Operating system")
    arch: str = Field(..., min_length=1, max_length=20, description="CPU architecture")
    cloud: str = Field(
        default="unknown",
        pattern="^(aws|gcp|azure|on_premises|other|unknown)$",
        description="Cloud provider",
    )
    compute: str = Field(
        default="unknown",
        pattern="^(ecs|eks|kubernetes|docker|podman|ec2|vm|unknown)$",
        description="Compute platform",
    )
    mode: str = Field(
        ...,
        pattern="^(with-gateway|registry-only)$",
        description="Deployment mode",
    )
    registry_mode: str = Field(
        ...,
        pattern="^(full|skills-only|mcp-servers-only|agents-only)$",
        description="Registry operating mode",
    )
    storage: str = Field(
        ...,
        pattern="^(file|documentdb|mongodb-ce|mongodb|mongodb-atlas)$",
        description="Storage backend",
    )
    auth: str = Field(..., min_length=1, max_length=50, description="Auth provider")
    federation: bool = Field(..., description="Federation enabled")
    search_queries_total: int = Field(
        default=0, ge=0, description="Lifetime semantic search query count"
    )
    search_queries_24h: int = Field(default=0, ge=0, description="Search queries in last 24 hours")
    search_queries_1h: int = Field(default=0, ge=0, description="Search queries in last hour")
    # Embeddings telemetry (added in schema v2, optional for backward compat)
    embeddings_provider: str | None = Field(
        default=None,
        max_length=100,
        description="Embeddings code path (sentence-transformers or litellm). Added in schema v2.",
    )
    embeddings_backend_kind: str | None = Field(
        default=None,
        pattern=_EMBEDDINGS_BACKEND_KIND_PATTERN,
        description=("Derived coarse-grained embeddings backend category. Added in schema v2."),
    )
    cloud_detection_method: str | None = Field(
        default=None,
        pattern=_CLOUD_DETECTION_METHOD_PATTERN,
        description=(
            "How the cloud value was detected. Added in schema v3 (issue #986). "
            "None for pre-v3 clients."
        ),
    )
    # Internal/workshop deployment classification (added in schema v5, issue #1216).
    # Optional so pre-v5 clients (which omit these) still validate.
    internal_only_deployment: bool | None = Field(
        default=None,
        description="Internal/workshop deployment flag. Added in schema v5.",
    )
    internal_deployment_type: str | None = Field(
        default=None,
        pattern=_INTERNAL_DEPLOYMENT_TYPE_PATTERN,
        description="Internal deployment classification. Added in schema v5.",
    )
    ts: str = Field(..., description="ISO 8601 timestamp")

    @field_validator("ts")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {e}") from e
        return v

    @model_validator(mode="after")
    def _validate_cloud_detection_consistency(self) -> "StartupEvent":
        """Reject payloads where cloud and cloud_detection_method disagree."""
        _check_cloud_detection_consistency(self.cloud, self.cloud_detection_method)
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event": "startup",
                "registry_id": "c546a650-8af9-4721-9efb-7df221b2a0d9",
                "v": "1.0.22",
                "py": "3.12",
                "os": "linux",
                "arch": "x86_64",
                "cloud": "aws",
                "cloud_detection_method": "imds",
                "compute": "ecs",
                "mode": "with-gateway",
                "registry_mode": "full",
                "storage": "documentdb",
                "auth": "keycloak",
                "federation": True,
                "embeddings_provider": "litellm",
                "embeddings_backend_kind": "bedrock",
                "search_queries_total": 150,
                "search_queries_24h": 12,
                "search_queries_1h": 3,
                "ts": "2026-03-18T00:00:00Z",
            }
        }
    )


class HeartbeatEvent(BaseModel):
    """
    Heartbeat telemetry event (Tier 2 - opt-in, default OFF).

    Sent every 24 hours when opted in to track:
    - Aggregate counts (servers, agents, skills, peers)
    - Search backend usage
    - Embeddings provider
    - Instance uptime
    - Deployment-shape fields (auth, arch, os, py, mode, registry_mode,
      storage, federation) -- added in schema v4. Pre-v4 clients omit
      these and the report's analyzer treats the absence as "unknown
      auth/arch/etc" rather than mislabeling the instance.
    """

    event: str = Field(..., pattern="^heartbeat$")
    registry_id: str | None = Field(default=None, max_length=36, description="Registry card UUID")
    v: str = Field(..., min_length=1, max_length=200, description="Registry version")
    # Deployment-shape fields (schema v4+). Optional because pre-v4 clients
    # (the existing fleet) don't send them; the heartbeat schema before v4
    # only carried runtime metrics. By accepting them here, an instance whose
    # last startup event predates the report window can still contribute its
    # auth/arch/etc to the analyzer instead of falling into the unknown
    # bucket. See registry/core/telemetry.py:_build_heartbeat_payload.
    py: str | None = Field(
        default=None,
        pattern=r"^\d+\.\d+$",
        description="Python version (major.minor). Added in schema v4.",
    )
    os: str | None = Field(
        default=None,
        pattern="^(linux|darwin|windows)$",
        description="Operating system. Added in schema v4.",
    )
    arch: str | None = Field(
        default=None,
        max_length=20,
        description="CPU architecture. Added in schema v4.",
    )
    mode: str | None = Field(
        default=None,
        pattern="^(with-gateway|registry-only)$",
        description="Deployment mode. Added in schema v4.",
    )
    registry_mode: str | None = Field(
        default=None,
        pattern="^(full|skills-only|mcp-servers-only|agents-only)$",
        description="Registry operating mode. Added in schema v4.",
    )
    storage: str | None = Field(
        default=None,
        pattern="^(file|documentdb|mongodb-ce)$",
        description="Storage backend. Added in schema v4.",
    )
    auth: str | None = Field(
        default=None,
        max_length=50,
        description="Auth provider. Added in schema v4.",
    )
    federation: bool | None = Field(
        default=None,
        description="Federation enabled. Added in schema v4.",
    )
    cloud: str = Field(
        default="unknown",
        pattern="^(aws|gcp|azure|on_premises|other|unknown)$",
        description="Cloud provider",
    )
    compute: str = Field(
        default="unknown",
        pattern="^(ecs|eks|kubernetes|docker|podman|ec2|vm|unknown)$",
        description="Compute platform",
    )
    servers_count: int = Field(..., ge=0, description="Number of registered MCP servers")
    agents_count: int = Field(..., ge=0, description="Number of registered agents")
    skills_count: int = Field(..., ge=0, description="Number of registered skills")
    peers_count: int = Field(..., ge=0, description="Number of federated peers")
    search_backend: str = Field(
        ...,
        pattern="^(faiss|documentdb)$",
        description="Search backend type",
    )
    # Schema v1 required this field; schema v2+ keeps it for backward compat
    # but relaxes to optional so future clients can omit it symmetrically
    # with StartupEvent. In practice v1.0.22+ always sets it.
    embeddings_provider: str | None = Field(
        default=None, max_length=100, description="Embeddings code path"
    )
    embeddings_backend_kind: str | None = Field(
        default=None,
        pattern=_EMBEDDINGS_BACKEND_KIND_PATTERN,
        description=("Derived coarse-grained embeddings backend category. Added in schema v2."),
    )
    cloud_detection_method: str | None = Field(
        default=None,
        pattern=_CLOUD_DETECTION_METHOD_PATTERN,
        description=(
            "How the cloud value was detected. Added in schema v3 (issue #986). "
            "None for pre-v3 clients."
        ),
    )
    uptime_hours: int = Field(..., ge=0, description="Instance uptime in hours")
    search_queries_total: int = Field(
        default=0, ge=0, description="Lifetime semantic search query count"
    )
    search_queries_24h: int = Field(default=0, ge=0, description="Search queries in last 24 hours")
    search_queries_1h: int = Field(default=0, ge=0, description="Search queries in last hour")
    # Internal/workshop deployment classification (added in schema v5, issue #1216).
    # Optional so pre-v5 clients (which omit these) still validate.
    internal_only_deployment: bool | None = Field(
        default=None,
        description="Internal/workshop deployment flag. Added in schema v5.",
    )
    internal_deployment_type: str | None = Field(
        default=None,
        pattern=_INTERNAL_DEPLOYMENT_TYPE_PATTERN,
        description="Internal deployment classification. Added in schema v5.",
    )
    ts: str = Field(..., description="ISO 8601 timestamp")

    @field_validator("ts")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {e}") from e
        return v

    @model_validator(mode="after")
    def _validate_cloud_detection_consistency(self) -> "HeartbeatEvent":
        """Reject payloads where cloud and cloud_detection_method disagree."""
        _check_cloud_detection_consistency(self.cloud, self.cloud_detection_method)
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event": "heartbeat",
                "registry_id": "c546a650-8af9-4721-9efb-7df221b2a0d9",
                "v": "1.0.22",
                "cloud": "aws",
                "cloud_detection_method": "imds",
                "compute": "ecs",
                "servers_count": 15,
                "agents_count": 8,
                "skills_count": 23,
                "peers_count": 2,
                "search_backend": "documentdb",
                "embeddings_provider": "sentence-transformers",
                "embeddings_backend_kind": "sentence-transformers",
                "uptime_hours": 48,
                "search_queries_total": 150,
                "search_queries_24h": 12,
                "search_queries_1h": 3,
                "ts": "2026-03-18T12:00:00Z",
            }
        }
    )
