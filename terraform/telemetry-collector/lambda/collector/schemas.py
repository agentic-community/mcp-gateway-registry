"""
Pydantic validation schemas for telemetry events.

Matches schemas from registry/core/telemetry.py (issue #558 client implementation).
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
        pattern="^(aws|gcp|azure|unknown)$",
        description="Cloud provider",
    )
    compute: str = Field(
        default="unknown",
        pattern="^(ecs|eks|kubernetes|docker|ec2|vm|unknown)$",
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
        pattern="^(file|documentdb|mongodb-ce)$",
        description="Storage backend",
    )
    auth: str = Field(..., min_length=1, max_length=50, description="Auth provider")
    federation: bool = Field(..., description="Federation enabled")
    search_queries_total: int = Field(
        default=0, ge=0, description="Lifetime semantic search query count"
    )
    search_queries_24h: int = Field(
        default=0, ge=0, description="Search queries in last 24 hours"
    )
    search_queries_1h: int = Field(
        default=0, ge=0, description="Search queries in last hour"
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event": "startup",
                "registry_id": "c546a650-8af9-4721-9efb-7df221b2a0d9",
                "v": "1.0.16",
                "py": "3.12",
                "os": "linux",
                "arch": "x86_64",
                "cloud": "aws",
                "compute": "ecs",
                "mode": "with-gateway",
                "registry_mode": "full",
                "storage": "documentdb",
                "auth": "keycloak",
                "federation": True,
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
    """

    event: str = Field(..., pattern="^heartbeat$")
    registry_id: str | None = Field(default=None, max_length=36, description="Registry card UUID")
    v: str = Field(..., min_length=1, max_length=200, description="Registry version")
    cloud: str = Field(
        default="unknown",
        pattern="^(aws|gcp|azure|unknown)$",
        description="Cloud provider",
    )
    compute: str = Field(
        default="unknown",
        pattern="^(ecs|eks|kubernetes|docker|ec2|vm|unknown)$",
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
    embeddings_provider: str = Field(..., min_length=1, max_length=100)
    uptime_hours: int = Field(..., ge=0, description="Instance uptime in hours")
    search_queries_total: int = Field(
        default=0, ge=0, description="Lifetime semantic search query count"
    )
    search_queries_24h: int = Field(
        default=0, ge=0, description="Search queries in last 24 hours"
    )
    search_queries_1h: int = Field(
        default=0, ge=0, description="Search queries in last hour"
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event": "heartbeat",
                "registry_id": "c546a650-8af9-4721-9efb-7df221b2a0d9",
                "v": "1.0.16",
                "cloud": "aws",
                "compute": "ecs",
                "servers_count": 15,
                "agents_count": 8,
                "skills_count": 23,
                "peers_count": 2,
                "search_backend": "documentdb",
                "embeddings_provider": "sentence-transformers",
                "uptime_hours": 48,
                "search_queries_total": 150,
                "search_queries_24h": 12,
                "search_queries_1h": 3,
                "ts": "2026-03-18T12:00:00Z",
            }
        }
    )
