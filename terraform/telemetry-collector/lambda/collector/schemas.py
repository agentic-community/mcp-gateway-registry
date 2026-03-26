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
    instance_id: str = Field(..., min_length=36, max_length=36)  # UUID v4
    v: str = Field(..., min_length=1, max_length=50, description="Registry version")
    py: str = Field(..., pattern=r"^\d+\.\d+$", description="Python version (major.minor)")
    os: str = Field(..., pattern="^(linux|darwin|windows)$", description="Operating system")
    arch: str = Field(..., min_length=1, max_length=20, description="CPU architecture")
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
                "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "v": "1.0.16",
                "py": "3.12",
                "os": "linux",
                "arch": "x86_64",
                "mode": "with-gateway",
                "registry_mode": "full",
                "storage": "documentdb",
                "auth": "keycloak",
                "federation": True,
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
    instance_id: str = Field(..., min_length=36, max_length=36)  # UUID v4
    v: str = Field(..., min_length=1, max_length=50, description="Registry version")
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
    search_queries_daily_7d_moving_avg: float | None = Field(
        default=None, description="7-day moving average of daily search queries"
    )
    search_queries_hourly_moving_avg: float | None = Field(
        default=None, description="Moving average of hourly search queries"
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
                "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "v": "1.0.16",
                "servers_count": 15,
                "agents_count": 8,
                "skills_count": 23,
                "peers_count": 2,
                "search_backend": "documentdb",
                "embeddings_provider": "sentence-transformers",
                "uptime_hours": 48,
                "search_queries_daily_7d_moving_avg": None,
                "search_queries_hourly_moving_avg": None,
                "ts": "2026-03-18T12:00:00Z",
            }
        }
    )
