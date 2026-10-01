"""
Backend session data models for virtual MCP server session management.

Defines schemas for storing and managing per-client backend MCP sessions
in MongoDB. Sessions map a client session ID + backend location to the
backend's MCP session ID, enabling session isolation and persistence.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field, model_validator


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class BackendSessionDocument(BaseModel):
    """MongoDB document for a backend MCP session or stateless initialization.

    Stored with _id = '<client_session_id>:<backend_key>' for fast lookups.
    TTL index on last_used_at auto-expires idle sessions.
    """

    client_session_id: str = Field(
        ...,
        description="Client-facing session ID (e.g., 'vs-abc123')",
    )
    backend_key: str = Field(
        ...,
        description="Backend location key (e.g., '/_vs_backend_weather_')",
    )
    backend_session_id: str | None = Field(
        default=None,
        description="Session ID returned by a stateful backend; null for a stateless backend",
    )
    stateless: bool = Field(default=False, description="Successful sessionless initialize")
    user_id: str = Field(
        ...,
        description="User identity from auth context (for audit)",
    )
    virtual_server_path: str = Field(
        ...,
        description="Virtual server path (e.g., '/virtual/my-server')",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the backend session was first created",
    )
    last_used_at: datetime = Field(
        default_factory=_utc_now,
        description="Last time this session was accessed (drives TTL expiry)",
    )

    @model_validator(mode="after")
    def _validate_state(self) -> "BackendSessionDocument":
        if self.stateless != (self.backend_session_id is None):
            raise ValueError("A stateless backend must not have a backend session ID")
        if self.backend_session_id == "":
            raise ValueError("A backend session ID cannot be empty")
        return self


class ClientSessionDocument(BaseModel):
    """MongoDB document for a client session.

    Stored with _id = 'client:<client_session_id>' for validation lookups.
    TTL index on last_used_at auto-expires idle sessions.
    """

    client_session_id: str = Field(
        ...,
        description="Client-facing session ID (e.g., 'vs-abc123')",
    )
    user_id: str = Field(
        ...,
        description="User identity from auth context",
    )
    virtual_server_path: str = Field(
        ...,
        description="Virtual server path this session was created for",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the client session was created",
    )
    last_used_at: datetime = Field(
        default_factory=_utc_now,
        description="Last time this session was accessed (drives TTL expiry)",
    )


class StoreSessionRequest(BaseModel):
    """Request body for storing an initialized backend session via internal API."""

    backend_session_id: str | None = Field(
        default=None,
        description="Session ID from a stateful backend; null for a stateless backend",
    )
    stateless: bool = Field(default=False, description="Successful sessionless initialize")
    client_session_id: str = Field(..., description="Client-facing session ID")
    user_id: str = Field(
        ...,
        description="User identity from auth context (required)",
    )
    virtual_server_path: str = Field(default="", description="Virtual server path")

    @model_validator(mode="after")
    def _validate_state(self) -> "StoreSessionRequest":
        if self.stateless != (self.backend_session_id is None):
            raise ValueError("A stateless backend must not have a backend session ID")
        if self.backend_session_id == "":
            raise ValueError("A backend session ID cannot be empty")
        return self


class CreateClientSessionRequest(BaseModel):
    """Request body for creating a client session via internal API."""

    user_id: str = Field(
        ...,
        description=(
            "User identity from auth context (required). Refuse to mint a "
            "session with no concrete owner rather than defaulting it."
        ),
    )
    virtual_server_path: str = Field(
        default="",
        description="Virtual server path this session is for",
    )


class CreateClientSessionResponse(BaseModel):
    """Response body after creating a client session."""

    client_session_id: str = Field(
        ...,
        description="Generated client session ID",
    )


class GetBackendSessionResponse(BaseModel):
    """Response body for an initialized backend session lookup."""

    backend_session_id: str | None = Field(default=None, description="Stateful backend session ID")
    stateless: bool = Field(default=False, description="Successful sessionless initialize")

    @model_validator(mode="after")
    def _validate_state(self) -> "GetBackendSessionResponse":
        if self.stateless != (self.backend_session_id is None):
            raise ValueError("A stateless backend must not have a backend session ID")
        if self.backend_session_id == "":
            raise ValueError("A backend session ID cannot be empty")
        return self
