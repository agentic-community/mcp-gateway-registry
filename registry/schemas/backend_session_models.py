"""
Backend session data models for virtual MCP server session management.

Defines schemas for storing and managing per-client backend MCP sessions
in MongoDB. Sessions map a client session ID + backend location to the
backend's MCP session ID, enabling session isolation and persistence.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class BackendSessionDocument(BaseModel):
    """MongoDB document for a backend MCP session.

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
    backend_session_id: str = Field(
        ...,
        description="Session ID returned by the backend MCP server",
    )
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
    """Request body for storing a backend session via internal API."""

    backend_session_id: str = Field(
        ...,
        description="Session ID from the backend MCP server",
    )
    client_session_id: str = Field(
        ...,
        description="Client-facing session ID",
    )
    user_id: str = Field(
        ...,
        description=(
            "User identity from auth context (required). A session must have a "
            "concrete owner; defaulting this would silently store a wrong-owner "
            "document if a caller omitted it."
        ),
    )
    virtual_server_path: str = Field(
        default="",
        description="Virtual server path",
    )


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
    """Response body for backend session lookup."""

    backend_session_id: str = Field(
        ...,
        description="Backend MCP session ID",
    )
