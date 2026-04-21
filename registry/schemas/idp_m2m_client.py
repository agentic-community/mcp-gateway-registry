"""IdP M2M Client model for MongoDB storage.

This module defines the schema for storing M2M client applications
and their group mappings in MongoDB. This allows the registry to track
service accounts from any IdP (Keycloak, Okta, Entra) and their permissions
without hardcoding them in authorization server expressions.

This collection serves as the authorization database for M2M clients.
"""

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

MANUAL_PROVIDER: str = "manual"

_CLIENT_ID_PATTERN: re.Pattern = re.compile(r"^[A-Za-z0-9_\-.:]{1,256}$")


def _validate_client_id(value: str) -> str:
    """Validate that client_id matches the allowed character set."""
    if not _CLIENT_ID_PATTERN.match(value):
        raise ValueError(
            "client_id must match ^[A-Za-z0-9_\\-.:]{1,256}$ "
            "(alphanumerics, dash, underscore, dot, colon only)"
        )
    return value


class IdPM2MClient(BaseModel):
    """IdP M2M client application with group mappings.

    Stores information about M2M service accounts from any identity provider
    including their client IDs, groups, and metadata. This data is used for
    authorization decisions when JWT tokens have empty groups claim.
    """

    client_id: str = Field(..., description="IdP application client ID")
    name: str = Field(..., description="Application name")
    description: str | None = Field(None, description="Application description")
    groups: list[str] = Field(default_factory=list, description="Groups this client belongs to")
    enabled: bool = Field(default=True, description="Whether client is active")
    provider: str = Field(..., description="Identity provider (okta, keycloak, entra, manual)")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When record was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When record was last updated"
    )
    idp_app_id: str | None = Field(None, description="IdP internal app ID")
    created_by: str | None = Field(
        default=None,
        description=(
            "Username of operator who registered this M2M client. "
            "Populated only for records with provider=manual."
        ),
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "client_id": "0oa1100req1AzfKaY698",
                "name": "ai-agent",
                "description": "AI agent with admin access",
                "groups": ["registry-admins"],
                "enabled": True,
                "provider": "okta",
                "idp_app_id": "0oa1100req1AzfKaY698",
            }
        }


class IdPM2MClientUpdate(BaseModel):
    """Payload for updating an IdP M2M client's group mappings (legacy)."""

    groups: list[str] = Field(..., description="New list of groups for this client", min_length=1)
    description: str | None = Field(None, description="Updated description")


class IdPM2MClientCreate(BaseModel):
    """Request body for POST /api/iam/m2m-clients.

    Creates a new M2M client record with provider=manual. Does not require
    any IdP Admin API token.
    """

    client_id: str = Field(
        ...,
        description="IdP application client ID",
        min_length=1,
        max_length=256,
    )
    client_name: str = Field(
        ...,
        description="Human-readable name for the client",
        min_length=1,
        max_length=256,
    )
    groups: list[str] = Field(
        default_factory=list,
        description="Groups this client belongs to (may be empty)",
    )
    description: str | None = Field(
        default=None,
        description="Optional human-readable description",
        max_length=1024,
    )

    @field_validator("client_id")
    @classmethod
    def _validate_client_id_format(cls, v: str) -> str:
        return _validate_client_id(v)


class IdPM2MClientPatch(BaseModel):
    """Request body for PATCH /api/iam/m2m-clients/{client_id}.

    Patch semantics use Pydantic v2's `model_dump(exclude_unset=True)` in the
    service, so fields not present in the request body are NOT written. Fields
    explicitly present (including None or empty list) ARE written.
    """

    client_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    groups: list[str] | None = Field(
        default=None,
        description="New groups list. Empty list clears groups.",
    )
    description: str | None = Field(
        default=None,
        max_length=1024,
    )
    enabled: bool | None = Field(default=None)


class M2MClientListResponse(BaseModel):
    """Paginated response envelope for GET /api/iam/m2m-clients."""

    total: int = Field(..., description="Total number of matching records")
    limit: int = Field(..., description="Limit applied to this page")
    skip: int = Field(..., description="Offset applied to this page")
    items: list[IdPM2MClient] = Field(
        default_factory=list,
        description="Records on this page",
    )
