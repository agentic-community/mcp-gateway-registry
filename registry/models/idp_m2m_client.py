"""IdP M2M Client model for MongoDB storage.

This module defines the schema for storing M2M client applications
and their group mappings in MongoDB. This allows the registry to track
service accounts from any IdP (Keycloak, Okta, Entra) and their permissions
without hardcoding them in authorization server expressions.

This collection serves as the authorization database for M2M clients.
"""

from datetime import datetime

from pydantic import BaseModel, Field


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
    provider: str = Field(..., description="Identity provider (okta, keycloak, entra)")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When record was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When record was last updated"
    )
    idp_app_id: str | None = Field(None, description="IdP internal app ID")

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
    """Payload for updating an IdP M2M client's group mappings."""

    groups: list[str] = Field(..., description="New list of groups for this client", min_length=1)
    description: str | None = Field(None, description="Updated description")
