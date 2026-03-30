"""Okta M2M Client models for API routes.

This module defines the request/response schemas for Okta M2M client
management endpoints.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class OktaM2MClient(BaseModel):
    """Okta M2M client application with group mappings."""

    client_id: str = Field(..., description="Okta application client ID")
    name: str = Field(..., description="Application name/label")
    description: str | None = Field(None, description="Application description")
    groups: list[str] = Field(default_factory=list, description="Groups this client belongs to")
    enabled: bool = Field(default=True, description="Whether client is active")
    okta_app_id: str | None = Field(None, description="Okta internal app ID")
    last_synced: datetime | None = Field(None, description="Last sync timestamp")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When record was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When record was last updated"
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
                "okta_app_id": "0oa1100req1AzfKaY698",
            }
        }


class OktaM2MClientUpdate(BaseModel):
    """Payload for updating an Okta M2M client's group mappings."""

    groups: list[str] = Field(..., description="New list of groups for this client", min_length=1)


class OktaSyncRequest(BaseModel):
    """Request payload for Okta M2M sync."""

    force_full_sync: bool = False


class OktaSyncResponse(BaseModel):
    """Response from Okta M2M sync operation."""

    synced_count: int
    added_count: int
    updated_count: int
    removed_count: int
    errors: list[str]
