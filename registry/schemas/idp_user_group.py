"""IdP user-group fallback model for MongoDB storage.

This module defines the schema for storing user-to-group mappings in MongoDB
for IdPs that do NOT carry group memberships in JWTs (e.g. PingFederate today).
The auth server reads from this collection to enrich the user context when the
JWT's groups claim is empty or missing for a configured fallback provider.

This collection is the user-side mirror of ``idp_m2m_clients``: it serves as
the authorization database for human users when group claims aren't available
on the token.

Tracked by issue #1127.
"""

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

MANUAL_PROVIDER: str = "manual"

_USERNAME_PATTERN: re.Pattern = re.compile(r"^[A-Za-z0-9_\-.@]{1,256}$")


def _validate_username(value: str) -> str:
    """Validate that username matches the allowed character set."""
    if not _USERNAME_PATTERN.match(value):
        raise ValueError(
            "username must match ^[A-Za-z0-9_\\-.@]{1,256}$ "
            "(alphanumerics, dash, underscore, dot, at-sign only)"
        )
    return value


class IdPUserGroup(BaseModel):
    """IdP user with fallback group mappings.

    Stores information about human users and their group memberships for
    identity providers that do not include groups in the JWT claims. This
    data is used for authorization decisions when the JWT's groups claim is
    empty or missing for a fallback-enabled provider.
    """

    username: str = Field(..., description="IdP username (sub, email, or login id)")
    groups: list[str] = Field(default_factory=list, description="Groups this user belongs to")
    enabled: bool = Field(default=True, description="Whether the fallback record is active")
    provider: str = Field(
        ...,
        description="Identity provider (pingfederate, okta, keycloak, entra, manual)",
    )
    email: str | None = Field(None, description="User email address (optional)")
    created_by: str | None = Field(
        default=None,
        description=(
            "Username of operator who registered this user-group record. "
            "Populated only for records with provider=manual."
        ),
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When record was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="When record was last updated"
    )

    @field_validator("username")
    @classmethod
    def _validate_username_format(cls, v: str) -> str:
        return _validate_username(v)

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "username": "alice@example.com",
                "groups": ["registry-admins"],
                "enabled": True,
                "provider": "pingfederate",
                "email": "alice@example.com",
            }
        }


class IdPUserGroupCreate(BaseModel):
    """Request body for POST /api/iam/user-groups.

    Creates a new user-group fallback record with provider=manual. Does not
    require any IdP Admin API token.
    """

    username: str = Field(
        ...,
        description="IdP username (sub, email, or login id)",
        min_length=1,
        max_length=256,
    )
    groups: list[str] = Field(
        default_factory=list,
        description="Groups this user belongs to (may be empty)",
    )
    email: str | None = Field(
        default=None,
        description="Optional user email address",
        max_length=512,
    )

    @field_validator("username")
    @classmethod
    def _validate_username_format(cls, v: str) -> str:
        return _validate_username(v)


class IdPUserGroupPatch(BaseModel):
    """Request body for PATCH /api/iam/user-groups/{username}.

    Patch semantics use Pydantic v2's `model_dump(exclude_unset=True)` in the
    service, so fields not present in the request body are NOT written. Fields
    explicitly present (including None or empty list) ARE written.
    """

    groups: list[str] | None = Field(
        default=None,
        description="New groups list. Empty list clears groups.",
    )
    email: str | None = Field(
        default=None,
        max_length=512,
    )
    enabled: bool | None = Field(default=None)


class UserGroupListResponse(BaseModel):
    """Paginated response envelope for GET /api/iam/user-groups."""

    total: int = Field(..., description="Total number of matching records")
    limit: int = Field(..., description="Limit applied to this page")
    skip: int = Field(..., description="Offset applied to this page")
    items: list[IdPUserGroup] = Field(
        default_factory=list,
        description="Records on this page",
    )


class PingFederateUserCreateRequest(BaseModel):
    """Request body for POST /api/iam/user-groups/{username}/pingfederate-user.

    Carries the password an admin chose for the new PingFederate user. The
    password is sent straight through to the PingFederate Simple Password
    Credential Validator (PCV) and is NOT stored in the registry.
    """

    password: str = Field(
        ...,
        min_length=8,
        max_length=256,
        description="Initial password for the PingFederate Simple PCV user.",
    )
