from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class M2MAccountRequest(BaseModel):
    """Payload for creating a Keycloak service account client."""

    name: str = Field(..., min_length=1)
    groups: list[str] = Field(..., min_length=1)
    description: str | None = None


class HumanUserRequest(BaseModel):
    """Payload for creating a Keycloak human user."""

    username: str = Field(..., min_length=1)
    email: EmailStr
    first_name: str = Field(..., min_length=1, alias="firstname")
    last_name: str = Field(..., min_length=1, alias="lastname")
    groups: list[str] = Field(..., min_length=1)
    password: str | None = Field(
        None, description="Initial password (optional, generated elsewhere)"
    )

    model_config = {"populate_by_name": True}


class UserDeleteResponse(BaseModel):
    """Standard response returned when a Keycloak user is deleted."""

    username: str
    deleted: bool = True


class UserSummary(BaseModel):
    """Subset of user information exposed through the API."""

    id: str
    username: str
    email: str | None = None
    firstName: str | None = None
    lastName: str | None = None
    enabled: bool = True
    groups: list[str] = Field(default_factory=list)


class UserListResponse(BaseModel):
    """Wrapper for list users endpoint."""

    users: list[UserSummary] = Field(default_factory=list)
    total: int


class GroupCreateRequest(BaseModel):
    """Payload for creating a group.

    Note: The backend currently only processes name and description.
    The scope_config field is accepted but not yet wired to
    scope_service.import_group(). Future work should pass
    server_access, group_mappings, and ui_permissions through
    to the scope service when creating a group.
    """

    name: str = Field(..., min_length=1)
    description: str | None = None
    scope_config: dict | None = Field(
        None,
        description="Scope configuration (accepted but not yet applied server-side)",
    )


class GroupSummary(BaseModel):
    """Group information."""

    id: str
    name: str
    path: str
    attributes: dict | None = None
    is_idp_managed: bool | None = Field(
        default=None,
        description=(
            "Whether the group is managed in the upstream identity provider. "
            "None for legacy records that predate the flag; True means "
            "PATCH/DELETE call the IdP, False means local-only. See issue #946."
        ),
    )


class GroupListResponse(BaseModel):
    """Response for listing groups."""

    groups: list[GroupSummary] = Field(default_factory=list)
    total: int


class GroupDeleteResponse(BaseModel):
    """Response when a Keycloak group is deleted."""

    name: str
    deleted: bool = True


class UpdateUserGroupsRequest(BaseModel):
    """Payload for updating a user's group memberships."""

    groups: list[str] = Field(..., description="List of group names to assign")


class UpdateUserGroupsResponse(BaseModel):
    """Response after updating user's group memberships."""

    username: str
    groups: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)


class GroupUpdateRequest(BaseModel):
    """Request to update a group."""

    description: str | None = None
    scope_config: dict | None = Field(
        None,
        description="Scope configuration (server_access, ui_permissions, etc.)",
    )


class GroupDetailResponse(BaseModel):
    """Detailed group information."""

    id: str
    name: str
    path: str | None = None
    description: str | None = None
    server_access: list | None = None
    group_mappings: list | None = None
    ui_permissions: dict | None = None
    agent_access: list | None = None
    is_idp_managed: bool | None = Field(
        default=None,
        description=(
            "Whether the group is managed in the upstream identity provider. "
            "None for legacy records that predate the flag; True means "
            "PATCH/DELETE call the IdP, False means local-only. See issue #946."
        ),
    )
