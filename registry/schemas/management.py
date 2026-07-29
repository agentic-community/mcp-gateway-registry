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


class ServerAccessRule(BaseModel):
    """One MCP-server access rule inside a scope's server_access list."""

    model_config = {"extra": "forbid"}

    server: str = Field(
        ...,
        min_length=1,
        description="Server name (leading/trailing slashes stripped by the caller).",
    )
    methods: list[str] = Field(
        default_factory=list,
        description="Allowed MCP methods (e.g. tools/call). Empty means none.",
    )
    tools: list[str] | str = Field(
        default_factory=list[str],
        description='Allowed tool names, or "*" for all tools on this server.',
    )


class AgentAccessRule(BaseModel):
    """One A2A-agent access rule inside a scope's server_access list."""

    model_config = {"extra": "forbid"}

    agent: str = Field(..., min_length=1, description="Agent path, or */all.")
    actions: list[str] = Field(
        default_factory=list,
        description="Allowed agent actions (list_agents, get_agent, invoke_agent, ...).",
    )


class ScopeConfig(BaseModel):
    """Validated scope configuration carried on group create/update.

    Replaces the previous free-form dict. Unknown keys are rejected so an
    operator gets a clear 422 instead of a silently-dropped field.

    All access fields default to None (omitted) rather than empty
    collections so the PATCH handler can distinguish "not provided --
    preserve existing values" from "explicitly provided". The create
    handler coerces None to empty collections.
    """

    model_config = {"extra": "forbid"}

    server_access: list[ServerAccessRule | AgentAccessRule] | None = Field(
        default=None,
        description="Per-server / per-agent invocation access rules.",
    )
    ui_permissions: dict[str, list[str]] | None = Field(
        default=None,
        description="UI permission name -> list of entity names/ids ('all' allowed).",
    )
    agent_access: list[str] | None = Field(
        default=None,
        description="Agent paths this scope may access.",
    )
    group_mappings: list[str] | None = Field(
        default=None,
        description="IdP group names/IDs. Defaults to [group_name] when omitted.",
    )
    create_in_idp: bool = Field(
        default=False,
        description="Create the group in the upstream IdP as well (default: local-only).",
    )


class GroupCreateRequest(BaseModel):
    """Payload for creating a group.

    scope_config (server_access, ui_permissions, agent_access, group_mappings,
    create_in_idp) is fully applied server-side via scope_service.import_group,
    and the change takes effect immediately (the handler triggers an
    auth-server scope reload).
    """

    name: str = Field(..., min_length=1)
    description: str | None = None
    scope_config: ScopeConfig | None = Field(
        None,
        description="Validated scope configuration, fully applied server-side.",
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
    scope_config: ScopeConfig | None = Field(
        None,
        description=(
            "Validated scope configuration (server_access, ui_permissions, "
            "agent_access, group_mappings). Omitted fields preserve existing values."
        ),
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
