"""Response models for the tool access listing (issue #2102).

Each row describes one tool on one MCP server, OpenAPI MCP server or virtual
MCP server, with the groups that can see it in the registry and the groups that
can call it through the gateway.
"""

from typing import Literal

from pydantic import BaseModel, Field

ServerType = Literal["mcp", "openapi", "virtual"]


class GroupRef(BaseModel):
    """A registry group and the identity provider groups mapped to it."""

    group: str = Field(..., description="Registry group name (the scope document id).")
    idp_groups: list[str] = Field(
        default_factory=list,
        description="Identity provider group names in the group's group_mappings.",
    )
    is_admin: bool = Field(
        default=False,
        description="True when the group makes its members registry admins.",
    )


class AccessCondition(BaseModel):
    """One check a member must pass, and the groups that pass it."""

    label: str = Field(..., description="What the check is, for display.")
    groups: list[str] = Field(
        default_factory=list,
        description="Registry groups that pass this check and are not already in 'alone'.",
    )


class AccessSummary(BaseModel):
    """Who has one kind of access (seeing or calling) to a tool.

    A user has access when they are in any group listed in ``alone``, or when
    they are in at least one group from every entry in ``together``.
    """

    alone: list[str] = Field(
        default_factory=list,
        description="Registry groups that grant this access by themselves.",
    )
    together: list[AccessCondition] = Field(
        default_factory=list,
        description=(
            "Checks a member must pass when no single group covers them all. "
            "Empty when combining groups adds nobody beyond 'alone'."
        ),
    )


class ToolAccessRow(BaseModel):
    """One tool on one server, with who can see it and who can call it."""

    tool_name: str = Field(..., description="Tool name a client calls (alias on a virtual server).")
    server_name: str = Field(..., description="Server display name.")
    server_path: str = Field(..., description="Registry path of the server.")
    server_type: ServerType = Field(..., description="mcp, openapi or virtual.")
    enabled: bool = Field(..., description="Whether the server is enabled.")
    blocked: bool = Field(
        default=False,
        description="True when an admin or a security scan blocked the tool, so nobody can call it.",
    )
    blocked_source: str | None = Field(
        default=None,
        description="'admin' or 'security_scan' when blocked.",
    )
    backend_server_path: str | None = Field(
        default=None,
        description="Virtual servers only: the backend server's path.",
    )
    backend_tool_name: str | None = Field(
        default=None,
        description="Virtual servers only: the tool's name on the backend server.",
    )
    can_see: AccessSummary = Field(
        default_factory=AccessSummary,
        description="Groups whose members see the tool in the registry UI, API and search.",
    )
    can_call: AccessSummary = Field(
        default_factory=AccessSummary,
        description="Groups whose members can call the tool through the gateway.",
    )


class ToolAccessListResponse(BaseModel):
    """A page of the tool access listing."""

    rows: list[ToolAccessRow] = Field(default_factory=list)
    groups: dict[str, GroupRef] = Field(
        default_factory=dict,
        description="Identity provider names and admin flag of every group named on this page.",
    )
    total_count: int = Field(..., description="Rows matching the filters, before paging.")
    limit: int = Field(..., description="Page size used.")
    offset: int = Field(..., description="Offset used.")
    has_next: bool = Field(..., description="True when more rows follow this page.")
    generated_at: str = Field(..., description="ISO-8601 UTC time the rows were computed.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "rows": [
                    {
                        "tool_name": "search_docs",
                        "server_name": "Cloudflare docs",
                        "server_path": "/cloudflare-docs/",
                        "server_type": "mcp",
                        "enabled": True,
                        "blocked": False,
                        "can_see": {"alone": ["docs-readers"], "together": []},
                        "can_call": {"alone": ["docs-readers"], "together": []},
                    }
                ],
                "groups": {
                    "docs-readers": {
                        "group": "docs-readers",
                        "idp_groups": ["eng-docs"],
                        "is_admin": False,
                    }
                },
                "total_count": 1,
                "limit": 50,
                "offset": 0,
                "has_next": False,
                "generated_at": "2026-11-22T18:00:00Z",
            }
        }
    }
