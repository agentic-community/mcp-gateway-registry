"""Pydantic models for mcpgw MCP server.

These models define the data structures returned by the registry API
and used by the MCP tools. Extra fields from the API are silently ignored.
"""

from pydantic import BaseModel, ConfigDict, Field


class ServerInfo(BaseModel):
    """Information about a registered MCP server."""

    model_config = ConfigDict(extra="ignore")

    server_name: str = Field(..., description="Display name of the server")
    path: str = Field(..., description="URL path for the server (e.g., '/fininfo')")
    description: str | None = Field(None, description="Server description")
    enabled: bool = Field(..., description="Whether the server is enabled")
    tags: list[str] = Field(default_factory=list, description="Server tags")
    tool_count: int | None = Field(None, description="Number of tools provided")


class AgentInfo(BaseModel):
    """Information about a registered agent."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Name of the agent")
    description: str | None = Field(None, description="Agent description")
    tags: list[str] = Field(default_factory=list, description="Agent tags")
    path: str | None = Field(None, description="Agent path")


class SkillInfo(BaseModel):
    """Information about a registered skill."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Name of the skill")
    path: str = Field(..., description="Skill path")
    description: str | None = Field(None, description="Skill description")
    tags: list[str] = Field(default_factory=list, description="Skill tags")
    visibility: str | None = Field(None, description="Visibility scope")
    is_enabled: bool = Field(True, description="Whether the skill is enabled")


class ToolSearchResult(BaseModel):
    """Search result for semantic tool search."""

    model_config = ConfigDict(extra="ignore")

    tool_name: str = Field(..., description="Name of the tool")
    server_name: str = Field(..., description="Server providing the tool")
    description: str | None = Field(None, description="Tool description")
    score: float | None = Field(None, description="Relevance score (0-1)")
    path: str | None = Field(None, description="Server path")


class RegistryStats(BaseModel):
    """Registry statistics and health information."""

    model_config = ConfigDict(extra="ignore")

    total_servers: int = Field(..., description="Total number of servers")
    enabled_servers: int | None = Field(None, description="Number of enabled servers")
    total_tools: int | None = Field(None, description="Total number of tools")
    health_status: str = Field(default="unknown", description="Health status")


class ErrorResponse(BaseModel):
    """Error response model."""

    model_config = ConfigDict(extra="ignore")

    error: str = Field(..., description="Error message")
    status: str = Field(default="failed", description="Status indicator")
    details: dict | None = Field(None, description="Additional error details")
