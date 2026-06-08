"""Pydantic models for the upstream MCP Registry server.json schema.

Based on the draft schema at:
https://raw.githubusercontent.com/modelcontextprotocol/registry/main/docs/reference/server-json/draft/server.schema.json
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class McpRegistryEnvironmentVariable(BaseModel):
    """Environment variable declaration for a package."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str = ""
    is_required: bool = Field(default=False, alias="isRequired")
    is_secret: bool = Field(default=False, alias="isSecret")
    default: str | None = None
    format: str | None = None


class McpRegistryPackageTransport(BaseModel):
    """Transport configuration inside a package."""

    type: str
    url: str | None = None


class McpRegistryPackage(BaseModel):
    """Package distribution entry (pypi, npm, oci, etc.)."""

    model_config = ConfigDict(populate_by_name=True)

    registry_type: str = Field(alias="registryType")
    identifier: str
    version: str = "1.0.0"
    transport: McpRegistryPackageTransport | None = None
    runtime_hint: str | None = Field(default=None, alias="runtimeHint")
    environment_variables: list[McpRegistryEnvironmentVariable] = Field(
        default_factory=list, alias="environmentVariables"
    )


class McpRegistryRemoteHeader(BaseModel):
    """HTTP header for a remote endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    value: str
    is_secret: bool = Field(default=False, alias="isSecret")


class McpRegistryRemoteVariable(BaseModel):
    """Variable definition for templated remote URLs/headers."""

    model_config = ConfigDict(populate_by_name=True)

    description: str = ""
    is_required: bool = Field(default=False, alias="isRequired")
    is_secret: bool = Field(default=False, alias="isSecret")


class McpRegistryRemote(BaseModel):
    """Remote endpoint definition (streamable-http, sse, etc.)."""

    type: str
    url: str
    headers: list[McpRegistryRemoteHeader] = Field(default_factory=list)
    variables: dict[str, McpRegistryRemoteVariable] = Field(default_factory=dict)


class McpRegistryRepository(BaseModel):
    """Source code repository reference."""

    url: str
    source: str = ""


class McpRegistryServerJson(BaseModel):
    """Top-level model for the upstream MCP Registry server.json format.

    Supports both pure upstream fields and passthrough gateway fields
    so users can provide a hybrid JSON (upstream spec + registry-specific).
    """

    model_config = ConfigDict(populate_by_name=True)

    # Upstream MCP Registry fields
    schema_url: str | None = Field(default=None, alias="$schema")
    name: str
    title: str | None = None
    description: str = ""
    version: str = "1.0.0"
    repository: McpRegistryRepository | None = None
    packages: list[McpRegistryPackage] = Field(default_factory=list)
    remotes: list[McpRegistryRemote] = Field(default_factory=list)
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")

    # Passthrough gateway fields (override derived values when present)
    path: str | None = None
    deployment: str | None = None
    proxy_pass_url: str | None = None
    transport: str | None = None
    supported_transports: list[str] | None = None
    auth_scheme: str | None = None
    visibility: str | None = None
    allowed_groups: list[str] | None = None
    status: str | None = None
    tags: list[str] | None = None
    num_tools: int | None = None
    tool_list: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
